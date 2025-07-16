# Week 3 Retrieval System - Code Chunk Retrieval with ChromaDB
# Focus: Simple, clean retrieval interface for code chunks

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeRetriever:
    """Simple and clean code chunk retrieval system"""
    
    def __init__(self, db_path: str = "data/chromadb", collection_name: str = "code_chunks"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="jinaai/jina-embeddings-v2-base-code"  # Using CodeBERT for code embeddings
            )
            
            # Get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            # Try to get metadata collection
            try:
                self.metadata_collection = self.client.get_collection(
                    name=f"{collection_name}_metadata",
                    embedding_function=self.embedding_function
                )
                logger.info(f"Connected to metadata collection")
            except Exception as e:
                logger.warning(f"Metadata collection not found: {e}")
                self.metadata_collection = None
            
            logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def search(self, query: str, n_results: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search for code chunks based on query
        
        Args:
            query: Search query string
            n_results: Number of results to return
            filters: Optional filters (e.g., {'language': 'python'})
            
        Returns:
            List of search results with metadata
        """
        try:
            # Build where clause from filters
            where_clause = filters if filters else None
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            if results.get('documents') and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'rank': i + 1,
                        'relevance_score': round(1 - distance, 3),  # Convert distance to relevance
                        'filename': metadata['filename'],
                        'file_path': metadata['file_path'],
                        'language': metadata['language'],
                        'repo_name': metadata['repo_name'],
                        'lines': f"{metadata['start_line']}-{metadata['end_line']}",
                        'chunk_type': metadata['chunk_type'],
                        'size_lines': metadata['size_lines'],
                        'content': doc,
                        'metadata': metadata
                    })
            
            logger.info(f"Found {len(formatted_results)} results for query: '{query}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_language(self, query: str, language: str, n_results: int = 5) -> List[Dict]:
        """Search for code chunks filtered by programming language"""
        return self.search(query, n_results, {'language': language})
    
    def search_by_repository(self, query: str, repo_name: str, n_results: int = 5) -> List[Dict]:
        """Search for code chunks filtered by repository"""
        return self.search(query, n_results, {'repo_name': repo_name})
    
    def intelligent_search(self, query: str, n_results: int = 5) -> Dict:
        """
        Intelligent search that considers repository context
        """
        try:
            # First, find relevant repositories if metadata collection exists
            relevant_repos = []
            if self.metadata_collection:
                repo_results = self.metadata_collection.query(
                    query_texts=[query],
                    n_results=3
                )
                
                if repo_results.get('metadatas'):
                    for metadata, distance in zip(repo_results['metadatas'][0], repo_results['distances'][0]):
                        relevant_repos.append({
                            'repo_name': metadata['repo_name'],
                            'relevance_score': 1 - distance,
                            'primary_language': metadata['primary_language'],
                            'description': metadata['description']
                        })
            
            # Search chunks with repository context
            where_clause = None
            if relevant_repos:
                repo_names = [repo['repo_name'] for repo in relevant_repos]
                where_clause = {"repo_name": {"$in": repo_names}}
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )
            
            # Format results with repository context
            formatted_results = []
            if results.get('documents') and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'rank': i + 1,
                        'relevance_score': round(1 - distance, 3),
                        'filename': metadata['filename'],
                        'file_path': metadata['file_path'],
                        'language': metadata['language'],
                        'repo_name': metadata['repo_name'],
                        'lines': f"{metadata['start_line']}-{metadata['end_line']}",
                        'chunk_type': metadata['chunk_type'],
                        'size_lines': metadata['size_lines'],
                        'content': doc,
                        'metadata': metadata,
                        'repo_context': {
                            'total_files': metadata.get('repo_total_files', 0),
                            'languages': metadata.get('repo_languages', []),
                            'dependencies': metadata.get('repo_dependencies', [])
                        }
                    })
            
            return {
                'results': formatted_results,
                'relevant_repositories': relevant_repos,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Intelligent search failed: {e}")
            return {'results': [], 'relevant_repositories': [], 'query': query}
    
    def get_available_languages(self) -> List[str]:
        """Get list of available programming languages"""
        try:
            all_items = self.collection.get()
            languages = {metadata.get('language', 'unknown') for metadata in all_items.get('metadatas', [])}
            return sorted(list(languages))
        except Exception as e:
            logger.error(f"Failed to get languages: {e}")
            return []
    
    def get_available_repositories(self) -> List[str]:
        """Get list of available repositories"""
        try:
            all_items = self.collection.get()
            repositories = {metadata.get('repo_name', 'unknown') for metadata in all_items.get('metadatas', [])}
            return sorted(list(repositories))
        except Exception as e:
            logger.error(f"Failed to get repositories: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get comprehensive collection statistics"""
        try:
            count = self.collection.count()
            all_items = self.collection.get()
            
            stats = {
                'total_chunks': count,
                'languages': {},
                'repositories': {},
                'file_types': {},
                'chunk_types': {}
            }
            
            for metadata in all_items.get('metadatas', []):
                # Language stats
                lang = metadata.get('language', 'unknown')
                stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
                
                # Repository stats
                repo = metadata.get('repo_name', 'unknown')
                stats['repositories'][repo] = stats['repositories'].get(repo, 0) + 1
                
                # File type stats (based on filename extension)
                filename = metadata.get('filename', '')
                if '.' in filename:
                    ext = filename.split('.')[-1]
                    stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
                
                # Chunk type stats
                chunk_type = metadata.get('chunk_type', 'unknown')
                stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

class RetrievalInterface:
    """User-friendly interface for code retrieval"""
    
    def __init__(self, db_path: str = "data/chromadb"):
        self.retriever = CodeRetriever(db_path)
    
    def search_interactive(self):
        """Interactive search interface"""
        print("🔍 Enhanced Code Retrieval System")
        print("=" * 60)
        
        while True:
            print("\nOptions:")
            print("1. Search code")
            print("2. Intelligent search (with repository context)")
            print("3. Search by language")
            print("4. Search by repository")
            print("5. View available languages")
            print("6. View available repositories")
            print("7. View collection stats")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                self._search_code()
            elif choice == '2':
                self._intelligent_search()
            elif choice == '3':
                self._search_by_language()
            elif choice == '4':
                self._search_by_repository()
            elif choice == '5':
                self._show_languages()
            elif choice == '6':
                self._show_repositories()
            elif choice == '7':
                self._show_stats()
            elif choice == '8':
                print("Goodbye! 👋")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _search_code(self):
        """Search for code chunks"""
        query = input("Enter your search query: ").strip()
        if not query:
            print("Query cannot be empty.")
            return
        
        try:
            limit = int(input("Number of results (default 5): ").strip() or "5")
        except ValueError:
            limit = 5
        
        results = self.retriever.search(query, limit)
        self._display_results(results, query)
    
    def _intelligent_search(self):
        """Intelligent search with repository context"""
        query = input("Enter your search query: ").strip()
        if not query:
            print("Query cannot be empty.")
            return
        
        try:
            limit = int(input("Number of results (default 5): ").strip() or "5")
        except ValueError:
            limit = 5
        
        results = self.retriever.intelligent_search(query, limit)
        
        # Display relevant repositories first
        if results.get('relevant_repositories'):
            print(f"\n📚 Relevant Repositories:")
            for repo in results['relevant_repositories']:
                print(f"  • {repo['repo_name']} ({repo['primary_language']}) - Score: {repo['relevance_score']:.3f}")
                print(f"    Description: {repo['description'][:100]}...")
        
        # Display search results
        self._display_intelligent_results(results['results'], query)
    
    def _search_by_language(self):
        """Search filtered by programming language"""
        languages = self.retriever.get_available_languages()
        
        print("\nAvailable languages:")
        for i, lang in enumerate(languages, 1):
            print(f"{i}. {lang}")
        
        try:
            lang_idx = int(input("Select language number: ").strip()) - 1
            if 0 <= lang_idx < len(languages):
                language = languages[lang_idx]
                query = input("Enter your search query: ").strip()
                
                if query:
                    results = self.retriever.search_by_language(query, language)
                    self._display_results(results, query, f"Language: {language}")
            else:
                print("Invalid language selection.")
        except ValueError:
            print("Invalid input.")
    
    def _search_by_repository(self):
        """Search filtered by repository"""
        repositories = self.retriever.get_available_repositories()
        
        print("\nAvailable repositories:")
        for i, repo in enumerate(repositories, 1):
            print(f"{i}. {repo}")
        
        try:
            repo_idx = int(input("Select repository number: ").strip()) - 1
            if 0 <= repo_idx < len(repositories):
                repository = repositories[repo_idx]
                query = input("Enter your search query: ").strip()
                
                if query:
                    results = self.retriever.search_by_repository(query, repository)
                    self._display_results(results, query, f"Repository: {repository}")
            else:
                print("Invalid repository selection.")
        except ValueError:
            print("Invalid input.")
    
    def _show_languages(self):
        """Show available programming languages"""
        languages = self.retriever.get_available_languages()
        print(f"\nAvailable languages ({len(languages)}):")
        for lang in languages:
            print(f"  • {lang}")
    
    def _show_repositories(self):
        """Show available repositories"""
        repositories = self.retriever.get_available_repositories()
        print(f"\nAvailable repositories ({len(repositories)}):")
        for repo in repositories:
            print(f"  • {repo}")
    
    def _show_stats(self):
        """Show collection statistics"""
        stats = self.retriever.get_collection_stats()
        
        print("\n📊 Collection Statistics")
        print("=" * 30)
        print(f"Total chunks: {stats.get('total_chunks', 0)}")
        
        print(f"\nLanguages ({len(stats.get('languages', {}))}):")
        for lang, count in sorted(stats.get('languages', {}).items()):
            print(f"  • {lang}: {count}")
        
        print(f"\nRepositories ({len(stats.get('repositories', {}))}):")
        for repo, count in sorted(stats.get('repositories', {}).items()):
            print(f"  • {repo}: {count}")
        
        print(f"\nFile types ({len(stats.get('file_types', {}))}):")
        for ext, count in sorted(stats.get('file_types', {}).items()):
            print(f"  • .{ext}: {count}")
    
    def _display_results(self, results: List[Dict], query: str, filter_info: str = ""):
        """Display search results in a formatted way"""
        if not results:
            print(f"\n❌ No results found for query: '{query}'")
            if filter_info:
                print(f"Filter: {filter_info}")
            return
        
        print(f"\n✅ Found {len(results)} results for query: '{query}'")
        if filter_info:
            print(f"Filter: {filter_info}")
        print("=" * 60)
        
        for result in results:
            print(f"\n{result['rank']}. {result['filename']} ({result['language']})")
            print(f"   Repository: {result['repo_name']}")
            print(f"   Lines: {result['lines']}")
            print(f"   Relevance: {result['relevance_score']}")
            print(f"   Chunk type: {result['chunk_type']}")
            
            self._show_preview(result['content'])
        
        # Ask if user wants to see full content
        if results:
            try:
                show_full = input("\nShow full content for result number (or press Enter to skip): ").strip()
                if show_full.isdigit():
                    idx = int(show_full) - 1
                    if 0 <= idx < len(results):
                        self._show_full_content(results[idx])
            except (ValueError, KeyboardInterrupt):
                pass
    
    def _display_intelligent_results(self, results: List[Dict], query: str):
        """Display intelligent search results with repository context"""
        if not results:
            print(f"\n❌ No results found for query: '{query}'")
            return
        
        print(f"\n✅ Found {len(results)} results for query: '{query}'")
        print("=" * 80)
        
        for result in results:
            print(f"\n{result['rank']}. {result['filename']} ({result['language']})")
            print(f"   Repository: {result['repo_name']}")
            print(f"   Lines: {result['lines']}")
            print(f"   Relevance: {result['relevance_score']}")
            print(f"   Context: {result['repo_context']['total_files']} files, Languages: {', '.join(result['repo_context']['languages'][:3])}")
            
            self._show_preview(result['content'])
    
    def _show_preview(self, content: str):
        """Show content preview"""
        preview = content[:300] + "..." if len(content) > 300 else content
        
        print(f"   Preview:")
        print("   " + "─" * 40)
        for line in preview.split('\n')[:10]:
            print(f"   {line}")
        if len(content.split('\n')) > 10:
            print("   ...")
        print("   " + "─" * 40)
    
    def _show_full_content(self, result: Dict):
        """Show full content of a specific result"""
        print(f"\n📄 Full content - {result['filename']}")
        print("=" * 60)
        print(f"Repository: {result['repo_name']}")
        print(f"Language: {result['language']}")
        print(f"Lines: {result['lines']}")
        print(f"Path: {result['file_path']}")
        print("─" * 60)
        print(result['content'])
        print("─" * 60)
    
    def quick_search(self, query: str, n_results: int = 5, language: str = None, repo: str = None):
        """Quick search method for programmatic use"""
        if language:
            results = self.retriever.search_by_language(query, language, n_results)
        elif repo:
            results = self.retriever.search_by_repository(query, repo, n_results)
        else:
            results = self.retriever.search(query, n_results)
        
        return results

# CLI Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Retrieval System")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--language", type=str, help="Filter by programming language")
    parser.add_argument("--repo", type=str, help="Filter by repository")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--db-path", type=str, default="data/chromadb", help="Database path")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    parser.add_argument("--list-languages", action="store_true", help="List available languages")
    parser.add_argument("--list-repos", action="store_true", help="List available repositories")
    
    args = parser.parse_args()
    
    try:
        interface = RetrievalInterface(args.db_path)
        
        if args.interactive:
            interface.search_interactive()
        elif args.stats:
            interface._show_stats()
        elif args.list_languages:
            interface._show_languages()
        elif args.list_repos:
            interface._show_repositories()
        elif args.query:
            results = interface.quick_search(
                args.query, 
                args.limit, 
                args.language, 
                args.repo
            )
            interface._display_results(results, args.query)
        else:
            print("Use --query to search, --interactive for interactive mode, or --help for options")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have processed some repositories first using week2_chunker.py")

if __name__ == "__main__":
    main()