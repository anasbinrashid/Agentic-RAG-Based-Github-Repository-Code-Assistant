# Fixed Code Retrieval System - Better Relevance and Accuracy
# Focus: Clean document format, consistent embeddings, filename-aware scoring

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
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

class QueryProcessor:
    """Simplified query processing for better code search"""
    
    def __init__(self):
        # Focus on essential programming terms
        self.stop_words = {
            'show', 'implementation', 'of', 'the', 'find', 'get', 'how', 'to',
            'what', 'is', 'where', 'when', 'which', 'why', 'code', 'file'
        }
        
        # Patterns for extracting code entities
        self.class_pattern = re.compile(r'\b([A-Z][a-zA-Z0-9_]*)\b')
        self.method_pattern = re.compile(r'\b([a-z][a-zA-Z0-9_]*)\s*\(')
    
    def extract_key_terms(self, query: str) -> Dict[str, List[str]]:
        """Extract key terms from query"""
        terms = {
            'classes': [],
            'methods': [],
            'keywords': [],
            'filename_hints': []
        }
        
        # Extract class names (PascalCase)
        class_matches = self.class_pattern.findall(query)
        terms['classes'] = [match for match in class_matches if match.lower() not in self.stop_words]
        
        # Extract method names (camelCase with parentheses)
        method_matches = self.method_pattern.findall(query)
        terms['methods'] = [match for match in method_matches if match.lower() not in self.stop_words]
        
        # Extract other keywords
        words = re.findall(r'\b\w+\b', query.lower())
        terms['keywords'] = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Check for potential filename hints
        for word in terms['keywords']:
            if any(word in filename for filename in ['.java', '.py', '.js', '.cpp']):
                terms['filename_hints'].append(word)
        
        return terms
    
    def create_search_queries(self, query: str) -> List[str]:
        """Create focused search queries"""
        terms = self.extract_key_terms(query)
        queries = [query]  # Original query first
        
        # Add class-focused query if class names found
        if terms['classes']:
            class_query = ' '.join(terms['classes'])
            queries.append(class_query)
        
        # Add method-focused query if methods found
        if terms['methods']:
            method_query = ' '.join(terms['methods'])
            queries.append(method_query)
        
        # Limit to 3 queries max to avoid noise
        return queries[:3]

class FixedCodeRetriever:
    """Fixed code retrieval with clean document format and better scoring"""
    
    def __init__(self, db_path: str = "data/chromadb", collection_name: str = "code_chunks"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.query_processor = QueryProcessor()
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Use the same embedding model as during indexing
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"  # Same as week2_chunker.py
            )
            
            # Get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def search_with_filename_boost(self, query: str, n_results: int = 5, 
                                  filters: Optional[Dict] = None) -> List[Dict]:
        """Enhanced search with filename-aware scoring"""
        try:
            # Create focused search queries
            search_queries = self.query_processor.create_search_queries(query)
            terms = self.query_processor.extract_key_terms(query)
            
            logger.info(f"Original query: '{query}'")
            logger.info(f"Search queries: {search_queries}")
            logger.info(f"Key terms: {terms}")
            
            # Search with the primary query and get more results for reranking
            results = self.collection.query(
                query_texts=[search_queries[0]],  # Use primary query
                n_results=min(n_results * 4, 50),  # Get more results for reranking
                where=filters
            )
            
            if not results.get('documents') or not results['documents'][0]:
                return []
            
            # Score and rerank results
            scored_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                chunk_id = results['ids'][0][i]
                base_score = 1 - distance  # Convert distance to similarity
                
                # Calculate filename boost
                filename_boost = self._calculate_filename_boost(
                    metadata.get('filename', ''), query, terms
                )
                
                # Calculate content boost
                content_boost = self._calculate_content_boost(
                    doc, query, terms
                )
                
                # Calculate final score
                final_score = base_score * filename_boost * content_boost
                
                scored_results.append({
                    'chunk_id': chunk_id,
                    'score': final_score,
                    'base_score': base_score,
                    'filename_boost': filename_boost,
                    'content_boost': content_boost,
                    'document': doc,
                    'metadata': metadata,
                    'distance': distance
                })
            
            # Sort by final score and format results
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            
            formatted_results = []
            for i, result in enumerate(scored_results[:n_results]):
                formatted_results.append({
                    'rank': i + 1,
                    'chunk_id': result['chunk_id'],
                    'relevance_score': round(result['score'], 3),
                    'filename': result['metadata']['filename'],
                    'file_path': result['metadata']['file_path'],
                    'language': result['metadata']['language'],
                    'repo_name': result['metadata']['repo_name'],
                    'lines': f"{result['metadata']['start_line']}-{result['metadata']['end_line']}",
                    'chunk_type': result['metadata']['chunk_type'],
                    'size_lines': result['metadata']['size_lines'],
                    'content': result['document'],
                    'metadata': result['metadata'],
                    'scoring_details': {
                        'base_score': round(result['base_score'], 3),
                        'filename_boost': round(result['filename_boost'], 3),
                        'content_boost': round(result['content_boost'], 3)
                    }
                })
            
            logger.info(f"Found {len(formatted_results)} results after reranking")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _calculate_filename_boost(self, filename: str, query: str, terms: Dict) -> float:
        """Calculate boost based on filename relevance"""
        if not filename:
            return 1.0
        
        boost = 1.0
        filename_lower = filename.lower()
        query_lower = query.lower()
        
        # Remove extension for comparison
        filename_base = filename_lower.replace('.java', '').replace('.py', '').replace('.js', '').replace('.cpp', '')
        
        # Strong boost for exact filename match
        if query_lower in filename_lower or filename_base in query_lower:
            boost *= 3.0
            logger.info(f"Filename exact match boost: {filename} matches '{query}'")
        
        # Boost for class name in filename
        for class_name in terms['classes']:
            if class_name.lower() in filename_lower:
                boost *= 2.5
                logger.info(f"Class name boost: {class_name} found in {filename}")
        
        # Boost for any keyword in filename
        for keyword in terms['keywords']:
            if keyword in filename_lower:
                boost *= 1.5
                logger.info(f"Keyword boost: {keyword} found in {filename}")
        
        return boost
    
    def _calculate_content_boost(self, document: str, query: str, terms: Dict) -> float:
        """Calculate boost based on content relevance"""
        boost = 1.0
        doc_lower = document.lower()
        
        # Boost for class definitions
        for class_name in terms['classes']:
            if re.search(rf'\bclass\s+{class_name}\b', document, re.IGNORECASE):
                boost *= 1.8
            elif re.search(rf'\b{class_name}\b', doc_lower):
                boost *= 1.3
        
        # Boost for method definitions
        for method_name in terms['methods']:
            if re.search(rf'\b{method_name}\s*\(', document, re.IGNORECASE):
                boost *= 1.4
        
        # Boost for multiple keyword matches
        keyword_matches = sum(1 for keyword in terms['keywords'] if keyword in doc_lower)
        if keyword_matches > 1:
            boost *= (1.1 ** keyword_matches)
        
        return boost
    
    def search(self, query: str, n_results: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Main search interface"""
        return self.search_with_filename_boost(query, n_results, filters)
    
    def search_by_language(self, query: str, language: str, n_results: int = 5) -> List[Dict]:
        """Search filtered by programming language"""
        return self.search(query, n_results, {'language': language})
    
    def search_by_repository(self, query: str, repo_name: str, n_results: int = 5) -> List[Dict]:
        """Search filtered by repository"""
        return self.search(query, n_results, {'repo_name': repo_name})
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Get a specific chunk by its ID"""
        try:
            result = self.collection.get(ids=[chunk_id])
            
            if result.get('documents') and result['documents']:
                return {
                    'chunk_id': chunk_id,
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            all_items = self.collection.get()
            
            stats = {
                'total_chunks': count,
                'languages': {},
                'repositories': {},
                'file_types': {}
            }
            
            for metadata in all_items.get('metadatas', []):
                # Language stats
                lang = metadata.get('language', 'unknown')
                stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
                
                # Repository stats
                repo = metadata.get('repo_name', 'unknown')
                stats['repositories'][repo] = stats['repositories'].get(repo, 0) + 1
                
                # File type stats
                filename = metadata.get('filename', '')
                if '.' in filename:
                    ext = filename.split('.')[-1]
                    stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

class FixedRetrievalInterface:
    """User-friendly interface for fixed retrieval system"""
    
    def __init__(self, db_path: str = "data/chromadb"):
        self.retriever = FixedCodeRetriever(db_path)
    
    def search(self, query: str, n_results: int = 5, language: str = None, repo: str = None, verbose: bool = False):
        """Search with optional filters"""
        if language:
            results = self.retriever.search_by_language(query, language, n_results)
        elif repo:
            results = self.retriever.search_by_repository(query, repo, n_results)
        else:
            results = self.retriever.search(query, n_results)
        
        if verbose:
            self._display_results_verbose(results, query)
        else:
            self._display_results(results, query)
        
        return results
    
    def _display_results(self, results: List[Dict], query: str):
        """Display search results"""
        if not results:
            print(f"\n❌ No results found for query: '{query}'")
            return
        
        print(f"\n✅ Found {len(results)} results for query: '{query}'")
        print("=" * 60)
        
        for result in results:
            print(f"\n{result['rank']}. {result['filename']} ({result['language']})")
            print(f"   Repository: {result['repo_name']}")
            print(f"   Lines: {result['lines']}")
            print(f"   Relevance: {result['relevance_score']}")
            
            # Show content preview
            content = result['content']
            
            # Clean up content display - remove metadata header
            content_lines = content.split('\n')
            code_lines = []
            skip_header = True
            
            for line in content_lines:
                if skip_header:
                    if line.strip() == '' and len(code_lines) == 0:
                        continue
                    if line.startswith('Repository:') or line.startswith('File:') or line.startswith('Language:'):
                        continue
                    skip_header = False
                
                code_lines.append(line)
            
            clean_content = '\n'.join(code_lines)
            
            if len(clean_content) > 200:
                preview = clean_content[:200] + "..."
            else:
                preview = clean_content
            
            print(f"   Preview:")
            print("   " + "─" * 40)
            for line in preview.split('\n')[:6]:
                print(f"   {line}")
            if len(clean_content.split('\n')) > 6:
                print("   ...")
            print("   " + "─" * 40)
    
    def _display_results_verbose(self, results: List[Dict], query: str):
        """Display search results with scoring details"""
        if not results:
            print(f"\n❌ No results found for query: '{query}'")
            return
        
        print(f"\n✅ Found {len(results)} results for query: '{query}'")
        print("=" * 80)
        
        for result in results:
            print(f"\n{result['rank']}. {result['filename']} ({result['language']})")
            print(f"   Repository: {result['repo_name']}")
            print(f"   Lines: {result['lines']}")
            print(f"   Final Score: {result['relevance_score']}")
            
            # Show scoring breakdown
            scoring = result['scoring_details']
            print(f"   Scoring Details:")
            print(f"     - Base Score: {scoring['base_score']}")
            print(f"     - Filename Boost: {scoring['filename_boost']}")
            print(f"     - Content Boost: {scoring['content_boost']}")
            
            # Show preview
            content = result['content']
            content_lines = content.split('\n')
            code_lines = []
            skip_header = True
            
            for line in content_lines:
                if skip_header:
                    if line.strip() == '' and len(code_lines) == 0:
                        continue
                    if line.startswith('Repository:') or line.startswith('File:') or line.startswith('Language:'):
                        continue
                    skip_header = False
                
                code_lines.append(line)
            
            clean_content = '\n'.join(code_lines)
            preview = clean_content[:150] + "..." if len(clean_content) > 150 else clean_content
            
            print(f"   Preview:")
            print("   " + "─" * 40)
            for line in preview.split('\n')[:4]:
                print(f"   {line}")
            print("   " + "─" * 40)

# CLI Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed Code Retrieval System")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--language", type=str, help="Filter by programming language")
    parser.add_argument("--repo", type=str, help="Filter by repository")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--db-path", type=str, default="data/chromadb", help="Database path")
    parser.add_argument("--verbose", action="store_true", help="Show detailed scoring")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    try:
        interface = FixedRetrievalInterface(args.db_path)
        
        if args.stats:
            stats = interface.retriever.get_collection_stats()
            print("\n📊 Collection Statistics")
            print("=" * 30)
            print(f"Total chunks: {stats.get('total_chunks', 0)}")
            
            print(f"\nLanguages:")
            for lang, count in sorted(stats.get('languages', {}).items()):
                print(f"  • {lang}: {count}")
            
            print(f"\nRepositories:")
            for repo, count in sorted(stats.get('repositories', {}).items()):
                print(f"  • {repo}: {count}")
            
            print(f"\nFile types:")
            for ext, count in sorted(stats.get('file_types', {}).items()):
                print(f"  • .{ext}: {count}")
        
        else:
            results = interface.search(
                args.query,
                args.limit,
                args.language,
                args.repo,
                args.verbose
            )
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have processed some repositories first using week2_chunker.py")

if __name__ == "__main__":
    main()