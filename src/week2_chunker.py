# Streamlined GitHub Repository Processor with ChromaDB and LangGraph
# Focus: ChromaDB primary storage, FAISS for specific search needs, LangGraph workflow

import git
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    chunk_id: str
    filename: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str
    size_chars: int
    size_lines: int
    created_at: str
    repo_name: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

class GitHubRepoProcessor:
    """Handles GitHub repository cloning and file processing"""
    
    def __init__(self, base_dir: str = "data/repos"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported file extensions
        self.supported_extensions = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.jsx': 'javascript', '.tsx': 'typescript', '.java': 'java',
            '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp',
            '.cs': 'csharp', '.go': 'go', '.rs': 'rust', '.php': 'php',
            '.rb': 'ruby', '.xml': 'xml', '.html': 'html', '.css': 'css',
            '.sql': 'sql', '.sh': 'bash', '.dockerfile': 'dockerfile'
        }
        
        # Directories to skip
        self.skip_dirs = {
            '.git', 'node_modules', '__pycache__', '.pytest_cache',
            'venv', 'env', '.venv', 'build', 'dist', 'target',
            '.idea', '.vscode', 'coverage', '.next', 'vendor', 'LICENSE', 'README.md', '.md', 'docs'
        }
    
    def clone_repository(self, repo_url: str, local_name: Optional[str] = None) -> Path:
        """Clone a GitHub repository"""
        if local_name is None:
            local_name = repo_url.split('/')[-1].replace('.git', '')
        
        repo_path = self.base_dir / local_name
        
        if repo_path.exists():
            import shutil
            shutil.rmtree(repo_path)
            logger.info(f"Removed existing directory: {repo_path}")
        
        try:
            logger.info(f"Cloning {repo_url} to {repo_path}")
            git.Repo.clone_from(repo_url, repo_path, depth=1)
            logger.info(f"Successfully cloned repository")
            return repo_path
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            raise
    
    def should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed"""
        # Skip ignored directories
        for part in file_path.parts:
            if part in self.skip_dirs:
                return False
        
        # Check file extension
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Skip large files (> 1MB)
        try:
            if file_path.stat().st_size > 1024 * 1024:
                return False
        except:
            return False
        
        return True
    
    def find_processable_files(self, repo_path: Path) -> List[Path]:
        """Find all files that should be processed"""
        processable_files = []
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and self.should_process_file(file_path):
                processable_files.append(file_path)
        
        logger.info(f"Found {len(processable_files)} processable files")
        return processable_files
    
    def read_file_safely(self, file_path: Path) -> Optional[str]:
        """Read file content safely"""
        encodings = ['utf-8', 'latin-1', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                return None
        
        return None
    
    def get_file_language(self, file_path: Path) -> str:
        """Determine the programming language of a file"""
        return self.supported_extensions.get(file_path.suffix.lower(), 'unknown')

class CodeChunker:
    """Handles chunking of code files"""
    
    def __init__(self, chunk_size: int = 75, overlap: int = 10):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def create_chunk_id(self, file_path: str, start_line: int, end_line: int, repo_name: str) -> str:
        """Create a unique ID for a chunk"""
        content = f"{repo_name}:{file_path}:{start_line}-{end_line}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def chunk_by_lines(self, content: str, file_path: Path, language: str, repo_name: str) -> List[CodeChunk]:
        """Chunk content by lines with overlap"""
        lines = content.split('\n')
        chunks = []
        
        if len(lines) <= self.chunk_size:
            # Small file - single chunk
            chunk_id = self.create_chunk_id(str(file_path), 1, len(lines), repo_name)
            chunk = CodeChunk(
                chunk_id=chunk_id,
                filename=file_path.name,
                file_path=str(file_path),
                content=content,
                start_line=1,
                end_line=len(lines),
                language=language,
                chunk_type='complete_file',
                size_chars=len(content),
                size_lines=len(lines),
                created_at=datetime.now().isoformat(),
                repo_name=repo_name
            )
            chunks.append(chunk)
        else:
            # Split into overlapping chunks
            start = 0
            chunk_num = 1
            
            while start < len(lines):
                end = min(start + self.chunk_size, len(lines))
                chunk_lines = lines[start:end]
                chunk_content = '\n'.join(chunk_lines)
                
                chunk_id = self.create_chunk_id(str(file_path), start + 1, end, repo_name)
                chunk = CodeChunk(
                    chunk_id=chunk_id,
                    filename=file_path.name,
                    file_path=str(file_path),
                    content=chunk_content,
                    start_line=start + 1,
                    end_line=end,
                    language=language,
                    chunk_type=f'chunk_{chunk_num}',
                    size_chars=len(chunk_content),
                    size_lines=len(chunk_lines),
                    created_at=datetime.now().isoformat(),
                    repo_name=repo_name
                )
                chunks.append(chunk)
                
                start = end - self.overlap
                chunk_num += 1
                
                if end >= len(lines):
                    break
        
        return chunks

class ChromaDBManager:
    """Manages ChromaDB collections for code chunks"""
    
    def __init__(self, db_path: str = "data/chromadb", collection_name: str = "code_chunks"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="microsoft/codebert-base"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        
        logger.info(f"ChromaDB initialized at {self.db_path}")
    
    def add_chunks(self, chunks: List[CodeChunk]) -> bool:
        """Add chunks to ChromaDB"""
        try:
            ids = [chunk.chunk_id for chunk in chunks]
            documents = []
            metadatas = []
            
            for chunk in chunks:
                # Create document for embedding
                document_text = f"Repository: {chunk.repo_name}\nFile: {chunk.filename}\nLanguage: {chunk.language}\n\n{chunk.content}"
                documents.append(document_text)
                
                # Create metadata
                metadata = {
                    "filename": chunk.filename,
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "size_chars": chunk.size_chars,
                    "size_lines": chunk.size_lines,
                    "created_at": chunk.created_at,
                    "repo_name": chunk.repo_name
                }
                metadatas.append(metadata)
            
            # Add in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))
                self.collection.add(
                    ids=ids[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                logger.info(f"Added batch {i//batch_size + 1} to ChromaDB")
            
            logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}")
            return False
    
    def query_chunks(self, query: str, n_results: int = 5, language_filter: Optional[str] = None) -> Dict:
        """Query chunks from ChromaDB"""
        try:
            where_clause = {}
            if language_filter:
                where_clause["language"] = language_filter
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return {}
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            all_results = self.collection.get()
            
            languages = {}
            repos = {}
            
            for metadata in all_results.get('metadatas', []):
                lang = metadata.get('language', 'unknown')
                repo = metadata.get('repo_name', 'unknown')
                
                languages[lang] = languages.get(lang, 0) + 1
                repos[repo] = repos.get(repo, 0) + 1
            
            return {
                'total_chunks': count,
                'languages': languages,
                'repositories': repos
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

# LangGraph State and Workflow
class ProcessingState(TypedDict):
    """State for the LangGraph processing workflow"""
    repo_url: str
    repo_name: str
    repo_path: Optional[str]
    files_found: int
    chunks_created: int
    chunks_embedded: int
    status: str
    error_message: Optional[str]
    chunks_data: List[Dict]  # Store chunks as dicts for serialization

class GitHubProcessingWorkflow:
    """LangGraph workflow for GitHub repository processing"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.repo_processor = GitHubRepoProcessor(f"{base_dir}/repos")
        self.chunker = CodeChunker()
        self.chroma_manager = ChromaDBManager(f"{base_dir}/chromadb")
        
        # Create workflow
        self.workflow = StateGraph(ProcessingState)
        
        # Add nodes
        self.workflow.add_node("clone_repo", self.clone_repo_node)
        self.workflow.add_node("process_files", self.process_files_node)
        self.workflow.add_node("create_chunks", self.create_chunks_node)
        self.workflow.add_node("embed_chunks", self.embed_chunks_node)
        self.workflow.add_node("finalize", self.finalize_node)
        
        # Add edges
        self.workflow.set_entry_point("clone_repo")
        self.workflow.add_edge("clone_repo", "process_files")
        self.workflow.add_edge("process_files", "create_chunks")
        self.workflow.add_edge("create_chunks", "embed_chunks")
        self.workflow.add_edge("embed_chunks", "finalize")
        self.workflow.add_edge("finalize", END)
        
        # Compile workflow
        self.app = self.workflow.compile()
    
    def clone_repo_node(self, state: ProcessingState) -> ProcessingState:
        """Node 1: Clone the repository"""
        try:
            logger.info(f"Cloning repository: {state['repo_url']}")
            repo_path = self.repo_processor.clone_repository(state['repo_url'], state['repo_name'])
            
            return {
                **state,
                "repo_path": str(repo_path),
                "status": "repo_cloned"
            }
            
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def process_files_node(self, state: ProcessingState) -> ProcessingState:
        """Node 2: Find and process files"""
        try:
            repo_path = Path(state['repo_path'])
            files = self.repo_processor.find_processable_files(repo_path)
            
            logger.info(f"Found {len(files)} processable files")
            
            return {
                **state,
                "files_found": len(files),
                "status": "files_processed"
            }
            
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def create_chunks_node(self, state: ProcessingState) -> ProcessingState:
        """Node 3: Create chunks from files"""
        try:
            repo_path = Path(state['repo_path'])
            files = self.repo_processor.find_processable_files(repo_path)
            
            all_chunks = []
            
            for file_path in files:
                content = self.repo_processor.read_file_safely(file_path)
                if content is None:
                    continue
                
                language = self.repo_processor.get_file_language(file_path)
                relative_path = file_path.relative_to(repo_path)
                
                chunks = self.chunker.chunk_by_lines(content, relative_path, language, state['repo_name'])
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks")
            
            # Convert to dicts for serialization
            chunks_data = [chunk.to_dict() for chunk in all_chunks]
            
            return {
                **state,
                "chunks_created": len(all_chunks),
                "chunks_data": chunks_data,
                "status": "chunks_created"
            }
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def embed_chunks_node(self, state: ProcessingState) -> ProcessingState:
        """Node 4: Embed chunks into ChromaDB"""
        try:
            # Convert back to CodeChunk objects
            chunks = [CodeChunk(**chunk_data) for chunk_data in state['chunks_data']]
            
            logger.info(f"Embedding {len(chunks)} chunks into ChromaDB")
            
            success = self.chroma_manager.add_chunks(chunks)
            
            if success:
                return {
                    **state,
                    "chunks_embedded": len(chunks),
                    "status": "chunks_embedded"
                }
            else:
                return {
                    **state,
                    "status": "error",
                    "error_message": "Failed to embed chunks into ChromaDB"
                }
                
        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def finalize_node(self, state: ProcessingState) -> ProcessingState:
        """Node 5: Finalize processing"""
        try:
            logger.info("Processing completed successfully!")
            logger.info(f"Repository: {state['repo_name']}")
            logger.info(f"Files processed: {state['files_found']}")
            logger.info(f"Chunks created: {state['chunks_created']}")
            logger.info(f"Chunks embedded: {state['chunks_embedded']}")
            
            return {
                **state,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in finalize: {e}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def process_repository(self, repo_url: str, repo_name: Optional[str] = None) -> Dict:
        """Process a complete repository through the LangGraph workflow"""
        if repo_name is None:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        initial_state = {
            "repo_url": repo_url,
            "repo_name": repo_name,
            "repo_path": None,
            "files_found": 0,
            "chunks_created": 0,
            "chunks_embedded": 0,
            "status": "initialized",
            "error_message": None,
            "chunks_data": []
        }
        
        # Run the workflow
        result = self.app.invoke(initial_state)
        
        return result
    
    def query_repository(self, query: str, n_results: int = 5, language_filter: Optional[str] = None) -> Dict:
        """Query the processed repository"""
        return self.chroma_manager.query_chunks(query, n_results, language_filter)
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.chroma_manager.get_collection_stats()

# MCP-Ready Interface
class MCPCodeProcessor:
    """MCP-compatible interface for the code processor"""
    
    def __init__(self, base_dir: str = "data"):
        self.workflow = GitHubProcessingWorkflow(base_dir)
    
    def process_repository(self, repo_url: str, repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a GitHub repository
        
        Args:
            repo_url: GitHub repository URL
            repo_name: Optional local name for the repository
            
        Returns:
            Dictionary with processing results
        """
        try:
            result = self.workflow.process_repository(repo_url, repo_name)
            
            return {
                "success": result["status"] == "completed",
                "repo_name": result["repo_name"],
                "files_processed": result["files_found"],
                "chunks_created": result["chunks_created"],
                "chunks_embedded": result["chunks_embedded"],
                "error": result.get("error_message")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_code(self, query: str, limit: int = 5, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for code chunks
        
        Args:
            query: Search query
            limit: Maximum number of results
            language: Filter by programming language
            
        Returns:
            Dictionary with search results
        """
        try:
            results = self.workflow.query_repository(query, limit, language)
            
            if not results.get('documents'):
                return {"success": True, "results": []}
            
            formatted_results = []
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            ):
                formatted_results.append({
                    "filename": metadata['filename'],
                    "language": metadata['language'],
                    "repo_name": metadata['repo_name'],
                    "lines": f"{metadata['start_line']}-{metadata['end_line']}",
                    "relevance_score": 1 - distance,  # Convert distance to relevance
                    "content_preview": doc[:200] + "..." if len(doc) > 200 else doc
                })
            
            return {
                "success": True,
                "results": formatted_results,
                "total_found": len(formatted_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """
        Get repository processing statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = self.workflow.get_stats()
            
            return {
                "success": True,
                "total_chunks": stats.get("total_chunks", 0),
                "languages": stats.get("languages", {}),
                "repositories": stats.get("repositories", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# CLI Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Code Processor with ChromaDB and LangGraph")
    parser.add_argument("command", choices=["process", "search", "stats"], 
                       help="Command to run")
    parser.add_argument("--repo", type=str, help="GitHub repository URL")
    parser.add_argument("--name", type=str, help="Local repository name")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--language", type=str, help="Filter by programming language")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--base-dir", type=str, default="data", help="Base directory")
    
    args = parser.parse_args()
    
    processor = MCPCodeProcessor(args.base_dir)
    
    if args.command == "process":
        if not args.repo:
            print("Error: --repo is required for process command")
            return
        
        print(f"🚀 Processing repository: {args.repo}")
        result = processor.process_repository(args.repo, args.name)
        
        if result["success"]:
            print(f"✅ Processing completed successfully!")
            print(f"Repository: {result['repo_name']}")
            print(f"Files processed: {result['files_processed']}")
            print(f"Chunks created: {result['chunks_created']}")
            print(f"Chunks embedded: {result['chunks_embedded']}")
        else:
            print(f"❌ Processing failed: {result['error']}")
    
    elif args.command == "search":
        if not args.query:
            print("Error: --query is required for search command")
            return
        
        print(f"🔍 Searching for: '{args.query}'")
        result = processor.search_code(args.query, args.limit, args.language)
        
        if result["success"]:
            results = result["results"]
            print(f"Found {len(results)} results:")
            
            for i, res in enumerate(results, 1):
                print(f"\n{i}. {res['filename']} ({res['language']})")
                print(f"   Repository: {res['repo_name']}")
                print(f"   Lines: {res['lines']}")
                print(f"   Relevance: {res['relevance_score']:.3f}")
                print(f"   Preview: {res['content_preview']}")
        else:
            print(f"❌ Search failed: {result['error']}")
    
    elif args.command == "stats":
        print("📊 Repository Statistics:")
        result = processor.get_repository_stats()
        
        if result["success"]:
            print(f"Total chunks: {result['total_chunks']}")
            print(f"Languages: {result['languages']}")
            print(f"Repositories: {result['repositories']}")
        else:
            print(f"❌ Failed to get stats: {result['error']}")

if __name__ == "__main__":
    main()