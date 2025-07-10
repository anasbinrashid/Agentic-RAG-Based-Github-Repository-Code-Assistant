# Week 2: GitHub Repository Processing with ChromaDB, Sentence-Transformers, and LangGraph - Updated Implementation
# This handles downloading GitHub repos, chunking, ChromaDB storage, and LangGraph EmbedFlow

import git
import os
import json
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import asyncio
from datetime import datetime
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import faiss
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
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
    chunk_type: str  # 'function', 'class', 'module', 'other'
    size_chars: int
    size_lines: int
    created_at: str
    repo_name: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

class GitHubRepoProcessor:
    """Handles GitHub repository cloning and processing"""
    
    def __init__(self, base_dir: str = "data/repos"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # File extensions to process (updated for more languages)
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.md': 'markdown',
            '.txt': 'text',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.sql': 'sql',
            '.sh': 'bash',
            '.dockerfile': 'dockerfile',
            '.toml': 'toml',
            '.ini': 'ini'
        }
        
        # Directories to skip
        self.skip_dirs = {
            '.git', 'node_modules', '__pycache__', '.pytest_cache',
            'venv', 'env', '.venv', 'build', 'dist', 'target',
            '.idea', '.vscode', 'coverage', '.nyc_output', '.next',
            'vendor', 'deps', '_build', 'tmp', 'temp'
        }
    
    def clone_repository(self, repo_url: str, local_name: Optional[str] = None) -> Path:
        """Clone a GitHub repository"""
        if local_name is None:
            # Extract repo name from URL
            local_name = repo_url.split('/')[-1].replace('.git', '')
        
        repo_path = self.base_dir / local_name
        
        # Remove existing directory if it exists
        if repo_path.exists():
            import shutil
            shutil.rmtree(repo_path)
            logger.info(f"Removed existing directory: {repo_path}")
        
        try:
            logger.info(f"Cloning {repo_url} to {repo_path}")
            git.Repo.clone_from(repo_url, repo_path, depth=1)  # Shallow clone
            logger.info(f"Successfully cloned repository to {repo_path}")
            return repo_path
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            raise
    
    def get_file_language(self, file_path: Path) -> str:
        """Determine the programming language of a file"""
        suffix = file_path.suffix.lower()
        
        # Special cases
        if file_path.name.lower() in ['dockerfile', 'makefile', 'readme']:
            return file_path.name.lower()
        
        return self.supported_extensions.get(suffix, 'unknown')
    
    def should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed"""
        # Skip if in ignored directory
        for part in file_path.parts:
            if part in self.skip_dirs:
                return False
        
        # Skip if not supported extension and not special file
        if (file_path.suffix.lower() not in self.supported_extensions and 
            file_path.name.lower() not in ['dockerfile', 'makefile', 'readme']):
            return False
        
        # Skip if file is too large (> 1MB)
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
        """Read file content safely, handling encoding issues"""
        encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                return None
        
        logger.warning(f"Could not read {file_path} with any encoding")
        return None

class SentenceTransformerEmbedder:
    """Handles embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        logger.info(f"Loading sentence-transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return []
    
    def get_embedding_function(self):
        """Get ChromaDB compatible embedding function"""
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )

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
        self.embedding_function = SentenceTransformerEmbedder().get_embedding_function()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        
        logger.info(f"ChromaDB initialized at {self.db_path}")
        logger.info(f"Collection: {self.collection_name}")
    
    def add_chunks(self, chunks: List[CodeChunk]) -> bool:
        """Add chunks to ChromaDB"""
        try:
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = []
            metadatas = []
            
            for chunk in chunks:
                # Create document text for embedding
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
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))
                batch_ids = ids[i:end_idx]
                batch_documents = documents[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                
                logger.info(f"Added batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} to ChromaDB")
            
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
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get language distribution
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
    
    def delete_collection(self) -> bool:
        """Delete the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

class FAISSManager:
    """Manages FAISS index for fast similarity search (stub implementation)"""
    
    def __init__(self, db_path: str = "data/faiss"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.db_path / "index.faiss"
        self.mapping_file = self.db_path / "doc_mapping.json"
        
        self.index = None
        self.doc_mapping = {}
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
        # Initialize embedder
        self.embedder = SentenceTransformerEmbedder()
        
        logger.info(f"FAISS manager initialized at {self.db_path}")
    
    def build_index(self, chunks: List[CodeChunk]) -> bool:
        """Build FAISS index from chunks"""
        try:
            logger.info(f"Building FAISS index for {len(chunks)} chunks")
            
            # Create index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            
            # Generate embeddings and build mapping
            embeddings = []
            doc_mapping = {}
            
            for i, chunk in enumerate(chunks):
                # Create embedding text
                text = f"Repository: {chunk.repo_name}\nFile: {chunk.filename}\nLanguage: {chunk.language}\n\n{chunk.content}"
                
                # Generate embedding
                embedding = self.embedder.embed_text(text)
                if embedding:
                    embeddings.append(embedding)
                    doc_mapping[str(i)] = {
                        'chunk_id': chunk.chunk_id,
                        'filename': chunk.filename,
                        'file_path': chunk.file_path,
                        'language': chunk.language,
                        'repo_name': chunk.repo_name
                    }
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(chunks)} embeddings")
            
            # Add embeddings to index
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings_array)
                
                # Add to index
                self.index.add(embeddings_array)
                
                # Save index and mapping
                faiss.write_index(self.index, str(self.index_file))
                
                with open(self.mapping_file, 'w') as f:
                    json.dump(doc_mapping, f, indent=2)
                
                self.doc_mapping = doc_mapping
                
                logger.info(f"FAISS index built successfully with {len(embeddings)} vectors")
                return True
            else:
                logger.error("No embeddings generated")
                return False
                
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return False
    
    def load_index(self) -> bool:
        """Load existing FAISS index"""
        try:
            if self.index_file.exists() and self.mapping_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                
                with open(self.mapping_file, 'r') as f:
                    self.doc_mapping = json.load(f)
                
                logger.info(f"FAISS index loaded with {self.index.ntotal} vectors")
                return True
            else:
                logger.warning("FAISS index files not found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        try:
            if self.index is None:
                logger.error("FAISS index not loaded")
                return []
            
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            scores, indices = self.index.search(query_vector, k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if str(idx) in self.doc_mapping:
                    result = self.doc_mapping[str(idx)].copy()
                    result['score'] = float(score)
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return []

class CodeChunker:
    """Handles chunking of code files"""
    
    def __init__(self, chunk_size: int = 75, overlap: int = 10):
        self.chunk_size = chunk_size  # lines per chunk
        self.overlap = overlap  # overlapping lines between chunks
    
    def create_chunk_id(self, file_path: str, start_line: int, end_line: int, repo_name: str) -> str:
        """Create a unique ID for a chunk"""
        content = f"{repo_name}:{file_path}:{start_line}-{end_line}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def chunk_by_lines(self, content: str, file_path: Path, language: str, repo_name: str) -> List[CodeChunk]:
        """Chunk content by lines with overlap"""
        lines = content.split('\n')
        chunks = []
        
        if len(lines) <= self.chunk_size:
            # File is small enough to be one chunk
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
                
                # Move start position (with overlap)
                start = end - self.overlap
                chunk_num += 1
                
                # Break if we've reached the end
                if end >= len(lines):
                    break
        
        return chunks

# LangGraph State and Nodes
class EmbedFlowState(TypedDict):
    """State for the LangGraph EmbedFlow - managers are not serializable"""
    chunks: List[Dict]  # Store as dicts instead of CodeChunk objects
    current_chunk_index: int
    embeddings_generated: int
    failed_embeddings: int
    status: str
    error_message: Optional[str]
    # Remove managers from state - they'll be accessed as instance variables

class LangGraphEmbedFlow:
    """LangGraph flow for embedding generation"""
    
    def __init__(self, chroma_manager: ChromaDBManager, faiss_manager: FAISSManager):
        self.chroma_manager = chroma_manager
        self.faiss_manager = faiss_manager
        
        # Create the workflow
        workflow = StateGraph(EmbedFlowState)
        
        # Add nodes
        workflow.add_node("load_chunks", self.load_chunks_node)
        workflow.add_node("embed_to_chroma", self.embed_to_chroma_node)
        workflow.add_node("build_faiss", self.build_faiss_node)
        workflow.add_node("finalize", self.finalize_node)
        
        # Add edges
        workflow.set_entry_point("load_chunks")
        workflow.add_edge("load_chunks", "embed_to_chroma")
        workflow.add_edge("embed_to_chroma", "build_faiss")
        workflow.add_edge("build_faiss", "finalize")
        workflow.add_edge("finalize", END)
        
        # Compile the graph without checkpointer to avoid serialization issues
        self.app = workflow.compile()
    
    def load_chunks_node(self, state: EmbedFlowState) -> EmbedFlowState:
        """Node 1: Load chunks from JSON file"""
        try:
            chunks_file = Path("data/chunks.json")
            if not chunks_file.exists():
                return {
                    **state,
                    "status": "error",
                    "error_message": "Chunks file not found"
                }
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            logger.info(f"Loaded {len(chunks_data)} chunks for embedding")
            
            return {
                **state,
                "chunks": chunks_data,  # Store as dict data
                "current_chunk_index": 0,
                "embeddings_generated": 0,
                "failed_embeddings": 0,
                "status": "loaded"
            }
            
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def embed_to_chroma_node(self, state: EmbedFlowState) -> EmbedFlowState:
        """Node 2: Embed chunks to ChromaDB"""
        try:
            chunks_data = state["chunks"]
            
            # Convert dict data back to CodeChunk objects
            chunks = [CodeChunk(**chunk_data) for chunk_data in chunks_data]
            
            logger.info(f"Adding {len(chunks)} chunks to ChromaDB")
            
            success = self.chroma_manager.add_chunks(chunks)
            
            if success:
                return {
                    **state,
                    "embeddings_generated": len(chunks),
                    "status": "chroma_complete"
                }
            else:
                return {
                    **state,
                    "status": "error",
                    "error_message": "Failed to add chunks to ChromaDB"
                }
                
        except Exception as e:
            logger.error(f"Error embedding to ChromaDB: {e}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def build_faiss_node(self, state: EmbedFlowState) -> EmbedFlowState:
        """Node 3: Build FAISS index"""
        try:
            chunks_data = state["chunks"]
            
            # Convert dict data back to CodeChunk objects
            chunks = [CodeChunk(**chunk_data) for chunk_data in chunks_data]
            
            logger.info(f"Building FAISS index for {len(chunks)} chunks")
            
            success = self.faiss_manager.build_index(chunks)
            
            if success:
                return {
                    **state,
                    "status": "faiss_complete"
                }
            else:
                return {
                    **state,
                    "status": "error",
                    "error_message": "Failed to build FAISS index"
                }
                
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def finalize_node(self, state: EmbedFlowState) -> EmbedFlowState:
        """Node 4: Finalize the embedding process"""
        try:
            embeddings_generated = state["embeddings_generated"]
            failed_embeddings = state["failed_embeddings"]
            
            logger.info(f"Embedding process complete!")
            logger.info(f"Embeddings generated: {embeddings_generated}")
            logger.info(f"Failed embeddings: {failed_embeddings}")
            
            return {
                **state,
                "status": "complete"
            }
            
        except Exception as e:
            logger.error(f"Error in finalize node: {e}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def run_embed_flow(self, chunks: List[CodeChunk]) -> Dict:
        """Run the embedding flow"""
        try:
            # Convert chunks to dict format for serialization
            chunks_data = [chunk.to_dict() for chunk in chunks]
            
            # Initial state
            initial_state = {
                "chunks": chunks_data,
                "current_chunk_index": 0,
                "embeddings_generated": 0,
                "failed_embeddings": 0,
                "status": "initialized",
                "error_message": None
            }
            
            # Run the workflow
            result = self.app.invoke(initial_state)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running embed flow: {e}")
            return {"status": "error", "error_message": str(e)}

class GitHubCodeProcessor:
    """Main processor that orchestrates the entire workflow"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.repo_processor = GitHubRepoProcessor(f"{base_dir}/repos")
        self.chunker = CodeChunker()
        self.chroma_manager = ChromaDBManager(f"{base_dir}/chromadb")
        self.faiss_manager = FAISSManager(f"{base_dir}/faiss")
        
        # Initialize LangGraph EmbedFlow
        self.embed_flow = LangGraphEmbedFlow(self.chroma_manager, self.faiss_manager)
        
        self.chunks_file = self.base_dir / "chunks.json"
        self.metadata_file = self.base_dir / "repo_metadata.json"
    
    def process_repository(self, repo_url: str, repo_name: Optional[str] = None) -> Tuple[int, int]:
        """Process a complete GitHub repository"""
        logger.info(f"Processing repository: {repo_url}")
        
        # Extract repo name if not provided
        if repo_name is None:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        # Step 1: Clone repository
        repo_path = self.repo_processor.clone_repository(repo_url, repo_name)
        
        # Step 2: Find processable files
        files = self.repo_processor.find_processable_files(repo_path)
        
        # Step 3: Process each file
        all_chunks = []
        processed_files = 0
        file_stats = {}
        
        for file_path in files:
            try:
                # Read file content
                content = self.repo_processor.read_file_safely(file_path)
                if content is None:
                    continue
                
                # Determine language
                language = self.repo_processor.get_file_language(file_path)
                
                # Create relative path from repo root
                relative_path = file_path.relative_to(repo_path)
                
                # Chunk the file
                chunks = self.chunker.chunk_by_lines(content, relative_path, language, repo_name)
                all_chunks.extend(chunks)
                processed_files += 1
                
                # Update stats
                file_stats[str(relative_path)] = {
                    'language': language,
                    'chunks': len(chunks),
                    'size': len(content),
                    'lines': len(content.split('\n'))
                }
                
                logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        # Step 4: Save chunks and metadata
        self.save_chunks(all_chunks)
        self.save_metadata(repo_url, repo_name, file_stats, len(all_chunks))
        
        logger.info(f"Processing complete: {processed_files} files, {len(all_chunks)} chunks")
        return processed_files, len(all_chunks)
    
    def save_chunks(self, chunks: List[CodeChunk]):
        """Save chunks to JSON file"""
        chunks_data = [chunk.to_dict() for chunk in chunks]
        
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {self.chunks_file}")
    
    def save_metadata(self, repo_url: str, repo_name: str, file_stats: Dict, total_chunks: int):
        """Save repository metadata"""
        metadata = {
            'repo_url': repo_url,
            'repo_name': repo_name,
            'processed_at': datetime.now().isoformat(),
            'total_chunks': total_chunks,
            'total_files': len(file_stats),
            'file_stats': file_stats,
            'languages': {}
        }
        
        # Count languages
        for file_info in file_stats.values():
            lang = file_info['language']
            if lang not in metadata['languages']:
                metadata['languages'][lang] = {'files': 0, 'chunks': 0}
            metadata['languages'][lang]['files'] += 1
            metadata['languages'][lang]['chunks'] += file_info['chunks']
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metadata to {self.metadata_file}")
    
    def load_chunks(self) -> List[CodeChunk]:
        """Load chunks from JSON file"""
        if not self.chunks_file.exists():
            return []
        
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks = []
        for chunk_data in chunks_data:
            chunk = CodeChunk(**chunk_data)
            chunks.append(chunk)
        
        logger.info(f"Loaded {len(chunks)} chunks from {self.chunks_file}")
        return chunks
    
    def load_metadata(self) -> Dict:
        """Load repository metadata"""
        if not self.metadata_file.exists():
            return {}
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_embedding_flow(self, chunks: Optional[List[CodeChunk]] = None) -> Dict:
        """Run the LangGraph embedding flow"""
        if chunks is None:
            chunks = self.load_chunks()
        
        if not chunks:
            logger.error("No chunks found to embed")
            return {"status": "error", "error_message": "No chunks found"}
        
        # Save chunks to file for LangGraph flow
        self.save_chunks(chunks)
        
        # Run the embedding flow
        result = self.embed_flow.run_embed_flow(chunks)
        
        return result
    
    def query_similar_chunks(self, query: str, n_results: int = 5, language_filter: Optional[str] = None) -> Dict:
        """Query similar chunks using ChromaDB"""
        return self.chroma_manager.query_chunks(query, n_results, language_filter)
    
    def search_with_faiss(self, query: str, k: int = 5) -> List[Dict]:
        """Search using FAISS index"""
        # Try to load existing index
        if not self.faiss_manager.load_index():
            logger.warning("FAISS index not found. Building new index...")
            chunks = self.load_chunks()
            if chunks:
                self.faiss_manager.build_index(chunks)
            else:
                logger.error("No chunks available for FAISS index")
                return []
        
        return self.faiss_manager.search(query, k)
    
    def get_collection_stats(self) -> Dict:
        """Get ChromaDB collection statistics"""
        return self.chroma_manager.get_collection_stats()
    
    def get_processing_stats(self) -> Dict:
        """Get overall processing statistics"""
        chunks = self.load_chunks()
        metadata = self.load_metadata()
        chroma_stats = self.get_collection_stats()
        
        stats = {
            'total_chunks': len(chunks),
            'chroma_chunks': chroma_stats.get('total_chunks', 0),
            'coverage': chroma_stats.get('total_chunks', 0) / len(chunks) if chunks else 0,
            'languages': {},
            'repo_info': metadata,
            'chroma_stats': chroma_stats
        }
        
        # Language breakdown
        for chunk in chunks:
            lang = chunk.language
            if lang not in stats['languages']:
                stats['languages'][lang] = {'count': 0}
            stats['languages'][lang]['count'] += 1
        
        return stats


# CLI Interface for Week 2 - Updated
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Code Processor - Week 2 (ChromaDB + LangGraph)")
    parser.add_argument("command", choices=["process", "embed", "query", "search", "info", "stats", "clear"], 
                       help="Command to run")
    parser.add_argument("--repo", type=str, 
                       help="GitHub repository URL")
    parser.add_argument("--name", type=str, 
                       help="Local repository name")
    parser.add_argument("--base-dir", type=str, default="data",
                       help="Base directory for data")
    parser.add_argument("--chunk-size", type=int, default=75,
                       help="Lines per chunk")
    parser.add_argument("--overlap", type=int, default=10,
                       help="Overlapping lines between chunks")
    parser.add_argument("--query", type=str,
                       help="Search query for ChromaDB/FAISS")
    parser.add_argument("--language", type=str,
                       help="Filter by programming language")
    parser.add_argument("--limit", type=int, default=5,
                       help="Number of results to return")
    parser.add_argument("--use-faiss", action="store_true",
                       help="Use FAISS for search instead of ChromaDB")
    
    args = parser.parse_args()
    
    processor = GitHubCodeProcessor(args.base_dir)
    
    if args.command == "process":
        if not args.repo:
            print("Error: --repo is required for process command")
            return
        
        # Update chunker settings if specified
        if args.chunk_size != 75 or args.overlap != 10:
            processor.chunker = CodeChunker(args.chunk_size, args.overlap)
        
        files_count, chunks_count = processor.process_repository(args.repo, args.name)
        print(f"✅ Processing complete!")
        print(f"Files processed: {files_count}")
        print(f"Chunks created: {chunks_count}")
        print(f"Chunks saved to: {processor.chunks_file}")
        print(f"Metadata saved to: {processor.metadata_file}")
    
    elif args.command == "embed":
        chunks = processor.load_chunks()
        if not chunks:
            print("No chunks found. Run 'process' command first.")
            return
        
        print(f"Running LangGraph embedding flow for {len(chunks)} chunks...")
        result = processor.run_embedding_flow(chunks)
        
        if result.get('status') == 'complete':
            print(f"✅ Embedding flow completed successfully!")
            print(f"Embeddings generated: {result.get('embeddings_generated', 0)}")
            print(f"Failed embeddings: {result.get('failed_embeddings', 0)}")
            
            # Show ChromaDB stats
            stats = processor.get_collection_stats()
            print(f"ChromaDB total chunks: {stats.get('total_chunks', 0)}")
            print(f"Languages: {stats.get('languages', {})}")
        else:
            print(f"❌ Embedding flow failed: {result.get('error_message', 'Unknown error')}")
    
    elif args.command == "query":
        if not args.query:
            print("Error: --query is required for query command")
            return
        
        print(f"Searching for: '{args.query}'")
        if args.use_faiss:
            results = processor.search_with_faiss(args.query, args.limit)
            print(f"Found {len(results)} results using FAISS:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['filename']} ({result['language']})")
                print(f"   Score: {result['score']:.4f}")
                print(f"   Repo: {result['repo_name']}")
        else:
            results = processor.query_similar_chunks(args.query, args.limit, args.language)
            if results.get('documents'):
                print(f"Found {len(results['documents'][0])} results using ChromaDB:")
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                ), 1):
                    print(f"\n{i}. {metadata['filename']} ({metadata['language']})")
                    print(f"   Distance: {distance:.4f}")
                    print(f"   Repo: {metadata['repo_name']}")
                    print(f"   Lines: {metadata['start_line']}-{metadata['end_line']}")
            else:
                print("No results found.")
    
    elif args.command == "search":
        # Alias for query command
        if not args.query:
            print("Error: --query is required for search command")
            return
        
        print(f"Searching for: '{args.query}'")
        results = processor.search_with_faiss(args.query, args.limit)
        print(f"Found {len(results)} results using FAISS:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']} ({result['language']})")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Repo: {result['repo_name']}")
    
    elif args.command == "info":
        chunks = processor.load_chunks()
        if not chunks:
            print("No chunks found.")
            return
        
        metadata = processor.load_metadata()
        
        print(f"Repository Information:")
        if metadata:
            print(f"Repository: {metadata.get('repo_name', 'Unknown')}")
            print(f"URL: {metadata.get('repo_url', 'Unknown')}")
            print(f"Processed: {metadata.get('processed_at', 'Unknown')}")
        
        print(f"\nChunk Statistics:")
        print(f"Total chunks: {len(chunks)}")
        
        # Language breakdown
        languages = {}
        for chunk in chunks:
            lang = chunk.language
            languages[lang] = languages.get(lang, 0) + 1
        
        print(f"Languages: {dict(sorted(languages.items(), key=lambda x: x[1], reverse=True))}")
        
        # ChromaDB stats
        chroma_stats = processor.get_collection_stats()
        if chroma_stats:
            print(f"\nChromaDB Statistics:")
            print(f"Total chunks in ChromaDB: {chroma_stats.get('total_chunks', 0)}")
            print(f"Repositories: {chroma_stats.get('repositories', {})}")
            print(f"Languages: {chroma_stats.get('languages', {})}")
        else:
            print("No ChromaDB data found.")
    
    elif args.command == "stats":
        stats = processor.get_processing_stats()
        print(f"Detailed Statistics:")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"ChromaDB chunks: {stats['chroma_chunks']}")
        print(f"Coverage: {stats['coverage']:.1%}")
        
        print(f"\nLanguage Breakdown:")
        for lang, info in sorted(stats['languages'].items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  {lang}: {info['count']} chunks")
        
        if stats['repo_info']:
            print(f"\nRepository Info:")
            repo_info = stats['repo_info']
            print(f"  Name: {repo_info.get('repo_name', 'Unknown')}")
            print(f"  Files: {repo_info.get('total_files', 0)}")
            print(f"  Processed: {repo_info.get('processed_at', 'Unknown')}")
        
        chroma_stats = stats.get('chroma_stats', {})
        if chroma_stats:
            print(f"\nChromaDB Details:")
            print(f"  Total chunks: {chroma_stats.get('total_chunks', 0)}")
            print(f"  Repositories: {list(chroma_stats.get('repositories', {}).keys())}")
    
    elif args.command == "clear":
        confirm = input("Are you sure you want to clear all ChromaDB data? (y/N): ")
        if confirm.lower() == 'y':
            success = processor.chroma_manager.delete_collection()
            if success:
                print("✅ ChromaDB collection cleared successfully")
            else:
                print("❌ Failed to clear ChromaDB collection")
        else:
            print("Operation cancelled")

if __name__ == "__main__":
    main()