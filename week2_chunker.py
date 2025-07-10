# Week 2: GitHub Repository Processing and Code Chunking - Updated for Groq Integration
# This handles downloading GitHub repos and breaking code into chunks

import git
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import httpx  # Changed from requests to httpx for async support
from datetime import datetime
import logging
from groq import Groq
from dotenv import load_dotenv
import asyncio

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
    repo_name: str  # Added repo name for better tracking
    
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

class GroqCodeChunker:
    """Handles chunking of code files with Groq integration"""
    
    def __init__(self, chunk_size: int = 75, overlap: int = 10):
        self.chunk_size = chunk_size  # lines per chunk
        self.overlap = overlap  # overlapping lines between chunks
        self.mcp_base_url = "http://localhost:8000"
        
        # Initialize Groq client for direct API calls if needed
        self.groq_client = None
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.groq_client = Groq(api_key=api_key)
    
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
    
    def chunk_file(self, file_path: Path, content: str, language: str, repo_name: str) -> List[CodeChunk]:
        """Chunk a single file"""
        logger.info(f"Chunking file: {file_path} ({language})")
        
        # For now, use simple line-based chunking
        # In future versions, we could add language-aware chunking
        return self.chunk_by_lines(content, file_path, language, repo_name)
    
    async def get_embedding_via_mcp(self, text: str) -> Optional[List[float]]:
        """Get embedding for text via MCP server (async)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.mcp_base_url}/embed",
                    json={"text": text, "model": "text-embedding-3-small"},
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    return result["embedding"]
                else:
                    logger.error(f"MCP embedding failed: {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"MCP embedding error: {e}")
            return None
    
    def get_embedding_via_mcp_sync(self, text: str) -> Optional[List[float]]:
        """Get embedding for text via MCP server (sync)"""
        try:
            import requests
            response = requests.post(
                f"{self.mcp_base_url}/embed",
                json={"text": text, "model": "text-embedding-3-small"},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result["embedding"]
            else:
                logger.error(f"MCP embedding failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"MCP embedding error: {e}")
            return None
    
    def get_embedding_direct_groq(self, text: str) -> Optional[List[float]]:
        """Get embedding directly from Groq API (if they support embeddings)"""
        if not self.groq_client:
            logger.warning("Groq client not initialized")
            return None
        
        try:
            # Note: This is a placeholder - Groq might not have embeddings API yet
            # For now, this will create a consistent hash-based embedding
            import hashlib
            import numpy as np
            
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            embedding = np.random.uniform(-1, 1, 768).tolist()
            
            return embedding
        except Exception as e:
            logger.error(f"Direct Groq embedding error: {e}")
            return None

class GitHubCodeProcessor:
    """Main processor that orchestrates the entire workflow"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.repo_processor = GitHubRepoProcessor(f"{base_dir}/repos")
        self.chunker = GroqCodeChunker()  # Updated to use Groq chunker
        
        self.chunks_file = self.base_dir / "chunks.json"
        self.embeddings_file = self.base_dir / "embeddings.json"
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
                chunks = self.chunker.chunk_file(relative_path, content, language, repo_name)
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
    
    async def generate_embeddings_async(self, chunks: Optional[List[CodeChunk]] = None) -> Dict[str, List[float]]:
        """Generate embeddings for all chunks via MCP (async)"""
        if chunks is None:
            chunks = self.load_chunks()
        
        embeddings = {}
        failed_count = 0
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks via MCP (async)...")
        
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Create tasks for batch
            tasks = []
            for chunk in batch:
                # Create embedding text (filename + content)
                embedding_text = f"Repository: {chunk.repo_name}\nFile: {chunk.filename}\nLanguage: {chunk.language}\n\n{chunk.content}"
                tasks.append(self.chunker.get_embedding_via_mcp(embedding_text))
            
            # Execute batch
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for chunk, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Embedding failed for chunk {chunk.chunk_id}: {result}")
                    failed_count += 1
                elif result:
                    embeddings[chunk.chunk_id] = result
                else:
                    failed_count += 1
                    logger.warning(f"Failed to get embedding for chunk {chunk.chunk_id}")
            
            # Progress logging
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        
        # Save embeddings
        with open(self.embeddings_file, 'w') as f:
            json.dump(embeddings, f, indent=2)
        
        logger.info(f"Generated {len(embeddings)} embeddings, {failed_count} failed")
        return embeddings
    
    def generate_embeddings_sync(self, chunks: Optional[List[CodeChunk]] = None) -> Dict[str, List[float]]:
        """Generate embeddings for all chunks via MCP (sync)"""
        if chunks is None:
            chunks = self.load_chunks()
        
        embeddings = {}
        failed_count = 0
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks via MCP (sync)...")
        
        for i, chunk in enumerate(chunks):
            # Create embedding text (filename + content)
            embedding_text = f"Repository: {chunk.repo_name}\nFile: {chunk.filename}\nLanguage: {chunk.language}\n\n{chunk.content}"
            
            # Get embedding via MCP
            embedding = self.chunker.get_embedding_via_mcp_sync(embedding_text)
            
            if embedding:
                embeddings[chunk.chunk_id] = embedding
            else:
                failed_count += 1
                logger.warning(f"Failed to get embedding for chunk {chunk.chunk_id}")
            
            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} embeddings")
        
        # Save embeddings
        with open(self.embeddings_file, 'w') as f:
            json.dump(embeddings, f, indent=2)
        
        logger.info(f"Generated {len(embeddings)} embeddings, {failed_count} failed")
        return embeddings
    
    def load_embeddings(self) -> Dict[str, List[float]]:
        """Load embeddings from JSON file"""
        if not self.embeddings_file.exists():
            return {}
        
        with open(self.embeddings_file, 'r') as f:
            embeddings = json.load(f)
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {self.embeddings_file}")
        return embeddings
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        chunks = self.load_chunks()
        embeddings = self.load_embeddings()
        metadata = self.load_metadata()
        
        stats = {
            'total_chunks': len(chunks),
            'total_embeddings': len(embeddings),
            'coverage': len(embeddings) / len(chunks) if chunks else 0,
            'languages': {},
            'repo_info': metadata
        }
        
        # Language breakdown
        for chunk in chunks:
            lang = chunk.language
            if lang not in stats['languages']:
                stats['languages'][lang] = {'count': 0, 'with_embeddings': 0}
            stats['languages'][lang]['count'] += 1
            if chunk.chunk_id in embeddings:
                stats['languages'][lang]['with_embeddings'] += 1
        
        return stats

# CLI Interface for Week 2 - Updated
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Code Processor - Week 2 (Groq Integration)")
    parser.add_argument("command", choices=["process", "embed", "embed-async", "info", "stats"], 
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
    
    args = parser.parse_args()
    
    processor = GitHubCodeProcessor(args.base_dir)
    
    if args.command == "process":
        if not args.repo:
            print("Error: --repo is required for process command")
            return
        
        # Update chunker settings if specified
        if args.chunk_size != 75 or args.overlap != 10:
            processor.chunker = GroqCodeChunker(args.chunk_size, args.overlap)
        
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
        
        embeddings = processor.generate_embeddings_sync(chunks)
        print(f"✅ Embeddings generated (sync)!")
        print(f"Embeddings created: {len(embeddings)}")
        print(f"Embeddings saved to: {processor.embeddings_file}")
    
    elif args.command == "embed-async":
        chunks = processor.load_chunks()
        if not chunks:
            print("No chunks found. Run 'process' command first.")
            return
        
        embeddings = asyncio.run(processor.generate_embeddings_async(chunks))
        print(f"✅ Embeddings generated (async)!")
        print(f"Embeddings created: {len(embeddings)}")
        print(f"Embeddings saved to: {processor.embeddings_file}")
    
    elif args.command == "info":
        chunks = processor.load_chunks()
        if not chunks:
            print("No chunks found.")
            return
        
        metadata = processor.load_metadata()
        
        print(f"📊 Repository Information:")
        if metadata:
            print(f"Repository: {metadata.get('repo_name', 'Unknown')}")
            print(f"URL: {metadata.get('repo_url', 'Unknown')}")
            print(f"Processed: {metadata.get('processed_at', 'Unknown')}")
        
        print(f"\n📋 Chunk Statistics:")
        print(f"Total chunks: {len(chunks)}")
        
        # Language breakdown
        languages = {}
        for chunk in chunks:
            lang = chunk.language
            languages[lang] = languages.get(lang, 0) + 1
        
        print(f"Languages: {dict(sorted(languages.items(), key=lambda x: x[1], reverse=True))}")
        
        # Check for embeddings
        embeddings = processor.load_embeddings()
        if embeddings:
            print(f"Embeddings available: {len(embeddings)}")
            coverage = len(embeddings) / len(chunks) * 100
            print(f"Coverage: {coverage:.1f}%")
        else:
            print("No embeddings found.")
    
    elif args.command == "stats":
        stats = processor.get_statistics()
        print(f"📊 Detailed Statistics:")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total embeddings: {stats['total_embeddings']}")
        print(f"Coverage: {stats['coverage']:.1%}")
        
        print(f"\n🔤 Language Breakdown:")
        for lang, info in sorted(stats['languages'].items(), key=lambda x: x[1]['count'], reverse=True):
            coverage = info['with_embeddings'] / info['count'] * 100 if info['count'] > 0 else 0
            print(f"  {lang}: {info['count']} chunks ({coverage:.1f}% embedded)")
        
        if stats['repo_info']:
            print(f"\n🏗️  Repository Info:")
            repo_info = stats['repo_info']
            print(f"  Name: {repo_info.get('repo_name', 'Unknown')}")
            print(f"  Files: {repo_info.get('total_files', 0)}")
            print(f"  Processed: {repo_info.get('processed_at', 'Unknown')}")

if __name__ == "__main__":
    main()
