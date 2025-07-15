# Week 4 Agent System - LangGraph Agent with Memory and Intelligent Ranking
# Focus: Agent workflow with persistent memory, repository metadata, and intelligent ranking

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any, TypedDict
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import hashlib

# ChromaDB and retrieval imports
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# Groq API imports
import requests
from dotenv import load_dotenv

# Import our existing components
from week3_retrieval import CodeRetriever

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agent State Definition
class AgentState(TypedDict):
    """State for the LangGraph agent workflow"""
    user_query: str
    query_embedding: Optional[List[float]]
    retrieved_chunks: List[Dict]
    ranked_chunks: List[Dict]
    reasoning_trace: List[str]
    scratchpad: str
    tool_calls: List[Dict]
    final_answer: str
    conversation_history: List[Dict]
    metadata_context: Dict
    error_message: Optional[str]
    current_step: str

@dataclass
class RepositoryMetadata:
    """Repository metadata for intelligent ranking"""
    repo_name: str
    languages: Dict[str, int]  # language -> chunk count
    file_structure: Dict[str, List[str]]  # directory -> files
    total_chunks: int
    main_language: str
    project_type: str  # inferred from structure
    key_files: List[str]  # important files like README, main.py, etc.
    dependencies: List[str]  # extracted from package files
    last_updated: str
    main_functions: List[Dict] = None  # NEW: Track actual main functions
    entry_points: List[Dict] = None    # NEW: Track entry points
    
    def __post_init__(self):
        if self.main_functions is None:
            self.main_functions = []
        if self.entry_points is None:
            self.entry_points = []
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_main_function_files(self) -> List[str]:
        """Get files that contain main functions"""
        return [mf['file'] for mf in self.main_functions]
    
    def get_main_function_chunks(self) -> List[str]:
        """Get chunk IDs that contain main functions"""
        return [mf['chunk_id'] for mf in self.main_functions if mf.get('chunk_id')]
    
    def has_main_functions(self) -> bool:
        """Check if repository has any main functions"""
        return len(self.main_functions) > 0

class RepositoryAnalyzer:
    """Analyzes repository structure and creates metadata for intelligent ranking"""
    
    def __init__(self, db_path: str = "data/chromadb"):
        self.retriever = CodeRetriever(db_path)
        self.metadata_path = Path("data/repo_metadata.json")
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
    def analyze_repository(self, repo_name: str) -> RepositoryMetadata:
        """Analyze a repository and create metadata"""
        try:
            # Get all chunks for this repository
            all_chunks = self.retriever.search("", n_results=10000, filters={"repo_name": repo_name})
            
            if not all_chunks:
                raise ValueError(f"No chunks found for repository: {repo_name}")
            
            # Analyze languages
            languages = {}
            file_structure = {}
            key_files = []
            dependencies = []
            main_functions = []  # Track actual main functions
            entry_points = []    # Track entry points
            
            for chunk in all_chunks:
                # Language analysis
                lang = chunk['language']
                languages[lang] = languages.get(lang, 0) + 1
                
                # File structure analysis
                file_path = chunk['file_path']
                directory = str(Path(file_path).parent)
                if directory not in file_structure:
                    file_structure[directory] = []
                if chunk['filename'] not in file_structure[directory]:
                    file_structure[directory].append(chunk['filename'])
                
                # Key files identification
                filename = chunk['filename'].lower()
                if any(key in filename for key in ['readme', 'main', 'index', 'app', 'server', 'client']):
                    if chunk['filename'] not in key_files:
                        key_files.append(chunk['filename'])
                
                # ENHANCED: Detect actual main functions and entry points
                content = chunk['content'].lower()
                
                # Java main method detection
                if 'public static void main(string[] args)' in content:
                    main_functions.append({
                        'file': chunk['filename'],
                        'type': 'java_main',
                        'chunk_id': chunk.get('id', ''),
                        'line_start': chunk.get('line_start', 0)
                    })
                
                # Python main detection
                if 'if __name__ == "__main__"' in content:
                    main_functions.append({
                        'file': chunk['filename'],
                        'type': 'python_main',
                        'chunk_id': chunk.get('id', ''),
                        'line_start': chunk.get('line_start', 0)
                    })
                
                # C/C++ main detection
                if any(pattern in content for pattern in ['int main(', 'void main(', 'int main (', 'void main (']):
                    main_functions.append({
                        'file': chunk['filename'],
                        'type': 'c_main',
                        'chunk_id': chunk.get('id', ''),
                        'line_start': chunk.get('line_start', 0)
                    })
                
                # JavaScript/Node.js entry patterns
                if any(pattern in content for pattern in ['process.argv', 'require.main === module', 'exports.main']):
                    entry_points.append({
                        'file': chunk['filename'],
                        'type': 'js_entry',
                        'chunk_id': chunk.get('id', ''),
                        'line_start': chunk.get('line_start', 0)
                    })
                
                # Spring Boot main class detection
                if '@springbootapplication' in content and 'springapplication.run' in content:
                    main_functions.append({
                        'file': chunk['filename'],
                        'type': 'springboot_main',
                        'chunk_id': chunk.get('id', ''),
                        'line_start': chunk.get('line_start', 0)
                    })
                
                # Dependencies extraction
                if filename in ['requirements.txt', 'package.json', 'setup.py', 'pom.xml', 'cargo.toml']:
                    deps = self._extract_dependencies(chunk['content'], filename)
                    dependencies.extend(deps)
            
            # Determine main language and project type
            main_language = max(languages, key=languages.get) if languages else 'unknown'
            project_type = self._infer_project_type(file_structure, main_language)
            
            metadata = RepositoryMetadata(
                repo_name=repo_name,
                languages=languages,
                file_structure=file_structure,
                total_chunks=len(all_chunks),
                main_language=main_language,
                project_type=project_type,
                key_files=key_files,
                dependencies=list(set(dependencies)),
                last_updated=datetime.now().isoformat(),
                main_functions=main_functions,  # NEW: Store actual main functions
                entry_points=entry_points       # NEW: Store entry points
            )
            
            # Save metadata
            self._save_metadata(metadata)
            
            logger.info(f"Analyzed repository: {repo_name}")
            logger.info(f"Main language: {main_language}, Project type: {project_type}")
            logger.info(f"Total chunks: {len(all_chunks)}, Key files: {len(key_files)}")
            logger.info(f"Found {len(main_functions)} main functions: {[mf['file'] for mf in main_functions]}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error analyzing repository {repo_name}: {e}")
            raise
    
    def _extract_dependencies(self, content: str, filename: str) -> List[str]:
        """Extract dependencies from package files"""
        dependencies = []
        
        try:
            if filename == 'requirements.txt':
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                        if dep:
                            dependencies.append(dep)
            
            elif filename == 'package.json':
                import json
                data = json.loads(content)
                deps = data.get('dependencies', {})
                dev_deps = data.get('devDependencies', {})
                dependencies.extend(list(deps.keys()) + list(dev_deps.keys()))
            
            elif filename == 'setup.py':
                # Simple regex-based extraction for setup.py
                import re
                install_requires = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if install_requires:
                    deps_str = install_requires.group(1)
                    deps = re.findall(r'[\'"]([^\'"]+)[\'"]', deps_str)
                    dependencies.extend(deps)
        
        except Exception as e:
            logger.warning(f"Failed to extract dependencies from {filename}: {e}")
        
        return dependencies
    
    def _infer_project_type(self, file_structure: Dict, main_language: str) -> str:
        """Infer project type from file structure and main language"""
        all_files = []
        for files in file_structure.values():
            all_files.extend(files)
        
        file_names = [f.lower() for f in all_files]
        
        # Web application patterns
        if any(f in file_names for f in ['app.py', 'main.py', 'server.py', 'wsgi.py', 'asgi.py']):
            if main_language == 'python':
                return 'python_web_app'
        
        if any(f in file_names for f in ['package.json', 'index.html', 'app.js', 'server.js']):
            return 'web_application'
        
        # Mobile app patterns
        if any(f in file_names for f in ['androidmanifest.xml', 'mainactivity.java']):
            return 'android_app'
        
        if any(f in file_names for f in ['info.plist', 'appdelegate.swift']):
            return 'ios_app'
        
        # Data science patterns
        if any(f in file_names for f in ['jupyter', 'notebook', 'analysis', 'model']):
            return 'data_science'
        
        # Library patterns
        if any(f in file_names for f in ['setup.py', '__init__.py', 'lib', 'library']):
            return 'library'
        
        # CLI tool patterns
        if any(f in file_names for f in ['cli.py', 'command', 'tool']):
            return 'cli_tool'
        
        # Default based on language
        language_defaults = {
            'python': 'python_project',
            'javascript': 'javascript_project',
            'java': 'java_project',
            'cpp': 'cpp_project',
            'c': 'c_project',
            'go': 'go_project',
            'rust': 'rust_project'
        }
        
        return language_defaults.get(main_language, 'general_project')
    
    def _save_metadata(self, metadata: RepositoryMetadata):
        """Save repository metadata to file"""
        try:
            # Load existing metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = {}
            
            # Update with new metadata
            all_metadata[metadata.repo_name] = metadata.to_dict()
            
            # Save back to file
            with open(self.metadata_path, 'w') as f:
                json.dump(all_metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def get_metadata(self, repo_name: str) -> Optional[RepositoryMetadata]:
        """Get repository metadata"""
        try:
            if not self.metadata_path.exists():
                return None
            
            with open(self.metadata_path, 'r') as f:
                all_metadata = json.load(f)
            
            if repo_name in all_metadata:
                return RepositoryMetadata(**all_metadata[repo_name])
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return None

class IntelligentRanker:
    """Intelligent ranking system for retrieved chunks"""
    
    def __init__(self, analyzer: RepositoryAnalyzer):
        self.analyzer = analyzer
    
    def rank_chunks(self, chunks: List[Dict], query: str, repo_metadata: RepositoryMetadata) -> List[Dict]:
        """Rank chunks based on relevance, metadata, and context"""
        if not chunks:
            return chunks
        
        # Calculate ranking scores
        ranked_chunks = []
        
        for chunk in chunks:
            base_score = chunk.get('relevance_score', 0)
            
            # Metadata-based scoring
            metadata_score = self._calculate_metadata_score(chunk, repo_metadata)
            
            # Query-context scoring
            context_score = self._calculate_context_score(chunk, query, repo_metadata)
            
            # File importance scoring
            importance_score = self._calculate_importance_score(chunk, repo_metadata)
            
            # NEW: Main function priority scoring
            main_function_score = self._calculate_main_function_score(chunk, query, repo_metadata)
            
            # Combined score with adjusted weights
            final_score = (
                base_score * 0.3 +               # Base semantic similarity
                metadata_score * 0.15 +          # Metadata relevance
                context_score * 0.25 +           # Query context
                importance_score * 0.1 +         # File importance
                main_function_score * 0.2        # NEW: Main function priority
            )
            
            chunk_copy = chunk.copy()
            chunk_copy['final_score'] = final_score
            chunk_copy['scoring_breakdown'] = {
                'base_score': base_score,
                'metadata_score': metadata_score,
                'context_score': context_score,
                'importance_score': importance_score,
                'main_function_score': main_function_score
            }
            
            ranked_chunks.append(chunk_copy)
        
        # Sort by final score
        ranked_chunks.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Update rank positions
        for i, chunk in enumerate(ranked_chunks):
            chunk['rank'] = i + 1
        
        return ranked_chunks
    
    def _calculate_main_function_score(self, chunk: Dict, query: str, repo_metadata: RepositoryMetadata) -> float:
        """NEW: Calculate score based on main function presence and query intent"""
        score = 0.0
        query_lower = query.lower()
        
        # Check if query is looking for main functions
        main_keywords = ['main', 'entry', 'start', 'beginning', 'program entry', 'main function', 'main method']
        is_main_query = any(keyword in query_lower for keyword in main_keywords)
        
        if is_main_query:
            # Check if this chunk contains a main function
            chunk_id = chunk.get('id', '')
            chunk_filename = chunk.get('filename', '')
            
            # Direct match with known main function chunks
            main_chunk_ids = repo_metadata.get_main_function_chunks()
            if chunk_id in main_chunk_ids:
                score += 0.8  # Very high score for actual main function chunks
            
            # Check if file contains main functions
            main_files = repo_metadata.get_main_function_files()
            if chunk_filename in main_files:
                score += 0.6  # High score for files with main functions
            
            # Content-based main function detection (fallback)
            content_lower = chunk['content'].lower()
            main_patterns = [
                'public static void main(string[] args)',
                'if __name__ == "__main__"',
                'int main(',
                'void main(',
                'springapplication.run',
                'process.argv'
            ]
            
            for pattern in main_patterns:
                if pattern in content_lower:
                    score += 0.5
                    break
            
            # Bonus for files with "main" in name
            if 'main' in chunk_filename.lower():
                score += 0.3
        
        # For non-main queries, slightly penalize main function chunks
        # as they might be less relevant for specific feature queries
        elif not is_main_query:
            chunk_id = chunk.get('id', '')
            main_chunk_ids = repo_metadata.get_main_function_chunks()
            if chunk_id in main_chunk_ids:
                score -= 0.1  # Small penalty for main functions in feature queries
        
        return min(score, 1.0)
    
    def _calculate_context_score(self, chunk: Dict, query: str, repo_metadata: RepositoryMetadata) -> float:
        """Calculate score based on query context - ENHANCED"""
        score = 0.0
        query_lower = query.lower()
        content_lower = chunk['content'].lower()
        filename_lower = chunk['filename'].lower()
        
        # Filename relevance
        query_words = query_lower.split()
        for word in query_words:
            if word in filename_lower:
                score += 0.2
        
        # Content keyword matching
        important_keywords = ['class', 'function', 'def', 'import', 'from', 'main', 'init']
        for keyword in important_keywords:
            if keyword in query_lower and keyword in content_lower:
                score += 0.1
        
        # Code structure relevance
        if 'function' in query_lower or 'method' in query_lower:
            if any(pattern in content_lower for pattern in ['def ', 'function ', 'method ', 'public ', 'private ']):
                score += 0.3
        
        if 'class' in query_lower:
            if any(pattern in content_lower for pattern in ['class ', 'interface ', 'struct ']):
                score += 0.3
        
        # ENHANCED: Main function specific scoring with better patterns
        if any(main_word in query_lower for main_word in ['main', 'entry', 'start']):
            # High-confidence main function patterns
            high_conf_patterns = [
                'public static void main(string[] args)',
                'public static void main(string args[])',
                'if __name__ == "__main__"',
                'int main(int argc, char *argv[])',
                'int main(void)',
                'springapplication.run'
            ]
            
            for pattern in high_conf_patterns:
                if pattern in content_lower:
                    score += 0.6  # High bonus for definitive main functions
                    break
            
            # Medium-confidence patterns
            med_conf_patterns = ['main(', 'main (', 'main\t(']
            for pattern in med_conf_patterns:
                if pattern in content_lower:
                    score += 0.4
                    break
            
            # File name bonus
            if 'main' in filename_lower:
                score += 0.3
        
        # NEW: Repository-specific context
        if repo_metadata.project_type == 'java_project':
            if 'main' in query_lower and 'public static void main' in content_lower:
                score += 0.4
        
        if repo_metadata.project_type == 'python_project':
            if 'main' in query_lower and '__main__' in content_lower:
                score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_metadata_score(self, chunk: Dict, repo_metadata: RepositoryMetadata) -> float:
        """Calculate score based on repository metadata"""
        score = 0.0
        
        # Language preference (main language gets bonus)
        if chunk['language'] == repo_metadata.main_language:
            score += 0.3
        
        # File structure relevance
        file_path = chunk['file_path']
        if any(key_dir in file_path for key_dir in ['src', 'lib', 'main', 'core']):
            score += 0.2
        
        # Avoid test files unless specifically asked
        if any(test_dir in file_path.lower() for test_dir in ['test', 'spec', '__test__']):
            score -= 0.1
        
        return min(score, 1.0)
    
    def _calculate_importance_score(self, chunk: Dict, repo_metadata: RepositoryMetadata) -> float:
        """Calculate score based on file importance"""
        score = 0.0
        filename = chunk['filename'].lower()
        
        # Key files get higher scores
        if chunk['filename'] in repo_metadata.key_files:
            score += 0.5
        
        # NEW: Main function files get importance boost
        if chunk['filename'] in repo_metadata.get_main_function_files():
            score += 0.4
        
        # Important file patterns
        important_patterns = {
            'main': 0.4,
            'index': 0.3,
            'app': 0.3,
            'server': 0.3,
            'client': 0.3,
            'config': 0.2,
            'setup': 0.2,
            'readme': 0.1
        }
        
        for pattern, bonus in important_patterns.items():
            if pattern in filename:
                score += bonus
        
        # Chunk type relevance
        if chunk.get('chunk_type') == 'complete_file':
            score += 0.1
        
        return min(score, 1.0)

class GroqLLMClient:
    """Client for Groq API using Llama models"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama3-8b-8192"  # Default model
        
    def generate_response(self, prompt: str, system_prompt: str = None, max_tokens: int = 1000) -> str:
        """Generate response using Groq API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"Error: Failed to generate response (Status: {response.status_code})"
                
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return f"Error: {str(e)}"

class PersistentMemory:
    """Persistent memory system for the agent"""
    
    def __init__(self, memory_path: str = "data/agent_memory.json"):
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing memory
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict:
        """Load memory from file"""
        try:
            if self.memory_path.exists():
                with open(self.memory_path, 'r') as f:
                    return json.load(f)
            return {
                'conversations': [],
                'reasoning_patterns': [],
                'user_preferences': {},
                'query_history': []
            }
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            return {'conversations': [], 'reasoning_patterns': [], 'user_preferences': {}, 'query_history': []}
    
    def _save_memory(self):
        """Save memory to file"""
        try:
            with open(self.memory_path, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def add_conversation(self, conversation: Dict):
        """Add a conversation to memory"""
        conversation['timestamp'] = datetime.now().isoformat()
        self.memory['conversations'].append(conversation)
        
        # Keep only last 50 conversations
        if len(self.memory['conversations']) > 50:
            self.memory['conversations'] = self.memory['conversations'][-50:]
        
        self._save_memory()
    
    def add_reasoning_pattern(self, pattern: Dict):
        """Add a reasoning pattern to memory"""
        self.memory['reasoning_patterns'].append(pattern)
        
        # Keep only last 100 patterns
        if len(self.memory['reasoning_patterns']) > 100:
            self.memory['reasoning_patterns'] = self.memory['reasoning_patterns'][-100:]
        
        self._save_memory()
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.memory['conversations'][-limit:]
    
    def get_similar_queries(self, query: str, limit: int = 5) -> List[Dict]:
        """Get similar queries from history"""
        query_lower = query.lower()
        similar = []
        
        for conv in self.memory['conversations']:
            if 'user_query' in conv:
                conv_query = conv['user_query'].lower()
                # Simple similarity check
                common_words = set(query_lower.split()) & set(conv_query.split())
                if len(common_words) >= 2:
                    similar.append(conv)
        
        return similar[-limit:]

class CodeAgent:
    """Main agent class with LangGraph workflow"""
    
    def __init__(self, db_path: str = "data/chromadb"):
        # Initialize components
        self.retriever = CodeRetriever(db_path)
        self.analyzer = RepositoryAnalyzer(db_path)
        self.ranker = IntelligentRanker(self.analyzer)
        self.llm = GroqLLMClient()
        self.memory = PersistentMemory()
        
        # Create LangGraph workflow
        self.workflow = StateGraph(AgentState)
        
        # Add nodes
        self.workflow.add_node("process_input", self._process_input_node)
        self.workflow.add_node("embed_query", self._embed_query_node)
        self.workflow.add_node("retrieve_chunks", self._retrieve_chunks_node)
        self.workflow.add_node("rank_chunks", self._rank_chunks_node)
        self.workflow.add_node("reasoning", self._reasoning_node)
        self.workflow.add_node("generate_answer", self._generate_answer_node)
        self.workflow.add_node("update_memory", self._update_memory_node)
        
        # Add edges
        self.workflow.set_entry_point("process_input")
        self.workflow.add_edge("process_input", "embed_query")
        self.workflow.add_edge("embed_query", "retrieve_chunks")
        self.workflow.add_edge("retrieve_chunks", "rank_chunks")
        self.workflow.add_edge("rank_chunks", "reasoning")
        self.workflow.add_edge("reasoning", "generate_answer")
        self.workflow.add_edge("generate_answer", "update_memory")
        self.workflow.add_edge("update_memory", END)
        
        # Compile workflow
        self.app = self.workflow.compile()
        
        logger.info("CodeAgent initialized successfully")
        
    def ensure_repository_metadata(self, repo_name: str) -> RepositoryMetadata:
        """Ensure repository metadata exists, generate if missing"""
        metadata = self.analyzer.get_metadata(repo_name)
        
        if metadata is None:
            logger.info(f"Metadata not found for {repo_name}, generating...")
            metadata = self.analyzer.analyze_repository(repo_name)
            logger.info(f"Generated metadata for {repo_name}: {len(metadata.main_functions)} main functions found")
        
        return metadata
    
    def _process_input_node(self, state: AgentState) -> AgentState:
        """Node 1: Process user input and load context"""
        try:
            # Clean the user query to remove command line artifacts
            user_query = state['user_query'].strip()
            
            # Remove command line patterns if they exist
            if user_query.startswith('python ') and '--query' in user_query:
                # Extract the actual query from command line format
                import re
                query_match = re.search(r'--query\s+["\']([^"\']+)["\']', user_query)
                if query_match:
                    user_query = query_match.group(1)
                else:
                    # Fallback: split by --query and take the last part
                    parts = user_query.split('--query')
                    if len(parts) > 1:
                        user_query = parts[-1].strip().strip('"\'')
            
            logger.info(f"Processing cleaned user query: {user_query}")
            
            # Update state with cleaned query
            state['user_query'] = user_query
            
            # Get conversation history
            history = self.memory.get_conversation_history()
            
            # Get similar queries
            similar_queries = self.memory.get_similar_queries(user_query)
            
            # Initialize reasoning trace
            reasoning_trace = [
                f"User query: {user_query}",
                f"Found {len(history)} previous conversations",
                f"Found {len(similar_queries)} similar queries"
            ]
            
            return {
                **state,
                "reasoning_trace": reasoning_trace,
                "conversation_history": history,
                "current_step": "input_processed"
            }
            
        except Exception as e:
            logger.error(f"Error in process_input_node: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "error"
            }
    
    def _embed_query_node(self, state: AgentState) -> AgentState:
        """Node 2: Embed the query (handled by ChromaDB)"""
        try:
            # ChromaDB handles embedding internally, so we just mark this step
            state["reasoning_trace"].append("Query embedding prepared for retrieval")
            
            return {
                **state,
                "current_step": "query_embedded"
            }
            
        except Exception as e:
            logger.error(f"Error in embed_query_node: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "error"
            }
    
    def _retrieve_chunks_node(self, state: AgentState) -> AgentState:
        """Node 3: Retrieve relevant chunks with intelligent filtering"""
        try:
            logger.info("Retrieving relevant chunks")
            
            query = state['user_query']
            query_lower = query.lower()
            
            # Check if this is a main function query
            is_main_query = any(keyword in query_lower for keyword in ['main', 'entry', 'start', 'beginning'])
            
            if is_main_query:
                # For main function queries, use targeted search
                logger.info("Detected main function query - using targeted search")
                
                # First, try to get repository metadata to find known main functions
                initial_chunks = self.retriever.search(query, n_results=50)  # Get more chunks initially
                
                if initial_chunks:
                    repo_name = initial_chunks[0]['repo_name']
                    repo_metadata = self.analyzer.get_metadata(repo_name)
                    
                    if repo_metadata and repo_metadata.has_main_functions():
                        # Prioritize chunks from files with main functions
                        main_files = repo_metadata.get_main_function_files()
                        main_chunks = [chunk for chunk in initial_chunks if chunk['filename'] in main_files]
                        
                        if main_chunks:
                            logger.info(f"Found {len(main_chunks)} chunks from main function files: {main_files}")
                            # Combine main function chunks with other relevant chunks
                            other_chunks = [chunk for chunk in initial_chunks if chunk['filename'] not in main_files]
                            chunks = main_chunks + other_chunks[:10]  # Prioritize main function chunks
                        else:
                            chunks = initial_chunks[:20]
                    else:
                        # Fallback to content-based search for main functions
                        chunks = self._search_for_main_functions(initial_chunks)
                else:
                    chunks = []
            else:
                # Regular search for non-main queries
                chunks = self.retriever.search(query, n_results=20)
            
            state["reasoning_trace"].append(f"Retrieved {len(chunks)} initial chunks")
            if is_main_query:
                state["reasoning_trace"].append("Used targeted main function search")
            
            return {
                **state,
                "retrieved_chunks": chunks,
                "current_step": "chunks_retrieved"
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_chunks_node: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "error"
            }

    def _search_for_main_functions(self, chunks: List[Dict]) -> List[Dict]:
        """Search for main functions in retrieved chunks"""
        main_function_patterns = [
            'public static void main(string[] args)',
            'public static void main(string args[])',
            'if __name__ == "__main__"',
            'int main(',
            'void main(',
            'springapplication.run'
        ]
        
        main_chunks = []
        other_chunks = []
        
        for chunk in chunks:
            content_lower = chunk['content'].lower()
            has_main = any(pattern in content_lower for pattern in main_function_patterns)
            
            if has_main:
                main_chunks.append(chunk)
            else:
                other_chunks.append(chunk)
        
        # Prioritize main function chunks
        return main_chunks + other_chunks[:15]  # Return up to 15 total chunks
    
    def _rank_chunks_node(self, state: AgentState) -> AgentState:
        """Node 4: Rank chunks using intelligent ranking"""
        try:
            logger.info("Ranking chunks with intelligent ranking system")
            
            chunks = state['retrieved_chunks']
            if not chunks:
                return {
                    **state,
                    "ranked_chunks": [],
                    "current_step": "chunks_ranked"
                }
            
            # Get repository metadata for the first chunk's repo
            repo_name = chunks[0]['repo_name']
            
            # ENHANCED: Always ensure metadata exists
            repo_metadata = self.ensure_repository_metadata(repo_name)
            
            # Rank chunks
            ranked_chunks = self.ranker.rank_chunks(chunks, state['user_query'], repo_metadata)
            
            # Take top 10 for further processing
            top_chunks = ranked_chunks[:10]
            
            # Enhanced logging for main function queries
            query_lower = state['user_query'].lower()
            is_main_query = any(keyword in query_lower for keyword in ['main', 'entry', 'start'])
            
            if is_main_query:
                logger.info("Main function query detected - ranking results:")
                for i, chunk in enumerate(top_chunks[:5], 1):
                    main_score = chunk['scoring_breakdown']['main_function_score']
                    logger.info(f"  {i}. {chunk['filename']} - Total: {chunk['final_score']:.3f} (Main: {main_score:.3f})")
            else:
                logger.info(f"Top 3 chunks after ranking:")
                for i, chunk in enumerate(top_chunks[:3], 1):
                    logger.info(f"  {i}. {chunk['filename']} - Score: {chunk['final_score']:.3f}")
            
            state["reasoning_trace"].append(f"Ranked chunks using metadata from {repo_name}")
            state["reasoning_trace"].append(f"Repository has {len(repo_metadata.main_functions)} main functions")
            state["reasoning_trace"].append(f"Top chunk: {top_chunks[0]['filename']} (score: {top_chunks[0]['final_score']:.3f})")
            
            return {
                **state,
                "ranked_chunks": top_chunks,
                "metadata_context": repo_metadata.to_dict(),
                "current_step": "chunks_ranked"
            }
            
        except Exception as e:
            logger.error(f"Error in rank_chunks_node: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "error"
            }
    
    def _reasoning_node(self, state: AgentState) -> AgentState:
        """Node 5: Reasoning and analysis"""
        try:
            logger.info("Performing reasoning and analysis")
            
            chunks = state['ranked_chunks']
            query = state['user_query']
            
            # Create reasoning scratchpad
            scratchpad = f"""
Query Analysis: {query}
Repository: {state['metadata_context']['repo_name']}
Main Language: {state['metadata_context']['main_language']}
Project Type: {state['metadata_context']['project_type']}

Top Retrieved Chunks:
"""
            
            for i, chunk in enumerate(chunks[:5], 1):
                scratchpad += f"""
{i}. {chunk['filename']} (Score: {chunk['final_score']:.3f})
   - Language: {chunk['language']}
   - Lines: {chunk['lines']}
   - Relevance: {chunk['relevance_score']:.3f}
   - Content preview: {chunk['content'][:100]}...
"""
            
            # Analyze query intent
            intent_analysis = self._analyze_query_intent(query, chunks)
            scratchpad += f"\nQuery Intent Analysis:\n{intent_analysis}"
            
            state["reasoning_trace"].append("Completed reasoning and analysis")
            state["reasoning_trace"].append(f"Query intent: {intent_analysis}")
            
            return {
                **state,
                "scratchpad": scratchpad,
                "current_step": "reasoning_complete"
            }
            
        except Exception as e:
            logger.error(f"Error in reasoning_node: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "error"
            }
    
    def _analyze_query_intent(self, query: str, chunks: List[Dict]) -> str:
        """Analyze the intent of the user query"""
        query_lower = query.lower()
        
        # Intent patterns
        if any(word in query_lower for word in ['how', 'what', 'explain', 'describe']):
            return "Explanation/Information seeking"
        elif any(word in query_lower for word in ['find', 'show', 'locate', 'search']):
            return "Code search/location"
        elif any(word in query_lower for word in ['implement', 'create', 'build', 'make']):
            return "Implementation guidance"
        elif any(word in query_lower for word in ['fix', 'debug', 'error', 'bug']):
            return "Debugging/Problem solving"
        elif any(word in query_lower for word in ['optimize', 'improve', 'performance']):
            return "Code optimization"
        else:
            return "General inquiry"
    
    def _generate_answer_node(self, state: AgentState) -> AgentState:
        """Node 6: Generate final answer using LLM"""
        try:
            logger.info("Generating final answer")
            
            # Create system prompt
            system_prompt = f"""You are a knowledgeable code assistant with access to repository information.

Repository Context:
- Name: {state['metadata_context']['repo_name']}
- Main Language: {state['metadata_context']['main_language']}
- Project Type: {state['metadata_context']['project_type']}
- Total Chunks: {state['metadata_context']['total_chunks']}

Your task is to provide accurate, helpful, and contextual answers based on the retrieved code chunks.
Be specific, reference the actual code when relevant, and explain concepts clearly.
If you cannot find sufficient information in the provided chunks, say so honestly.
"""
            
            # Create user prompt with context
            user_prompt = f"""
User Query: {state['user_query']}

Retrieved Code Context:
{state['scratchpad']}

Reasoning Trace:
{chr(10).join(state['reasoning_trace'])}

Please provide a comprehensive answer based on the above context.
"""
            
            # Generate response
            response = self.llm.generate_response(user_prompt, system_prompt)
            
            state["reasoning_trace"].append("Generated final answer using LLM")
            
            return {
                **state,
                "final_answer": response,
                "current_step": "answer_generated"
            }
            
        except Exception as e:
            logger.error(f"Error in generate_answer_node: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "error"
            }
    
    def _update_memory_node(self, state: AgentState) -> AgentState:
        """Node 7: Update persistent memory"""
        try:
            logger.info("Updating persistent memory")
            
            # Create conversation record
            conversation = {
                "user_query": state['user_query'],
                "final_answer": state['final_answer'],
                "chunks_used": len(state['ranked_chunks']),
                "repository": state['metadata_context']['repo_name'],
                "reasoning_trace": state['reasoning_trace']
            }
            
            # Add to memory
            self.memory.add_conversation(conversation)
            
            # Create reasoning pattern
            reasoning_pattern = {
                "query_type": self._analyze_query_intent(state['user_query'], state['ranked_chunks']),
                "successful_chunks": [chunk['filename'] for chunk in state['ranked_chunks'][:3]],
                "repository_type": state['metadata_context']['project_type'],
                "main_language": state['metadata_context']['main_language']
            }
            
            # Add reasoning pattern to memory
            self.memory.add_reasoning_pattern(reasoning_pattern)
            
            state["reasoning_trace"].append("Updated persistent memory")
            
            return {
                **state,
                "current_step": "memory_updated"
            }
            
        except Exception as e:
            logger.error(f"Error in update_memory_node: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "error"
            }
    
    def query(self, user_query: str) -> Dict:
        """Main query method for the agent"""
        try:
            # Initial state
            initial_state = {
                "user_query": user_query,
                "query_embedding": None,
                "retrieved_chunks": [],
                "ranked_chunks": [],
                "reasoning_trace": [],
                "scratchpad": "",
                "tool_calls": [],
                "final_answer": "",
                "conversation_history": [],
                "metadata_context": {},
                "error_message": None,
                "current_step": "starting"
            }
            
            # Run the workflow
            result = self.app.invoke(initial_state)
            
            return {
                "query": user_query,
                "answer": result.get("final_answer", "No answer generated"),
                "chunks_used": len(result.get("ranked_chunks", [])),
                "reasoning_trace": result.get("reasoning_trace", []),
                "repository": result.get("metadata_context", {}).get("repo_name", "Unknown"),
                "status": "success" if not result.get("error_message") else "error",
                "error": result.get("error_message")
            }
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return {
                "query": user_query,
                "answer": f"Error processing query: {str(e)}",
                "chunks_used": 0,
                "reasoning_trace": [],
                "repository": "Unknown",
                "status": "error",
                "error": str(e)
            }

class MCPToolManager:
    """Tool manager for MCP (Model Context Protocol) integration"""
    
    def __init__(self, agent: CodeAgent):
        self.agent = agent
        self.tools = {
            "file_search": self._file_search_tool,
            "repo_summarizer": self._repo_summarizer_tool,
            "code_analyzer": self._code_analyzer_tool,
            "dependency_finder": self._dependency_finder_tool
        }
    
    def _file_search_tool(self, filename: str, repo_name: str = None) -> Dict:
        """Search for specific files in the repository"""
        try:
            filters = {"filename": filename}
            if repo_name:
                filters["repo_name"] = repo_name
            
            results = self.agent.retriever.search("", n_results=10, filters=filters)
            
            return {
                "tool": "file_search",
                "query": filename,
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "tool": "file_search",
                "query": filename,
                "error": str(e),
                "status": "error"
            }
    
    def _repo_summarizer_tool(self, repo_name: str) -> Dict:
        """Provide a comprehensive summary of a repository"""
        try:
            metadata = self.agent.analyzer.get_metadata(repo_name)
            if not metadata:
                metadata = self.agent.analyzer.analyze_repository(repo_name)
            
            # Get some sample chunks
            sample_chunks = self.agent.retriever.search_by_repository("", repo_name, n_results=5)
            
            summary = {
                "repository": repo_name,
                "main_language": metadata.main_language,
                "project_type": metadata.project_type,
                "total_chunks": metadata.total_chunks,
                "languages": metadata.languages,
                "key_files": metadata.key_files,
                "dependencies": metadata.dependencies,
                "file_structure": metadata.file_structure,
                "sample_files": [chunk['filename'] for chunk in sample_chunks]
            }
            
            return {
                "tool": "repo_summarizer",
                "repository": repo_name,
                "summary": summary,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "tool": "repo_summarizer",
                "repository": repo_name,
                "error": str(e),
                "status": "error"
            }
    
    def _code_analyzer_tool(self, code_query: str, language: str = None) -> Dict:
        """Analyze code patterns and provide insights"""
        try:
            # Search for code patterns
            if language:
                results = self.agent.retriever.search_by_language(code_query, language, n_results=10)
            else:
                results = self.agent.retriever.search(code_query, n_results=10)
            
            # Analyze patterns
            patterns = {}
            for result in results:
                content = result['content']
                
                # Simple pattern analysis
                if 'class ' in content:
                    patterns['classes'] = patterns.get('classes', 0) + 1
                if 'def ' in content or 'function ' in content:
                    patterns['functions'] = patterns.get('functions', 0) + 1
                if 'import ' in content or 'from ' in content:
                    patterns['imports'] = patterns.get('imports', 0) + 1
            
            return {
                "tool": "code_analyzer",
                "query": code_query,
                "language": language,
                "patterns": patterns,
                "examples": results[:3],
                "status": "success"
            }
            
        except Exception as e:
            return {
                "tool": "code_analyzer",
                "query": code_query,
                "error": str(e),
                "status": "error"
            }
    
    def _dependency_finder_tool(self, package_name: str) -> Dict:
        """Find dependencies and their usage"""
        try:
            # Search for dependency usage
            results = self.agent.retriever.search(package_name, n_results=15)
            
            usage_patterns = []
            for result in results:
                content = result['content']
                if package_name.lower() in content.lower():
                    # Extract lines containing the package
                    lines = content.split('\n')
                    relevant_lines = [line.strip() for line in lines if package_name.lower() in line.lower()]
                    usage_patterns.extend(relevant_lines[:3])  # Take first 3 relevant lines
            
            return {
                "tool": "dependency_finder",
                "package": package_name,
                "usage_patterns": list(set(usage_patterns)),  # Remove duplicates
                "found_in_files": [result['filename'] for result in results[:5]],
                "status": "success"
            }
            
        except Exception as e:
            return {
                "tool": "dependency_finder",
                "package": package_name,
                "error": str(e),
                "status": "error"
            }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict:
        """Execute a specific tool"""
        if tool_name not in self.tools:
            return {
                "tool": tool_name,
                "error": f"Unknown tool: {tool_name}",
                "status": "error"
            }
        
        return self.tools[tool_name](**kwargs)

class AgentInterface:
    """User-friendly interface for the agent"""
    
    def __init__(self, db_path: str = "data/chromadb"):
        self.agent = CodeAgent(db_path)
        self.tool_manager = MCPToolManager(self.agent)
    
    def interactive_mode(self):
        """Run agent in interactive mode"""
        print("🤖 Code Agent - Interactive Mode")
        print("=" * 50)
        print("Available commands:")
        print("  /help     - Show this help")
        print("  /stats    - Show collection statistics")
        print("  /repos    - List available repositories")
        print("  /tools    - List available tools")
        print("  /memory   - Show recent memory")
        print("  /exit     - Exit the agent")
        print("  Or just ask any question about the code!")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🔍 Query: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                # Process query
                print("\n🤔 Thinking...")
                result = self.agent.query(user_input)
                
                # Display results
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _handle_command(self, command: str):
        """Handle special commands"""
        if command == '/help':
            print("\n📖 Help:")
            print("  Ask questions about code, functions, classes, or implementations")
            print("  Examples:")
            print("    - 'How does authentication work in this project?'")
            print("    - 'Show me the main function'")
            print("    - 'Find all database models'")
            print("    - 'Explain the API endpoints'")
        
        elif command == '/stats':
            stats = self.agent.retriever.get_collection_stats()
            print(f"\n📊 Collection Statistics:")
            print(f"Total chunks: {stats.get('total_chunks', 0)}")
            print(f"Languages: {len(stats.get('languages', {}))}")
            print(f"Repositories: {len(stats.get('repositories', {}))}")
        
        elif command == '/repos':
            repos = self.agent.retriever.get_available_repositories()
            print(f"\n📁 Available Repositories ({len(repos)}):")
            for repo in repos:
                print(f"  • {repo}")
        
        elif command == '/tools':
            tools = list(self.tool_manager.tools.keys())
            print(f"\n🛠️ Available Tools ({len(tools)}):")
            for tool in tools:
                print(f"  • {tool}")
        
        elif command == '/memory':
            history = self.agent.memory.get_conversation_history(5)
            print(f"\n💭 Recent Memory ({len(history)} conversations):")
            for i, conv in enumerate(history, 1):
                print(f"  {i}. {conv.get('user_query', 'Unknown')[:50]}...")
        
        elif command == '/exit':
            print("Goodbye! 👋")
            exit()
        
        else:
            print(f"Unknown command: {command}")
    
    def _display_result(self, result: Dict):
        """Display query result in a formatted way"""
        print(f"\n✅ Result for: '{result['query']}'")
        print("=" * 60)
        
        if result['status'] == 'error':
            print(f"❌ Error: {result['error']}")
            return
        
        # Show answer
        print(f"🎯 Answer:")
        print(result['answer'])
        
        # Show metadata
        print(f"\n📊 Metadata:")
        print(f"  Repository: {result['repository']}")
        print(f"  Chunks used: {result['chunks_used']}")
        print(f"  Status: {result['status']}")
        
        # Show reasoning trace
        if result['reasoning_trace']:
            print(f"\n🧠 Reasoning Trace:")
            for i, step in enumerate(result['reasoning_trace'], 1):
                print(f"  {i}. {step}")
    
    def query(self, question: str) -> Dict:
        """Single query method"""
        return self.agent.query(question)

# CLI Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Agent System")
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--db-path", type=str, default="data/chromadb", help="Database path")
    parser.add_argument("--analyze-repo", type=str, help="Analyze a specific repository")
    parser.add_argument("--tool", type=str, help="Execute a specific tool")
    parser.add_argument("--tool-args", type=str, help="Tool arguments (JSON format)")
    
    args = parser.parse_args()
    
    try:
        interface = AgentInterface(args.db_path)
        
        if args.interactive:
            interface.interactive_mode()
        
        elif args.query:
            result = interface.query(args.query)
            interface._display_result(result)
        
        elif args.analyze_repo:
            analyzer = RepositoryAnalyzer(args.db_path)
            metadata = analyzer.analyze_repository(args.analyze_repo)
            print(f"\n📊 Repository Analysis: {args.analyze_repo}")
            print("=" * 50)
            print(f"Main Language: {metadata.main_language}")
            print(f"Project Type: {metadata.project_type}")
            print(f"Total Chunks: {metadata.total_chunks}")
            print(f"Key Files: {', '.join(metadata.key_files)}")
            print(f"Dependencies: {', '.join(metadata.dependencies)}")
        
        elif args.tool:
            tool_args = {}
            if args.tool_args:
                import json
                tool_args = json.loads(args.tool_args)
            
            result = interface.tool_manager.execute_tool(args.tool, **tool_args)
            print(f"\n🛠️ Tool Result: {args.tool}")
            print("=" * 50)
            print(json.dumps(result, indent=2))
        
        else:
            print("Use --query for single query, --interactive for interactive mode, or --help for options")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Processed repositories with week2_chunker.py")
        print("2. Set up ChromaDB with week3_retrieval.py")
        print("3. Set GROQ_API_KEY environment variable")

if __name__ == "__main__":
    main()