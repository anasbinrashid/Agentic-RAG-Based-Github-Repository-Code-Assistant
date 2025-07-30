# Streamlit Interface for Intelligent Code Assistant
# Modern web interface for the Groq-powered RAG code assistant with session management

import streamlit as st
import os
import json
import time
from datetime import datetime
from pathlib import Path
import plotly.express as px
import pandas as pd
import threading
from queue import Queue
import git
from urllib.parse import urlparse
import re
from dataclasses import asdict
import logging
from typing import Optional
import uuid
import shutil

# Import the agent with error handling
try:
    from agent import GroqCodeAgent, AgentResponse
    from chunker_embedder import GitHubRepoProcessor, CodeChunker, ChromaDBManager
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please check that all dependencies are installed correctly.")
    st.stop()
    
# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Add helper methods to ChromaDBManager
setattr(ChromaDBManager, "is_repository_already_stored", lambda self, repo_name: repo_name in self.get_all_repositories())
setattr(ChromaDBManager, "get_languages_for_repo", lambda self, repo_name: self.get_all_repositories().get(repo_name, {}).get('languages', {}))

# Session Management Functions
class SessionManager:
    def __init__(self, base_sessions_dir: str = "data/sessions"):
        self.base_sessions_dir = Path(base_sessions_dir)
        self.base_sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def create_new_session(self) -> str:
        """Create a new session folder and return session ID"""
        session_id = str(uuid.uuid4())[:8]  # Short UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{timestamp}_{session_id}"
        
        session_path = self.base_sessions_dir / session_name
        session_path.mkdir(exist_ok=True)
        
        # Create session metadata
        metadata = {
            "session_id": session_id,
            "session_name": session_name,
            "created_at": datetime.now().isoformat(),
            "repositories_used": [],
            "total_queries": 0,
            "last_activity": datetime.now().isoformat()
        }
        
        with open(session_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create subdirectories
        (session_path / "conversations").mkdir(exist_ok=True)
        (session_path / "exports").mkdir(exist_ok=True)
        
        return session_name
    
    def save_conversation(self, session_name: str, query: str, response: AgentResponse):
        """Save a conversation to the session"""
        session_path = self.base_sessions_dir / session_name
        conversations_path = session_path / "conversations"
        
        # Create conversation entry
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": asdict(response)
        }
        
        # Save individual conversation
        conv_filename = f"conv_{int(time.time())}_{len(os.listdir(conversations_path)) + 1}.json"
        with open(conversations_path / conv_filename, "w") as f:
            json.dump(conversation_entry, f, indent=2)
        
        # Update session metadata
        self.update_session_metadata(session_name, response)
    
    def update_session_metadata(self, session_name: str, response: AgentResponse):
        """Update session metadata"""
        session_path = self.base_sessions_dir / session_name
        metadata_path = session_path / "metadata.json"
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Update metadata
        metadata["total_queries"] += 1
        metadata["last_activity"] = datetime.now().isoformat()
        
        # Update repositories used
        for source in response.sources:
            repo_name = source.get('repository', 'unknown')
            if repo_name not in metadata["repositories_used"]:
                metadata["repositories_used"].append(repo_name)
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def get_all_sessions(self) -> list:
        """Get all available sessions"""
        sessions = []
        for session_dir in self.base_sessions_dir.iterdir():
            if session_dir.is_dir():
                metadata_path = session_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    sessions.append(metadata)
        
        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)
    
    def load_session_conversations(self, session_name: str) -> list:
        """Load all conversations from a session"""
        session_path = self.base_sessions_dir / session_name
        conversations_path = session_path / "conversations"
        
        conversations = []
        if conversations_path.exists():
            for conv_file in sorted(conversations_path.glob("*.json")):
                with open(conv_file, "r") as f:
                    conversation = json.load(f)
                conversations.append(conversation)
        
        return conversations

# Extend GitHubRepoProcessor with safer clone method
def clone_repository(self, repo_url: str, local_name: Optional[str] = None) -> Path:
    """Clone a GitHub repository safely"""
    if local_name is None:
        local_name = repo_url.split('/')[-1].replace('.git', '')

    repo_path = self.base_dir / local_name

    if repo_path.exists():
        import shutil
        try:
            shutil.rmtree(repo_path)
            logger.info(f"Removed existing directory: {repo_path}")
        except Exception as cleanup_error:
            logger.warning(f"Could not remove existing directory {repo_path}: {cleanup_error}")
            return repo_path  # Fallback to existing path

    try:
        logger.info(f"Cloning {repo_url} to {repo_path}")
        git.Repo.clone_from(repo_url, repo_path, depth=1)
        logger.info("Successfully cloned repository")
        return repo_path
    except Exception as e:
        logger.error(f"Failed to clone repository: {e}")
        raise

GitHubRepoProcessor.clone_repository = clone_repository

def get_repository_files(repo_name: str, db_path: str) -> list:
    """Get list of files in a repository from ChromaDB"""
    try:
        chroma_manager = ChromaDBManager(db_path)
        # This would require implementing a method in ChromaDBManager to get file list
        # For now, we'll return a placeholder
        return [f"File listing for {repo_name} - Implementation needed in ChromaDBManager"]
    except Exception as e:
        logger.error(f"Error getting repository files: {e}")
        return []

def process_repository(repo_url: str, progress_queue: Queue, settings: dict):
    """Process a repository by cloning and chunking it, or load if already processed"""
    try:
        repo_name = extract_repo_name(repo_url)
        chroma_manager = ChromaDBManager(settings['db_path'])

        # Check if repository is already processed
        if chroma_manager.is_repository_already_stored(repo_name):
            # Get files from filesystem for already processed repos
            files = get_repository_files_from_filesystem(repo_name, f"{settings['base_dir']}/repos")
            progress_queue.put({
                "status": "success",
                "message": f"Repository '{repo_name}' already processed. Loaded from backend.",
                "stats": {
                    "files_processed": 0,
                    "chunks_created": 0,
                    "languages": chroma_manager.get_languages_for_repo(repo_name)
                },
                "files": files,
                "repo_name": repo_name
            })
            return

        progress_queue.put({"status": "cloning", "message": "Cloning repository..."})

        # Initialize processors
        repo_processor = GitHubRepoProcessor(f"{settings['base_dir']}/repos")
        chunker = CodeChunker()

        # Clone and process
        repo_path = repo_processor.clone_repository(repo_url, repo_name)
        progress_queue.put({"status": "cloned", "message": f"Repository cloned to {repo_path}"})

        # Get list of all files immediately after cloning
        all_files = get_repository_files_from_filesystem(repo_name, f"{settings['base_dir']}/repos")
        progress_queue.put({
            "status": "files_listed", 
            "message": f"Found {len(all_files)} files",
            "files": all_files,
            "repo_name": repo_name
        })

        progress_queue.put({"status": "processing", "message": "Processing files..."})
        files = repo_processor.find_processable_files(repo_path)
        all_chunks = []

        for file_path in files:
            content = repo_processor.read_file_safely(file_path)
            if content is None:
                continue
            language = repo_processor.get_file_language(file_path)
            relative_path = file_path.relative_to(repo_path)
            chunks = chunker.chunk_by_lines(content, relative_path, language, repo_name)
            all_chunks.extend(chunks)

        progress_queue.put({"status": "chunking", "message": f"Created {len(all_chunks)} chunks"})
        progress_queue.put({"status": "embedding", "message": "Embedding chunks..."})

        repo_metadata = repo_processor.extract_metadata(repo_path, repo_name, repo_url)
        repo_metadata.total_chunks = len(all_chunks)

        chroma_manager.add_repository_metadata(repo_metadata)
        chroma_manager.add_chunks_with_metadata(all_chunks, repo_metadata)

        progress_queue.put({
            "status": "success",
            "message": f"Successfully processed repository! Added {len(all_chunks)} chunks.",
            "stats": {
                "files_processed": len(files),
                "chunks_created": len(all_chunks),
                "languages": repo_metadata.languages
            },
            "files": all_files,
            "repo_name": repo_name
        })

    except Exception as e:
        progress_queue.put({"status": "error", "message": f"Error processing repository: {str(e)}"})

# Page config
st.set_page_config(
    page_title="🤖 Intelligent Code Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Intelligent Code Assistant\nPowered by Groq API and Llama models"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .session-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .current-session {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .file-list {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.2rem 0;
        font-family: monospace;
        font-size: 0.8rem;
    }
    
    .query-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .response-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    .source-card {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 3px solid #6c757d;
        margin: 0.5rem 0;
    }
    
    .metrics-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()
    
    if 'current_session' not in st.session_state:
        st.session_state.current_session = st.session_state.session_manager.create_new_session()
    
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_response' not in st.session_state:
        st.session_state.current_response = None
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'model': 'llama3-70b-8192',
            'db_path': 'data/chromadb',
            'base_dir': 'data',
            'max_history': 50
        }
    if 'stats' not in st.session_state:
        st.session_state.stats = {}
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None
    if 'processing_queue' not in st.session_state:
        st.session_state.processing_queue = Queue()
    if 'repo_files' not in st.session_state:
        st.session_state.repo_files = {}
def load_agent():
    """Load and initialize the agent with better error handling"""
    try:
        if st.session_state.agent is None:
            with st.spinner("🔄 Initializing agent..."):
                # Check if GROQ_API_KEY exists
                if not os.getenv("GROQ_API_KEY"):
                    st.error("❌ GROQ_API_KEY environment variable is required!")
                    return False
                
                # Try to initialize with fallback options
                try:
                    st.session_state.agent = GroqCodeAgent(
                        db_path=st.session_state.settings['db_path'],
                        model=st.session_state.settings['model']
                    )
                    st.success("✅ Agent initialized successfully!")
                    return True
                    
                except Exception as model_error:
                    st.error(f"❌ Failed to initialize with model {st.session_state.settings['model']}")
                    st.error(f"Error details: {str(model_error)}")
                    return False
        return True
    except Exception as e:
        st.error(f"❌ Critical error initializing agent: {str(e)}")
        st.error("Please check your environment setup and try again.")
        return False
def get_repository_files_from_filesystem(repo_name: str, base_dir: str = "data/repos") -> list:
    """Get list of files in a repository from the filesystem"""
    try:
        repo_path = Path(base_dir) / repo_name
        if not repo_path.exists():
            return []
        
        files = []
        # Walk through all files in the repository
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                # Skip common directories we don't want to show
                skip_dirs = {'.git', '__pycache__', 'node_modules', '.vscode', '.idea', 'dist', 'build'}
                if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                    continue
                
                # Get relative path from repo root
                relative_path = file_path.relative_to(repo_path)
                files.append(str(relative_path))
        
        return sorted(files)
    except Exception as e:
        logger.error(f"Error getting repository files from filesystem: {e}")
        return []
def get_confidence_color(confidence: float) -> str:
    """Get color class based on confidence score"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def format_response_time(seconds: float) -> str:
    """Format response time in a readable way"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"

def render_main_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Intelligent Code Assistant</h1>
        <p>Powered by Groq API & Llama Models | RAG-Enhanced Code Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with session management and repository info"""
    with st.sidebar:
        # Current Session Info
        st.header("📝 Current Session")
        
        current_session_info = f"""
        <div class="current-session">
            <strong>Session ID:</strong> {st.session_state.current_session}<br>
            <strong>Queries:</strong> {len(st.session_state.conversation_history)}<br>
            <strong>Created:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
        """
        st.markdown(current_session_info, unsafe_allow_html=True)
        
        # Session Management
        if st.button("🆕 New Session", type="secondary"):
            st.session_state.current_session = st.session_state.session_manager.create_new_session()
            st.session_state.conversation_history = []
            st.session_state.current_response = None
            st.rerun()
        
        st.divider()
        
        # Repository Information
        st.header("📚 Repository Files")
        
        # Get available repositories from both stats and repo_files
        available_repos = set()
        if st.session_state.stats:
            available_repos.update(st.session_state.stats.get('repositories', {}).keys())
        if st.session_state.repo_files:
            available_repos.update(st.session_state.repo_files.keys())
        
        if available_repos:
            selected_repo = st.selectbox(
                "Select Repository:",
                options=list(available_repos),
                key="repo_selector"
            )
            
            if selected_repo:
                st.markdown(f"**Repository:** {selected_repo}")
                
                # Show chunk count if available
                repositories = st.session_state.stats.get('repositories', {})
                if selected_repo in repositories:
                    st.markdown(f"**Chunks:** {repositories[selected_repo]}")
                
                # Get languages for this repo
                languages = st.session_state.stats.get('languages', {})
                if languages:
                    st.markdown("**Languages:**")
                    for lang, count in languages.items():
                        st.markdown(f"• {lang}: {count} files")
                
                # Display actual files from the repository
                if selected_repo in st.session_state.repo_files:
                    files = st.session_state.repo_files[selected_repo]
                    
                    with st.expander(f"📄 Files ({len(files)} total)", expanded=True):
                        # Add search functionality
                        search_term = st.text_input("🔍 Search files:", key=f"search_{selected_repo}")
                        
                        # Filter files based on search
                        if search_term:
                            filtered_files = [f for f in files if search_term.lower() in f.lower()]
                        else:
                            filtered_files = files
                        
                        # Group files by directory for better organization
                        file_tree = {}
                        for file_path in filtered_files:
                            parts = file_path.split('/')
                            if len(parts) == 1:
                                # Root level file
                                if '📄 Root Files' not in file_tree:
                                    file_tree['📄 Root Files'] = []
                                file_tree['📄 Root Files'].append(file_path)
                            else:
                                # File in subdirectory
                                dir_name = f"📁 {parts[0]}/"
                                if dir_name not in file_tree:
                                    file_tree[dir_name] = []
                                file_tree[dir_name].append(file_path)
                        
                        # Display files organized by directory
                        for dir_name, dir_files in sorted(file_tree.items()):
                            if len(dir_files) > 5:  # Use expander for directories with many files
                                with st.expander(f"{dir_name} ({len(dir_files)} files)"):
                                    for file_path in sorted(dir_files):
                                        st.markdown(f"<div class='file-list'>📄 {file_path}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"**{dir_name}**")
                                for file_path in sorted(dir_files):
                                    st.markdown(f"<div class='file-list'>📄 {file_path}</div>", unsafe_allow_html=True)
                        
                        if search_term and not filtered_files:
                            st.info(f"No files found matching '{search_term}'")
                else:
                    # Try to load files from filesystem if not in session state
                    files = get_repository_files_from_filesystem(selected_repo)
                    if files:
                        st.session_state.repo_files[selected_repo] = files
                        st.rerun()  # Refresh to show the files
                    else:
                        st.info("No files found for this repository.")
        else:
            st.info("No repositories loaded. Add repositories in the Repository Manager tab.")
        
        st.divider()
        
        # Model Settings (simplified)
        st.header("⚙️ Settings")
        
        available_models = [
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        
        new_model = st.selectbox(
            "Model:",
            available_models,
            index=available_models.index(st.session_state.settings['model']),
        )
        
        if new_model != st.session_state.settings['model']:
            st.session_state.settings['model'] = new_model
            st.session_state.agent = None  # Force re-initialization
            st.rerun()
        
        # Refresh Stats
        if st.button("🔄 Refresh Stats"):
            if st.session_state.agent:
                try:
                    with st.spinner("Loading stats..."):
                        st.session_state.stats = st.session_state.agent.get_stats()
                except Exception as e:
                    st.error(f"Failed to load stats: {e}")

def render_query_examples():
    """Render example queries"""
    with st.expander("💡 Example Queries", expanded=False):
        examples = [
            "Explain how authentication works in this codebase",
            "Find examples of error handling in Python",
            "How to implement a REST API in this project?",
            "Show me database connection patterns",
            "Compare different sorting algorithms used here",
            "What are the security best practices implemented?",
            "Find examples of async/await usage",
            "How is logging configured in this application?"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                    st.session_state.current_query = example

def render_response(response: AgentResponse):
    """Render the agent response"""
    st.markdown('<div class="response-box">', unsafe_allow_html=True)
    
    # Response header with metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        confidence_class = get_confidence_color(response.confidence)
        st.markdown(f"""
        <div class="metrics-card">
            <h4>Confidence</h4>
            <p class="{confidence_class}">{response.confidence:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metrics-card">
            <h4>Response Time</h4>
            <p>{format_response_time(response.response_time)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metrics-card">
            <h4>Model Used</h4>
            <p>{response.model_used}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # with col4:
    #     st.markdown(f"""
    #     <div class="metrics-card">
    #         <h4>Sources</h4>
    #         <p>{len(response.sources)}</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Main answer
    st.subheader("📝 Answer")
    st.markdown(response.answer)
    
    # Reasoning
    if response.reasoning:
        st.subheader("🧠 Reasoning")
        st.info(response.reasoning)
    
    # # Sources
    # if response.sources:
    #     st.subheader("📚 Sources Used")
        
    #     # Create DataFrame for sources table
    #     sources_data = []
    #     for i, source in enumerate(response.sources, 1):
    #         sources_data.append({
    #             "#": i,
    #             "File": source['filename'],
    #             "Repository": source['repository'],
    #             "Language": source['language'],
    #             "Lines": source['lines'],
    #             "Relevance": f"{source['relevance_score']:.3f}"
    #         })
        
    #     sources_df = pd.DataFrame(sources_data)
    #     st.dataframe(sources_df, use_container_width=True)
        
    #     # Detailed source cards
    #     with st.expander("🔍 Detailed Source Information"):
    #         for i, source in enumerate(response.sources, 1):
    #             st.markdown(f"""
    #             <div class="source-card">
    #                 <strong>Source {i}: {source['filename']}</strong><br>
    #                 📁 Repository: {source['repository']}<br>
    #                 🔤 Language: {source['language']}<br>
    #                 📏 Lines: {source['lines']}<br>
    #                 📊 Relevance Score: {source['relevance_score']:.3f}<br>
    #                 📂 Path: {source.get('file_path', 'N/A')}
    #             </div>
    #             """, unsafe_allow_html=True)
    
    # Export options
    st.subheader("💾 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Copy Answer"):
            st.write("```")
            st.write(response.answer)
            st.write("```")
    
    with col2:
        # Download as JSON
        response_json = json.dumps(asdict(response), indent=2)
        st.download_button(
            "📥 Download JSON",
            response_json,
            file_name=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        # Create markdown report
        markdown_report = f"""# Query Response Report

## Query
{response.query}

## Answer
{response.answer}

## Details
- **Confidence:** {response.confidence:.2f}
- **Response Time:** {format_response_time(response.response_time)}
- **Model Used:** {response.model_used}
- **Sources:** {len(response.sources)}

## Reasoning
{response.reasoning}
"""
## Sources
#"""
        # for i, source in enumerate(response.sources, 1):
        #     markdown_report += f"\n{i}. **{source['filename']}** ({source['language']})\n"
        #     markdown_report += f"   - Repository: {source['repository']}\n"
        #     markdown_report += f"   - Lines: {source['lines']}\n"
        #     markdown_report += f"   - Relevance: {source['relevance_score']:.3f}\n"
        
        st.download_button(
            "📝 Download Report",
            markdown_report,
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

def render_conversation_history():
    """Render conversation history"""
    if not st.session_state.conversation_history:
        st.info("No conversation history yet. Start by asking a question!")
        return
    
    st.subheader("📚 Current Session History")
    
    # History controls
    col1, col2, col3 = st.columns(3)
    with col1:
        show_count = st.selectbox("Show", [5, 10, 20, "All"], index=0)
    with col2:
        sort_order = st.selectbox("Sort", ["Newest First", "Oldest First"], index=0)
    with col3:
        filter_confidence = st.selectbox("Filter by Confidence", ["All", "High (>0.7)", "Medium (0.4-0.7)", "Low (<0.4)"], index=0)
    
    # Filter and sort history
    history = st.session_state.conversation_history.copy()
    
    # Apply confidence filter
    if filter_confidence != "All":
        if filter_confidence == "High (>0.7)":
            history = [h for h in history if h['response'].confidence > 0.7]
        elif filter_confidence == "Medium (0.4-0.7)":
            history = [h for h in history if 0.4 <= h['response'].confidence <= 0.7]
        elif filter_confidence == "Low (<0.4)":
            history = [h for h in history if h['response'].confidence < 0.4]
    
    # Sort
    if sort_order == "Oldest First":
        history = history[::-1]
    
    # Limit count
    if show_count != "All":
        history = history[:int(show_count)]
    
    # Display history
    for i, item in enumerate(history):
        response = item['response']
        timestamp = item['timestamp']
        
        with st.expander(f"🔍 {response.query[:60]}... | Confidence: {response.confidence:.2f} | {timestamp}"):
            st.markdown(f"**Query:** {response.query}")
            st.markdown(f"**Answer:** {response.answer}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confidence", f"{response.confidence:.2f}")
            with col2:
                st.metric("Time", format_response_time(response.response_time))
            with col3:
                st.metric("Model", response.model_used)
            with col4:
                st.metric("Sources", len(response.sources))
            
            if st.button(f"🔄 Re-run Query", key=f"rerun_{i}"):
                st.session_state.current_query = response.query
                st.rerun()

def render_session_history_tab():
    """Render the session history tab"""
    st.header("📂 Session History")
    
    all_sessions = st.session_state.session_manager.get_all_sessions()
    
    if not all_sessions:
        st.info("No previous sessions found.")
        return
    
    st.subheader("🗂️ Previous Sessions")
    
    for session in all_sessions:
        with st.expander(f"📅 Session: {session['session_name']} | Created: {session['created_at'][:16]} | Queries: {session['total_queries']}"):
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Session ID:** {session['session_id']}")
                st.markdown(f"**Total Queries:** {session['total_queries']}")
                st.markdown(f"**Last Activity:** {session['last_activity'][:16]}")
            
            with col2:
                st.markdown("**Repositories Used:**")
                if session['repositories_used']:
                    for repo in session['repositories_used']:
                        st.markdown(f"• {repo}")
                else:
                    st.markdown("• No repositories used")
            
            # Load conversations button
            if st.button(f"📖 Load Conversations", key=f"load_{session['session_name']}"):
                conversations = st.session_state.session_manager.load_session_conversations(session['session_name'])
                
                if conversations:
                    st.markdown("### 💬 Conversations")
                    for conv in conversations:
                        st.markdown(f"**Query:** {conv['query']}")
                        st.markdown(f"**Answer:** {conv['response']['answer'][:200]}...")
                        st.markdown(f"**Timestamp:** {conv['timestamp']}")
                        st.divider()
                else:
                    st.info("No conversations found in this session.")

def is_valid_git_url(url: str) -> bool:
    """Validate if the URL is a valid git repository URL"""
    git_patterns = [
        r'^https://github\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?',
        r'^https://gitlab\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?',
        r'^https://bitbucket\.org/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?',
        r'^git@github\.com:[\w\-\.]+/[\w\-\.]+\.git',
        r'^git@gitlab\.com:[\w\-\.]+/[\w\-\.]+\.git',
    ]
    
    return any(re.match(pattern, url.strip()) for pattern in git_patterns)

def extract_repo_name(url: str) -> str:
    """Extract repository name from URL"""
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if path.endswith('.git'):
        path = path[:-4]
    return path.split('/')[-1] if '/' in path else path

def render_repository_manager():
    """Render repository management interface"""
    st.header("📂 Repository Manager")
    st.markdown("Add new repositories to your knowledge base for analysis.")
    
    # Repository URL input
    repo_url = st.text_input(
        "🔗 Repository URL",
        placeholder="https://github.com/username/repository-name",
        help="Enter a valid Git repository URL (GitHub, GitLab, Bitbucket supported)"
    )
    
    # Validation and preview
    if repo_url:
        if is_valid_git_url(repo_url):
            st.success(f"✅ Valid repository URL")
            repo_name = extract_repo_name(repo_url)
            st.info(f"📁 Repository name: **{repo_name}**")
        else:
            st.error("❌ Invalid repository URL. Please enter a valid Git repository URL.")
            return
    
    # Process button and status
    col1, col2 = st.columns([1, 3])
    
    with col1:
        process_btn = st.button(
            "🚀 Process Repository", 
            disabled=not repo_url or not is_valid_git_url(repo_url),
            type="primary"
        )
    
    with col2:
        if process_btn:
            # Clear previous queue items
            while not st.session_state.processing_queue.empty():
                try:
                    st.session_state.processing_queue.get_nowait()
                except:
                    break
            
            # Start processing thread with settings passed as parameter
            processing_thread = threading.Thread(
                target=process_repository,
                args=(repo_url, st.session_state.processing_queue, st.session_state.settings.copy())
            )
            processing_thread.daemon = True  # Make thread daemon so it doesn't block app shutdown
            processing_thread.start()
            st.session_state.processing_status = "running"
            st.rerun()  # Refresh to show processing status
    
    # Display processing status
    if hasattr(st.session_state, 'processing_status') and st.session_state.processing_status == "running":
        status_container = st.container()
        
        with status_container:
            # Auto-refresh every 2 seconds while processing
            placeholder = st.empty()
            
            # Check for updates from processing thread
            updates_received = False
            try:
                while not st.session_state.processing_queue.empty():
                    update = st.session_state.processing_queue.get_nowait()
                    updates_received = True
                    
                    if update["status"] == "cloning":
                        st.info(f"🔄 {update['message']}")
                    elif update["status"] == "cloned":
                        st.success(f"✅ {update['message']}")
                    elif update["status"] == "files_listed":
                        st.info(f"📄 {update['message']}")
                        # Store files in session state immediately
                        if 'files' in update and 'repo_name' in update:
                            st.session_state.repo_files[update['repo_name']] = update['files']
                    elif update["status"] == "processing":
                        st.info(f"⚙️ {update['message']}")
                    elif update["status"] == "chunking":
                        st.info(f"✂️ {update['message']}")
                    elif update["status"] == "embedding":
                        st.info(f"📥 {update['message']}")
                    elif update["status"] == "success":
                        st.success(f"🎉 {update['message']}")
                        # Store files in session state
                        if 'files' in update and 'repo_name' in update:
                            st.session_state.repo_files[update['repo_name']] = update['files']
                        
                        if 'stats' in update:
                            stats = update['stats']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Files Processed", stats['files_processed'])
                            with col2:
                                st.metric("Chunks Created", stats['chunks_created'])
                            with col3:
                                st.metric("Languages", len(stats['languages']))
                        
                        st.session_state.processing_status = "completed"
                        st.session_state.agent = None  # Force agent reinitialization
                        st.session_state.stats = {}  # Clear stats to force refresh
                        
                    elif update["status"] == "error":
                        st.error(f"❌ {update['message']}")
                        st.session_state.processing_status = "error"
                        
            except Exception as e:
                pass
            
            # Show progress indicator if still running
            if st.session_state.processing_status == "running":
                with placeholder:
                    st.info("🔄 Processing repository... This may take a few minutes.")
                    st.write("The page will automatically refresh to show progress.")
                
                # Auto-refresh after 2 seconds
                time.sleep(2)
                st.rerun()
    
    # Repository list
    st.divider()
    st.subheader("📚 Existing Repositories")
    
    if st.session_state.stats:
        repositories = st.session_state.stats.get('repositories', {})
        
        if repositories:
            # Create DataFrame for repositories
            repo_data = []
            for repo_name, count in repositories.items():
                # Add file count if available
                file_count = len(st.session_state.repo_files.get(repo_name, []))
                repo_data.append({
                    "Repository": repo_name,
                    "Chunks": count,
                    "Files": file_count if file_count > 0 else "Loading..."
                })
            
            repo_df = pd.DataFrame(repo_data)
            st.dataframe(repo_df, use_container_width=True, hide_index=True)
        else:
            st.info("No repositories found in the database. Process a repository to get started!")
    else:
        st.info("No repository statistics available. Process a repository or refresh stats.")

def main():
    """Main application"""
    init_session_state()
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("❌ GROQ_API_KEY environment variable is required!")
        st.error("Please set it in your .env file or environment variables.")
        st.stop()
    
    # Render UI
    render_main_header()
    render_sidebar()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Query Assistant", "📂 Repository Manager", "📚 Current Session", "🗂️ Session History", "ℹ️ About"])
    
    with tab1:
        # Load agent
        if not load_agent():
            st.stop()
        
        # Query examples
        render_query_examples()
        
        # Main query interface
        st.subheader("🔍 Ask Your Question")
        
        # Initialize current_query if it doesn't exist
        if 'current_query' not in st.session_state:
            st.session_state.current_query = ""
        
        # Query input
        query = st.text_area(
            "Enter your question about the codebase:",
            value=st.session_state.current_query,
            height=100,
            placeholder="e.g., How does authentication work in this codebase?",
            key="query_input"
        )
        
        # Query button
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            submit_query = st.button("🚀 Ask Assistant", type="primary", use_container_width=True)
        with col2:
            clear_query = st.button("🗑️ Clear", use_container_width=True)
        with col3:
            if st.session_state.current_response:
                regenerate = st.button("🔄 Regenerate", use_container_width=True)
            else:
                regenerate = False
        
        # Handle button actions
        if clear_query:
            st.session_state.current_query = ""
            st.rerun()
        
        if submit_query or regenerate:
            if query.strip():
                with st.spinner("🤖 Processing your query..."):
                    try:
                        # Process query
                        start_time = time.time()
                        response = st.session_state.agent.query(query)
                        end_time = time.time()
                        
                        # Update response time (in case it's not accurate from agent)
                        response.response_time = end_time - start_time
                        
                        # Store response
                        st.session_state.current_response = response
                        
                        # Save to session
                        st.session_state.session_manager.save_conversation(
                            st.session_state.current_session, 
                            query, 
                            response
                        )
                        
                        # Add to history
                        history_item = {
                            'query': query,
                            'response': response,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Remove duplicate if regenerating
                        if regenerate and st.session_state.conversation_history:
                            if st.session_state.conversation_history[-1]['query'] == query:
                                st.session_state.conversation_history[-1] = history_item
                            else:
                                st.session_state.conversation_history.append(history_item)
                        else:
                            st.session_state.conversation_history.append(history_item)
                        
                        # Limit history size
                        max_history = st.session_state.settings['max_history']
                        if len(st.session_state.conversation_history) > max_history:
                            st.session_state.conversation_history = st.session_state.conversation_history[-max_history:]
                        
                        # Clear current query
                        st.session_state.current_query = ""
                        
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
                        st.exception(e)
            else:
                st.warning("⚠️ Please enter a query.")
        
        # Display current response
        if st.session_state.current_response:
            st.divider()
            render_response(st.session_state.current_response)
    
    with tab2:
        render_repository_manager()
    
    with tab3:
        render_conversation_history()
    
    with tab4:
        render_session_history_tab()
    
    with tab5:
        st.header("ℹ️ About")
        
        st.markdown("""
        ## 🤖 Intelligent Code Assistant
        
        This application provides an intelligent interface to analyze and understand your codebase using:

        ### 🚀 Technologies
        - **Groq API**: Lightning-fast inference
        - **Llama Models**: State-of-the-art language understanding
        - **RAG (Retrieval Augmented Generation)**: Context-aware responses
        - **ChromaDB**: Efficient vector storage and retrieval
        - **Streamlit**: Modern web interface

        ### 🎯 Features
        - **Session Management**: Each session is automatically saved with conversation history
        - **Intelligent Query Processing**: Analyzes your questions to determine the best retrieval strategy
        - **Multi-Language Support**: Works with Python, JavaScript, Java, Go, C++, Rust, and more
        - **Context-Aware Responses**: Provides detailed answers based on your actual codebase
        - **Source Attribution**: Shows exactly which code files informed each response
        - **Conversation History**: Keeps track of your queries and responses across sessions
        - **Repository Management**: Add and process new repositories through the UI

        ### 📊 Session Features
        - **Automatic Session Creation**: New session created on startup
        - **Persistent Storage**: All conversations saved to disk
        - **Session History**: Access previous sessions and their conversations
        - **Repository Tracking**: Track which repositories were used in each session

        ### 📂 File Structure
        ```
        data/
        ├── sessions/
        │   ├── 20241225_143022_a1b2c3d4/
        │   │   ├── metadata.json
        │   │   ├── conversations/
        │   │   │   ├── conv_1703516222_1.json
        │   │   │   └── conv_1703516245_2.json
        │   │   └── exports/
        │   └── 20241225_150015_e5f6g7h8/
        └── chromadb/
        ```

        ### 💡 Tips for Better Results
        - Be specific in your queries
        - Mention the programming language when relevant
        - Ask about specific files or components for targeted results
        - Use technical terms related to your codebase

        ---
        
        **Version**: 2.0 | **Powered by**: Groq + Llama | **Built with**: Streamlit
        """)

if __name__ == "__main__":
    main()