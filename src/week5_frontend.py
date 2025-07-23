# Week 5 Frontend - Streamlit Interface for Intelligent Code Assistant
# Modern web interface for the Groq-powered RAG code assistant

import streamlit as st
import os
import json
import time
from datetime import datetime
from pathlib import Path
import plotly.express as px
import pandas as pd
import subprocess
import threading
from queue import Queue
import tempfile
import git
from urllib.parse import urlparse
import re
from dataclasses import asdict

# Import the agent
from week4_agent import GroqCodeAgent, AgentResponse
from week2_chunker import GitHubRepoProcessor, CodeChunker, ChromaDBManager

# Page config
st.set_page_config(
    page_title="🤖 Intelligent Code Assistant",
    page_icon="🤖",
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
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e9ecef;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
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
    """Render the sidebar with settings and stats"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        available_models = [
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        
        new_model = st.selectbox(
            "🧠 Model",
            available_models,
            index=available_models.index(st.session_state.settings['model']),
            help="Choose the language model for processing queries"
        )
        
        if new_model != st.session_state.settings['model']:
            st.session_state.settings['model'] = new_model
            st.session_state.agent = None  # Force re-initialization
            st.rerun()
        
        # Database path
        db_path = st.text_input(
            "💾 Database Path",
            value=st.session_state.settings['db_path'],
            help="Path to your ChromaDB database"
        )
        
        if db_path != st.session_state.settings['db_path']:
            st.session_state.settings['db_path'] = db_path
            st.session_state.agent = None  # Force re-initialization
        
        # Base directory
        base_dir = st.text_input(
            "📁 Base Directory",
            value=st.session_state.settings['base_dir'],
            help="Base directory for storing repositories and data"
        )
        
        if base_dir != st.session_state.settings['base_dir']:
            st.session_state.settings['base_dir'] = base_dir
        
        # Max history
        st.session_state.settings['max_history'] = st.slider(
            "📚 Max History",
            min_value=10,
            max_value=100,
            value=st.session_state.settings['max_history'],
            help="Maximum number of queries to keep in history"
        )
        
        st.divider()
        
        # Stats section
        st.header("📊 Statistics")
        
        if st.session_state.agent and st.button("🔄 Refresh Stats"):
            try:
                with st.spinner("Loading stats..."):
                    st.session_state.stats = st.session_state.agent.get_stats()
            except Exception as e:
                st.error(f"Failed to load stats: {e}")
        
        if st.session_state.stats:
            stats = st.session_state.stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", stats.get('total_chunks', 0))
            with col2:
                st.metric("Languages", len(stats.get('languages', {})))
            
            st.metric("Repositories", len(stats.get('repositories', {})))
            
            # Top languages chart
            languages = stats.get('languages', {})
            if languages:
                st.subheader("🔤 Language Distribution")
                lang_df = pd.DataFrame(
                    list(languages.items()), 
                    columns=['Language', 'Count']
                ).sort_values('Count', ascending=False).head(8)
                
                fig = px.pie(
                    lang_df, 
                    values='Count', 
                    names='Language',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Session info
        st.header("📝 Session Info")
        st.metric("Queries Asked", len(st.session_state.conversation_history))
        
        if st.session_state.conversation_history:
            avg_confidence = sum(
                item['response'].confidence 
                for item in st.session_state.conversation_history
            ) / len(st.session_state.conversation_history)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.conversation_history = []
            st.session_state.current_response = None
            st.rerun()

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
    
    with col4:
        st.markdown(f"""
        <div class="metrics-card">
            <h4>Sources</h4>
            <p>{len(response.sources)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Main answer
    st.subheader("📝 Answer")
    st.markdown(response.answer)
    
    # Reasoning
    if response.reasoning:
        st.subheader("🧠 Reasoning")
        st.info(response.reasoning)
    
    # Sources
    if response.sources:
        st.subheader("📚 Sources Used")
        
        # Create DataFrame for sources table
        sources_data = []
        for i, source in enumerate(response.sources, 1):
            sources_data.append({
                "#": i,
                "File": source['filename'],
                "Repository": source['repository'],
                "Language": source['language'],
                "Lines": source['lines'],
                "Relevance": f"{source['relevance_score']:.3f}"
            })
        
        sources_df = pd.DataFrame(sources_data)
        st.dataframe(sources_df, use_container_width=True)
        
        # Detailed source cards
        with st.expander("🔍 Detailed Source Information"):
            for i, source in enumerate(response.sources, 1):
                st.markdown(f"""
                <div class="source-card">
                    <strong>Source {i}: {source['filename']}</strong><br>
                    📁 Repository: {source['repository']}<br>
                    🔤 Language: {source['language']}<br>
                    📏 Lines: {source['lines']}<br>
                    📊 Relevance Score: {source['relevance_score']:.3f}<br>
                    📂 Path: {source.get('file_path', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
    
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

## Sources
"""
        for i, source in enumerate(response.sources, 1):
            markdown_report += f"\n{i}. **{source['filename']}** ({source['language']})\n"
            markdown_report += f"   - Repository: {source['repository']}\n"
            markdown_report += f"   - Lines: {source['lines']}\n"
            markdown_report += f"   - Relevance: {source['relevance_score']:.3f}\n"
        
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
    
    st.subheader("📚 Conversation History")
    
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

def is_valid_git_url(url: str) -> bool:
    """Validate if the URL is a valid git repository URL"""
    git_patterns = [
        r'^https://github\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?$',
        r'^https://gitlab\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?$',
        r'^https://bitbucket\.org/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?$',
        r'^git@github\.com:[\w\-\.]+/[\w\-\.]+\.git$',
        r'^git@gitlab\.com:[\w\-\.]+/[\w\-\.]+\.git$',
    ]
    
    return any(re.match(pattern, url.strip()) for pattern in git_patterns)

def extract_repo_name(url: str) -> str:
    """Extract repository name from URL"""
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if path.endswith('.git'):
        path = path[:-4]
    return path.split('/')[-1] if '/' in path else path

def process_repository(repo_url: str, progress_queue: Queue, settings: dict):
    """Process a repository by cloning and chunking it"""
    try:
        progress_queue.put({"status": "cloning", "message": "Cloning repository..."})
        
        # Initialize processors with passed settings
        repo_processor = GitHubRepoProcessor(f"{settings['base_dir']}/repos")
        chunker = CodeChunker()
        chroma_manager = ChromaDBManager(settings['db_path'])
        
        # Clone repository
        repo_name = extract_repo_name(repo_url)
        repo_path = repo_processor.clone_repository(repo_url, repo_name)
        progress_queue.put({"status": "cloned", "message": f"Repository cloned to {repo_path}"})
        
        # Process files and create chunks
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
        
        # Extract metadata and add to ChromaDB
        progress_queue.put({"status": "embedding", "message": "Embedding chunks..."})
        repo_metadata = repo_processor.extract_metadata(repo_path, repo_name, repo_url)
        repo_metadata.total_chunks = len(all_chunks)
        
        # Add to ChromaDB
        chroma_manager.add_repository_metadata(repo_metadata)
        chroma_manager.add_chunks_with_metadata(all_chunks, repo_metadata)
        
        progress_queue.put({
            "status": "success", 
            "message": f"Successfully processed repository! Added {len(all_chunks)} chunks.",
            "stats": {
                "files_processed": len(files),
                "chunks_created": len(all_chunks),
                "languages": repo_metadata.languages
            }
        })
        
    except Exception as e:
        progress_queue.put({"status": "error", "message": f"Error processing repository: {str(e)}"})


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
                    elif update["status"] == "processing":
                        st.info(f"⚙️ {update['message']}")
                    elif update["status"] == "chunking":
                        st.info(f"✂️ {update['message']}")
                    elif update["status"] == "embedding":
                        st.info(f"📥 {update['message']}")
                    elif update["status"] == "success":
                        st.success(f"🎉 {update['message']}")
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
                repo_data.append({
                    "Repository": repo_name,
                    "Chunks": count
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
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query Assistant", "📂 Repository Manager", "📚 History", "ℹ️ About"])
    
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
        - **Intelligent Query Processing**: Analyzes your questions to determine the best retrieval strategy
        - **Multi-Language Support**: Works with Python, JavaScript, Java, Go, C++, Rust, and more
        - **Context-Aware Responses**: Provides detailed answers based on your actual codebase
        - **Source Attribution**: Shows exactly which code files informed each response
        - **Conversation History**: Keeps track of your queries and responses
        - **Repository Management**: Add and process new repositories through the UI

        ### 📊 Query Types Supported
        - **Code Explanation**: Understanding how specific code works
        - **Implementation Guidance**: How to implement features or patterns
        - **Debugging Help**: Finding and fixing issues in code
        - **Code Search**: Locating specific functionality or patterns
        - **Architecture Analysis**: Understanding system design and structure
        - **Best Practices**: Learning recommended approaches and patterns

        ### ⚙️ Setup Requirements
        1. **GROQ_API_KEY**: Set your Groq API key in environment variables
        2. **Git**: Required for repository cloning
        3. **Python Dependencies**: Install required Python packages

        ### 💡 Tips for Better Results
        - Be specific in your queries
        - Mention the programming language when relevant
        - Ask about specific files or components for targeted results
        - Use technical terms related to your codebase

        ---
        
        **Version**: 1.0 | **Powered by**: Groq + Llama | **Built with**: Streamlit
        """)

if __name__ == "__main__":
    main()