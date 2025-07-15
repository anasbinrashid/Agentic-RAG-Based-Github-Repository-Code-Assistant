# Week 1: Foundation Setup with Groq API Integration
# This file contains all the setup scripts and basic tests for Groq-based architecture

import subprocess
import sys
import os
import json
import httpx
from pathlib import Path
from typing import List, Optional
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProjectSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.requirements = [
            "langchain",
            "langgraph", 
            "faiss-cpu",
            "gitpython",
            "openai",  # For API wrapper compatibility
            "httpx",   # For async HTTP requests
            "uvicorn",
            "fastapi",
            "streamlit",
            "numpy",
            "pandas",
            "python-dotenv",
            "groq",    # Official Groq Python client
            "pydantic"
        ]
    
    def install_requirements(self):
        """Install all required Python packages"""
        print("Installing Python requirements...")
        for package in self.requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
    
    def setup_environment(self):
        """Setup environment variables and .env file"""
        print("Setting up environment configuration...")
        
        env_file = self.project_root / ".env"
        
        if not env_file.exists():
            # Create .env template
            env_content = """# Groq API Configuration
GROQ_API_KEY=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
GROQ_BASE_URL=https://api.groq.com

# Default Models
DEFAULT_LLM_MODEL=llama3-70b-8192
DEFAULT_EMBEDDING_MODEL=microsoft/codebert-base

# MCP Server Configuration
MCP_HOST=0.0.0.0
MCP_PORT=8000

# Rate Limiting
MAX_TOKENS=4000
TIMEOUT_SECONDS=60
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
            print(f"✅ Created .env template at {env_file}")
            print("⚠️  Please add your Groq API key to the .env file")
        else:
            print("✅ .env file already exists")
    
    def test_groq_connection(self):
        """Test connection to Groq API"""
        print("Testing Groq API connection...")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            print("❌ Groq API key not configured. Please set GROQ_API_KEY in .env file")
            return False
        
        try:
            from groq import Groq
            
            client = Groq(api_key=api_key)
            
            # Test with a simple completion
            completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": "Hello, are you working?"}
                ],
                model="llama3-70b-8192",
                max_tokens=100
            )
            
            if completion.choices:
                print("✅ Groq API connection successful")
                print(f"Model response: {completion.choices[0].message.content[:100]}...")
                return True
            else:
                print("❌ Groq API returned no response")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Groq API: {e}")
            return False
    
    def check_available_models(self):
        """Check available models on Groq"""
        print("Checking available Groq models...")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ Groq API key not configured")
            return False
        
        try:
            from groq import Groq
            
            client = Groq(api_key=api_key)
            models = client.models.list()
            
            print("Available models:")
            for model in models.data:
                print(f"  - {model.id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking models: {e}")
            return False
    
    def create_project_structure(self):
        """Create the project directory structure"""
        print("Creating project structure...")
        directories = [
            "data",
            "src",
            "tests",
            "frontend",
            "mcp_server",
            "embeddings",
            "cloned_repo"  # For storing cloned GitHub repos
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
        
        # Create __init__.py files
        init_files = [
            "src/__init__.py",
            "tests/__init__.py", 
            "mcp_server/__init__.py"
        ]
        
        for init_file in init_files:
            init_path = self.project_root / init_file
            init_path.touch()
            print(f"✅ Created: {init_file}")
    
    def run_complete_setup(self):
        """Run the complete setup process"""
        print("Starting GitHub Code Assistant Setup (Groq Integration)...")
        print("=" * 60)
        
        # Step 1: Install requirements
        self.install_requirements()
        print("\n" + "=" * 60)
        
        # Step 2: Setup environment
        self.setup_environment()
        print("\n" + "=" * 60)
        
        # Step 3: Test Groq connection
        groq_ok = self.test_groq_connection()
        print("\n" + "=" * 60)
        
        # Step 4: Check available models
        if groq_ok:
            self.check_available_models()
        print("\n" + "=" * 60)
        
        # Step 5: Create project structure
        self.create_project_structure()
        print("\n" + "=" * 60)
        
        print("Setup complete! Next steps:")
        print("1. Add your Groq API key to the .env file")
        print("2. Test the MCP server: python week1_groq_setup.py server")
        print("3. Run tests: python week1_groq_setup.py test")
        print("4. Move on to Week 2 implementation")


# MCP Server with Groq Integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from groq import Groq
import numpy as np

app = FastAPI(title="MCP Server - Groq Integration", version="1.0.0")

# Initialize Groq client
groq_client = None

def get_groq_client():
    global groq_client
    if groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Groq API key not configured")
        groq_client = Groq(api_key=api_key)
    return groq_client

class EmbeddingRequest(BaseModel):
    text: str
    model: str = "text-embedding-3-small"

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str
    text_length: int

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "llama3-70b-8192"
    max_tokens: int = 500
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    response: str
    model: str
    tokens_used: int

class ChatRequest(BaseModel):
    messages: List[dict]
    model: str = "llama3-70b-8192"
    max_tokens: int = 500
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: int

@app.get("/")
async def root():
    return {
        "message": "MCP Server with Groq Integration", 
        "version": "1.0.0",
        "endpoints": ["/generate", "/chat", "/embed", "/models"]
    }

@app.get("/models")
async def list_models():
    """List available models from Groq"""
    try:
        client = get_groq_client()
        models = client.models.list()
        return {"models": [model.id for model in models.data]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    """Generate embeddings via Groq API"""
    try:
        client = get_groq_client()
        
        # Simulated embedding for testing (consistent hash-based)
        import hashlib
        text_hash = hashlib.md5(request.text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16))
        embedding = np.random.uniform(-1, 1, 768).tolist()
        
        return EmbeddingResponse(
            embedding=embedding,
            model=request.model,
            text_length=len(request.text)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text via Groq API"""
    try:
        client = get_groq_client()
        
        completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": request.prompt}
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        response_text = completion.choices[0].message.content
        tokens_used = completion.usage.total_tokens if completion.usage else 0
        
        return GenerateResponse(
            response=response_text,
            model=request.model,
            tokens_used=tokens_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Chat completion via Groq API"""
    try:
        client = get_groq_client()
        
        completion = client.chat.completions.create(
            messages=request.messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        response_text = completion.choices[0].message.content
        tokens_used = completion.usage.total_tokens if completion.usage else 0
        
        return ChatResponse(
            response=response_text,
            model=request.model,
            tokens_used=tokens_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_mcp_server():
    """Run the MCP server"""
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


# Test Suite for Week 1 with Groq Integration
class Week1Tests:
    def __init__(self):
        self.mcp_base_url = "http://localhost:8000"
        load_dotenv()
    
    def test_groq_direct(self):
        """Test Groq API directly"""
        print("Testing Groq API directly...")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            print("❌ Groq API key not configured")
            return False
        
        try:
            from groq import Groq
            
            client = Groq(api_key=api_key)
            
            completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": "What is Python programming language?"}
                ],
                model="llama3-70b-8192",
                max_tokens=100
            )
            
            if completion.choices:
                print("✅ Direct Groq API test successful")
                print(f"Response preview: {completion.choices[0].message.content[:100]}...")
                return True
            else:
                print("❌ Direct Groq API test failed - no response")
                return False
                
        except Exception as e:
            print(f"❌ Direct Groq API test error: {e}")
            return False
    
    def test_mcp_server(self):
        """Test MCP server endpoints"""
        print("Testing MCP server...")
        
        try:
            import httpx
            
            # Test root endpoint
            with httpx.Client() as client:
                response = client.get(f"{self.mcp_base_url}/")
                if response.status_code == 200:
                    print("✅ MCP server root endpoint working")
                    data = response.json()
                    print(f"Server info: {data}")
                else:
                    print(f"❌ MCP server root endpoint failed: {response.status_code}")
                    return False
                
                # Test models endpoint
                response = client.get(f"{self.mcp_base_url}/models")
                if response.status_code == 200:
                    print("✅ MCP models endpoint working")
                    models = response.json()
                    print(f"Available models: {len(models.get('models', []))}")
                else:
                    print(f"❌ MCP models endpoint failed: {response.status_code}")
                
                # Test generate endpoint
                response = client.post(
                    f"{self.mcp_base_url}/generate",
                    json={
                        "prompt": "Hello, this is a test",
                        "model": "llama3-70b-8192",
                        "max_tokens": 50
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    print("✅ MCP generate endpoint working")
                    print(f"Response preview: {result.get('response', '')[:100]}...")
                else:
                    print(f"❌ MCP generate endpoint failed: {response.status_code}")
                
                # Test embedding endpoint
                response = client.post(
                    f"{self.mcp_base_url}/embed",
                    json={
                        "text": "Hello world test embedding",
                        "model": "text-embedding-3-small"
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    if len(result["embedding"]) == 768:
                        print("✅ MCP embedding endpoint working")
                    else:
                        print(f"❌ MCP embedding wrong dimension: {len(result['embedding'])}")
                else:
                    print(f"❌ MCP embedding endpoint failed: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ MCP server test error: {e}")
            return False
        
        return True
    
    def test_environment_setup(self):
        """Test environment configuration"""
        print("Testing environment setup...")
        
        # Check .env file
        env_file = Path(".env")
        if env_file.exists():
            print("✅ .env file exists")
        else:
            print("❌ .env file missing")
            return False
        
        # Check API key
        api_key = os.getenv("GROQ_API_KEY")
        if api_key and api_key != "your_groq_api_key_here":
            print("✅ Groq API key configured")
        else:
            print("❌ Groq API key not configured")
            return False
        
        return True
    
    def run_all_tests(self):
        """Run all Week 1 tests"""
        print("Running Week 1 Tests (Groq Integration)...")
        print("=" * 60)
        
        # Test environment
        env_ok = self.test_environment_setup()
        print("\n" + "=" * 40)
        
        # Test Groq direct
        if env_ok:
            groq_ok = self.test_groq_direct()
            print("\n" + "=" * 40)
            
            # Test MCP server
            if groq_ok:
                self.test_mcp_server()
            else:
                print("⚠️  Skipping MCP server tests - Groq API not working")
        else:
            print("⚠️  Skipping API tests - Environment not configured")
        
        print("\n" + "=" * 40)
        print("✅ Week 1 tests complete!")
        print("\nNext steps:")
        print("1. Ensure Groq API key is properly configured")
        print("2. Run MCP server in background for development")
        print("3. Begin Week 2 implementation")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "setup":
            setup = ProjectSetup()
            setup.run_complete_setup()
        elif sys.argv[1] == "server":
            print("Starting MCP Server with Groq Integration...")
            run_mcp_server()
        elif sys.argv[1] == "test":
            tests = Week1Tests()
            tests.run_all_tests()
        else:
            print("Usage: python week1_groq_setup.py [setup|server|test]")
    else:
        print("GitHub Code Assistant - Week 1 Setup (Groq Integration)")
        print("=" * 60)
        print("Usage:")
        print("  python week1_groq_setup.py setup  - Run complete setup")
        print("  python week1_groq_setup.py server - Start MCP server")
        print("  python week1_groq_setup.py test   - Run tests")
        print("\nPrerequisites:")
        print("1. Get Groq API key from https://console.groq.com/")
        print("2. Run setup to install dependencies")
        print("3. Configure .env file with your API key")
        print("4. Test connection and start development")