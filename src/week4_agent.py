# Week 4 Agent - Intelligent Code Assistant with Groq API and Llama
# Focus: RAG-based code assistant using stored chunk metadata for accurate retrieval

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import re

from groq import Groq
from dotenv import load_dotenv

# Import our retrieval system
from week3_retrieval import CodeRetriever, RetrievalInterface

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    """Structured response from the agent"""
    query: str
    answer: str
    sources: List[Dict]
    reasoning: str
    confidence: float
    response_time: float
    model_used: str

class GroqCodeAgent:
    """Intelligent code assistant using Groq API and retrieval system"""
    
    def __init__(self, db_path: str = "data/chromadb", model: str = "llama3-8b-8192"):
        # Initialize Groq client
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.client = Groq(api_key=self.groq_api_key)
        self.model = model
        
        # Initialize retrieval system
        self.retriever = CodeRetriever(db_path)
        self.retrieval_interface = RetrievalInterface(db_path)
        
        # Available models
        self.available_models = [
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        
        logger.info(f"Initialized GroqCodeAgent with model: {model}")
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the user query to determine intent and optimal retrieval strategy"""
        
        # Intent patterns
        intent_patterns = {
            'code_explanation': [
                r'explain|what does|how does|understand|clarify|breakdown',
                r'what is this|what\'s this|describe'
            ],
            'code_search': [
                r'find|search|look for|locate|show me',
                r'where is|where can I find'
            ],
            'implementation': [
                r'how to|how do I|implement|create|build|make',
                r'write|code|develop|generate'
            ],
            'debugging': [
                r'debug|fix|error|bug|issue|problem|wrong',
                r'not working|broken|fails'
            ],
            'comparison': [
                r'compare|difference|versus|vs|better|alternative',
                r'which is|what\'s the difference'
            ],
            'documentation': [
                r'document|docs|readme|api|reference',
                r'documentation|manual|guide'
            ]
        }
        
        # Language detection
        language_patterns = {
            'python': r'\b(python|py|pip|django|flask|numpy|pandas|def |class |import )\b',
            'javascript': r'\b(javascript|js|node|npm|react|vue|angular|function|const|let|var)\b',
            'java': r'\b(java|class|public|private|static|void|String|int|boolean)\b',
            'go': r'\b(go|golang|func|package|import|var|type|struct)\b',
            'cpp': r'\b(c\+\+|cpp|include|namespace|class|public|private|int|char|void)\b',
            'rust': r'\b(rust|cargo|fn|let|mut|struct|impl|trait|pub)\b'
        }
        
        query_lower = query.lower()
        
        # Determine intent
        detected_intent = 'general'
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_intent = intent
                    break
            if detected_intent != 'general':
                break
        
        # Determine language preference
        detected_language = None
        for lang, pattern in language_patterns.items():
            if re.search(pattern, query_lower):
                detected_language = lang
                break
        
        # Determine complexity and retrieval needs
        complexity_indicators = [
            'complex', 'detailed', 'comprehensive', 'advanced', 'deep',
            'architecture', 'design pattern', 'best practice', 'optimization'
        ]
        
        is_complex = any(indicator in query_lower for indicator in complexity_indicators)
        
        return {
            'intent': detected_intent,
            'language': detected_language,
            'complexity': 'high' if is_complex else 'medium',
            'requires_multiple_sources': is_complex or detected_intent in ['comparison', 'implementation'],
            'retrieval_count': 8 if is_complex else 5
        }
    
    def _build_retrieval_strategy(self, query: str, analysis: Dict) -> Dict:
        """Build retrieval strategy based on query analysis"""
        
        # Base retrieval parameters
        strategy = {
            'query': query,
            'n_results': analysis['retrieval_count'],
            'use_intelligent_search': True,
            'language_filter': analysis['language'],
            'intent': analysis['intent']
        }
        
        # Adjust strategy based on intent
        if analysis['intent'] == 'code_explanation':
            strategy['focus'] = 'semantic_similarity'
            strategy['prefer_complete_files'] = True
            
        elif analysis['intent'] == 'implementation':
            strategy['focus'] = 'functional_similarity'
            strategy['prefer_examples'] = True
            
        elif analysis['intent'] == 'debugging':
            strategy['focus'] = 'error_patterns'
            strategy['include_context'] = True
            
        elif analysis['intent'] == 'comparison':
            strategy['n_results'] = strategy['n_results'] * 2  # Need more examples
            strategy['diversity'] = True
            
        return strategy
    
    def _retrieve_relevant_chunks(self, strategy: Dict) -> List[Dict]:
        """Retrieve relevant code chunks based on strategy"""
        
        try:
            if strategy['use_intelligent_search']:
                # Use intelligent search with repository context
                results = self.retriever.intelligent_search(
                    strategy['query'], 
                    strategy['n_results']
                )
                chunks = results.get('results', [])
                
                # Add repository context to each chunk
                for chunk in chunks:
                    chunk['repository_context'] = results.get('relevant_repositories', [])
                
            else:
                # Use basic search with language filter
                if strategy['language_filter']:
                    chunks = self.retriever.search_by_language(
                        strategy['query'],
                        strategy['language_filter'],
                        strategy['n_results']
                    )
                else:
                    chunks = self.retriever.search(
                        strategy['query'],
                        strategy['n_results']
                    )
            
            # Apply post-processing based on intent
            if strategy.get('prefer_complete_files'):
                chunks = sorted(chunks, key=lambda x: x.get('chunk_type', '') == 'complete_file', reverse=True)
            
            if strategy.get('diversity'):
                # Ensure diversity in repositories and languages
                seen_repos = set()
                seen_languages = set()
                diverse_chunks = []
                
                for chunk in chunks:
                    repo = chunk.get('repo_name', '')
                    lang = chunk.get('language', '')
                    
                    if len(diverse_chunks) < strategy['n_results'] // 2:
                        diverse_chunks.append(chunk)
                    elif repo not in seen_repos or lang not in seen_languages:
                        diverse_chunks.append(chunk)
                        seen_repos.add(repo)
                        seen_languages.add(lang)
                
                chunks = diverse_chunks
            
            return chunks[:strategy['n_results']]
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def _build_context_prompt(self, query: str, chunks: List[Dict], analysis: Dict) -> str:
        """Build context-aware prompt for the LLM"""
        
        # System prompt based on intent
        intent_prompts = {
            'code_explanation': "You are an expert code analyst. Explain code clearly and thoroughly, focusing on functionality, purpose, and key concepts.",
            'code_search': "You are a helpful code search assistant. Help users find relevant code examples and explain their relevance.",
            'implementation': "You are a senior software engineer. Provide practical implementation guidance with clear examples and best practices.",
            'debugging': "You are a debugging expert. Analyze code for potential issues and provide solutions with explanations.",
            'comparison': "You are a technical consultant. Compare different approaches objectively, highlighting pros and cons.",
            'documentation': "You are a technical writer. Provide clear, comprehensive documentation and explanations.",
            'general': "You are a knowledgeable programming assistant. Provide accurate, helpful responses about code and programming concepts."
        }
        
        system_prompt = intent_prompts.get(analysis['intent'], intent_prompts['general'])
        
        # Build context from retrieved chunks
        context_sections = []
        
        for i, chunk in enumerate(chunks, 1):
            # Extract key information
            repo_name = chunk.get('repo_name', 'Unknown')
            filename = chunk.get('filename', 'Unknown')
            language = chunk.get('language', 'Unknown')
            lines = chunk.get('lines', 'Unknown')
            relevance = chunk.get('relevance_score', 0.0)
            
            # Get repository context if available
            repo_context = chunk.get('repo_context', {})
            repo_langs = repo_context.get('languages', [])
            repo_deps = repo_context.get('dependencies', [])
            
            context_section = f"""
CODE CHUNK {i}:
Repository: {repo_name}
File: {filename}
Language: {language}
Lines: {lines}
Relevance Score: {relevance:.3f}
Repository Languages: {', '.join(repo_langs[:3])}
Dependencies: {', '.join(repo_deps[:3])}

```{language}
{chunk.get('content', '')}
```
"""
            context_sections.append(context_section)
        
        # Build repository summary
        repo_summary = self._build_repository_summary(chunks)
        
        # Build final prompt
        prompt = f"""
{system_prompt}

CONTEXT INFORMATION:
{repo_summary}

RELEVANT CODE CHUNKS:
{''.join(context_sections)}

USER QUERY: {query}

INSTRUCTIONS:
1. Analyze the provided code chunks in relation to the user's query
2. Provide a comprehensive, accurate answer based on the retrieved code
3. Reference specific code examples when relevant
4. Explain your reasoning process
5. If the code chunks don't fully answer the query, clearly state what information is missing
6. Provide practical insights and best practices where appropriate
7. Always cite which code chunks you're referencing (use "Code Chunk X" format)

RESPONSE FORMAT:
- Start with a direct answer to the query
- Provide detailed explanation with code references
- Include relevant examples from the retrieved code
- End with additional insights or recommendations

Answer:"""
        
        return prompt
    
    def _build_repository_summary(self, chunks: List[Dict]) -> str:
        """Build a summary of the repositories and their context"""
        
        repo_info = {}
        
        for chunk in chunks:
            repo_name = chunk.get('repo_name', 'Unknown')
            if repo_name not in repo_info:
                repo_info[repo_name] = {
                    'languages': set(),
                    'files': set(),
                    'total_chunks': 0,
                    'context': chunk.get('repo_context', {})
                }
            
            repo_info[repo_name]['languages'].add(chunk.get('language', 'Unknown'))
            repo_info[repo_name]['files'].add(chunk.get('filename', 'Unknown'))
            repo_info[repo_name]['total_chunks'] += 1
        
        summary_parts = []
        for repo_name, info in repo_info.items():
            summary_parts.append(
                f"Repository '{repo_name}': {info['total_chunks']} chunks, "
                f"Languages: {', '.join(info['languages'])}, "
                f"Files: {', '.join(list(info['files'])[:3])}"
            )
        
        return "REPOSITORY SUMMARY:\n" + '\n'.join(summary_parts) + "\n"
    
    def _call_groq_api(self, prompt: str) -> Tuple[str, Dict]:
        """Call Groq API with error handling and response parsing"""
        
        try:
            start_time = datetime.now()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.1,  # Low temperature for more accurate, factual responses
                top_p=0.9,
                stream=False
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            content = response.choices[0].message.content
            
            # Extract usage information
            usage_info = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
                'response_time': response_time,
                'model': self.model
            }
            
            return content, usage_info
            
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}", {}
    
    def _calculate_confidence(self, query: str, chunks: List[Dict], response: str) -> float:
        """Calculate confidence score based on various factors"""
        
        # Base confidence factors
        factors = {
            'relevance_scores': sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks) if chunks else 0,
            'chunk_count': min(len(chunks) / 5, 1.0),  # Normalized by expected count
            'response_length': min(len(response) / 500, 1.0),  # Longer responses often more comprehensive
            'code_references': len(re.findall(r'Code Chunk \d+', response)) / len(chunks) if chunks else 0,
            'query_complexity': 0.8 if len(query.split()) > 10 else 0.6
        }
        
        # Weight the factors
        weights = {
            'relevance_scores': 0.4,
            'chunk_count': 0.2,
            'response_length': 0.1,
            'code_references': 0.2,
            'query_complexity': 0.1
        }
        
        confidence = sum(factors[key] * weights[key] for key in factors)
        return min(confidence, 1.0)
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from the response"""
        
        # Look for reasoning patterns
        reasoning_patterns = [
            r'because\s+(.+?)(?:\.|$)',
            r'since\s+(.+?)(?:\.|$)',
            r'due to\s+(.+?)(?:\.|$)',
            r'this is\s+(.+?)(?:\.|$)',
            r'the reason\s+(.+?)(?:\.|$)'
        ]
        
        reasoning_parts = []
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            reasoning_parts.extend(matches)
        
        if reasoning_parts:
            return '. '.join(reasoning_parts[:3])  # Top 3 reasoning points
        
        # Fallback: extract first explanatory sentence
        sentences = response.split('. ')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['explain', 'shows', 'demonstrates', 'indicates']):
                return sentence
        
        return "Based on analysis of the retrieved code chunks"
    
    def query(self, query: str) -> AgentResponse:
        """Process a user query and return a comprehensive response"""
        
        start_time = datetime.now()
        
        try:
            # Step 1: Analyze query intent and build strategy
            analysis = self._analyze_query_intent(query)
            logger.info(f"Query analysis: {analysis}")
            
            # Step 2: Build retrieval strategy
            strategy = self._build_retrieval_strategy(query, analysis)
            logger.info(f"Retrieval strategy: {strategy}")
            
            # Step 3: Retrieve relevant chunks
            chunks = self._retrieve_relevant_chunks(strategy)
            logger.info(f"Retrieved {len(chunks)} code chunks")
            
            if not chunks:
                return AgentResponse(
                    query=query,
                    answer="I couldn't find relevant code chunks for your query. Please make sure repositories have been processed and try rephrasing your question.",
                    sources=[],
                    reasoning="No relevant code chunks found in the database",
                    confidence=0.0,
                    response_time=0.0,
                    model_used=self.model
                )
            
            # Step 4: Build context-aware prompt
            prompt = self._build_context_prompt(query, chunks, analysis)
            
            # Step 5: Call Groq API
            response_content, usage_info = self._call_groq_api(prompt)
            
            # Step 6: Process response and calculate metrics
            confidence = self._calculate_confidence(query, chunks, response_content)
            reasoning = self._extract_reasoning(response_content)
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            # Step 7: Prepare source information
            sources = []
            for chunk in chunks:
                sources.append({
                    'filename': chunk.get('filename', 'Unknown'),
                    'repository': chunk.get('repo_name', 'Unknown'),
                    'language': chunk.get('language', 'Unknown'),
                    'lines': chunk.get('lines', 'Unknown'),
                    'relevance_score': chunk.get('relevance_score', 0.0),
                    'file_path': chunk.get('file_path', 'Unknown')
                })
            
            return AgentResponse(
                query=query,
                answer=response_content,
                sources=sources,
                reasoning=reasoning,
                confidence=confidence,
                response_time=total_time,
                model_used=self.model
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return AgentResponse(
                query=query,
                answer=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                reasoning="Error occurred during processing",
                confidence=0.0,
                response_time=0.0,
                model_used=self.model
            )

class InteractiveCodeAgent:
    """Interactive interface for the code agent"""
    
    def __init__(self, db_path: str = "data/chromadb", model: str = "llama3-8b-8192"):
        self.agent = GroqCodeAgent(db_path, model)
        self.conversation_history = []
    
    def start_interactive_session(self):
        """Start an interactive session with the agent"""
        
        print("🤖 Intelligent Code Assistant (Powered by Groq + Llama)")
        print("=" * 60)
        print("Ask me anything about your codebase! Type 'help' for commands or 'quit' to exit.")
        print()
        
        while True:
            try:
                user_input = input("\n🔍 Your query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                
                if user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                if user_input.lower().startswith('model '):
                    self._change_model(user_input[6:])
                    continue
                
                # Process the query
                print("\n🔄 Processing your query...")
                response = self.agent.query(user_input)
                
                # Display response
                self._display_response(response)
                
                # Add to conversation history
                self.conversation_history.append({
                    'query': user_input,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                })
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 Available Commands:")
        print("  help     - Show this help message")
        print("  stats    - Show collection statistics")
        print("  history  - Show conversation history")
        print("  model <name> - Change the model (llama3-8b-8192, llama3-70b-8192, etc.)")
        print("  quit     - Exit the assistant")
        print("\n💡 Query Examples:")
        print("  'Explain how authentication works in this codebase'")
        print("  'Find examples of error handling in Python'")
        print("  'How to implement a REST API in this project?'")
        print("  'Debug this SQL query performance issue'")
        print("  'Compare different sorting algorithms used here'")
    
    def _show_stats(self):
        """Show collection statistics"""
        stats = self.agent.retriever.get_collection_stats()
        
        print("\n📊 Codebase Statistics:")
        print(f"  Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  Languages: {len(stats.get('languages', {}))}")
        print(f"  Repositories: {len(stats.get('repositories', {}))}")
        
        # Show top languages
        languages = stats.get('languages', {})
        if languages:
            print("\n🔤 Top Languages:")
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {lang}: {count}")
    
    def _show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("No conversation history yet.")
            return
        
        print(f"\n📝 Conversation History ({len(self.conversation_history)} queries):")
        for i, item in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            print(f"{i}. {item['query']}")
            print(f"   Confidence: {item['response'].confidence:.2f}")
            print(f"   Sources: {len(item['response'].sources)}")
    
    def _change_model(self, model_name: str):
        """Change the model"""
        model_name = model_name.strip()
        
        if model_name in self.agent.available_models:
            self.agent.model = model_name
            print(f"✅ Model changed to: {model_name}")
        else:
            print(f"❌ Invalid model. Available models: {', '.join(self.agent.available_models)}")
    
    def _display_response(self, response: AgentResponse):
        """Display the agent response in a formatted way"""
        print("\n" + "=" * 80)
        print(f"🤖 Assistant Response")
        print("=" * 80)
        
        print(f"\n📝 Answer:")
        print(response.answer)
        
        print(f"\n📊 Response Details:")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Response Time: {response.response_time:.2f}s")
        print(f"  Model Used: {response.model_used}")
        print(f"  Sources Used: {len(response.sources)}")
        
        if response.reasoning:
            print(f"\n🧠 Reasoning: {response.reasoning}")
        
        if response.sources:
            print(f"\n📚 Sources:")
            for i, source in enumerate(response.sources, 1):
                print(f"  {i}. {source['filename']} ({source['language']})")
                print(f"     Repository: {source['repository']}")
                print(f"     Lines: {source['lines']}")
                print(f"     Relevance: {source['relevance_score']:.3f}")
        
        print("=" * 80)

# CLI Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Code Assistant")
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--model", type=str, default="llama3-8b-8192", 
                       help="Model to use (llama3-8b-8192, llama3-70b-8192, etc.)")
    parser.add_argument("--db-path", type=str, default="data/chromadb", 
                       help="Database path")
    parser.add_argument("--interactive", action="store_true", 
                       help="Start interactive session")
    parser.add_argument("--output", type=str, help="Output file for response")
    
    args = parser.parse_args()
    
    # Check for GROQ_API_KEY
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable is required")
        print("Please set it in your .env file or environment")
        return
    
    try:
        if args.interactive or not args.query:
            # Start interactive session
            agent = InteractiveCodeAgent(args.db_path, args.model)
            agent.start_interactive_session()
        else:
            # Process single query
            agent = GroqCodeAgent(args.db_path, args.model)
            print(f"🔄 Processing query: {args.query}")
            
            response = agent.query(args.query)
            
            # Display response
            print("\n" + "=" * 80)
            print(f"🤖 Assistant Response")
            print("=" * 80)
            print(f"\n📝 Answer:")
            print(response.answer)
            
            print(f"\n📊 Response Details:")
            print(f"  Confidence: {response.confidence:.2f}")
            print(f"  Response Time: {response.response_time:.2f}s")
            print(f"  Model Used: {response.model_used}")
            print(f"  Sources Used: {len(response.sources)}")
            
            if response.sources:
                print(f"\n📚 Sources:")
                for i, source in enumerate(response.sources, 1):
                    print(f"  {i}. {source['filename']} ({source['language']})")
                    print(f"     Repository: {source['repository']}")
                    print(f"     Relevance: {source['relevance_score']:.3f}")
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(asdict(response), f, indent=2)
                print(f"\n💾 Response saved to: {args.output}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("1. Set GROQ_API_KEY in your environment")
        print("2. Processed repositories using week2_chunker.py")
        print("3. Verified ChromaDB is accessible")

if __name__ == "__main__":
    main()