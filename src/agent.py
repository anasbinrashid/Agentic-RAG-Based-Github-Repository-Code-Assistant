# Intelligent Code Assistant with Groq API and Llama
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
from retrieval import CodeRetriever, RetrievalInterface

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
    
class QueryExpansionLogger:
    """Logs query expansions to a text file"""
    
    def __init__(self, log_file: str = "query_expansions.txt"):
        self.log_file = log_file
        self.ensure_log_file()
    
    def ensure_log_file(self):
        """Create log file with header if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("=== QUERY EXPANSION LOG ===\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
    
    def log_expansion(self, original_query: str, expanded_queries: List[str], 
                     language: Optional[str], intent: str, user_session: str = None):
        """Log a query expansion session"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
                if user_session:
                    f.write(f"Session: {user_session}\n")
                f.write(f"Original Query: {original_query}\n")
                f.write(f"Language: {language or 'Any'}\n")
                f.write(f"Intent: {intent}\n")
                f.write(f"Expansions Generated: {len(expanded_queries)}\n")
                f.write("Expanded Queries:\n")
                
                for i, query in enumerate(expanded_queries, 1):
                    f.write(f"  {i}. {query}\n")
                
                f.write("-" * 40 + "\n\n")
                
        except Exception as e:
            logger.warning(f"Failed to log query expansion: {e}")
    
    def log_retrieval_results(self, original_query: str, chunks_found: int, 
                            top_chunks: List[Dict], expansion_effectiveness: Dict):
        """Log retrieval effectiveness of expanded queries"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"RETRIEVAL RESULTS for: {original_query}\n")
                f.write(f"Total chunks found: {chunks_found}\n")
                f.write(f"Expansion effectiveness:\n")
                
                for query, stats in expansion_effectiveness.items():
                    f.write(f"  '{query}': {stats['chunks']} chunks, "
                           f"avg_score: {stats['avg_score']:.3f}\n")
                
                if top_chunks:
                    f.write("Top 3 chunks:\n")
                    for i, chunk in enumerate(top_chunks[:3], 1):
                        f.write(f"  {i}. {chunk.get('filename', 'Unknown')} "
                               f"(score: {chunk.get('final_relevance_score', 0):.3f})\n")
                
                f.write("-" * 40 + "\n\n")
                
        except Exception as e:
            logger.warning(f"Failed to log retrieval results: {e}")
    
    def get_expansion_stats(self) -> Dict:
        """Get statistics about query expansions from log file"""
        stats = {
            'total_expansions': 0,
            'total_queries': 0,
            'avg_expansions_per_query': 0,
            'common_intents': {},
            'common_languages': {}
        }
        
        try:
            if not os.path.exists(self.log_file):
                return stats
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Count total queries
                stats['total_queries'] = content.count('Original Query:')
                
                # Count total expansions
                import re
                expansions = re.findall(r'Expansions Generated: (\d+)', content)
                stats['total_expansions'] = sum(int(x) for x in expansions)
                
                # Calculate average
                if stats['total_queries'] > 0:
                    stats['avg_expansions_per_query'] = stats['total_expansions'] / stats['total_queries']
                
                # Count intents and languages
                intents = re.findall(r'Intent: (\w+)', content)
                languages = re.findall(r'Language: (\w+)', content)
                
                for intent in intents:
                    stats['common_intents'][intent] = stats['common_intents'].get(intent, 0) + 1
                
                for lang in languages:
                    if lang != 'Any':
                        stats['common_languages'][lang] = stats['common_languages'].get(lang, 0) + 1
                
        except Exception as e:
            logger.warning(f"Failed to get expansion stats: {e}")
        
        return stats

class QueryExpansionEngine:
    """Advanced query expansion for better code retrieval"""
    
    def __init__(self, groq_client, model="llama3-70b-8192", log_file="query_expansions.txt"):
        self.client = groq_client
        self.model = model
        self.logger = QueryExpansionLogger(log_file)
        
        # Enhanced programming domain synonyms
        self.code_synonyms = {
            'function': ['method', 'procedure', 'routine', 'subroutine'],
            'class': ['object', 'type', 'component', 'entity'],
            'interface': ['contract', 'protocol', 'API', 'abstraction'],
            'variable': ['field', 'attribute', 'property', 'parameter'],
            'implementation': ['code', 'logic', 'algorithm', 'solution'],
            'error': ['exception', 'bug', 'issue', 'problem'],
            'database': ['db', 'storage', 'persistence', 'data'],
            'authentication': ['auth', 'login', 'security', 'access'],
            'library': ['framework', 'package', 'module', 'dependency']
        }
        
        # Code-specific action words
        self.action_synonyms = {
            'implement': ['create', 'build', 'develop', 'code'],
            'explain': ['describe', 'clarify', 'detail', 'breakdown'],
            'find': ['search', 'locate', 'get', 'retrieve'],
            'use': ['utilize', 'apply', 'employ', 'work with'],
            'handle': ['manage', 'process', 'deal with', 'control']
        }
    
    def expand_query(self, query: str, language: Optional[str] = None, 
                    intent: str = 'general', user_session: str = None) -> List[str]:
        """Generate better expanded versions of the query"""
        
        # Start with original query
        expanded_queries = [query]
        
        # Extract key terms from the query
        key_terms = self._extract_key_terms(query)
        
        # 1. Generate focused expansions (most important)
        focused_expansions = self._generate_focused_expansions(query, key_terms, language, intent)
        expanded_queries.extend(focused_expansions)
        
        # 2. Generate semantic variations
        semantic_variations = self._generate_semantic_variations(query, key_terms)
        expanded_queries.extend(semantic_variations)
        
        # 3. Generate context-specific variations
        context_variations = self._generate_context_variations(query, intent, language)
        expanded_queries.extend(context_variations)
        
        # 4. Use LLM for complex queries only (improved prompt)
        if len(query.split()) > 5 and intent != 'general':
            llm_expansions = self._improved_llm_expansion(query, language, intent)
            expanded_queries.extend(llm_expansions)
        
        # Clean and deduplicate
        final_queries = self._clean_and_filter_queries(expanded_queries, query)
        
        # Log the expansion
        self.logger.log_expansion(query, final_queries, language, intent, user_session)
        
        return final_queries
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key programming terms from the query"""
        # Remove stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'how', 'what', 'why', 'when', 'where'}
        
        # Clean query
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
        words = cleaned.split()
        
        # Filter meaningful terms
        key_terms = []
        for word in words:
            if (len(word) > 2 and 
                word not in stop_words and 
                not word.isdigit()):
                key_terms.append(word)
        
        return key_terms
    
    def _generate_focused_expansions(self, query: str, key_terms: List[str], 
                                   language: Optional[str], intent: str) -> List[str]:
        """Generate focused expansions based on key terms"""
        expansions = []
        
        # Focus on the most important terms
        core_terms = key_terms[:3]  # Take top 3 terms
        
        for term in core_terms:
            # Synonym replacement
            if term in self.code_synonyms:
                for synonym in self.code_synonyms[term][:2]:  # Limit to 2 synonyms
                    new_query = query.lower().replace(term, synonym)
                    expansions.append(new_query)
            
            # Action word replacement
            if term in self.action_synonyms:
                for synonym in self.action_synonyms[term][:2]:
                    new_query = query.lower().replace(term, synonym)
                    expansions.append(new_query)
        
        # Language-specific expansions
        if language:
            lang_expansions = self._generate_language_specific_expansions(query, language)
            expansions.extend(lang_expansions)
        
        return expansions[:4]  # Limit to 4 focused expansions
    
    def _generate_language_specific_expansions(self, query: str, language: str) -> List[str]:
        """Generate language-specific query expansions"""
        expansions = []
        
        # Language-specific term mappings
        lang_mappings = {
            'java': {
                'class': 'Java class',
                'method': 'Java method',
                'interface': 'Java interface',
                'exception': 'Java exception',
                'package': 'Java package'
            },
            'python': {
                'function': 'Python function',
                'class': 'Python class',
                'module': 'Python module',
                'decorator': 'Python decorator'
            },
            'javascript': {
                'function': 'JavaScript function',
                'class': 'JavaScript class',
                'module': 'JavaScript module',
                'promise': 'JavaScript promise'
            }
        }
        
        if language in lang_mappings:
            for term, lang_term in lang_mappings[language].items():
                if term in query.lower():
                    expansions.append(query.lower().replace(term, lang_term))
        
        # Add language prefix/suffix
        if not language.lower() in query.lower():
            expansions.append(f"{language} {query}")
            expansions.append(f"{query} {language}")
        
        return expansions
    
    def _generate_semantic_variations(self, query: str, key_terms: List[str]) -> List[str]:
        """Generate semantic variations of the query"""
        variations = []
        
        # Question transformations
        question_patterns = {
            'how to': ['implementing', 'creating', 'building'],
            'what is': ['definition of', 'explanation of', 'understanding'],
            'why': ['reason for', 'purpose of', 'rationale behind'],
            'when': ['timing of', 'conditions for', 'scenarios for']
        }
        
        query_lower = query.lower()
        for pattern, replacements in question_patterns.items():
            if pattern in query_lower:
                for replacement in replacements:
                    variations.append(query_lower.replace(pattern, replacement))
        
        return variations[:3]  # Limit to 3 variations
    
    def _generate_context_variations(self, query: str, intent: str, language: Optional[str]) -> List[str]:
        """Generate context-specific variations"""
        variations = []
        
        # Intent-based prefixes/suffixes
        intent_contexts = {
            'code_explanation': ['how does', 'explain', 'understanding'],
            'implementation': ['implement', 'create', 'build'],
            'debugging': ['fix', 'debug', 'solve'],
            'code_search': ['find', 'search', 'locate']
        }
        
        if intent in intent_contexts:
            for context in intent_contexts[intent][:2]:
                if context not in query.lower():
                    variations.append(f"{context} {query}")
        
        return variations
    
    def _improved_llm_expansion(self, query: str, language: Optional[str], intent: str) -> List[str]:
        """Improved LLM-based expansion with better prompting"""
        try:
            # Much more focused and specific prompt
            expansion_prompt = f"""You are a code search expert. Generate 2 alternative search queries that would help find relevant code for this programming question.

Original query: "{query}"
Programming language: {language or 'any'}
User intent: {intent}

Rules:
1. Keep alternatives SHORT and FOCUSED (max 10 words each)
2. Use specific programming terminology
3. Focus on the core technical concept
4. Avoid unnecessary words

Alternative 1: [concise alternative focusing on main concept]
Alternative 2: [concise alternative using different terminology]"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": expansion_prompt}],
                max_tokens=100,  # Reduced from 200
                temperature=0.3,
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            
            # Extract alternatives with better regex
            alternatives = re.findall(r'Alternative \d+:\s*(.+)', content)
            
            # Clean up alternatives
            cleaned_alternatives = []
            for alt in alternatives:
                cleaned = alt.strip().strip('[]"\'')
                if cleaned and len(cleaned.split()) <= 10:  # Enforce word limit
                    cleaned_alternatives.append(cleaned)
            
            return cleaned_alternatives
            
        except Exception as e:
            logging.warning(f"LLM expansion failed: {e}")
            return []
    
    def _clean_and_filter_queries(self, queries: List[str], original: str) -> List[str]:
        """Clean and filter expanded queries"""
        cleaned = []
        seen = set()
        
        for query in queries:
            # Clean the query
            cleaned_query = query.strip().lower()
            
            # Skip if empty, too short, or duplicate
            if (not cleaned_query or 
                len(cleaned_query) < 3 or 
                cleaned_query in seen or
                len(cleaned_query.split()) > 20):  # Skip very long queries
                continue
            
            # Skip if too similar to original
            if self._similarity_score(cleaned_query, original.lower()) > 0.9:
                continue
            
            cleaned.append(cleaned_query)
            seen.add(cleaned_query)
        
        # Always include original first, then up to 5 expansions
        final_queries = [original] + cleaned[:5]
        return final_queries
    
    def _similarity_score(self, query1: str, query2: str) -> float:
        """Calculate simple similarity score between two queries"""
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
class GroqCodeAgent:
    """Intelligent code assistant using Groq API and retrieval system"""
    
    def __init__(self, db_path: str = "data/chromadb", model: str = "llama3-70b-8192"):
        # Initialize Groq client
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.client = Groq(api_key=self.groq_api_key)
        self.model = model
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.query_expander = QueryExpansionEngine(self.client, "llama3-70b-8192", 
                                                 f"query_expansions_{self.session_id}.txt")

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
        """Enhanced retrieval with query expansion"""
        
        try:
            # Step 1: Expand the query
            expanded_queries = self.query_expander.expand_query(
                strategy['query'], 
                strategy.get('language_filter'),
                strategy.get('intent', 'general'),
                self.session_id
            )
            
            logger.info(f"Expanded query into {len(expanded_queries)} variants")
            
            # Step 2: Retrieve chunks for each expanded query
            all_chunks = []
            chunk_scores = {}  # Track chunks and their cumulative scores
            expansion_effectiveness = {}  # Track how effective each expansion was
            
            for i, expanded_query in enumerate(expanded_queries):
                # Weight: original query gets highest weight, others decrease
                weight = 1.0 if i == 0 else 0.7 / i
                
                if strategy['use_intelligent_search']:
                    results = self.retriever.intelligent_search(
                        expanded_query, 
                        strategy['n_results'] // 2  # Fewer per query to allow diversity
                    )
                    chunks = results.get('results', [])
                    
                    # Add repository context
                    for chunk in chunks:
                        chunk['repository_context'] = results.get('relevant_repositories', [])
                        
                else:
                    if strategy['language_filter']:
                        chunks = self.retriever.search_by_language(
                            expanded_query,
                            strategy['language_filter'],
                            strategy['n_results'] // 2
                        )
                    else:
                        chunks = self.retriever.search(
                            expanded_query,
                            strategy['n_results'] // 2
                        )
                
                # Process and weight the chunks
                query_chunks = []
                for chunk in chunks:
                    chunk_id = f"{chunk.get('filename', '')}:{chunk.get('lines', '')}"
                    original_score = chunk.get('relevance_score', 0.0)
                    weighted_score = original_score * weight
                    
                    query_chunks.append(chunk)
                    
                    if chunk_id in chunk_scores:
                        # Boost score for chunks found in multiple expansions
                        chunk_scores[chunk_id]['boosted_score'] += weighted_score * 0.5
                        chunk_scores[chunk_id]['query_matches'] += 1
                    else:
                        chunk_scores[chunk_id] = {
                            'chunk': chunk,
                            'original_score': original_score,
                            'boosted_score': weighted_score,
                            'query_matches': 1,
                            'expansion_query': expanded_query
                        }
                
                # Track expansion effectiveness
                expansion_effectiveness[expanded_query] = {
                    'chunks': len(query_chunks),
                    'avg_score': sum(c.get('relevance_score', 0) for c in query_chunks) / len(query_chunks) if query_chunks else 0
                }
            
            # Step 3: Rank chunks by combined score
            ranked_chunks = []
            for chunk_id, score_info in chunk_scores.items():
                chunk = score_info['chunk']
                
                # Calculate final score combining relevance and query matches
                final_score = (
                    score_info['boosted_score'] * 0.7 +  # Weighted relevance
                    (score_info['query_matches'] / len(expanded_queries)) * 0.3  # Query coverage
                )
                
                chunk['final_relevance_score'] = final_score
                chunk['query_matches'] = score_info['query_matches']
                chunk['expansion_source'] = score_info['expansion_query']
                
                ranked_chunks.append(chunk)
            
            # Sort by final score
            ranked_chunks.sort(key=lambda x: x['final_relevance_score'], reverse=True)
            
            # Step 4: Apply post-processing filters
            final_chunks = ranked_chunks[:strategy['n_results']]
            
            if strategy.get('prefer_complete_files'):
                final_chunks = sorted(final_chunks, 
                                    key=lambda x: x.get('chunk_type', '') == 'complete_file', 
                                    reverse=True)
            
            if strategy.get('diversity'):
                final_chunks = self._ensure_diversity(final_chunks, strategy['n_results'])
            
            # Step 5: Log retrieval results
            self.query_expander.logger.log_retrieval_results(
                strategy['query'], 
                len(final_chunks), 
                final_chunks, 
                expansion_effectiveness
            )
            
            logger.info(f"Retrieved {len(final_chunks)} chunks after expansion and ranking")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {e}")
            # Fallback to original method
            return self._retrieve_relevant_chunks_original(strategy)
    
    def _ensure_diversity(self, chunks: List[Dict], max_results: int) -> List[Dict]:
        """Ensure diversity in retrieved chunks"""
        diverse_chunks = []
        seen_repos = set()
        seen_files = set()
        seen_languages = set()
        
        # First pass: prioritize diverse sources
        for chunk in chunks:
            if len(diverse_chunks) >= max_results:
                break
                
            repo = chunk.get('repo_name', '')
            file_path = chunk.get('file_path', '')
            language = chunk.get('language', '')
            
            # Check if this adds diversity
            adds_diversity = (
                repo not in seen_repos or 
                file_path not in seen_files or 
                language not in seen_languages
            )
            
            if adds_diversity or len(diverse_chunks) < max_results // 2:
                diverse_chunks.append(chunk)
                seen_repos.add(repo)
                seen_files.add(file_path)
                seen_languages.add(language)
        
        # Second pass: fill remaining slots with highest scores
        for chunk in chunks:
            if len(diverse_chunks) >= max_results:
                break
            if chunk not in diverse_chunks:
                diverse_chunks.append(chunk)
        
        return diverse_chunks[:max_results]
    
    def _retrieve_relevant_chunks_original(self, strategy: Dict) -> List[Dict]:
        """Original retrieval method as fallback"""
        # This is the original method implementation for fallback
        # (keeping the existing logic from the original _retrieve_relevant_chunks)
        try:
            if strategy['use_intelligent_search']:
                results = self.retriever.intelligent_search(
                    strategy['query'], 
                    strategy['n_results']
                )
                chunks = results.get('results', [])
                
                for chunk in chunks:
                    chunk['repository_context'] = results.get('relevant_repositories', [])
                
            else:
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
            
            return chunks[:strategy['n_results']]
            
        except Exception as e:
            logger.error(f"Error in original retrieval: {e}")
            return []
    def _build_repository_summary(self, chunks: List[Dict]) -> str:
        """Build a summary of repositories represented in the chunks"""
        repos = {}
        languages = set()
        
        for chunk in chunks:
            repo_name = chunk.get('repo_name', 'Unknown')
            lang = chunk.get('language', 'Unknown')
            
            if repo_name not in repos:
                repos[repo_name] = {'files': set(), 'languages': set()}
            
            repos[repo_name]['files'].add(chunk.get('filename', ''))
            repos[repo_name]['languages'].add(lang)
            languages.add(lang)
        
        summary = f"Analyzing code from {len(repos)} repositories with {len(languages)} languages: {', '.join(list(languages)[:5])}\n"
        
        for repo, info in list(repos.items())[:3]:  # Show top 3 repositories
            summary += f"- {repo}: {len(info['files'])} files ({', '.join(list(info['languages'])[:3])})\n"
        
        return summary
    def _build_enhanced_context_prompt(self, query: str, chunks: List[Dict], 
                                     analysis: Dict, expanded_queries: List[str]) -> str:
        """Build enhanced context prompt with query expansion information"""
        
        base_prompt = self._build_context_prompt(query, chunks, analysis)
        
        # Add query expansion context
        expansion_context = f"""
QUERY EXPANSION CONTEXT:
Original Query: {query}
Expanded Queries: {', '.join(expanded_queries[1:4])}  # Show first 3 expansions

Note: The retrieved code chunks were found using multiple query variations to ensure comprehensive coverage.
"""
        
        # Insert expansion context after the system prompt
        lines = base_prompt.split('\n')
        system_prompt_end = next(i for i, line in enumerate(lines) if line.startswith('CONTEXT INFORMATION:'))
        
        lines.insert(system_prompt_end, expansion_context)
        
        return '\n'.join(lines)
    
    def _build_context_prompt(self, query: str, chunks: List[Dict], analysis: Dict) -> str:
        """Build context-aware prompt for the LLM"""
        
        # Base system prompt for GitHub repository code assistant
        base_system_prompt = """You are an intelligent GitHub Repository Code Assistant specializing in Java, Python, C, C++, and shell scripting. Your core capabilities include:

• Deep understanding of repository structures, code patterns, and best practices
• Expertise in analyzing, explaining, and troubleshooting code across multiple languages
• Ability to provide context-aware recommendations based on existing codebase patterns
• Focus on delivering accurate, concise, and actionable responses

Your approach:
1. Analyze user queries in detail to understand their exact needs
2. Examine the provided code chunks for relevant patterns and context
3. Synthesize information to provide summarized, average-length responses
4. Ensure responses are well-formatted, presentable, and directly address user requirements
5. Reference specific code examples when helpful for clarity

Always maintain professional tone while being approachable and helpful."""

        # Intent-specific enhancements
        intent_enhancements = {
            'code_explanation': " Focus on breaking down complex code logic into understandable concepts, highlighting key design patterns and architectural decisions.",
            'code_search': " Help locate relevant code examples and explain their significance within the repository context.",
            'implementation': " Provide practical implementation guidance that follows existing codebase conventions and best practices.",
            'debugging': " Analyze code systematically for potential issues, providing clear explanations and actionable solutions.",
            'comparison': " Compare different approaches objectively, considering repository-specific context and constraints.",
            'documentation': " Create clear, comprehensive explanations that would help both current and future developers.",
            'general': " Provide comprehensive assistance while maintaining focus on repository-specific context and patterns."
        }
        
        # Combine base prompt with intent-specific enhancement
        system_prompt = base_system_prompt + intent_enhancements.get(analysis['intent'], intent_enhancements['general'])
        
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
        prompt = f"""{system_prompt}

REPOSITORY CONTEXT:
{repo_summary}

CONTEXT INFORMATION:
{''.join(context_sections)}

USER QUERY: {query}

RESPONSE GUIDELINES:
1. Provide a direct, well-structured answer that addresses the user's specific needs
2. Keep responses at average length - comprehensive but not overwhelming
3. Use clear formatting with appropriate headers, bullet points, or code blocks
4. Reference specific code chunks using "Code Chunk X" format when relevant
5. Include practical examples or recommendations based on the retrieved code
6. If information is incomplete, clearly state limitations and suggest next steps
7. Maintain focus on accuracy and actionable insights

Provide your response now:"""
        
        return prompt
    
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
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base"""
        
        try:
            # Get all repositories and their metadata
            all_repositories = self.retriever.chroma_manager.get_all_repositories()
            
            # Initialize stats structure
            stats = {
                'total_chunks': 0,
                'repositories': {},
                'languages': {},
                'total_repositories': 0,
                'average_chunks_per_repo': 0,
                'repository_details': {}
            }
            
            # Process each repository
            for repo_name, repo_info in all_repositories.items():
                chunk_count = repo_info.get('total_chunks', 0)
                languages = repo_info.get('languages', {})
                
                # Add to repositories stats
                stats['repositories'][repo_name] = chunk_count
                stats['total_chunks'] += chunk_count
                
                # Add language information
                for language, count in languages.items():
                    if language in stats['languages']:
                        stats['languages'][language] += count
                    else:
                        stats['languages'][language] = count
                
                # Detailed repository information
                stats['repository_details'][repo_name] = {
                    'chunks': chunk_count,
                    'languages': languages,
                    'created_at': repo_info.get('created_at', 'Unknown'),
                    'url': repo_info.get('url', 'Unknown'),
                    'total_files': repo_info.get('total_files', 0)
                }
            
            # Calculate derived stats
            stats['total_repositories'] = len(all_repositories)
            if stats['total_repositories'] > 0:
                stats['average_chunks_per_repo'] = stats['total_chunks'] / stats['total_repositories']
            
            # Get additional ChromaDB stats if available
            try:
                collection = self.retriever.chroma_manager.collection
                collection_count = collection.count()
                
                # Verify consistency
                if collection_count != stats['total_chunks']:
                    logger.warning(f"Chunk count mismatch: metadata={stats['total_chunks']}, collection={collection_count}")
                    stats['total_chunks'] = collection_count  # Use actual collection count
                    
            except Exception as e:
                logger.warning(f"Could not get collection stats: {e}")
            
            # Add performance metrics if available
            if hasattr(self, 'query_expander') and hasattr(self.query_expander, 'logger'):
                try:
                    expansion_stats = self.query_expander.logger.get_session_stats()
                    stats['query_expansion'] = expansion_stats
                except Exception as e:
                    logger.debug(f"Could not get expansion stats: {e}")
            
            logger.info(f"Generated stats: {stats['total_repositories']} repos, {stats['total_chunks']} chunks")
            return stats
            
        except Exception as e:
            logger.error(f"Error generating stats: {e}")
            # Return empty stats structure on error
            return {
                'total_chunks': 0,
                'repositories': {},
                'languages': {},
                'total_repositories': 0,
                'average_chunks_per_repo': 0,
                'repository_details': {},
                'error': str(e)
            }
    def _call_groq_api(self, prompt: str) -> Tuple[str, Dict]:
        """Call the Groq API with the prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3,
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            usage_info = {
                'prompt_tokens': response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                'completion_tokens': response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                'total_tokens': response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
            return content, usage_info
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return f"I encountered an error while processing your request: {str(e)}", {}     
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
            
            # Step 3: Retrieve relevant chunks with expansion
            chunks = self._retrieve_relevant_chunks(strategy)
            
            # Get expanded queries for context
            expanded_queries = self.query_expander.expand_query(
                query, 
                analysis.get('language'),
                analysis.get('intent', 'general'),
                self.session_id
            )
            
            logger.info(f"Retrieved {len(chunks)} code chunks using query expansion")
            
            if not chunks:
                return AgentResponse(
                    query=query,
                    answer="I couldn't find relevant code chunks for your query. Please make sure repositories have been processed and try rephrasing your question.",
                    sources=[],
                    reasoning="No relevant code chunks found in the database after query expansion",
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
    
    def __init__(self, db_path: str = "data/chromadb", model: str = "llama3-70b-8192"):
        self.agent = GroqCodeAgent(db_path, model)
        self.conversation_history = []
    
    def start_interactive_session(self):
        """Start an interactive session with the agent"""
        
        print("Intelligent Code Assistant (Powered by Groq + Llama)")
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
                print(f"\nError: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("\nAvailable Commands:")
        print("  help     - Show this help message")
        print("  stats    - Show collection statistics")
        print("  history  - Show conversation history")
        print("  model <name> - Change the model (llama3-8b-8192, llama3-70b-8192, etc.)")
        print("  quit     - Exit the assistant")
        print("\nQuery Examples:")
        print("  'Explain how authentication works in this codebase'")
        print("  'Find examples of error handling in Python'")
        print("  'How to implement a REST API in this project?'")
        print("  'Debug this SQL query performance issue'")
        print("  'Compare different sorting algorithms used here'")
    
    def _show_stats(self):
        """Show collection statistics"""
        stats = self.agent.retriever.get_collection_stats()
        
        print("\nCodebase Statistics:")
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
        
        print(f"\nConversation History ({len(self.conversation_history)} queries):")
        for i, item in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            print(f"{i}. {item['query']}")
            print(f"   Confidence: {item['response'].confidence:.2f}")
            print(f"   Sources: {len(item['response'].sources)}")
    
    def _change_model(self, model_name: str):
        """Change the model"""
        model_name = model_name.strip()
        
        if model_name in self.agent.available_models:
            self.agent.model = model_name
            print(f"Model changed to: {model_name}")
        else:
            print(f"Invalid model. Available models: {', '.join(self.agent.available_models)}")
    
    def _display_response(self, response: AgentResponse):
        """Display the agent response in a formatted way"""
        print("\n" + "=" * 80)
        print(f"Assistant Response")
        print("=" * 80)
        
        print(f"\nAnswer:")
        print(response.answer)
        
        print(f"\nResponse Details:")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Response Time: {response.response_time:.2f}s")
        print(f"  Model Used: {response.model_used}")
        print(f"  Sources Used: {len(response.sources)}")
        
        if response.reasoning:
            print(f"\nReasoning: {response.reasoning}")
        
        if response.sources:
            print(f"\nSources:")
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
    parser.add_argument("--model", type=str, default="llama3-70b-8192", 
                       help="Model to use (llama3-8b-8192, llama3-70b-8192, etc.)")
    parser.add_argument("--db-path", type=str, default="data/chromadb", 
                       help="Database path")
    parser.add_argument("--interactive", action="store_true", 
                       help="Start interactive session")
    parser.add_argument("--output", type=str, help="Output file for response")
    
    args = parser.parse_args()
    
    # Check for GROQ_API_KEY
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is required")
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
            print(f"Processing query: {args.query}")
            
            response = agent.query(args.query)
            
            # Display response
            print("\n" + "=" * 80)
            print(f"Assistant Response")
            print("=" * 80)
            print(f"\nAnswer:")
            print(response.answer)
            
            print(f"\nResponse Details:")
            print(f"  Confidence: {response.confidence:.2f}")
            print(f"  Response Time: {response.response_time:.2f}s")
            print(f"  Model Used: {response.model_used}")
            print(f"  Sources Used: {len(response.sources)}")
            
            if response.sources:
                print(f"\nSources:")
                for i, source in enumerate(response.sources, 1):
                    print(f"  {i}. {source['filename']} ({source['language']})")
                    print(f"     Repository: {source['repository']}")
                    print(f"     Relevance: {source['relevance_score']:.3f}")
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(asdict(response), f, indent=2)
                print(f"\nResponse saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Set GROQ_API_KEY in your environment")
        print("2. Processed repositories using week2_chunker.py")
        print("3. Verified ChromaDB is accessible")

if __name__ == "__main__":
    main()