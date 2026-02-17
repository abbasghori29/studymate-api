"""
Chatbot service with RAG (Retrieval Augmented Generation)
Includes memory for past Q&A

Now uses LangGraph for proper conversation handling and follow-up support.
The key improvement is query contextualization - rewriting follow-up questions
like "why?" into full standalone questions before vector retrieval.
"""
import os
import sys
from typing import List, Optional, Dict
from pathlib import Path

# Add project root to path to import train_vector_db
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from app.core.config import settings
import re

# Language detection
try:
    from langdetect import detect, LangDetectException
    LANG_SUPPORT = True
except ImportError:
    LANG_SUPPORT = False
    print("Warning: Language detection library not installed. Install langdetect for bilingual support.")

# Import memory services (Pinecone preferred, FAISS fallback)
if settings.PINECONE_API_KEY:
    from app.services.pinecone_memory import get_pinecone_memory_service as get_memory_service
    from app.services.pinecone_memory import PineconeMemoryService as MemoryService
    MEMORY_TYPE = "pinecone"
else:
    from app.services.memory import get_memory_service, MemoryService
    MEMORY_TYPE = "faiss"

# Flag to enable LangGraph-based conversation handling
USE_LANGGRAPH = True


class ChatbotService:
    """Chatbot service with RAG capabilities and memory"""
    
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.embeddings = None
        self.memory: Optional[MemoryService] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize LLM and vector store"""
        # Initialize LLM with OpenAI as primary, Groq as fallback
        primary_llm = None
        fallback_llm = None
        
        # Try to initialize OpenAI as primary
        if settings.OPENAI_API_KEY:
            try:
                primary_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.5,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    api_key=settings.OPENAI_API_KEY,
                )
                print("✓ OpenAI LLM initialized as primary")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI LLM: {e}")
        
        # Initialize Groq as fallback (or primary if OpenAI not available)
        if settings.GROQ_API_KEY:
            try:
                fallback_llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    groq_api_key=settings.GROQ_API_KEY,
                )
                print("✓ Groq LLM initialized as fallback")
            except Exception as e:
                print(f"Warning: Could not initialize Groq LLM: {e}")
        
        # Set up LLM with fallback chain
        if primary_llm and fallback_llm:
            self.llm = primary_llm.with_fallbacks([fallback_llm])
            print("✓ LLM configured with OpenAI primary + Groq fallback")
        elif primary_llm:
            self.llm = primary_llm
            print("✓ Using OpenAI LLM only (no fallback)")
        elif fallback_llm:
            self.llm = fallback_llm
            print("✓ Using Groq LLM only (OpenAI not configured)")
        else:
            raise ValueError("No LLM API key configured. Set OPENAI_API_KEY or GROQ_API_KEY in .env")
        
        # Initialize OpenAI embeddings (much better semantic search quality)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY,
        )
        print("✓ Using OpenAI embeddings (text-embedding-3-large)")
        
        # Skip vector store load (delegating to LangGraph service)
        print("✓ Delegating vector store management to LangGraph service")
        
        # Initialize memory service (Pinecone with OpenAI embeddings)
        try:
            self.memory = get_memory_service(embeddings=self.embeddings)
            print(f"✓ Memory service initialized ({MEMORY_TYPE})")
        except Exception as e:
            print(f"Warning: Could not initialize memory service ({MEMORY_TYPE}): {e}")
            self.memory = None
    

    
    def _normalize_markdown_to_html(self, text: str) -> str:
        """Convert any markdown that slipped through to HTML"""
        # First, convert markdown bold to HTML (do this before line processing)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        
        # Process line by line
        lines = text.split('\n')
        result_lines = []
        in_ul = False
        in_ol = False
        in_p = False
        current_paragraph = []
        
        for line in lines:
            stripped = line.strip()
            
            # Check for markdown heading
            heading_match = re.match(r'^#{1,6}\s+(.+)$', stripped)
            if heading_match:
                # Close any open tags
                if in_p:
                    result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
                    current_paragraph = []
                    in_p = False
                if in_ul:
                    result_lines.append('</ul>')
                    in_ul = False
                if in_ol:
                    result_lines.append('</ol>')
                    in_ol = False
                result_lines.append(f'<h3>{heading_match.group(1)}</h3>')
                continue
            
            # Check for unordered list item
            ul_match = re.match(r'^[-*]\s+(.+)$', stripped)
            if ul_match:
                if in_p:
                    result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
                    current_paragraph = []
                    in_p = False
                if in_ol:
                    result_lines.append('</ol>')
                    in_ol = False
                if not in_ul:
                    result_lines.append('<ul>')
                    in_ul = True
                result_lines.append(f'  <li>{ul_match.group(1)}</li>')
                continue
            
            # Check for ordered list item
            ol_match = re.match(r'^\d+\.\s+(.+)$', stripped)
            if ol_match:
                if in_p:
                    result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
                    current_paragraph = []
                    in_p = False
                if in_ul:
                    result_lines.append('</ul>')
                    in_ul = False
                if not in_ol:
                    result_lines.append('<ol>')
                    in_ol = True
                result_lines.append(f'  <li>{ol_match.group(1)}</li>')
                continue
            
            # Empty line - close open tags
            if not stripped:
                if in_p:
                    result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
                    current_paragraph = []
                    in_p = False
                if in_ul:
                    result_lines.append('</ul>')
                    in_ul = False
                if in_ol:
                    result_lines.append('</ol>')
                    in_ol = False
                continue
            
            # Check if line is already HTML
            if stripped.startswith('<') and ('</' in stripped or stripped.endswith('>')):
                # Close open tags
                if in_p:
                    result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
                    current_paragraph = []
                    in_p = False
                if in_ul:
                    result_lines.append('</ul>')
                    in_ul = False
                if in_ol:
                    result_lines.append('</ol>')
                    in_ol = False
                result_lines.append(line)
                continue
            
            # Regular text - add to paragraph
            if not in_p:
                in_p = True
            current_paragraph.append(stripped)
        
        # Close any remaining open tags
        if in_p:
            result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
        if in_ul:
            result_lines.append('</ul>')
        if in_ol:
            result_lines.append('</ol>')
        
        return '\n'.join(result_lines)
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up the LLM answer - normalize to HTML and remove junk"""
        
        # Step 1: Remove document references
        doc_patterns = [
            r'【Document\s*\d+】\.?', r'\[Document\s*\d+\]\.?', r'\(Document\s*\d+\)\.?',
            r'Document\s*\d+:?', r'Source\s*\d+:?', r'\[Source:\s*[^\]]+\]',
            r'\(Source:\s*[^\)]+\)', r'Page\s*\d+:?', r'\[p\.\s*\d+\]', r'\(p\.\s*\d+\)',
        ]
        for pattern in doc_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        # Step 2: Normalize any markdown to HTML
        # Check if answer contains markdown patterns
        has_markdown = bool(re.search(r'(\*\*|__|^#{1,6}\s|^[-*]\s|^\d+\.\s)', answer, re.MULTILINE))
        if has_markdown:
            answer = self._normalize_markdown_to_html(answer)
        
        # Step 3: Ensure proper spacing after HTML block elements
        answer = answer.replace('</h3>', '</h3>\n')
        answer = answer.replace('</p>', '</p>\n')
        answer = answer.replace('</ul>', '</ul>\n')
        answer = answer.replace('</ol>', '</ol>\n')
        answer = answer.replace('</li>', '</li>\n')
        
        # Step 4: Clean up excessive whitespace
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        answer = answer.strip()
        
        return answer
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        if not LANG_SUPPORT:
            return "en"  # Default to English
        
        try:
            # Detect language
            lang = detect(text)
            # Normalize to 'en' or 'fr'
            if lang == 'fr':
                return 'fr'
            else:
                return 'en'  # Default to English for all other languages
        except (LangDetectException, Exception) as e:
            print(f"Language detection error: {e}, defaulting to English")
            return "en"
    
    def _translate_query(self, text: str, target_lang: str = "en") -> str:
        """Translate text to target language using LLM (for vector search)"""
        try:
            # Ensure LLM is initialized
            if not self.llm:
                print("LLM not initialized, cannot translate. Using original text.")
                return text
            
            # Detect source language
            source_lang = self._detect_language(text) if LANG_SUPPORT else "en"
            
            # If already in target language, return as is
            if source_lang == target_lang:
                return text
            
            # Use LLM to translate
            translation_prompt = f"""Translate the following text from {source_lang.upper()} to {target_lang.upper()}. 
Return ONLY the translation, nothing else. No explanations, no additional text, just the translated text.

Text to translate: {text}

Translation:"""
            
            response = self.llm.invoke(translation_prompt)
            
            # Extract translation from response
            if hasattr(response, 'content'):
                translated = response.content.strip()
            else:
                translated = str(response).strip()
            
            print(f"Translated query: '{text}' ({source_lang}) -> '{translated}' ({target_lang})")
            return translated
        except Exception as e:
            print(f"Translation error: {e}, using original text")
            return text
    

    

    
    def chat(
        self,
        question: str,
        chat_history: Optional[List] = None,
        k: int = 5,
        use_memory: bool = True,
        store_in_memory: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Chat with the RAG-powered chatbot
        
        Args:
            question: User's question
            chat_history: List of previous messages in format [("human", "..."), ("ai", "...")]
            k: Number of documents to retrieve
            use_memory: Whether to search memory for similar past conversations
            store_in_memory: Whether to store this conversation in memory
            user_id: Anonymous user ID for personalized memory
            session_id: Session ID for grouping conversations
        
        Returns:
            Dictionary with answer, sources, and memory info
        """
        # Use LangGraph service
        try:
            from app.services.langgraph_rag import get_langgraph_service
            langgraph_service = get_langgraph_service()
            return langgraph_service.chat(
                question=question,
                chat_history=chat_history,
                k=k,
                use_memory=use_memory,
                store_in_memory=store_in_memory,
                user_id=user_id,
                session_id=session_id,
            )
        except Exception as e:
            print(f"⚠️ LangGraph service error: {e}")
            raise e
    



# Global chatbot instance (singleton)
_chatbot_service: Optional[ChatbotService] = None


def get_chatbot_service() -> ChatbotService:
    """Get or create chatbot service instance"""
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service

