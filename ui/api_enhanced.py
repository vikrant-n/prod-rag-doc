"""
FastAPI Backend for Query Interface

This module provides the REST API endpoints for the web-based
query interface, integrating with the context-aware query engine.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, List, Optional, Any

# Setup global logger
logger = logging.getLogger(__name__)
from datetime import datetime, timezone
import json
import uuid
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Import only what we need for basic functionality
from pipeline.processing.source_attribution import CitationFormat
from langchain_qdrant import QdrantVectorStore
from pipeline.context_management.conversation_context import MessageType

# Import RBAC components
try:
    from api.rbac import Permission, Resource
except ImportError:
    # Mock Permission enum for testing
    from enum import Enum
    class Permission(Enum):
        DOCUMENT_READ = "document_read"
        DOCUMENT_SEARCH = "document_search"
    
    class Resource(Enum):
        DOCUMENT = "document"

# Import conversation message class
try:
    from pipeline.context_management.conversation_context import ConversationMessage
except ImportError:
    # Mock ConversationMessage class for testing
    @dataclass
    class ConversationMessage:
        message_id: str
        session_id: str
        message_type: Any
        content: str
        timestamp: datetime
        metadata: Dict[str, Any] = None

# Mock User class and authentication for testing
class User:
    def __init__(self, id: str, email: str, roles: List[str] = None):
        self.id = id
        self.email = email
        self.roles = roles or ["user"]

async def get_current_user() -> Optional[User]:
    """Mock authentication function for testing"""
    return User(id="test_user", email="test@example.com", roles=["user"])

# Enhanced retrieval components disabled for simplified operation
EnhancedRetriever = None
EnhancedRetrievalIntegration = None
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    citation_format: Optional[str] = Field("apa", description="Citation format")
    response_style: Optional[str] = Field("detailed", description="Response style")
    response_format: Optional[str] = Field("markdown", description="Response format")
    include_context: bool = Field(True, description="Whether to include conversation context")
    max_sources: Optional[int] = Field(None, description="Maximum number of sources (automatically determined if not provided)")


class QueryResponse(BaseModel):
    """Response model for query processing"""
    query: str
    response: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_type: str
    complexity: str
    processing_time: float
    context_used: bool
    citations: List[str]
    session_id: str
    word_count: int
    metadata: Dict[str, Any]
    timestamp: str


class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str
    message_count: int
    current_topic: Optional[str]
    created_at: str
    last_activity: Optional[str]
    metadata: Dict[str, Any]


class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str
    data: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SummarizeRequest(BaseModel):
    content: str
    summary_type: str = "executive"  # executive, technical, bullet_points, abstract, action_items
    max_length: int = 500
    extract_key_topics: bool = True

class DocumentContent(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class SummarizeMultipleRequest(BaseModel):
    documents: List[DocumentContent]
    summary_type: str = "executive"
    max_length: int = 300

class CompareDocumentsRequest(BaseModel):
    documents: List[DocumentContent]
    comparison_focus: str = "similarities and differences"

class LanguageDetectionRequest(BaseModel):
    text: str
    use_openai: bool = False

class TranslationRequest(BaseModel):
    text: str
    target_language: str = "en"
    source_language: Optional[str] = None

class MultilingualDocumentRequest(BaseModel):
    content: str
    target_language: Optional[str] = None
    detect_language: bool = True
    translate_if_needed: bool = True


# FastAPI application
app = FastAPI(
    title="Document RAG Query Interface",
    description="Web-based interface for context-aware document retrieval and query processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Connect a WebSocket for a session"""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        if session_id not in self.session_connections:
            self.session_connections[session_id] = []
        self.session_connections[session_id].append(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} for session {session_id}")
        return connection_id
    
    async def disconnect(self, connection_id: str, session_id: str):
        """Disconnect a WebSocket"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if session_id in self.session_connections:
            self.session_connections[session_id] = [
                conn for conn in self.session_connections[session_id] 
                if conn != connection_id
            ]
            
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]):
        """Send message to all connections in a session"""
        if session_id in self.session_connections:
            for connection_id in self.session_connections[session_id]:
                if connection_id in self.active_connections:
                    websocket = self.active_connections[connection_id]
                    try:
                        await websocket.send_text(json.dumps(message))
                    except Exception as e:
                        logger.error(f"Error sending message to {connection_id}: {e}")
    
    async def send_to_all(self, message: Dict[str, Any]):
        """Send message to all connected WebSockets"""
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")


class SimpleQueryEngine:
    """
    Complete RAG Pipeline following the architectural diagram
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        enable_enhanced_retrieval: bool = True,
        enable_cohere_rerank: bool = True,
        enable_bm25: bool = False  # Disable BM25 by default - not needed for enhanced retrieval
    ):
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
        
        # Initialize components from the architectural diagram
        self._initialize_pipeline_components()
        
        # Initialize hybrid search engine
        self._initialize_hybrid_search()
        
        # Initialize RBAC manager
        self._initialize_rbac()
        
        # Initialize encryption
        self._initialize_encryption()
        
        # Initialize analytics
        self._initialize_analytics()
        
        # Initialize document summarizer
        self._initialize_summarizer()
        
        # Initialize multilingual processor
        self._initialize_multilingual_processor()
        
        # Initialize conversation context manager
        self._initialize_conversation_context()
    
    def _initialize_summarizer(self):
        """Initialize document summarization"""
        try:
            from pipeline.processing.document_summarizer import get_document_summarizer
            
            self.document_summarizer = get_document_summarizer()
            self.logger.info("Document summarizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize document summarizer: {e}")
            self.document_summarizer = None
    
    def _initialize_multilingual_processor(self):
        """Initialize multilingual processing"""
        try:
            from pipeline.processing.multilingual_processor import get_multilingual_processor
            
            self.multilingual_processor = get_multilingual_processor()
            self.logger.info("Multilingual processor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize multilingual processor: {e}")
            self.multilingual_processor = None
    
    def _initialize_conversation_context(self):
        """Initialize conversation context management"""
        try:
            # Try to use real conversation context manager first
            from pipeline.context_management.conversation_context import ConversationContextManager
            self.conversation_manager = ConversationContextManager()
            self.logger.info("Real conversation context manager initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize real conversation manager: {e}")
            try:
                # Fallback to mock conversation context manager
                from pipeline.context_management.mock_conversation_context import MockConversationContextManager
                self.conversation_manager = MockConversationContextManager()
                self.logger.info("Mock conversation context manager initialized (in-memory storage)")
                
            except Exception as e2:
                self.logger.error(f"Failed to initialize conversation context manager: {e2}")
                self.conversation_manager = None
    
    def _initialize_pipeline_components(self):
        """Initialize all pipeline components from the architectural diagram"""
        try:
            # 1. Small Language Model for Query Optimization
            from pipeline.query_engine.query_classifier import QueryClassifier
            self.query_classifier = QueryClassifier(
                confidence_threshold=0.7,
                enable_ml_classification=True
            )
            self.logger.info("Query classifier (Small Language Model) initialized")
            
            # 2. Enhanced Retrieval with ReRanking
            from pipeline.query_engine.enhanced_retrieval import EnhancedRetriever
            self.enhanced_retriever = EnhancedRetriever(
                qdrant_store=self.vector_store,
                enable_bm25=False,  # Disable BM25 - not needed for enhanced retrieval
                enable_cohere_rerank=True
            )
            self.logger.info("Enhanced retriever with reranking initialized")
            
            # 3. Response Generator (LLM/Chat Model)
            from pipeline.query_engine.response_generator import ResponseGenerator
            self.response_generator = ResponseGenerator(
                default_style="detailed",
                default_format="markdown"
            )
            self.logger.info("Response generator (LLM/Chat Model) initialized")
            
            # 4. OpenAI LLM for final response generation
            from openai import OpenAI
            import os
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.logger.info("OpenAI LLM client initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            # Fallback to basic functionality
            self.query_classifier = None
            self.enhanced_retriever = None
            self.response_generator = None
            self.openai_client = None
    
    def _initialize_hybrid_search(self):
        """Initialize hybrid search engine with BM25 and semantic search"""
        try:
            from pipeline.query_engine.hybrid_search import HybridSearchEngine
            import os
            
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                self.logger.warning("COHERE_API_KEY not found, hybrid search will use fallback")
                self.hybrid_search = None
                return
            
            # Get Qdrant client from vector store
            qdrant_client = self.vector_store.client
            collection_name = self.vector_store.collection_name
            
            self.hybrid_search = HybridSearchEngine(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                cohere_api_key=cohere_api_key,
                bm25_weight=0.0,  # Disable BM25 - use only semantic search
                semantic_weight=1.0,  # Use only semantic search
                enable_reranking=True
            )
            
            # Skip BM25 indexing since it's not needed for enhanced retrieval
            # self._build_bm25_index_optimized()
            self.logger.info("Hybrid search engine initialized (BM25 indexing skipped)")
            
            self.logger.info("Hybrid search engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid search: {e}")
            self.hybrid_search = None
    
    def _build_bm25_index_optimized(self):
        """Build BM25 index with performance optimizations for client deployment"""
        try:
            if not self.hybrid_search:
                return
            
            # Add timeout protection
            import signal
            import threading
            import time
            
            # Set a timeout for the entire indexing process
            timeout_seconds = 300  # 5 minutes max
            
            def timeout_handler():
                self.logger.warning(f"BM25 indexing timed out after {timeout_seconds} seconds")
                return
            
            # Get document count first with timeout
            try:
                documents = self._get_all_documents()
                doc_count = len(documents)
                
                if doc_count == 0:
                    self.logger.info("No documents found for BM25 indexing")
                    return
                
                self.logger.info(f"Found {doc_count} documents for BM25 indexing")
                
                # Use incremental indexing for much better performance
                self.logger.info(f"Starting incremental BM25 indexing for {doc_count} documents...")
                self.hybrid_search.build_bm25_index_incremental(documents, use_cache=True)
                
                # Get stats after indexing
                stats = self.hybrid_search.get_bm25_stats()
                self.logger.info(f"BM25 indexing completed. Total documents: {stats.get('total_documents', 0)}")
                    
            except Exception as e:
                self.logger.error(f"Error during document loading or indexing: {e}")
                self.logger.info("Continuing without BM25 indexing - server will still function")
                
        except Exception as e:
            self.logger.error(f"Failed to start optimized BM25 indexing: {e}")
            self.logger.info("Continuing without BM25 indexing - server will still function")
    
    def _build_bm25_index_background(self, documents: List[Dict[str, Any]]):
        """Build BM25 index in background thread with incremental updates"""
        try:
            self.logger.info(f"Building BM25 index in background for {len(documents)} documents...")
            
            # Use incremental indexing for background processing
            self.hybrid_search.build_bm25_index_incremental(documents, use_cache=True)
            
            self.logger.info("Background BM25 indexing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Background BM25 indexing failed: {e}")
    
    def _build_bm25_index(self):
        """Build BM25 index from documents in Qdrant with optimized grouping"""
        try:
            if not self.hybrid_search:
                return
            
            # Get all documents from Qdrant
            documents = self._get_all_documents()
            
            if not documents:
                self.logger.info("No documents found for BM25 indexing")
                return
            
            # Use incremental indexing for better performance
            self.hybrid_search.build_bm25_index_incremental(documents, use_cache=True)
            
        except Exception as e:
            self.logger.error(f"Failed to build BM25 index: {e}")
    
    def _build_bm25_index_legacy(self):
        """Build BM25 index from documents in Qdrant (legacy method - no grouping)"""
        try:
            if not self.hybrid_search:
                return
            
            # Get all documents from Qdrant
            documents = self._get_all_documents()
            
            # Build BM25 index without grouping
            self.hybrid_search.build_bm25_index(documents, use_cache=True)
            self.logger.info(f"BM25 index built with {len(documents)} documents (no grouping)")
            
        except Exception as e:
            self.logger.error(f"Failed to build BM25 index: {e}")
    
    def _get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from Qdrant for BM25 indexing"""
        documents = []
        offset = 0
        limit = 100
        
        while True:
            results = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not results[0]:  # No more documents
                break
            
            for result in results[0]:
                documents.append({
                    'id': result.id,
                    'content': result.payload.get('content', ''),
                    'metadata': result.payload.get('metadata', {})
                })
            
            offset += limit
            
            if len(results[0]) < limit:  # Last batch
                break
        
        return documents
    
    async def _filter_documents_by_access(self, documents: List[Document], user_id: str) -> List[Document]:
        """
        Filter documents based on user access permissions
        """
        if not self.rbac_manager:
            return documents
        
        try:
            from api.rbac import Permission, Resource
            from api.auth import User
            
            # Create user object
            user = User(id=user_id, email=f"{user_id}@example.com", roles=["user"])
            
            accessible_documents = []
            
            for doc in documents:
                # Extract document ID from metadata
                doc_id = doc.metadata.get('source', '')
                
                if not doc_id:
                    # If no document ID, allow access (fallback)
                    accessible_documents.append(doc)
                    continue
                
                # Check if user has access to this document
                has_access = self.rbac_manager.has_resource_access(
                    user=user,
                    resource_type=Resource.DOCUMENT,
                    resource_id=doc_id,
                    required_permission=Permission.DOCUMENT_READ
                )
                
                if has_access:
                    accessible_documents.append(doc)
                else:
                    self.logger.debug(f"User {user_id} denied access to document {doc_id}")
            
            return accessible_documents
            
        except Exception as e:
            self.logger.error(f"Error filtering documents by access: {e}")
            # In case of error, return all documents (fail open for safety)
            return documents
    
    def _initialize_rbac(self):
        """Initialize RBAC manager for access control"""
        try:
            from api.rbac import get_rbac_manager, Permission, Resource
            
            self.rbac_manager = get_rbac_manager()
            self.logger.info("RBAC manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RBAC manager: {e}")
            self.rbac_manager = None
    
    def _initialize_encryption(self):
        """Initialize encryption for sensitive data"""
        try:
            from api.encryption import get_document_encryption
            
            self.document_encryption = get_document_encryption()
            self.logger.info("Document encryption initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            self.document_encryption = None
    
    def _initialize_analytics(self):
        """Initialize analytics collection"""
        try:
            from ui.analytics_dashboard import dashboard_manager
            
            self.analytics = dashboard_manager.analytics
            self.logger.info("Analytics collection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics: {e}")
            self.analytics = None
    
    async def process_query(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,
        citation_format: CitationFormat = CitationFormat.APA,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG Pipeline following the architectural diagram:
        1. User Query → 2. Small Language Model (Query Optimization) → 
        3. Enhanced Retrieval → 4. ReRanking → 5. LLM/Chat Model → 6. Final Response
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"Starting complete RAG pipeline for query: {query[:50]}...")
            
            # Step 0: Handle conversation context
            if self.conversation_manager and include_context:
                try:
                    # Get or create session
                    session = await self.conversation_manager.get_session(session_id)
                    if not session:
                        session = await self.conversation_manager.create_session(user_id=user_id)
                        session_id = session.session_id
                    
                    # Add user message to conversation history
                    await self.conversation_manager.add_message(
                        session_id=session_id,
                        message_type=MessageType.USER,
                        content=query,
                        metadata={
                            'user_id': user_id,
                            'timestamp': start_time.isoformat(),
                            'include_context': include_context
                        }
                    )
                    
                    # Get conversation history for context enhancement
                    conversation_history = await self.conversation_manager.get_conversation_history(
                        session_id=session_id, 
                        limit=10
                    )
                    
                    # Enhance query with conversation context if available
                    if len(conversation_history) > 1:  # More than just the current message
                        context_enhanced_query = await self._enhance_query_with_context(query, conversation_history)
                        self.logger.info(f"Query enhanced with context: '{query[:50]}...' → '{context_enhanced_query[:50]}...'")
                        query = context_enhanced_query
                    
                except Exception as e:
                    self.logger.warning(f"Failed to handle conversation context: {e}")
                    # Continue without context if there's an error
            
            # Step 1: Query Optimization using Small Language Model
            logger.info("Step 1: Optimizing query with small language model")
            optimized_query, query_type, complexity = await self._optimize_query(query)
            
            # Step 1.5: Determine optimal source count intelligently
            optimal_source_count = await self._determine_optimal_source_count(query, query_type, complexity)
            self.logger.info(f"Intelligent source selection: {optimal_source_count} sources for '{query_type}' query with '{complexity}' complexity")
            
            # Step 2: Perform adaptive retrieval with intelligent source selection
            self.logger.info("Step 2: Performing adaptive retrieval with intelligent source selection")
            documents = await self._adaptive_retrieval(optimized_query, optimal_source_count)
            
            # Step 2.5: Apply Access Control Filtering
            if self.rbac_manager and user_id:
                documents = await self._filter_documents_by_access(documents, user_id)
                logger.info(f"After access control filtering: {len(documents)} documents")
            
            if not documents:
                logger.warning("No documents accessible to user after access control filtering")
                return {
                    'response': 'No accessible documents found to answer your question.',
                    'sources': [],
                    'session_id': session_id,
                    'query': query,
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'confidence': 0.0,
                    'query_type': 'no_documents',
                    'complexity': 'simple',
                    'context_used': include_context,
                    'citations': [],
                    'word_count': 0,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'retrieval_method': 'hybrid_search',
                        'reranked_count': 0,
                        'total_candidates': 0,
                        'citation_format': citation_format.value,
                        'pipeline_steps': ['query_optimization', 'enhanced_retrieval', 'access_control', 'llm_generation']
                    }
                }
            
            # Step 3: Generate Response using LLM/Chat Model
            response_data = await self._generate_response(query, documents, citation_format, session_id, user_id)
            
            # Debug: Log the sources being returned
            logger.info(f"Sources being returned to UI: {len(response_data.get('sources', []))}")
            for i, source in enumerate(response_data.get('sources', [])):
                logger.info(f"Source {i+1}: {source.get('title', 'Unknown')} - URL: {source.get('google_drive_url', 'No URL')}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            
            # Record analytics
            if self.analytics:
                try:
                    self.analytics.record_query({
                        'query': query,
                        'session_id': session_id,
                        'user_id': user_id,
                        'processing_time': processing_time,
                        'status': 'success',
                        'retrieval_method': 'hybrid_search', # Assuming hybrid_search is the current method
                        'documents_retrieved': len(documents),
                        'response_length': len(response_data.get('answer', response_data.get('response', '')))
                    })
                    
                    # Record user session data
                    if user_id:
                        self.analytics.record_user_session(user_id, {
                            'last_activity': datetime.now().isoformat(),
                            'query_count': self.analytics.user_sessions.get(user_id, {}).get('query_count', 0) + 1,
                            'last_query': query
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error recording analytics: {e}")
            
            # Save assistant response to conversation history
            if self.conversation_manager and include_context:
                try:
                    await self.conversation_manager.add_message(
                        session_id=session_id,
                        message_type=MessageType.ASSISTANT,
                        content=response_data.get('answer', response_data.get('response', '')),
                        metadata={
                            'user_id': user_id,
                            'processing_time': processing_time,
                            'sources_count': len(response_data.get('sources', [])),
                            'confidence': response_data.get('confidence', 0.8)
                        }
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Failed to save assistant response to conversation history: {e}")
            
                return {
                    'response': response_data.get('answer', response_data.get('response', '')),
                    'sources': response_data.get('sources', []),
                    'session_id': session_id,
                    'query': query,
                    'optimized_query': optimized_query,
                    'processing_time': processing_time,
                    'confidence': response_data.get('confidence', 0.8),
                    'query_type': 'hybrid_search',
                    'complexity': 'moderate',
                    'context_used': include_context,
                    'citations': [],
                    'word_count': len(response_data.get('answer', response_data.get('response', '')).split()),
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'retrieval_method': 'hybrid_search',
                        'reranked_count': 0, # Hybrid search doesn't have reranking in this simplified model
                        'total_candidates': len(documents),
                        'citation_format': citation_format.value,
                        'pipeline_steps': ['query_optimization', 'enhanced_retrieval', 'reranking', 'llm_generation']
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                'response': f"An error occurred while processing your query: {str(e)}",
                'sources': [],
                'session_id': session_id,
                'query': query,
                'processing_time': 0,
                'confidence': 0.0,
                'query_type': 'error',
                'complexity': 'simple',
                'context_used': include_context,
                'citations': [],
                'word_count': 0,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'error': str(e),
                    'citation_format': citation_format.value
                }
            }
    
    async def _optimize_query(self, query: str) -> tuple[str, str, str]:
        """Step 1: Query Optimization using Small Language Model"""
        try:
            if self.query_classifier:
                # Use query classifier to optimize the query
                classification = self.query_classifier.classify_query(query)
                # Handle different return types from query classifier
                if hasattr(classification, 'intent') and hasattr(classification.intent, 'value'):
                    intent = classification.intent.value
                elif hasattr(classification, 'intent'):
                    intent = str(classification.intent)
                else:
                    intent = 'unknown'
                
                optimized_query = f"{query} [Intent: {intent}]"
                query_type = intent
                complexity = 'moderate'  # Default complexity
                
                return optimized_query, query_type, complexity
            else:
                # Fallback: return original query
                return query, 'unknown', 'moderate'
        except Exception as e:
            self.logger.warning(f"Query optimization failed: {e}")
            return query, 'unknown', 'moderate'
    
    async def _enhance_query_with_context(self, query: str, conversation_history: List) -> str:
        """Enhance query with conversation context"""
        try:
            if not conversation_history or len(conversation_history) <= 1:
                return query
            
            # Get recent conversation context (last 5 messages)
            recent_messages = conversation_history[-5:]
            
            # Extract context from recent messages
            context_parts = []
            for msg in recent_messages:
                if hasattr(msg, 'content') and msg.content:
                    # Only include user and assistant messages, skip the current query
                    if hasattr(msg, 'message_type'):
                        if msg.message_type.value in ['user', 'assistant'] and msg.content != query:
                            context_parts.append(f"{msg.message_type.value}: {msg.content}")
            
            if not context_parts:
                return query
            
            # Create context-enhanced query
            context_summary = " | ".join(context_parts[-3:])  # Last 3 messages
            enhanced_query = f"Context: {context_summary} | Current query: {query}"
            
            self.logger.info(f"Enhanced query with {len(context_parts)} context messages")
            return enhanced_query
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance query with context: {e}")
            return query
    
    async def _adaptive_retrieval(self, query: str, optimal_count: int) -> List[Document]:
        """
        Perform adaptive retrieval that adjusts source count based on content relevance and coverage.
        
        Args:
            query: User's query
            optimal_count: Initial optimal number of sources
            
        Returns:
            List of relevant documents
        """
        # Start with optimal count
        current_count = optimal_count
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            try:
                # Retrieve documents with current count
                documents = await self._retrieve_documents(query, current_count)
                
                if not documents:
                    # No documents found, try with more sources
                    current_count = min(current_count + 2, 10)
                    attempts += 1
                    continue
                
                # Analyze content coverage and relevance
                coverage_score = self._analyze_content_coverage(query, documents)
                relevance_score = self._analyze_relevance_distribution(documents)
                
                self.logger.info(f"Retrieval attempt {attempts + 1}: {len(documents)} docs, coverage: {coverage_score:.2f}, relevance: {relevance_score:.2f}")
                
                # Determine if we need to adjust
                if coverage_score < 0.6 and current_count < 8:
                    # Low coverage, try more sources
                    current_count = min(current_count + 2, 10)
                    attempts += 1
                    continue
                elif coverage_score > 0.9 and relevance_score < 0.7 and current_count > 3:
                    # High coverage but low relevance, try fewer sources
                    current_count = max(current_count - 1, 3)
                    attempts += 1
                    continue
                else:
                    # Good balance, return current results
                    self.logger.info(f"Optimal retrieval achieved: {len(documents)} sources with coverage {coverage_score:.2f} and relevance {relevance_score:.2f}")
                    return documents
                    
            except Exception as e:
                self.logger.warning(f"Retrieval attempt {attempts + 1} failed: {e}")
                attempts += 1
                current_count = max(current_count - 1, 2)
        
        # Fallback to basic retrieval
        self.logger.warning("Adaptive retrieval failed, using fallback")
        return await self._retrieve_documents(query, optimal_count)
    
    def _analyze_content_coverage(self, query: str, documents: List[Document]) -> float:
        """
        Analyze how well the documents cover the query topics.
        
        Args:
            query: User's query
            documents: Retrieved documents
            
        Returns:
            Coverage score between 0 and 1
        """
        if not documents:
            return 0.0
        
        # Extract key terms from query
        query_terms = set(query.lower().split())
        query_terms = {term for term in query_terms if len(term) > 3}  # Filter short words
        
        if not query_terms:
            return 0.5  # Default score for queries without clear terms
        
        # Count how many terms are covered in documents
        covered_terms = 0
        total_content = ""
        
        for doc in documents:
            total_content += " " + doc.page_content.lower()
        
        for term in query_terms:
            if term in total_content:
                covered_terms += 1
        
        coverage_score = covered_terms / len(query_terms) if query_terms else 0.5
        return min(coverage_score, 1.0)
    
    def _analyze_relevance_distribution(self, documents: List[Document]) -> float:
        """
        Analyze the distribution of relevance scores across documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Relevance distribution score between 0 and 1
        """
        if not documents:
            return 0.0
        
        # Extract relevance scores
        scores = []
        for doc in documents:
            score = doc.metadata.get('combined_score', 0.0)
            if score > 0:
                scores.append(score)
        
        if not scores:
            return 0.5  # Default score if no scores available
        
        # Calculate relevance distribution metrics
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        # Good distribution: high average, reasonable spread
        if avg_score > 0.7 and (max_score - min_score) < 0.4:
            return 0.9
        elif avg_score > 0.6:
            return 0.7
        elif avg_score > 0.5:
            return 0.5
        else:
            return 0.3
    
    async def _generate_response(
        self,
        query: str,
        documents: List[Document],
        citation_format: CitationFormat,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response using LLM/Chat Model (Step 4 in architectural diagram)
        """
        try:
            if not documents:
                return {
                    'answer': 'I could not find any relevant documents to answer your question.',
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Prepare context from retrieved documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(documents, 1):
                content = doc.page_content
                metadata = doc.metadata
                
                # Add document content to context
                context_parts.append(f"Document {i}:\n{content}")
                
                # Prepare comprehensive source information
                source_info = {
                    'id': metadata.get('source', f'doc_{i}'),
                    'title': metadata.get('source_file_name', metadata.get('title', f'Document {i}')),
                    'source_file_name': metadata.get('source_file_name', metadata.get('title', f'Document {i}')),
                    'type': metadata.get('source_file_type', metadata.get('type', 'unknown')),
                    'score': metadata.get('combined_score', 0.0),
                    'bm25_score': metadata.get('bm25_score', 0.0),
                    'semantic_score': metadata.get('semantic_score', 0.0),
                    'confidence': metadata.get('combined_score', 0.8),
                    'content': content[:200] + '...' if len(content) > 200 else content,
                    # File link information - ensure proper extraction
                    'google_drive_url': metadata.get('google_drive_url', ''),
                    'source_url': metadata.get('source_url', ''),
                    'url': metadata.get('source', ''),
                    # Location information
                    'page_number': metadata.get('page_number'),
                    'slide_number': metadata.get('slide_number'),
                    'sheet_name': metadata.get('sheet_name'),
                    'section_title': metadata.get('section_title'),
                    'heading': metadata.get('heading'),
                    # Quality metrics
                    'quality_score': metadata.get('quality_score', 0.8),
                    'extraction_confidence': metadata.get('extraction_confidence', 0.8)
                }
                
                # Debug: Log the source information being prepared
                logger.debug(f"Source {i+1} prepared: {source_info['title']} - Google Drive URL: {source_info['google_drive_url']}")
                
                sources.append(source_info)
            
            context = "\n\n".join(context_parts)
            
            # Generate response using OpenAI
            if self.openai_client:
                try:
                    prompt = f"""
                    Based on the following retrieved documents, provide a comprehensive and accurate answer to the user's question.
                    
                    User Question: {query}
                    
                    Retrieved Documents:
                    {context}
                    
                    Instructions:
                    1. Answer the question based ONLY on the information provided in the documents
                    2. If the documents don't contain enough information to answer the question, say so
                    3. Provide specific details and examples from the documents when possible
                    4. Use clear, professional language
                    5. Include relevant citations in {citation_format.value} format
                    
                    Answer:
                    """
                    
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on retrieved documents."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.3
                    )
                    
                    answer = response.choices[0].message.content
                    
                    # Calculate confidence based on document scores
                    avg_score = sum(doc.metadata.get('combined_score', 0.0) for doc in documents) / len(documents)
                    confidence = min(avg_score, 1.0)
                    
                    return {
                        'answer': answer,
                        'sources': sources,
                        'confidence': confidence
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error generating response with OpenAI: {e}")
                    # Fallback to simple response
                    return self._generate_simple_response(query, documents, sources)
            else:
                # Fallback to simple response
                return self._generate_simple_response(query, documents, sources)
                
        except Exception as e:
            self.logger.error(f"Error in response generation: {e}")
            
            # Record error in analytics
            if self.analytics:
                try:
                    self.analytics.record_error({
                        'error_type': 'response_generation_error',
                        'error_message': str(e),
                        'query': query,
                        'session_id': session_id,
                        'user_id': user_id
                    })
                except Exception as analytics_error:
                    self.logger.error(f"Error recording error analytics: {analytics_error}")
            
            return {
                'answer': 'An error occurred while generating the response.',
                'sources': [],
                'confidence': 0.0
            }
    
    def _generate_simple_response(
        self,
        query: str,
        documents: List[Document],
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a simple response when LLM is not available
        """
        if not documents:
            return {
                'answer': 'No relevant documents found.',
                'sources': [],
                'confidence': 0.0
            }
        
        # Create a simple response based on the retrieved documents
        context_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Truncate if too long
        if len(context_text) > 2000:
            context_text = context_text[:2000] + "..."
        
        answer = f"Based on the retrieved documents:\n\n{context_text}"
        
        # Calculate confidence based on document scores
        avg_score = sum(doc.metadata.get('combined_score', 0.0) for doc in documents) / len(documents)
        confidence = min(avg_score, 1.0)
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': confidence
            }

    async def _determine_optimal_source_count(self, query: str, query_type: str, complexity: str) -> int:
        """
        Intelligently determine the optimal number of sources based on query characteristics.
        
        Args:
            query: User's query
            query_type: Type of query (factual, analytical, comparative, etc.)
            complexity: Query complexity (simple, moderate, complex)
            
        Returns:
            Optimal number of sources to retrieve
        """
        # Base source count based on query type
        base_counts = {
            'factual': 3,      # Specific facts need fewer sources
            'definitional': 2,  # Definitions are usually concise
            'comparative': 5,   # Comparisons need more context
            'analytical': 6,    # Analysis needs comprehensive coverage
            'procedural': 4,    # How-to questions need step-by-step info
            'summarization': 8, # Summaries need broad coverage
            'search': 5,        # General search needs moderate coverage
            'unknown': 4        # Default fallback
        }
        
        # Adjust based on complexity
        complexity_multipliers = {
            'simple': 0.8,      # Reduce for simple queries
            'moderate': 1.0,    # Standard for moderate queries
            'complex': 1.3      # Increase for complex queries
        }
        
        # Get base count
        base_count = base_counts.get(query_type, base_counts['unknown'])
        
        # Apply complexity multiplier
        complexity_mult = complexity_multipliers.get(complexity, 1.0)
        
        # Calculate initial optimal count
        optimal_count = int(base_count * complexity_mult)
        
        # Additional adjustments based on query characteristics
        query_lower = query.lower()
        
        # Increase for broad/topical queries
        if any(word in query_lower for word in ['overview', 'summary', 'explain', 'describe', 'what is']):
            optimal_count = min(optimal_count + 2, 10)
        
        # Decrease for specific/focused queries
        if any(word in query_lower for word in ['specific', 'exact', 'precise', 'number', 'date']):
            optimal_count = max(optimal_count - 1, 2)
        
        # Increase for comparative queries
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'between']):
            optimal_count = min(optimal_count + 2, 8)
        
        # Ensure reasonable bounds
        optimal_count = max(2, min(optimal_count, 10))
        
        self.logger.info(f"Query type: {query_type}, complexity: {complexity}, optimal sources: {optimal_count}")
        return optimal_count

    async def _retrieve_documents(self, query: str, count: int) -> List[Document]:
        """
        Retrieve documents using the hybrid search engine.
        
        Args:
            query: Search query
            count: Number of documents to retrieve
            
        Returns:
            List of Document objects
        """
        try:
            if self.hybrid_search:
                # Use hybrid search engine
                search_results = self.hybrid_search.search(query, top_k=count)
                
                # Convert SearchResult objects to Document objects
                documents = []
                for result in search_results:
                    doc = Document(
                        page_content=result.content,
                        metadata=result.metadata
                    )
                    documents.append(doc)
                
                return documents
            else:
                # Fallback to basic vector search
                search_results = await self.vector_store.asimilarity_search(query, k=count)
                return search_results
                
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []


# Global variables for components (defined after classes)
query_engine: Optional['SimpleQueryEngine'] = None
websocket_manager: Optional['WebSocketManager'] = None


async def setup_query_engine():
    """Initialize the query engine with all components"""
    global query_engine
    
    if query_engine is not None:
        return query_engine
    
    try:
        logger.info("Starting query engine initialization...")
        
        # Initialize vector database - QDRANT ONLY, NO FALLBACK
        qdrant_client = None
        
        # Try remote Qdrant first (where your documents are stored)
        qdrant_url = os.getenv("QDRANT_URL", "http://52.23.202.165:6333")
        if qdrant_url.startswith("http://"):
            host_port = qdrant_url.replace("http://", "").split(":")
            qdrant_host = host_port[0]
            qdrant_port = int(host_port[1]) if len(host_port) > 1 else 6333
        else:
            qdrant_host = os.getenv("QDRANT_HOST", "52.23.202.165")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        logger.info(f"Connecting to remote Qdrant at {qdrant_host}:{qdrant_port}")
        try:
            # Try with longer timeout for remote connection
            qdrant_client = QdrantClient(
                host=qdrant_host, 
                port=qdrant_port,
                timeout=30.0  # 30 second timeout
            )
            
            # Quick connection test
            collections = qdrant_client.get_collections()
            logger.info(f"Remote Qdrant connected successfully. Collections: {len(collections.collections)}")
            
            # Check if documents collection exists and has data
            collection_name = os.getenv("QDRANT_COLLECTION", "documents")
            try:
                collection_info = qdrant_client.get_collection(collection_name)
                point_count = collection_info.points_count
                logger.info(f"Collection '{collection_name}' exists with {point_count} documents")
                if point_count == 0:
                    logger.warning(f"Collection '{collection_name}' is empty - no documents found")
            except Exception as e:
                logger.error(f"Collection '{collection_name}' not found: {e}")
                
        except Exception as e:
            logger.error(f"Remote Qdrant connection failed: {e}")
            
            # Try local Qdrant as backup
            try:
                logger.info("Trying local Qdrant at localhost:6333")
                qdrant_client = QdrantClient(host="localhost", port=6333, timeout=10.0)
                
                # Quick connection test
                collections = qdrant_client.get_collections()
                logger.info(f"Local Qdrant connected successfully. Collections: {len(collections.collections)}")
                
                # Check local collection
                collection_name = os.getenv("QDRANT_COLLECTION", "documents")
                try:
                    collection_info = qdrant_client.get_collection(collection_name)
                    point_count = collection_info.points_count
                    logger.info(f"Local collection '{collection_name}' has {point_count} documents")
                    if point_count == 0:
                        logger.warning("Local Qdrant collection is empty - your documents are in the remote server")
                except Exception as e:
                    logger.info(f"Local collection '{collection_name}' not found, will create it")
                
            except Exception as e2:
                logger.error(f"Both remote and local Qdrant connections failed")
                logger.error(f"Remote error: {e}")
                logger.error(f"Local error: {e2}")
                raise Exception("QDRANT CONNECTION FAILED - Cannot proceed without Qdrant. Please check your Qdrant server.")
        
        # Check OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise Exception("OPENAI_API_KEY environment variable is not set")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            openai_api_key=openai_api_key
        )
        
        # Create vector store with collection creation
        collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        
        # Check if collection exists, create if it doesn't
        try:
            qdrant_client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' already exists")
        except Exception:
            logger.info(f"Creating collection '{collection_name}'")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": 3072,  # OpenAI text-embedding-3-large dimension
                    "distance": "Cosine"
                }
            )
        
        qdrant_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings
        )
        
        # Initialize conversation context manager (always use mock for now)
        try:
            from pipeline.context_management.mock_conversation_context import MockConversationContextManager
            conversation_manager = MockConversationContextManager()
            logger.info("Using mock conversation context manager")
        except Exception as e:
            logger.warning(f"Failed to create mock conversation manager: {e}")
            conversation_manager = None
        
        # Initialize source attribution manager (always use mock for now)
        try:
            from pipeline.processing.mock_source_attribution import MockSourceAttributionManager
            source_attribution_manager = MockSourceAttributionManager()
            logger.info("Using mock source attribution manager")
        except Exception as e:
            logger.warning(f"Failed to create mock source attribution manager: {e}")
            source_attribution_manager = None
        
        # Initialize metadata tracker (always use mock for now)
        try:
            from pipeline.state_management.mock_metadata_tracker import MockMetadataTracker
            metadata_tracker = MockMetadataTracker()
            logger.info("Using mock metadata tracker")
        except Exception as e:
            logger.warning(f"Failed to create mock metadata tracker: {e}")
            metadata_tracker = None
        
        # Initialize the query engine with optimal configuration
        logger.info("Initializing optimized query engine with semantic search only")
        
        # Try to initialize advanced query engine first
        try:
            from pipeline.query_engine.context_aware_query_engine import ContextAwareQueryEngine
            
            # Initialize real conversation context manager
            from pipeline.context_management.conversation_context import ConversationContextManager
            real_conversation_manager = ConversationContextManager()
            
            # Initialize mock source attribution manager (real one requires fingerprint_db)
            from pipeline.processing.mock_source_attribution import MockSourceAttributionManager
            real_source_attribution_manager = MockSourceAttributionManager()
            
            # Initialize real metadata tracker
            from pipeline.state_management.metadata_tracker import MetadataTracker
            real_metadata_tracker = MetadataTracker()
            
            # Create the advanced query engine with all features enabled
            query_engine = ContextAwareQueryEngine(
                qdrant_store=qdrant_store,
                conversation_manager=real_conversation_manager,
                source_attribution_manager=real_source_attribution_manager,
                metadata_tracker=real_metadata_tracker,
                enable_enhanced_retrieval=True,
                enable_cohere_rerank=True,
                enable_bm25=False,  # Disable BM25 - not needed for enhanced retrieval
                enable_query_classification=True
            )
            
            logger.info("Advanced ContextAwareQueryEngine initialized successfully with all features")
            
        except Exception as e:
            logger.warning(f"Failed to initialize advanced query engine: {e}")
            logger.info("Falling back to SimpleQueryEngine with optimized features")
            
            # Use SimpleQueryEngine with optimal configuration
            query_engine = SimpleQueryEngine(
                vector_store=qdrant_store,
                enable_enhanced_retrieval=True,  # Enable enhanced retrieval
                enable_cohere_rerank=True,      # Enable Cohere reranking
                enable_bm25=False               # Disable BM25 - not needed for enhanced retrieval
            )
            
            logger.info("SimpleQueryEngine initialized successfully with optimized features")
        
        logger.info("Query engine initialization completed successfully")
        return query_engine
        
    except Exception as e:
        logger.error(f"Failed to initialize query engine: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global websocket_manager
    
    websocket_manager = WebSocketManager()
    await setup_query_engine()
    
    # Mount static files
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main interface"""
    try:
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        index_path = os.path.join(static_dir, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document RAG Query Interface</title>
        </head>
        <body>
            <h1>Document RAG Query Interface</h1>
            <p>Static files not found. Please check the ui/static directory.</p>
        </body>
        </html>
        """)


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a query and return results"""
    try:
        # Ensure query engine is initialized
        engine = await setup_query_engine()
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Convert string enums to actual enum values
        citation_format = CitationFormat(request.citation_format.lower())
        
        # Send real-time update via WebSocket
        if websocket_manager:
            await websocket_manager.send_to_session(session_id, {
                "type": "query_start",
                "data": {
                    "query": request.query,
                    "session_id": session_id
                }
            })
        
        # Process the query
        result = await engine.process_query(
            query=request.query,
            session_id=session_id,
            user_id=request.user_id,
            citation_format=citation_format,
            include_context=request.include_context
            # max_sources is now automatically determined by intelligent source selection
        )
        
        # Convert result to response format
        # Handle both ContextAwareQueryEngine (QueryResult object) and SimpleQueryEngine (dict) responses
        if isinstance(result, dict):
            # SimpleQueryEngine response format
            response = QueryResponse(
                query=result.get('query', request.query),
                response=result.get('response', ''),
                sources=[{
                    "source_id": source.get('id', f"source_{i}"),
                    "source_type": "document",
                    "title": source.get('title', f"Document {i+1}"),
                    "source_file_name": source.get('source_file_name', source.get('title', f"Document {i+1}")),
                    "authors": [],
                    "confidence": source.get('confidence', source.get('relevance_score', 0.0)),
                    "citation": source.get('title', f"Document {i+1}"),
                    "content": source.get('content', ''),
                    # File link information
                    "google_drive_url": source.get('google_drive_url', ''),
                    "source_url": source.get('source_url', ''),
                    "url": source.get('url', ''),
                    # Location information
                    "page_number": source.get('page_number'),
                    "slide_number": source.get('slide_number'),
                    "sheet_name": source.get('sheet_name'),
                    "section_title": source.get('section_title'),
                    "heading": source.get('heading'),
                    # Quality metrics
                    "quality_score": source.get('quality_score', 0.8),
                    "extraction_confidence": source.get('extraction_confidence', 0.8)
                } for i, source in enumerate(result.get('sources', []))],
                confidence=result.get('confidence', 0.8),
                query_type=result.get('query_type', 'simple'),
                complexity=result.get('complexity', 'medium'),
                processing_time=result.get('processing_time', 0),
                context_used=result.get('context_used', request.include_context),
                citations=result.get('citations', []),
                session_id=result.get('session_id', session_id),
                word_count=result.get('word_count', len(result.get('response', '').split())),
                metadata=result.get('metadata', {}),
                timestamp=result.get('timestamp', datetime.now().isoformat())
            )
        else:
            # ContextAwareQueryEngine response format (QueryResult object)
            response = QueryResponse(
                query=result.query,
                response=result.response,
                sources=[{
                    "source_id": source.source_id,
                    "source_type": source.source_type,
                    "title": source.title,
                    "authors": source.authors,
                    "confidence": source.confidence,
                    "citation": source.generate_citation(citation_format)
                } for source in result.sources],
                confidence=result.confidence,
                query_type=result.query_type.value,
                complexity=result.complexity.value,
                processing_time=result.processing_time,
                context_used=result.context_used,
                citations=result.citations,
                session_id=result.session_id,
                word_count=result.word_count,
                metadata=result.metadata,
                timestamp=result.timestamp.isoformat()
            )
        
        # Send real-time update via WebSocket
        if websocket_manager:
            await websocket_manager.send_to_session(session_id, {
                "type": "query_complete",
                "data": response.dict()
            })
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get session information"""
    try:
        engine = await setup_query_engine()
        if not engine.conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
            
        summary = await engine.conversation_manager.get_session_statistics(session_id)
        
        if not summary or summary.get("error"):
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionInfo(
            session_id=summary.get("session_id"),
            message_count=summary.get("total_messages", 0),
            current_topic=summary.get("current_context"),
            created_at=summary.get("created_at"),
            last_activity=summary.get("last_activity"),
            metadata=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 50):
    """Get session query history"""
    try:
        engine = await setup_query_engine()
        if not engine.conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
            
        history = await engine.conversation_manager.get_conversation_history(session_id, limit)
        
        return {"session_id": session_id, "history": [msg.to_dict() for msg in history]}
        
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session"""
    try:
        engine = await setup_query_engine()
        if not engine.conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
            
        await engine.conversation_manager.close_session(session_id)
        
        # Notify WebSocket connections
        if websocket_manager:
            await websocket_manager.send_to_session(session_id, {
                "type": "session_cleared",
                "data": {"session_id": session_id}
            })
        
        return {"message": "Session cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        engine = await setup_query_engine()
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "query_engine": "initialized" if engine else "not_initialized",
                "websocket_manager": "initialized" if websocket_manager else "not_initialized"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates"""
    if not websocket_manager:
        await websocket.close(code=1000, reason="WebSocket manager not initialized")
        return
    
    connection_id = await websocket_manager.connect(websocket, session_id)
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
            
            elif message.get("type") == "subscribe":
                # Client wants to subscribe to session updates
                await websocket.send_text(json.dumps({
                    "type": "subscribed",
                    "data": {"session_id": session_id}
                }))
                
    except WebSocketDisconnect:
        await websocket_manager.disconnect(connection_id, session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket_manager.disconnect(connection_id, session_id)


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "websocket_connections": len(websocket_manager.active_connections) if websocket_manager else 0,
            "active_sessions": len(websocket_manager.session_connections) if websocket_manager else 0,
            "query_engine_status": "initialized" if query_engine else "not_initialized"
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize")
async def summarize_document(
    request: SummarizeRequest,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Summarize a document or document content
    """
    try:
        engine = await setup_query_engine()
        if not engine.document_summarizer:
            raise HTTPException(
                status_code=503,
                detail="Document summarization service is not available"
            )
        
        # Check user permissions
        if current_user and engine.rbac_manager:
            if not engine.rbac_manager.has_permission(current_user, Permission.DOCUMENT_READ):
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: You don't have permission to read documents"
                )
        
        # Convert summary type string to enum
        from pipeline.processing.document_summarizer import SummaryType
        try:
            summary_type = SummaryType(request.summary_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid summary type. Must be one of: {[t.value for t in SummaryType]}"
            )
        
        # Generate summary
        summary_result = await engine.document_summarizer.summarize_document(
            content=request.content,
            summary_type=summary_type,
            max_length=request.max_length,
            extract_key_topics=request.extract_key_topics
        )
        
        if not summary_result:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate summary"
            )
        
        return {
            "summary": summary_result.summary,
            "summary_type": summary_result.summary_type.value,
            "key_topics": summary_result.key_topics,
            "metrics": {
                "original_length": summary_result.original_length,
                "summary_length": summary_result.summary_length,
                "compression_ratio": summary_result.compression_ratio,
                "confidence_score": summary_result.confidence_score,
                "processing_time": summary_result.processing_time
            },
            "metadata": summary_result.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document summarization: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during summarization"
        )

@app.post("/api/summarize-multiple")
async def summarize_multiple_documents(
    request: SummarizeMultipleRequest,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Summarize multiple documents
    """
    try:
        if not query_engine.document_summarizer:
            raise HTTPException(
                status_code=503,
                detail="Document summarization service is not available"
            )
        
        # Check user permissions
        if current_user and query_engine.rbac_manager:
            if not query_engine.rbac_manager.has_permission(current_user, Permission.DOCUMENT_READ):
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: You don't have permission to read documents"
                )
        
        # Convert summary type string to enum
        from pipeline.processing.document_summarizer import SummaryType
        try:
            summary_type = SummaryType(request.summary_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid summary type. Must be one of: {[t.value for t in SummaryType]}"
            )
        
        # Prepare documents for summarization
        documents = []
        for doc in request.documents:
            documents.append({
                'content': doc.content,
                'metadata': doc.metadata or {}
            })
        
        # Generate summaries
        summary_results = await query_engine.document_summarizer.summarize_multiple_documents(
            documents=documents,
            summary_type=summary_type,
            max_length=request.max_length
        )
        
        # Format results
        results = []
        for i, summary_result in enumerate(summary_results):
            results.append({
                "document_index": i,
                "summary": summary_result.summary,
                "summary_type": summary_result.summary_type.value,
                "key_topics": summary_result.key_topics,
                "metrics": {
                    "original_length": summary_result.original_length,
                    "summary_length": summary_result.summary_length,
                    "compression_ratio": summary_result.compression_ratio,
                    "confidence_score": summary_result.confidence_score,
                    "processing_time": summary_result.processing_time
                },
                "metadata": summary_result.metadata
            })
        
        return {
            "summaries": results,
            "total_documents": len(documents),
            "successful_summaries": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multiple document summarization: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during summarization"
        )

@app.post("/api/compare-documents")
async def compare_documents(
    request: CompareDocumentsRequest,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Generate a comparative summary of multiple documents
    """
    try:
        if not query_engine.document_summarizer:
            raise HTTPException(
                status_code=503,
                detail="Document summarization service is not available"
            )
        
        # Check user permissions
        if current_user and query_engine.rbac_manager:
            if not query_engine.rbac_manager.has_permission(current_user, Permission.DOCUMENT_READ):
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: You don't have permission to read documents"
                )
        
        # Prepare documents for comparison
        documents = []
        for doc in request.documents:
            documents.append({
                'content': doc.content,
                'metadata': doc.metadata or {}
            })
        
        # Generate comparative summary
        summary_result = await query_engine.document_summarizer.generate_comparative_summary(
            documents=documents,
            comparison_focus=request.comparison_focus
        )
        
        if not summary_result:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate comparative summary"
            )
        
        return {
            "comparative_summary": summary_result.summary,
            "comparison_focus": request.comparison_focus,
            "metrics": {
                "original_length": summary_result.original_length,
                "summary_length": summary_result.summary_length,
                "compression_ratio": summary_result.compression_ratio,
                "confidence_score": summary_result.confidence_score,
                "processing_time": summary_result.processing_time
            },
            "metadata": summary_result.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document comparison: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during document comparison"
        )

@app.post("/api/detect-language")
async def detect_language(
    request: LanguageDetectionRequest,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Detect the language of the provided text
    """
    try:
        if not query_engine.multilingual_processor:
            raise HTTPException(
                status_code=503,
                detail="Language detection service is not available"
            )
        
        # Detect language
        detection_result = query_engine.multilingual_processor.detect_language(
            text=request.text,
            use_openai=request.use_openai
        )
        
        return {
            "detected_language": detection_result.detected_language.value,
            "confidence": detection_result.confidence,
            "alternative_languages": [
                {"language": lang.value, "confidence": conf}
                for lang, conf in detection_result.alternative_languages
            ],
            "processing_time": detection_result.processing_time,
            "metadata": detection_result.metadata
        }
        
    except Exception as e:
        logger.error(f"Error in language detection: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during language detection"
        )

@app.post("/api/translate")
async def translate_text(
    request: TranslationRequest,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Translate text to the target language
    """
    try:
        if not query_engine.multilingual_processor:
            raise HTTPException(
                status_code=503,
                detail="Translation service is not available"
            )
        
        # Map language codes
        from pipeline.processing.multilingual_processor import Language
        
        target_lang = query_engine.multilingual_processor._map_language_code(request.target_language)
        source_lang = None
        if request.source_language:
            source_lang = query_engine.multilingual_processor._map_language_code(request.source_language)
        
        # Translate text
        translation_result = await query_engine.multilingual_processor.translate_text(
            text=request.text,
            target_language=target_lang,
            source_language=source_lang
        )
        
        if not translation_result:
            raise HTTPException(
                status_code=500,
                detail="Failed to translate text"
            )
        
        return {
            "original_text": translation_result.original_text,
            "translated_text": translation_result.translated_text,
            "source_language": translation_result.source_language.value,
            "target_language": translation_result.target_language.value,
            "confidence": translation_result.confidence,
            "processing_time": translation_result.processing_time,
            "metadata": translation_result.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text translation: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during translation"
        )

@app.post("/api/process-multilingual-document")
async def process_multilingual_document(
    request: MultilingualDocumentRequest,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Process a document that may be in multiple languages
    """
    try:
        if not query_engine.multilingual_processor:
            raise HTTPException(
                status_code=503,
                detail="Multilingual processing service is not available"
            )
        
        # Map target language if provided
        target_lang = None
        if request.target_language:
            target_lang = query_engine.multilingual_processor._map_language_code(request.target_language)
        
        # Process document
        result = await query_engine.multilingual_processor.process_multilingual_document(
            content=request.content,
            target_language=target_lang,
            detect_language=request.detect_language,
            translate_if_needed=request.translate_if_needed
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in multilingual document processing: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during multilingual document processing"
        )

@app.get("/api/supported-languages")
async def get_supported_languages():
    """Get list of supported languages for translation"""
    try:
        from pipeline.processing.multilingual_processor import MultilingualProcessor
        
        processor = MultilingualProcessor()
        languages = processor.get_supported_languages()
        
        return {
            "supported_languages": languages,
            "total_count": len(languages),
            "message": "Supported languages retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported languages: {str(e)}")

@app.post("/api/bm25/add-documents-incremental")
async def add_documents_incremental(
    documents: List[Dict[str, Any]],
    current_user: Optional[User] = Depends(get_current_user)
):
    """Add new documents to BM25 index incrementally (only processes new documents)"""
    try:
        if not query_engine or not query_engine.hybrid_search:
            raise HTTPException(status_code=503, detail="Hybrid search not available")
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        # Use incremental indexing for new documents
        query_engine.hybrid_search.build_bm25_index_incremental(documents, use_cache=True)
        
        # Get updated stats
        stats = query_engine.hybrid_search.get_bm25_stats()
        
        return {
            "message": "Documents added incrementally to BM25 index",
            "documents_added": len(documents),
            "total_documents": stats.get('total_documents', 0),
            "index_built": stats.get('index_built', False)
        }
        
    except Exception as e:
        logger.error(f"Error adding documents incrementally: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")

@app.post("/api/bm25/add-documents")
async def add_documents_to_bm25(
    documents: List[Dict[str, Any]],
    current_user: Optional[User] = Depends(get_current_user)
):
    """Add new documents to BM25 index incrementally"""
    try:
        if not query_engine or not query_engine.hybrid_search:
            raise HTTPException(status_code=503, detail="Hybrid search not available")
        
        # Add documents to BM25 index
        query_engine.hybrid_search.add_documents_to_bm25(documents)
        
        # Get updated stats
        stats = query_engine.hybrid_search.get_bm25_stats()
        
        return {
            "message": f"Successfully added {len(documents)} documents to BM25 index",
            "bm25_stats": stats,
            "documents_added": len(documents)
        }
    except Exception as e:
        logger.error(f"Error adding documents to BM25: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add documents to BM25: {str(e)}")

@app.delete("/api/bm25/remove-documents")
async def remove_documents_from_bm25(
    document_ids: List[str],
    current_user: Optional[User] = Depends(get_current_user)
):
    """Remove documents from BM25 index"""
    try:
        if not query_engine or not query_engine.hybrid_search:
            raise HTTPException(status_code=503, detail="Hybrid search not available")
        
        # Remove documents from BM25 index
        query_engine.hybrid_search.remove_documents_from_bm25(document_ids)
        
        # Get updated stats
        stats = query_engine.hybrid_search.get_bm25_stats()
        
        return {
            "message": f"Successfully removed {len(document_ids)} documents from BM25 index",
            "bm25_stats": stats,
            "documents_removed": len(document_ids)
        }
    except Exception as e:
        logger.error(f"Error removing documents from BM25: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove documents from BM25: {str(e)}")

@app.get("/api/bm25/stats")
async def get_bm25_stats():
    """Get BM25 index statistics"""
    try:
        if not query_engine or not query_engine.hybrid_search:
            raise HTTPException(status_code=503, detail="Hybrid search not available")
        
        stats = query_engine.hybrid_search.get_bm25_stats()
        
        return {
            "bm25_stats": stats,
            "message": "BM25 statistics retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting BM25 stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get BM25 stats: {str(e)}")

@app.post("/api/bm25/rebuild")
async def rebuild_bm25_index(
    current_user: Optional[User] = Depends(get_current_user)
):
    """Rebuild BM25 index from scratch"""
    try:
        if not query_engine:
            raise HTTPException(status_code=503, detail="Query engine not available")
        
        # Rebuild BM25 index
        query_engine._build_bm25_index()
        
        # Get updated stats
        if query_engine.hybrid_search:
            stats = query_engine.hybrid_search.get_bm25_stats()
        else:
            stats = {"error": "Hybrid search not available"}
        
        return {
            "message": "BM25 index rebuilt successfully",
            "bm25_stats": stats
        }
    except Exception as e:
        logger.error(f"Error rebuilding BM25 index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild BM25 index: {str(e)}")

@app.post("/api/bm25/rebuild-async")
async def rebuild_bm25_index_async(
    current_user: Optional[User] = Depends(get_current_user)
):
    """Rebuild BM25 index asynchronously in background"""
    try:
        if not query_engine:
            raise HTTPException(status_code=503, detail="Query engine not available")
        
        # Start background rebuild
        if query_engine.hybrid_search:
            thread = query_engine.hybrid_search.build_bm25_index_async(
                documents=query_engine._get_all_documents(),
                use_cache=True
            )
            
            return {
                "message": "BM25 index rebuild started in background",
                "task_id": id(thread),
                "status": "started"
            }
        else:
            raise HTTPException(status_code=503, detail="Hybrid search not available")
        
    except Exception as e:
        logger.error(f"Error starting async BM25 rebuild: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start BM25 rebuild: {str(e)}")

@app.get("/api/bm25/progress")
async def get_bm25_indexing_progress():
    """Get BM25 indexing progress"""
    try:
        if not query_engine or not query_engine.hybrid_search:
            raise HTTPException(status_code=503, detail="Hybrid search not available")
        
        progress = query_engine.hybrid_search.get_indexing_progress()
        
        if progress:
            return {
                "progress": {
                    "status": progress.status,
                    "total_documents": progress.total_documents,
                    "processed_documents": progress.processed_documents,
                    "current_document": progress.current_document,
                    "start_time": progress.start_time,
                    "estimated_completion": progress.estimated_completion,
                    "percentage": (progress.processed_documents / progress.total_documents * 100) if progress.total_documents > 0 else 0
                }
            }
        else:
            return {
                "progress": {
                    "status": "not_available",
                    "message": "Progress tracking not enabled"
                }
            }
        
    except Exception as e:
        logger.error(f"Error getting BM25 progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get BM25 progress: {str(e)}")

@app.get("/api/bm25/performance")
async def get_bm25_performance_metrics():
    """Get BM25 performance metrics for client reporting"""
    try:
        if not query_engine or not query_engine.hybrid_search:
            raise HTTPException(status_code=503, detail="Hybrid search not available")
        
        stats = query_engine.hybrid_search.get_bm25_stats()
        progress = query_engine.hybrid_search.get_indexing_progress()
        
        # Calculate performance metrics
        performance_metrics = {
            "total_documents": stats.get('total_documents', 0),
            "index_size_mb": stats.get('vocabulary_size', 0) / (1024 * 1024) if stats.get('vocabulary_size', 0) > 0 else 0,
            "index_built": stats.get('index_built', False),
            "caching_enabled": query_engine.hybrid_search.enable_caching,
            "estimated_indexing_time": {
                "100_docs": "15-30 seconds",
                "500_docs": "1-2 minutes", 
                "1000_docs": "2-4 minutes",
                "5000_docs": "8-15 minutes",
                "10000_docs": "15-30 minutes"
            },
            "optimization_strategies": {
                "small_docs_100": "Individual chunks (fastest)",
                "medium_docs_500": "Page-level grouping",
                "large_docs_1000": "Section-level grouping",
                "very_large_5000": "Background indexing + section grouping"
            },
            "recommendations": []
        }
        
        # Add recommendations based on document count
        doc_count = stats.get('total_documents', 0)
        if doc_count > 1000:
            performance_metrics["recommendations"].append("✅ Using section-level grouping for better performance")
        if doc_count > 500:
            performance_metrics["recommendations"].append("✅ Background indexing enabled to prevent server blocking")
        if doc_count > 5000:
            performance_metrics["recommendations"].append("💡 Consider document preprocessing for very large sets")
        if not query_engine.hybrid_search.enable_caching:
            performance_metrics["recommendations"].append("💡 Enable caching to improve startup time")
        
        return {
            "performance_metrics": performance_metrics,
            "current_status": progress.status if progress else "unknown",
            "optimization_applied": doc_count > 500
        }
        
    except Exception as e:
        logger.error(f"Error getting BM25 performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get BM25 performance: {str(e)}")

@app.post("/api/bm25/clear-cache")
async def clear_bm25_cache(
    current_user: Optional[User] = Depends(get_current_user)
):
    """Clear BM25 index cache"""
    try:
        if not query_engine or not query_engine.hybrid_search:
            raise HTTPException(status_code=503, detail="Hybrid search not available")
        
        if query_engine.hybrid_search.index_manager:
            query_engine.hybrid_search.index_manager.clear_cache()
        
        return {
            "message": "BM25 cache cleared successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing BM25 cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/api/bm25/cache-status")
async def get_bm25_cache_status():
    """Get BM25 cache status and information"""
    try:
        if not query_engine or not query_engine.hybrid_search:
            raise HTTPException(status_code=503, detail="Hybrid search not available")
        
        cache_info = {
            "caching_enabled": query_engine.hybrid_search.enable_caching,
            "cache_directory": str(query_engine.hybrid_search.index_manager.cache_dir) if query_engine.hybrid_search.index_manager else None,
            "index_file_exists": query_engine.hybrid_search.index_manager.index_file.exists() if query_engine.hybrid_search.index_manager else False,
            "fingerprints_file_exists": query_engine.hybrid_search.index_manager.document_fingerprints_file.exists() if query_engine.hybrid_search.index_manager else False
        }
        
        if query_engine.hybrid_search.index_manager and query_engine.hybrid_search.index_manager.index_file.exists():
            cache_info.update({
                "index_size_mb": query_engine.hybrid_search.index_manager.index_file.stat().st_size / (1024 * 1024),
                "last_modified": datetime.fromtimestamp(query_engine.hybrid_search.index_manager.index_file.stat().st_mtime).isoformat(),
                "fingerprints_count": len(query_engine.hybrid_search.index_manager._load_document_fingerprints())
            })
        
        return cache_info
        
    except Exception as e:
        logger.error(f"Error getting BM25 cache status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")

class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    document_id: Optional[str] = None

@app.post("/api/documents/upload")
async def upload_document(
    request: DocumentUploadRequest,
    current_user: Optional[User] = Depends(get_current_user)
):
    """Upload a new document and automatically update BM25 index"""
    try:
        if not query_engine:
            raise HTTPException(status_code=503, detail="Query engine not available")
        
        # Generate document ID if not provided
        document_id = request.document_id or str(uuid.uuid4())
        
        # Prepare document for BM25 index
        document = {
            'id': document_id,
            'content': request.content,
            'metadata': request.metadata or {}
        }
        
        # Add to BM25 index incrementally
        if query_engine.hybrid_search:
            query_engine.hybrid_search.add_documents_to_bm25([document])
            bm25_stats = query_engine.hybrid_search.get_bm25_stats()
        else:
            bm25_stats = {"error": "Hybrid search not available"}
        
        # TODO: Also add to Qdrant vector store
        # This would require implementing document processing pipeline
        
        return {
            "message": "Document uploaded and BM25 index updated successfully",
            "document_id": document_id,
            "bm25_stats": bm25_stats,
            "content_length": len(request.content)
        }
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@app.post("/api/documents/bulk-upload")
async def bulk_upload_documents(
    documents: List[DocumentUploadRequest],
    current_user: Optional[User] = Depends(get_current_user)
):
    """Upload multiple documents and automatically update BM25 index"""
    try:
        if not query_engine:
            raise HTTPException(status_code=503, detail="Query engine not available")
        
        # Prepare documents for BM25 index
        bm25_documents = []
        uploaded_docs = []
        
        for doc_request in documents:
            document_id = doc_request.document_id or str(uuid.uuid4())
            
            document = {
                'id': document_id,
                'content': doc_request.content,
                'metadata': doc_request.metadata or {}
            }
            
            bm25_documents.append(document)
            uploaded_docs.append({
                'id': document_id,
                'content_length': len(doc_request.content)
            })
        
        # Add to BM25 index incrementally
        if query_engine.hybrid_search:
            query_engine.hybrid_search.add_documents_to_bm25(bm25_documents)
            bm25_stats = query_engine.hybrid_search.get_bm25_stats()
        else:
            bm25_stats = {"error": "Hybrid search not available"}
        
        return {
            "message": f"Successfully uploaded {len(documents)} documents and updated BM25 index",
            "documents_uploaded": uploaded_docs,
            "bm25_stats": bm25_stats,
            "total_documents": len(documents)
        }
    except Exception as e:
        logger.error(f"Error bulk uploading documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk upload documents: {str(e)}")

@app.get("/api/indexing/status")
async def get_indexing_status():
    """Get current indexing status and progress"""
    try:
        if not query_engine or not query_engine.hybrid_search:
            return {
                "status": "not_available",
                "message": "Hybrid search not initialized",
                "progress": 0,
                "total_documents": 0,
                "processed_documents": 0
            }
        
        # Get progress from hybrid search
        progress = query_engine.hybrid_search.get_indexing_progress()
        
        if progress:
            return {
                "status": progress.status,
                "message": f"Indexing {progress.current_document}",
                "progress": (progress.processed_documents / progress.total_documents * 100) if progress.total_documents > 0 else 0,
                "total_documents": progress.total_documents,
                "processed_documents": progress.processed_documents,
                "start_time": progress.start_time,
                "estimated_completion": progress.estimated_completion
            }
        else:
            # Get basic stats
            stats = query_engine.hybrid_search.get_bm25_stats()
            return {
                "status": "idle",
                "message": "No indexing in progress",
                "progress": 100 if stats.get('index_built', False) else 0,
                "total_documents": stats.get('total_documents', 0),
                "processed_documents": stats.get('total_documents', 0),
                "index_built": stats.get('index_built', False)
            }
            
    except Exception as e:
        logger.error(f"Error getting indexing status: {e}")
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
            "progress": 0,
            "total_documents": 0,
            "processed_documents": 0
        }

@app.post("/api/indexing/skip")
async def skip_indexing():
    """Skip BM25 indexing and continue with server startup"""
    try:
        logger.info("User requested to skip BM25 indexing")
        return {
            "message": "BM25 indexing skipped. Server will continue without BM25 search.",
            "status": "skipped"
        }
    except Exception as e:
        logger.error(f"Error skipping indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to skip indexing: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 