#!/usr/bin/env python3
"""
Integrated FastAPI Backend with Enhanced Retrieval

This module integrates the enhanced pipeline components for superior
document retrieval and query processing.
"""

import asyncio
import logging
import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced pipeline components
try:
    from pipeline.query_engine.hybrid_search import BM25IndexManager
    from pipeline.query_engine.context_aware_query_engine import ContextAwareQueryEngine
    from pipeline.query_engine.enhanced_retrieval import EnhancedRetriever
    from pipeline.query_engine.response_generator import ResponseGenerator
    from pipeline.context_management.conversation_context import ConversationContext, MessageType
    from pipeline.processing.source_attribution import CitationFormat, SourceAttributionProcessor
    logger.info("✅ Successfully imported enhanced pipeline components")
except ImportError as e:
    logger.warning(f"⚠️ Could not import enhanced components: {e}")
    # Fallback to basic components
    from enum import Enum
    
    class CitationFormat(Enum):
        APA = "apa"
        MLA = "mla"
        CHICAGO = "chicago"
    
    class MessageType(Enum):
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"
    
    # Mock enhanced components
    BM25IndexManager = None
    ContextAwareQueryEngine = None
    EnhancedRetriever = None
    ResponseGenerator = None
    ConversationContext = None
    SourceAttributionProcessor = None

# Basic imports that should always work
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
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
    max_sources: Optional[int] = Field(5, description="Maximum number of sources")

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

# FastAPI application
app = FastAPI(
    title="Enhanced Document RAG Query Interface",
    description="Advanced web-based interface with enhanced retrieval and context-aware processing",
    version="2.0.0",
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

# Mount static files
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# Global variables for the enhanced query engine components
enhanced_query_engine = None
conversation_contexts: Dict[str, Any] = {}
sessions: Dict[str, SessionInfo] = {}

class EnhancedQueryEngine:
    """Enhanced query engine using advanced pipeline components."""
    
    def __init__(self):
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("COLLECTION_NAME", "documents")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        
        # Initialize components
        self.vector_store = None
        self.hybrid_search = None
        self.context_aware_engine = None
        self.response_generator = None
        self.source_attributor = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize enhanced components"""
        try:
            # Initialize basic vector store
            embeddings = OpenAIEmbeddings(model=self.embedding_model)
            
            self.vector_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.collection_name,
                embedding=embeddings,
                url=f"http://{self.qdrant_host}:{self.qdrant_port}",
            )
            
            logger.info(f"✅ Successfully connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
            
            # Initialize enhanced components if available
            if BM25IndexManager:
                self.hybrid_search = BM25IndexManager()
                logger.info("✅ Initialized BM25 hybrid search")
            
            if ContextAwareQueryEngine:
                self.context_aware_engine = ContextAwareQueryEngine(
                    vector_store=self.vector_store,
                    embeddings=embeddings
                )
                logger.info("✅ Initialized context-aware query engine")
            
            if ResponseGenerator:
                self.response_generator = ResponseGenerator()
                logger.info("✅ Initialized enhanced response generator")
            
            if SourceAttributionProcessor:
                self.source_attributor = SourceAttributionProcessor()
                logger.info("✅ Initialized source attribution processor")
                
        except Exception as e:
            logger.error(f"❌ Error initializing enhanced query engine: {e}")
            # Fallback to basic functionality
            self.vector_store = None
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query using enhanced retrieval components"""
        start_time = datetime.now()
        
        try:
            if not self.vector_store:
                raise HTTPException(status_code=503, detail="Vector store not available")
            
            # Get or create conversation context
            session_id = request.session_id or str(uuid.uuid4())
            conversation_context = conversation_contexts.get(session_id)
            
            if ConversationContext and not conversation_context:
                conversation_context = ConversationContext(session_id=session_id)
                conversation_contexts[session_id] = conversation_context
            
            # Use enhanced retrieval if available
            if self.context_aware_engine and conversation_context:
                # Add user message to context
                if hasattr(conversation_context, 'add_message'):
                    conversation_context.add_message(
                        message_type=MessageType.USER,
                        content=request.query,
                        metadata={"timestamp": datetime.now().isoformat()}
                    )
                
                # Process with context-aware engine
                result = await self._process_with_enhanced_engine(request, conversation_context)
            else:
                # Fallback to basic retrieval
                result = await self._process_with_basic_engine(request)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update result with processing time
            result.processing_time = processing_time
            result.session_id = session_id
            result.timestamp = datetime.now(timezone.utc).isoformat()
            
            # Add assistant response to context
            if conversation_context and hasattr(conversation_context, 'add_message'):
                conversation_context.add_message(
                    message_type=MessageType.ASSISTANT,
                    content=result.response,
                    metadata={
                        "sources": result.sources,
                        "confidence": result.confidence,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    async def _process_with_enhanced_engine(self, request: QueryRequest, context) -> QueryResponse:
        """Process query with enhanced components"""
        try:
            # Use context-aware query processing
            query_result = await self.context_aware_engine.process_query(
                query=request.query,
                conversation_context=context,
                max_sources=request.max_sources
            )
            
            # Generate enhanced response
            if self.response_generator:
                response_text = await self.response_generator.generate_response(
                    query=request.query,
                    retrieved_docs=query_result.get('documents', []),
                    context=context,
                    style=request.response_style
                )
            else:
                response_text = await self._generate_basic_response(
                    request.query, 
                    query_result.get('documents', [])
                )
            
            # Enhanced source attribution
            sources = []
            if self.source_attributor and query_result.get('documents'):
                sources = self.source_attributor.process_sources(
                    documents=query_result['documents'],
                    query=request.query,
                    citation_format=CitationFormat(request.citation_format)
                )
            else:
                # Fallback source processing
                for i, doc in enumerate(query_result.get('documents', [])):
                    source = {
                        "id": f"doc_{i}",
                        "title": doc.metadata.get("title", f"Document {i+1}"),
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "confidence": query_result.get('scores', [0.8])[i] if i < len(query_result.get('scores', [])) else 0.8,
                        "metadata": doc.metadata,
                        "page_number": doc.metadata.get("page", None),
                        "source_url": doc.metadata.get("source", None),
                        "google_drive_url": doc.metadata.get("google_drive_url", None)
                    }
                    sources.append(source)
            
            # Create citations
            citations = self._create_citations(query_result.get('documents', []), request.citation_format)
            
            return QueryResponse(
                query=request.query,
                response=response_text,
                sources=sources,
                confidence=query_result.get('confidence', 0.8),
                query_type=query_result.get('query_type', 'enhanced_retrieval'),
                complexity=query_result.get('complexity', 'medium'),
                processing_time=0.0,  # Will be updated by caller
                context_used=request.include_context,
                citations=citations,
                session_id="",  # Will be updated by caller
                word_count=len(response_text.split()),
                metadata={
                    "sources_count": len(sources),
                    "enhanced_processing": True
                },
                timestamp=""  # Will be updated by caller
            )
            
        except Exception as e:
            logger.error(f"Enhanced processing failed, falling back to basic: {e}")
            return await self._process_with_basic_engine(request)
    
    async def _process_with_basic_engine(self, request: QueryRequest) -> QueryResponse:
        """Fallback to basic query processing"""
        # Perform similarity search
        docs = self.vector_store.similarity_search(
            request.query, 
            k=request.max_sources or 5
        )
        
        # Create sources from documents
        sources = []
        for i, doc in enumerate(docs):
            source = {
                "id": f"doc_{i}",
                "title": doc.metadata.get("title", f"Document {i+1}"),
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "confidence": 0.8,  # Placeholder confidence
                "metadata": doc.metadata,
                "page_number": doc.metadata.get("page", None),
                "source_url": doc.metadata.get("source", None),
                "google_drive_url": doc.metadata.get("google_drive_url", None)
            }
            sources.append(source)
        
        # Generate response using retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        response_text = await self._generate_basic_response(request.query, docs)
        
        # Create citations
        citations = self._create_citations(docs, request.citation_format)
        
        return QueryResponse(
            query=request.query,
            response=response_text,
            sources=sources,
            confidence=0.8,  # Placeholder
            query_type="basic_retrieval",
            complexity="medium",
            processing_time=0.0,  # Will be updated by caller
            context_used=request.include_context,
            citations=citations,
            session_id="",  # Will be updated by caller
            word_count=len(response_text.split()),
            metadata={"sources_count": len(sources), "enhanced_processing": False},
            timestamp=""  # Will be updated by caller
        )
    
    async def _generate_basic_response(self, query: str, docs: List[Document]) -> str:
        """Generate response using OpenAI GPT"""
        try:
            import openai
            
            client = openai.OpenAI()
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""Based on the following context, please answer the user's question accurately and comprehensively.

Context:
{context}

Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            context = "\n\n".join([doc.page_content for doc in docs])
            return f"Based on the retrieved documents, here's what I found relevant to your query: {context[:500]}..."
    
    def _create_citations(self, docs: List[Document], format_type: str) -> List[str]:
        """Create citations for the sources"""
        citations = []
        for i, doc in enumerate(docs):
            title = doc.metadata.get("title", f"Document {i+1}")
            source = doc.metadata.get("source", "Unknown source")
            citations.append(f"[{i+1}] {title} - {source}")
        return citations

# Initialize enhanced query engine
enhanced_query_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced query engine on startup"""
    global enhanced_query_engine
    logger.info("🚀 Starting Enhanced Document RAG API...")
    
    try:
        enhanced_query_engine = EnhancedQueryEngine()
        logger.info("✅ Enhanced query engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize enhanced query engine: {e}")
        enhanced_query_engine = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("ui/static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Enhanced Document RAG API</h1><p>Static files not found. Please ensure ui/static/index.html exists.</p>")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return enhanced results"""
    if not enhanced_query_engine:
        raise HTTPException(status_code=503, detail="Enhanced query engine not initialized")
    
    # Create session if not exists
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = SessionInfo(
            session_id=session_id,
            message_count=0,
            current_topic=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            last_activity=datetime.now(timezone.utc).isoformat(),
            metadata={}
        )
    
    # Update session
    sessions[session_id].message_count += 1
    sessions[session_id].last_activity = datetime.now(timezone.utc).isoformat()
    
    # Process query with enhanced engine
    response = await enhanced_query_engine.process_query(request)
    
    return response

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    conversation_context = conversation_contexts.get(session_id)
    conversation_history = []
    
    if conversation_context and hasattr(conversation_context, 'get_messages'):
        conversation_history = [
            {
                "type": msg.message_type.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in conversation_context.get_messages()
        ]
    
    return {
        "session": sessions[session_id],
        "conversation_history": conversation_history
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if enhanced_query_engine and enhanced_query_engine.vector_store else "unhealthy"
    components = {
        "vector_store": enhanced_query_engine.vector_store is not None if enhanced_query_engine else False,
        "hybrid_search": enhanced_query_engine.hybrid_search is not None if enhanced_query_engine else False,
        "context_aware_engine": enhanced_query_engine.context_aware_engine is not None if enhanced_query_engine else False,
        "response_generator": enhanced_query_engine.response_generator is not None if enhanced_query_engine else False,
        "source_attributor": enhanced_query_engine.source_attributor is not None if enhanced_query_engine else False
    }
    
    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "components": components
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)