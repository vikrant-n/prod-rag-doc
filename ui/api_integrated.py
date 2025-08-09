@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    with tracer.start_as_current_span("api.health_check") as span:
        span.set_attribute("endpoint", "/api/health")
        
        status = "healthy" if enhanced_query_engine and enhanced_query_engine.vector_store else "unhealthy"
        
        # Check backend service health
        backend_status = "unknown"
        with tracer.start_as_current_span("backend.health_check") as backend_span:
            try:
                import httpx
                backend_url = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8001")
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{backend_url}/status", timeout=5.0)
                    if response.status_code == 200:
                        backend_status = "healthy"
                        backend_span.set_attribute("backend.status", "healthy")
                    else:
                        backend_status = "unhealthy"
                        backend_span.set_attribute("backend.status", "unhealthy")
                        backend_span.set_attribute("backend.status_code", response.status_code)
            except Exception as e:
                backend_span.record_exception(e)
                backend_status = "unreachable"
                backend_span.set_attribute("backend.status", "unreachable")
        
        components = {
            "vector_store": enhanced_query_engine.vector_store is not None if enhanced_query_engine else False,
            "hybrid_search": enhanced_query_engine.hybrid_search is not None if enhanced_query_engine else False,
            "context_aware_engine": enhanced_query_engine.context_aware_engine is not None if enhanced_query_engine else False,
            "response_generator": enhanced_query_engine.response_generator is not None if enhanced_query_engine else False,
            "source_attributor": enhanced_query_engine.source_attributor is not None if enhanced_query_engine else False,
            "backend_service": backend_status
        }
        
        health_data = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "components": components,
            "backend_service": {
                "status": backend_status,
                "url": os.getenv("BACKEND_SERVICE_URL", "http://localhost:8001")
            }
        }
        
        span.set_attribute("health.status", status)
        span.set_attribute("health.backend_status", backend_status)
        span.set_attribute("health.components", json.dumps(components))
        
        # Record health check metrics
        record_cache_event("health_check", status == "healthy")
        record_cache_event("backend_health_check", backend_status == "healthy")
        
        return health_data

@app.get("/api/backend/status")
async def get_backend_status():
    """Get backend service status (proxied call)"""
    with tracer.start_as_current_span("api.get_backend_status") as span:
        span.set_attribute("endpoint", "/api/backend/status")
        
        try:
            import httpx
            backend_url = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8001")
            
            with tracer.start_as_current_span("backend.status_call") as backend_span:
                backend_span.set_attribute("backend.url", backend_url)
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(f"{backend_url}/scan", timeout=30.0)
                    
                    backend_span.set_attribute("http.status_code", response.status_code)
                    backend_span.set_attribute("http.method", "POST")
                    backend_span.set_attribute("http.url", f"{backend_url}/scan")
                    
                    if response.status_code == 200:
                        scan_result = response.json()
                        span.set_attribute("backend.scan_triggered", True)
                        logger.info("Backend scan triggered successfully via API")
                        return scan_result
                    else:
                        span.set_attribute("api.error", "backend_scan_failed")
                        raise HTTPException(status_code=502, detail=f"Backend scan failed with status {response.status_code}")
                        
        except httpx.TimeoutException:
            span.set_attribute("api.error", "backend_timeout")
            raise HTTPException(status_code=504, detail="Backend scan request timeout")
        except httpx.ConnectError:
            span.set_attribute("api.error", "backend_unreachable")
            raise HTTPException(status_code=502, detail="Backend service unreachable")
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("api.error", "backend_scan_call_failed")
            raise HTTPException(status_code=502, detail=f"Failed to trigger backend scan: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    with tracer.start_as_current_span("app.main_startup") as span:
        span.set_attribute("app.mode", "main")
        logger.info("Starting API server directly")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
                
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{backend_url}/status", timeout=10.0)
                
                backend_span.set_attribute("http.status_code", response.status_code)
                backend_span.set_attribute("http.method", "GET")
                backend_span.set_attribute("http.url", f"{backend_url}/status")
                
                if response.status_code == 200:
                    backend_data = response.json()
                    span.set_attribute("backend.files_processed", backend_data.get("processing", {}).get("files_processed_session", 0))
                    span.set_attribute("backend.is_running", backend_data.get("service", {}).get("is_running", False))
                    return backend_data
                else:
                    span.set_attribute("api.error", "backend_unhealthy")
                    raise HTTPException(status_code=502, detail=f"Backend service returned {response.status_code}")
                        
        except httpx.TimeoutException:
            span.set_attribute("api.error", "backend_timeout")
            raise HTTPException(status_code=504, detail="Backend service timeout")
        except httpx.ConnectError:
            span.set_attribute("api.error", "backend_unreachable")
            raise HTTPException(status_code=502, detail="Backend service unreachable")
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("api.error", "backend_call_failed")
            raise HTTPException(status_code=502, detail=f"Failed to call backend service: {str(e)}")

@app.post("/api/backend/scan")
async def trigger_backend_scan():
    """Trigger backend scan (proxied call)"""
    with tracer.start_as_current_span("api.trigger_backend_scan") as span:
        span.set_attribute("endpoint", "/api/backend/scan")
        
        try:
            import httpx
            backend_url = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8001")
            
            with tracer.start_as_current_span("backend.scan_call") as backend_span:
                backend_span.set_attribute("backend.url", backend_url)
                #!/usr/bin/env python3
"""
Integrated FastAPI Backend with Enhanced Retrieval - Instrumented with OpenTelemetry

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

# Initialize OpenTelemetry
from otel_config import initialize_opentelemetry, get_tracer, add_trace_correlation_to_log
from metrics import (
    rag_metrics, time_query_processing, record_query_processed, 
    record_cache_event
)

# Initialize OpenTelemetry for API service
tracer, meter = initialize_opentelemetry(
    service_name="document-rag-api",
    service_version="2.0.0",
    environment="development"
)

# Setup logging with trace correlation
logging.basicConfig(level=logging.INFO)
logger = add_trace_correlation_to_log(logging.getLogger(__name__))

# Import enhanced pipeline components
try:
    from pipeline.query_engine.hybrid_search import BM25IndexManager
    from pipeline.query_engine.context_aware_query_engine import ContextAwareQueryEngine
    from pipeline.query_engine.enhanced_retrieval import EnhancedRetriever
    from pipeline.query_engine.response_generator import ResponseGenerator
    from pipeline.context_management.conversation_context import ConversationContext, MessageType
    from pipeline.processing.source_attribution import CitationFormat, SourceAttributionProcessor
    logger.info("Successfully imported enhanced pipeline components")
except ImportError as e:
    logger.warning(f"Could not import enhanced components: {e}")
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
        with tracer.start_as_current_span("query_engine.init") as span:
            self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            self.collection_name = os.getenv("COLLECTION_NAME", "documents")
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
            
            span.set_attribute("qdrant.host", self.qdrant_host)
            span.set_attribute("qdrant.port", self.qdrant_port)
            span.set_attribute("qdrant.collection", self.collection_name)
            span.set_attribute("embedding.model", self.embedding_model)
            
            # Initialize components
            self.vector_store = None
            self.hybrid_search = None
            self.context_aware_engine = None
            self.response_generator = None
            self.source_attributor = None
            
            self._initialize_components()
            span.set_attribute("init.status", "success")
    
    def _initialize_components(self):
        """Initialize enhanced components"""
        with tracer.start_as_current_span("query_engine.initialize_components") as span:
            try:
                # Initialize basic vector store
                with tracer.start_as_current_span("embeddings.init") as emb_span:
                    embeddings = OpenAIEmbeddings(model=self.embedding_model)
                    emb_span.set_attribute("model", self.embedding_model)
                
                with tracer.start_as_current_span("vector_store.connect") as vs_span:
                    qdrant_url = f"http://{self.qdrant_host}:{self.qdrant_port}"
                    vs_span.set_attribute("qdrant.url", qdrant_url)
                    vs_span.set_attribute("collection.name", self.collection_name)
                    
                    self.vector_store = QdrantVectorStore.from_existing_collection(
                        collection_name=self.collection_name,
                        embedding=embeddings,
                        url=qdrant_url,
                    )
                    vs_span.set_attribute("connection.status", "success")
                    logger.info(f"Successfully connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
                
                # Initialize enhanced components if available
                enhanced_components = {}
                
                if BM25IndexManager:
                    with tracer.start_as_current_span("hybrid_search.init"):
                        self.hybrid_search = BM25IndexManager()
                        enhanced_components["hybrid_search"] = True
                        logger.info("Initialized BM25 hybrid search")
                
                if ContextAwareQueryEngine:
                    with tracer.start_as_current_span("context_aware_engine.init"):
                        self.context_aware_engine = ContextAwareQueryEngine(
                            vector_store=self.vector_store,
                            embeddings=embeddings
                        )
                        enhanced_components["context_aware_engine"] = True
                        logger.info("Initialized context-aware query engine")
                
                if ResponseGenerator:
                    with tracer.start_as_current_span("response_generator.init"):
                        self.response_generator = ResponseGenerator()
                        enhanced_components["response_generator"] = True
                        logger.info("Initialized enhanced response generator")
                
                if SourceAttributionProcessor:
                    with tracer.start_as_current_span("source_attributor.init"):
                        self.source_attributor = SourceAttributionProcessor()
                        enhanced_components["source_attributor"] = True
                        logger.info("Initialized source attribution processor")
                
                span.set_attribute("enhanced_components", json.dumps(enhanced_components))
                span.set_attribute("components.vector_store", True)
                span.set_attribute("init.status", "success")
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("init.status", "failed")
                logger.error(f"Error initializing enhanced query engine: {e}")
                # Fallback to basic functionality
                self.vector_store = None
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query using enhanced retrieval components"""
        with tracer.start_as_current_span("query.process") as span:
            start_time = datetime.now()
            
            span.set_attribute("query.text", request.query[:100] + "..." if len(request.query) > 100 else request.query)
            span.set_attribute("query.max_sources", request.max_sources or 5)
            span.set_attribute("query.citation_format", request.citation_format)
            span.set_attribute("query.include_context", request.include_context)
            span.set_attribute("query.response_style", request.response_style)
            
            try:
                if not self.vector_store:
                    span.set_attribute("query.error", "vector_store_unavailable")
                    raise HTTPException(status_code=503, detail="Vector store not available")
                
                # Get or create conversation context
                session_id = request.session_id or str(uuid.uuid4())
                conversation_context = conversation_contexts.get(session_id)
                
                span.set_attribute("session.id", session_id)
                span.set_attribute("session.has_context", conversation_context is not None)
                
                if ConversationContext and not conversation_context:
                    with tracer.start_as_current_span("session.create_context"):
                        conversation_context = ConversationContext(session_id=session_id)
                        conversation_contexts[session_id] = conversation_context
                        span.add_event("conversation_context_created")
                
                # Use enhanced retrieval if available
                if self.context_aware_engine and conversation_context:
                    span.set_attribute("processing.type", "enhanced")
                    # Add user message to context
                    if hasattr(conversation_context, 'add_message'):
                        with tracer.start_as_current_span("context.add_user_message"):
                            conversation_context.add_message(
                                message_type=MessageType.USER,
                                content=request.query,
                                metadata={"timestamp": datetime.now().isoformat()}
                            )
                    
                    # Process with context-aware engine
                    result = await self._process_with_enhanced_engine(request, conversation_context)
                else:
                    span.set_attribute("processing.type", "basic")
                    # Fallback to basic retrieval
                    result = await self._process_with_basic_engine(request)
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Update result with processing time
                result.processing_time = processing_time
                result.session_id = session_id
                result.timestamp = datetime.now(timezone.utc).isoformat()
                
                span.set_attribute("query.processing_time", processing_time)
                span.set_attribute("query.result.sources_count", len(result.sources))
                span.set_attribute("query.result.confidence", result.confidence)
                span.set_attribute("query.result.word_count", result.word_count)
                
                # Record metrics
                record_query_processed(
                    query_type=result.query_type,
                    result_count=len(result.sources),
                    confidence=result.confidence
                )
                
                # Add assistant response to context
                if conversation_context and hasattr(conversation_context, 'add_message'):
                    with tracer.start_as_current_span("context.add_assistant_message"):
                        conversation_context.add_message(
                            message_type=MessageType.ASSISTANT,
                            content=result.response,
                            metadata={
                                "sources": result.sources,
                                "confidence": result.confidence,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                
                span.set_attribute("query.status", "success")
                logger.info(f"Query processed successfully: {processing_time:.2f}s, {len(result.sources)} sources")
                
                return result
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("query.status", "error")
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    async def _process_with_enhanced_engine(self, request: QueryRequest, context) -> QueryResponse:
        """Process query with enhanced components"""
        with tracer.start_as_current_span("query.process_enhanced") as span:
            try:
                # Use context-aware query processing
                with tracer.start_as_current_span("context_aware_engine.process_query") as cae_span:
                    query_result = await self.context_aware_engine.process_query(
                        query=request.query,
                        conversation_context=context,
                        max_sources=request.max_sources
                    )
                    cae_span.set_attribute("query_result.documents_count", len(query_result.get('documents', [])))
                    cae_span.set_attribute("query_result.confidence", query_result.get('confidence', 0.0))
                
                # Generate enhanced response
                if self.response_generator:
                    with tracer.start_as_current_span("response_generator.generate") as rg_span:
                        response_text = await self.response_generator.generate_response(
                            query=request.query,
                            retrieved_docs=query_result.get('documents', []),
                            context=context,
                            style=request.response_style
                        )
                        rg_span.set_attribute("response.length", len(response_text))
                        rg_span.set_attribute("response.word_count", len(response_text.split()))
                else:
                    response_text = await self._generate_basic_response(
                        request.query, 
                        query_result.get('documents', [])
                    )
                
                # Enhanced source attribution
                sources = []
                if self.source_attributor and query_result.get('documents'):
                    with tracer.start_as_current_span("source_attributor.process") as sa_span:
                        sources = self.source_attributor.process_sources(
                            documents=query_result['documents'],
                            query=request.query,
                            citation_format=CitationFormat(request.citation_format)
                        )
                        sa_span.set_attribute("sources.processed_count", len(sources))
                else:
                    # Fallback source processing
                    with tracer.start_as_current_span("sources.fallback_processing"):
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
                
                span.set_attribute("enhanced.processing_success", True)
                span.set_attribute("enhanced.sources_count", len(sources))
                span.set_attribute("enhanced.citations_count", len(citations))
                
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
                span.record_exception(e)
                span.set_attribute("enhanced.processing_success", False)
                logger.error(f"Enhanced processing failed, falling back to basic: {e}")
                return await self._process_with_basic_engine(request)
    
    async def _process_with_basic_engine(self, request: QueryRequest) -> QueryResponse:
        """Fallback to basic query processing"""
        with tracer.start_as_current_span("query.process_basic") as span:
            span.set_attribute("processing.type", "basic_fallback")
            
            # Perform similarity search
            with tracer.start_as_current_span("vector_store.similarity_search") as search_span:
                docs = self.vector_store.similarity_search(
                    request.query, 
                    k=request.max_sources or 5
                )
                search_span.set_attribute("search.query", request.query[:100] + "..." if len(request.query) > 100 else request.query)
                search_span.set_attribute("search.k", request.max_sources or 5)
                search_span.set_attribute("search.results_count", len(docs))
            
            # Create sources from documents
            sources = []
            with tracer.start_as_current_span("sources.basic_processing") as sources_span:
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
                sources_span.set_attribute("sources.created_count", len(sources))
            
            # Generate response using retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            response_text = await self._generate_basic_response(request.query, docs)
            
            # Create citations
            citations = self._create_citations(docs, request.citation_format)
            
            span.set_attribute("basic.sources_count", len(sources))
            span.set_attribute("basic.response_length", len(response_text))
            span.set_attribute("basic.word_count", len(response_text.split()))
            
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
        with tracer.start_as_current_span("response.generate_basic") as span:
            span.set_attribute("response.query", query[:100] + "..." if len(query) > 100 else query)
            span.set_attribute("response.docs_count", len(docs))
            
            try:
                import openai
                
                with tracer.start_as_current_span("openai.chat_completion") as openai_span:
                    client = openai.OpenAI()
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    openai_span.set_attribute("openai.model", "gpt-4o-mini")
                    openai_span.set_attribute("openai.context_length", len(context))
                    
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
                    
                    response_text = response.choices[0].message.content
                    openai_span.set_attribute("openai.response_length", len(response_text))
                    openai_span.set_attribute("openai.tokens_used", response.usage.total_tokens if hasattr(response, 'usage') else 0)
                    
                    span.set_attribute("response.generation_status", "success")
                    return response_text
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("response.generation_status", "fallback")
                logger.error(f"Error generating response: {e}")
                context = "\n\n".join([doc.page_content for doc in docs])
                return f"Based on the retrieved documents, here's what I found relevant to your query: {context[:500]}..."
    
    def _create_citations(self, docs: List[Document], format_type: str) -> List[str]:
        """Create citations for the sources"""
        with tracer.start_as_current_span("citations.create") as span:
            span.set_attribute("citations.format", format_type)
            span.set_attribute("citations.docs_count", len(docs))
            
            citations = []
            for i, doc in enumerate(docs):
                title = doc.metadata.get("title", f"Document {i+1}")
                source = doc.metadata.get("source", "Unknown source")
                citations.append(f"[{i+1}] {title} - {source}")
            
            span.set_attribute("citations.created_count", len(citations))
            return citations

# Initialize enhanced query engine
enhanced_query_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced query engine on startup"""
    with tracer.start_as_current_span("fastapi.startup") as span:
        global enhanced_query_engine
        logger.info("Starting Enhanced Document RAG API...")
        
        try:
            enhanced_query_engine = EnhancedQueryEngine()
            span.set_attribute("startup.query_engine", "success")
            logger.info("Enhanced query engine initialized successfully")
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("startup.query_engine", "failed")
            logger.error(f"Failed to initialize enhanced query engine: {e}")
            enhanced_query_engine = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with tracer.start_as_current_span("api.serve_root") as span:
        span.set_attribute("endpoint", "/")
        
        try:
            with open("ui/static/index.html", "r") as f:
                content = f.read()
                span.set_attribute("static.file_served", "index.html")
                span.set_attribute("static.content_length", len(content))
                return HTMLResponse(content=content)
        except FileNotFoundError:
            span.set_attribute("static.error", "file_not_found")
            return HTMLResponse(content="<h1>Enhanced Document RAG API</h1><p>Static files not found. Please ensure ui/static/index.html exists.</p>")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return enhanced results"""
    with tracer.start_as_current_span("api.process_query") as span:
        span.set_attribute("endpoint", "/api/query")
        span.set_attribute("request.query", request.query[:100] + "..." if len(request.query) > 100 else request.query)
        span.set_attribute("request.session_id", request.session_id or "new")
        span.set_attribute("request.max_sources", request.max_sources or 5)
        
        if not enhanced_query_engine:
            span.set_attribute("api.error", "query_engine_not_initialized")
            raise HTTPException(status_code=503, detail="Enhanced query engine not initialized")
        
        # Create session if not exists
        session_id = request.session_id or str(uuid.uuid4())
        
        with tracer.start_as_current_span("session.manage") as session_span:
            if session_id not in sessions:
                sessions[session_id] = SessionInfo(
                    session_id=session_id,
                    message_count=0,
                    current_topic=None,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    last_activity=datetime.now(timezone.utc).isoformat(),
                    metadata={}
                )
                session_span.add_event("session_created")
                session_span.set_attribute("session.created", True)
            else:
                session_span.set_attribute("session.created", False)
            
            # Update session
            sessions[session_id].message_count += 1
            sessions[session_id].last_activity = datetime.now(timezone.utc).isoformat()
            
            session_span.set_attribute("session.message_count", sessions[session_id].message_count)
        
        # Process query with enhanced engine
        with time_query_processing(session_id=session_id, query_type="api_request"):
            response = await enhanced_query_engine.process_query(request)
        
        span.set_attribute("api.response.processing_time", response.processing_time)
        span.set_attribute("api.response.sources_count", len(response.sources))
        span.set_attribute("api.response.confidence", response.confidence)
        
        logger.info(f"API query processed: {response.processing_time:.2f}s, confidence: {response.confidence:.2f}")
        
        return response

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    with tracer.start_as_current_span("api.get_session") as span:
        span.set_attribute("endpoint", "/api/sessions/{session_id}")
        span.set_attribute("session.id", session_id)
        
        if session_id not in sessions:
            span.set_attribute("session.found", False)
            raise HTTPException(status_code=404, detail="Session not found")
        
        span.set_attribute("session.found", True)
        
        conversation_context = conversation_contexts.get(session_id)
        conversation_history = []
        
        if conversation_context and hasattr(conversation_context, 'get_messages'):
            with tracer.start_as_current_span("session.get_conversation_history"):
                messages = conversation_context.get_messages()
                conversation_history = [
                    {
                        "type": msg.message_type.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata
                    }
                    for msg in messages
                ]
                span.set_attribute("conversation.message_count", len(conversation_history))
        
        session_info = {
            "session": sessions[session_id],
            "conversation_history": conversation_history
        }
        
        span.set_attribute("session.response_size", len(json.dumps(session_info, default=str)))
        return session_info

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    with tracer.start_as_current_span("api.health_check") as span:
        span.set_attribute("endpoint", "/api/health")
        
        status = "healthy" if enhanced_query_engine and enhanced_query_engine.vector_store else "unhealthy"
        
        components = {
            "vector_store": enhanced_query_engine.vector_store is not None if enhanced_query_engine else False,
            "hybrid_search": enhanced_query_engine.hybrid_search is not None if enhanced_query_engine else False,
            "context_aware_engine": enhanced_query_engine.context_aware_engine is not None if enhanced_query_engine else False,
            "response_generator": enhanced_query_engine.response_generator is not None if enhanced_query_engine else False,
            "source_attributor": enhanced_query_engine.source_attributor is not None if enhanced_query_engine else False
        }
        
        health_data = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "components": components
        }
        
        span.set_attribute("health.status", status)
        span.set_attribute("health.components", json.dumps(components))
        
        # Record health check metrics
        record_cache_event("health_check", status == "healthy")
        
        return health_data

if __name__ == "__main__":
    import uvicorn
    
    with tracer.start_as_current_span("app.main_startup") as span:
        span.set_attribute("app.mode", "main")
        logger.info("Starting API server directly")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)