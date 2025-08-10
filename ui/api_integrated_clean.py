#!/usr/bin/env python3
"""
ENHANCED API SERVICE: Complete Integration with W3C Trace Propagation
Creates hierarchical query processing with granular service components and full trace correlation
"""

import asyncio
import logging
import os
import json
import uuid
import httpx
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Force service name early
os.environ["OTEL_SERVICE_NAME"] = "document-rag-api"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enhanced OpenTelemetry configuration with W3C propagation
from otel_config import (
    initialize_opentelemetry, get_service_tracer, traced_function, 
    trace_http_call, trace_health_check, get_current_trace_id,
    add_trace_correlation_to_log, inject_trace_context, extract_trace_context,
    SERVICE_HIERARCHY
)
from metrics import (
    rag_metrics, time_query_processing, record_query_processed, 
    record_cache_event
)

# Basic imports
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.docstore.document import Document

# Initialize OpenTelemetry for API service
tracer, meter = initialize_opentelemetry(
    service_name="document-rag-api",
    service_version="2.0.0",
    environment="production"
)

# Setup enhanced logging with trace correlation
logger = add_trace_correlation_to_log(logging.getLogger(__name__))

# Pydantic models
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
    session_id: str
    message_count: int
    current_topic: Optional[str]
    created_at: str
    last_activity: Optional[str]
    metadata: Dict[str, Any]

class QueryProcessor:
    """Query processing service component with enhanced W3C tracing"""
    
    def __init__(self):
        self.tracer = get_service_tracer("query-processor")
        self.service_name = "query-processor"
    
    @traced_function(service_name="query-processor")
    async def process_user_query(self, request: QueryRequest, vector_store) -> Dict[str, Any]:
        """Process user query with detailed W3C trace correlation"""
        with self.tracer.start_as_current_span("query_processor.process_user_query") as span:
            span.set_attribute("query.text", request.query[:100])
            span.set_attribute("query.max_sources", request.max_sources)
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-api")
            span.set_attribute("service.hierarchy.level", 2)
            
            # Add W3C trace context attributes
            span.set_attribute("w3c.trace_id", get_current_trace_id())
            
            # Validate query
            with self.tracer.start_as_current_span("query_processor.validate_query") as validate_span:
                validate_span.set_attribute("service.component", self.service_name)
                
                if not request.query or len(request.query.strip()) < 3:
                    validate_span.set_attribute("validation.result", "too_short")
                    raise HTTPException(status_code=400, detail="Query too short")
                validate_span.set_attribute("validation.result", "valid")
            
            # Perform vector search with enhanced tracing
            with self.tracer.start_as_current_span("query_processor.vector_search") as search_span:
                search_span.set_attribute("service.component", self.service_name)
                search_span.set_attribute("service.external", "qdrant-database")
                search_span.set_attribute("search.query", request.query[:50])
                search_span.set_attribute("search.k", request.max_sources or 5)
                
                docs = vector_store.similarity_search(request.query, k=request.max_sources or 5)
                search_span.set_attribute("search.results_count", len(docs))
                search_span.set_attribute("search.status", "success")
            
            # Create sources with enhanced metadata
            with self.tracer.start_as_current_span("query_processor.create_sources") as sources_span:
                sources_span.set_attribute("service.component", self.service_name)
                
                sources = []
                for i, doc in enumerate(docs):
                    # Enhanced source processing with W3C context
                    with self.tracer.start_as_current_span(f"process_source_{i}") as source_span:
                        source_span.set_attribute("source.index", i)
                        source_span.set_attribute("source.content_length", len(doc.page_content))
                        
                        source = {
                            "id": f"doc_{i}",
                            "title": doc.metadata.get("source_file_name", doc.metadata.get("title", f"Document {i+1}")),
                            "source_file_name": doc.metadata.get("source_file_name", f"Document {i+1}"),
                            "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                            "confidence": doc.metadata.get("combined_score", 0.8),
                            "metadata": doc.metadata,
                            "page_number": doc.metadata.get("page_number"),
                            "slide_number": doc.metadata.get("slide_number"),
                            "sheet_name": doc.metadata.get("sheet_name"),
                            "section_title": doc.metadata.get("section_title"),
                            "heading": doc.metadata.get("heading"),
                            # Enhanced file link information with better extraction
                            "google_drive_url": doc.metadata.get("google_drive_url", doc.metadata.get("source", "")),
                            "source_url": doc.metadata.get("source_url", ""),
                            "url": doc.metadata.get("source", ""),
                            # Quality metrics
                            "quality_score": doc.metadata.get("quality_score", 0.8),
                            "extraction_confidence": doc.metadata.get("extraction_confidence", 0.8),
                            # Trace correlation
                            "trace_id": get_current_trace_id(),
                            "processed_by_service": "document-rag-api"
                        }
                        
                        source_span.set_attribute("source.title", source["title"])
                        source_span.set_attribute("source.confidence", source["confidence"])
                        
                        sources.append(source)
                
                sources_span.set_attribute("sources.created", len(sources))
                sources_span.set_attribute("sources.status", "success")
            
            return {"documents": docs, "sources": sources}

class ResponseGenerator:
    """Response generation service component with OpenAI integration"""
    
    def __init__(self):
        self.tracer = get_service_tracer("response-generator")
        self.service_name = "response-generator"
    
    @traced_function(service_name="response-generator")
    async def generate_response(self, query: str, documents: List[Document]) -> str:
        """Generate AI response using OpenAI with enhanced W3C tracing"""
        with self.tracer.start_as_current_span("response_generator.generate_response") as span:
            span.set_attribute("query.text", query[:100])
            span.set_attribute("documents.count", len(documents))
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-api")
            span.set_attribute("service.hierarchy.level", 2)
            
            try:
                import openai
                
                with self.tracer.start_as_current_span("openai_completions.chat_completion") as openai_span:
                    client = openai.OpenAI()
                    context = "\n\n".join([doc.page_content for doc in documents])
                    
                    openai_span.set_attribute("openai.model", "gpt-4o-mini")
                    openai_span.set_attribute("openai.context_length", len(context))
                    openai_span.set_attribute("service.external", "openai-api")
                    openai_span.set_attribute("service.component", self.service_name)
                    
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
                    openai_span.set_attribute("openai.status", "success")
                    
                    span.set_attribute("response.generation_status", "success")
                    span.set_attribute("response.length", len(response_text))
                    span.set_attribute("response.word_count", len(response_text.split()))
                    
                    return response_text
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("response.generation_status", "fallback")
                span.set_attribute("response.error", str(e))
                
                # Fallback response
                context = "\n\n".join([doc.page_content for doc in documents])
                fallback_response = f"Based on the retrieved documents, here's what I found relevant to your query: {context[:500]}..."
                
                span.set_attribute("response.fallback_length", len(fallback_response))
                return fallback_response

class SessionManager:
    """Session management service component with enhanced tracking"""
    
    def __init__(self):
        self.tracer = get_service_tracer("session-manager")
        self.service_name = "session-manager"
        self.sessions: Dict[str, SessionInfo] = {}
    
    @traced_function(service_name="session-manager")
    def get_or_create_session(self, session_id: Optional[str]) -> str:
        """Get existing session or create new one with W3C trace correlation"""
        with self.tracer.start_as_current_span("session_manager.get_or_create_session") as span:
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-api")
            
            if not session_id:
                session_id = str(uuid.uuid4())
                span.set_attribute("session.created", True)
            else:
                span.set_attribute("session.created", False)
            
            span.set_attribute("session.id", session_id)
            span.set_attribute("w3c.trace_id", get_current_trace_id())
            
            if session_id not in self.sessions:
                # Create new session with trace context
                with self.tracer.start_as_current_span("session_manager.create_session") as create_span:
                    self.sessions[session_id] = SessionInfo(
                        session_id=session_id,
                        message_count=0,
                        current_topic=None,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        last_activity=datetime.now(timezone.utc).isoformat(),
                        metadata={
                            "created_trace_id": get_current_trace_id(),
                            "service": self.service_name
                        }
                    )
                    create_span.set_attribute("session.status", "new")
                    create_span.set_attribute("session.trace_id", get_current_trace_id())
            else:
                span.set_attribute("session.status", "existing")
            
            # Update session activity
            self.sessions[session_id].message_count += 1
            self.sessions[session_id].last_activity = datetime.now(timezone.utc).isoformat()
            self.sessions[session_id].metadata["last_trace_id"] = get_current_trace_id()
            
            span.set_attribute("session.message_count", self.sessions[session_id].message_count)
            return session_id
    
    @traced_function(service_name="session-manager")
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information with trace correlation"""
        with self.tracer.start_as_current_span("session_manager.get_session_info") as span:
            span.set_attribute("session.id", session_id)
            span.set_attribute("service.component", self.service_name)
            
            if session_id in self.sessions:
                span.set_attribute("session.found", True)
                span.set_attribute("session.message_count", self.sessions[session_id].message_count)
                return self.sessions[session_id]
            else:
                span.set_attribute("session.found", False)
                return None

class BackendProxy:
    """Backend service proxy component with enhanced W3C propagation"""
    
    def __init__(self):
        self.tracer = get_service_tracer("backend-proxy")
        self.service_name = "backend-proxy"
        self.backend_url = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8001")
    
    @trace_http_call("GET", "backend_status", "backend-proxy")
    async def get_backend_status(self) -> Dict[str, Any]:
        """Get backend service status with comprehensive W3C trace propagation"""
        with self.tracer.start_as_current_span("backend_proxy.get_backend_status") as span:
            span.set_attribute("backend.url", self.backend_url)
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-api")
            span.set_attribute("http.target_service", "document-rag-backend")
            span.set_attribute("w3c.propagation", "enabled")
            
            try:
                async with httpx.AsyncClient() as client:
                    # Enhanced trace context injection with W3C headers
                    headers = inject_trace_context({})
                    headers.update({
                        "X-Trace-ID": get_current_trace_id(),
                        "X-Service-Chain": "document-rag-orchestrator -> document-rag-api -> backend-proxy -> document-rag-backend",
                        "X-Request-Source": "api-backend-proxy"
                    })
                    
                    response = await client.get(
                        f"{self.backend_url}/status", 
                        headers=headers, 
                        timeout=10.0
                    )
                    
                    span.set_attribute("http.status_code", response.status_code)
                    span.set_attribute("http.response_time", response.elapsed.total_seconds())
                    span.set_attribute("backend.response_size", len(response.content))
                    
                    if response.status_code == 200:
                        backend_data = response.json()
                        
                        # Extract enhanced backend metrics
                        processing_info = backend_data.get("processing", {})
                        service_info = backend_data.get("service", {})
                        
                        span.set_attribute("backend.files_processed", processing_info.get("files_processed_session", 0))
                        span.set_attribute("backend.is_running", service_info.get("is_running", False))
                        span.set_attribute("backend.uptime_seconds", service_info.get("uptime_seconds", 0))
                        span.set_attribute("backend.status", "healthy")
                        
                        return backend_data
                    else:
                        span.set_attribute("backend.status", "unhealthy")
                        span.set_attribute("backend.error", f"HTTP {response.status_code}")
                        return {
                            "status": "unhealthy", 
                            "status_code": response.status_code,
                            "error": f"Backend returned HTTP {response.status_code}"
                        }
                        
            except httpx.TimeoutException:
                span.set_attribute("backend.status", "timeout")
                span.set_attribute("backend.error", "request_timeout")
                return {"status": "timeout", "error": "Backend request timed out"}
            except httpx.ConnectError:
                span.set_attribute("backend.status", "unreachable")
                span.set_attribute("backend.error", "connection_failed")
                return {"status": "unreachable", "error": "Cannot connect to backend service"}
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("backend.status", "error")
                span.set_attribute("backend.error", str(e))
                return {"status": "error", "error": str(e)}
    
    @trace_http_call("POST", "backend_scan", "backend-proxy")
    async def trigger_backend_scan(self) -> Dict[str, Any]:
        """Trigger backend scan with enhanced W3C trace propagation"""
        with self.tracer.start_as_current_span("backend_proxy.trigger_scan") as span:
            span.set_attribute("backend.url", self.backend_url)
            span.set_attribute("action", "trigger_scan")
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("http.target_service", "document-rag-backend")
            
            try:
                async with httpx.AsyncClient() as client:
                    # Enhanced trace context with scan-specific metadata
                    headers = inject_trace_context({})
                    headers.update({
                        "X-Trace-ID": get_current_trace_id(),
                        "X-Service-Chain": "document-rag-orchestrator -> document-rag-api -> backend-proxy -> document-rag-backend",
                        "X-Request-Source": "api-scan-trigger",
                        "X-Action": "manual-scan"
                    })
                    
                    response = await client.post(
                        f"{self.backend_url}/scan", 
                        headers=headers, 
                        timeout=30.0
                    )
                    
                    span.set_attribute("http.status_code", response.status_code)
                    span.set_attribute("http.response_time", response.elapsed.total_seconds())
                    
                    if response.status_code == 200:
                        scan_result = response.json()
                        span.set_attribute("scan.triggered", True)
                        span.set_attribute("scan.status", "success")
                        
                        # Extract scan result metadata
                        if isinstance(scan_result, dict):
                            span.set_attribute("scan.message", scan_result.get("message", "Scan triggered"))
                            span.set_attribute("scan.trace_id", scan_result.get("trace_id", ""))
                        
                        return scan_result
                    else:
                        span.set_attribute("scan.triggered", False)
                        span.set_attribute("scan.status", "failed")
                        span.set_attribute("scan.error", f"HTTP {response.status_code}")
                        
                        return {
                            "status": "failed", 
                            "status_code": response.status_code,
                            "error": f"Backend scan failed with HTTP {response.status_code}"
                        }
                        
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("scan.triggered", False)
                span.set_attribute("scan.status", "error")
                span.set_attribute("scan.error", str(e))
                return {"status": "error", "error": str(e)}

class EnhancedQueryEngine:
    """Main query engine orchestrator with full W3C trace correlation"""
    
    def __init__(self):
        # Initialize main API service tracer with proper hierarchy
        self.tracer, self.meter = initialize_opentelemetry(
            service_name="document-rag-api",
            service_version="2.0.0",
            environment="production"
        )
        
        self.service_name = "document-rag-api"
        
        # Configuration with enhanced metadata
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("COLLECTION_NAME", "documents")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        
        # Initialize service components with proper hierarchy
        self.query_processor = QueryProcessor()
        self.response_generator = ResponseGenerator()
        self.session_manager = SessionManager()
        self.backend_proxy = BackendProxy()
        
        # Vector store
        self.vector_store = None
        
        # Setup enhanced logger with trace correlation
        self.logger = add_trace_correlation_to_log(logging.getLogger(__name__))
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("Enhanced API service initialized with full W3C trace propagation")
    
    @traced_function(service_name="document-rag-api")
    def _initialize_components(self):
        """Initialize vector store and components with enhanced tracing"""
        with self.tracer.start_as_current_span("api.initialize_components") as span:
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-orchestrator")
            span.set_attribute("initialization.type", "enhanced")
            
            try:
                # Initialize embeddings with tracing
                with self.tracer.start_as_current_span("openai_embeddings.init") as emb_span:
                    embeddings = OpenAIEmbeddings(model=self.embedding_model)
                    emb_span.set_attribute("embedding.model", self.embedding_model)
                    emb_span.set_attribute("service.external", "openai-api")
                    emb_span.set_attribute("embedding.dimension", 3072)  # text-embedding-3-large
                
                # Initialize vector store with enhanced connection details
                with self.tracer.start_as_current_span("qdrant_client.connect") as vs_span:
                    qdrant_url = f"http://{self.qdrant_host}:{self.qdrant_port}"
                    vs_span.set_attribute("qdrant.url", qdrant_url)
                    vs_span.set_attribute("qdrant.collection", self.collection_name)
                    vs_span.set_attribute("qdrant.host", self.qdrant_host)
                    vs_span.set_attribute("qdrant.port", self.qdrant_port)
                    vs_span.set_attribute("service.external", "qdrant-database")
                    
                    self.vector_store = QdrantVectorStore.from_existing_collection(
                        collection_name=self.collection_name,
                        embedding=embeddings,
                        url=qdrant_url,
                    )
                    vs_span.set_attribute("connection.status", "success")
                    self.logger.info(f"Connected to Qdrant at {qdrant_url}")
                
                span.set_attribute("init.status", "success")
                span.set_attribute("init.components_ready", True)
                span.set_attribute("init.vector_store", self.vector_store is not None)
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("init.status", "failed")
                span.set_attribute("init.error", str(e))
                self.logger.error(f"Component initialization failed: {e}")
                self.vector_store = None
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process query through enhanced service pipeline with full W3C correlation"""
        with self.tracer.start_as_current_span("api.process_query") as span:
            start_time = datetime.now()
            
            span.set_attribute("query.text", request.query[:100])
            span.set_attribute("query.max_sources", request.max_sources or 5)
            span.set_attribute("query.include_context", request.include_context)
            span.set_attribute("query.citation_format", request.citation_format)
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.hierarchy", "document-rag-orchestrator -> document-rag-api")
            span.set_attribute("w3c.trace_id", get_current_trace_id())
            
            try:
                if not self.vector_store:
                    span.set_attribute("api.error", "vector_store_unavailable")
                    span.set_attribute("api.status", "service_unavailable")
                    raise HTTPException(status_code=503, detail="Vector store not available")
                
                # Enhanced session management with trace correlation
                with self.tracer.start_as_current_span("api.session_management") as session_span:
                    session_id = self.session_manager.get_or_create_session(request.session_id)
                    session_span.set_attribute("session.id", session_id)
                    session_span.set_attribute("session.trace_id", get_current_trace_id())
                
                # Enhanced query processing with full context propagation
                with self.tracer.start_as_current_span("api.query_processing") as query_span:
                    query_result = await self.query_processor.process_user_query(request, self.vector_store)
                    query_span.set_attribute("query.documents_found", len(query_result["documents"]))
                    query_span.set_attribute("query.sources_created", len(query_result["sources"]))
                
                # Enhanced response generation with OpenAI integration
                with self.tracer.start_as_current_span("api.response_generation") as response_span:
                    response_text = await self.response_generator.generate_response(
                        request.query, 
                        query_result["documents"]
                    )
                    response_span.set_attribute("response.length", len(response_text))
                    response_span.set_attribute("response.word_count", len(response_text.split()))
                
                # Create enhanced citations with trace context
                with self.tracer.start_as_current_span("api.create_citations") as citation_span:
                    citations = self._create_citations(query_result["documents"], request.citation_format)
                    citation_span.set_attribute("citations.count", len(citations))
                    citation_span.set_attribute("citations.format", request.citation_format)
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Build enhanced response with comprehensive metadata
                result = QueryResponse(
                    query=request.query,
                    response=response_text,
                    sources=query_result["sources"],
                    confidence=0.8,  # Could be calculated from source confidences
                    query_type="enhanced_vector_search",
                    complexity="medium",
                    processing_time=processing_time,
                    context_used=request.include_context,
                    citations=citations,
                    session_id=session_id,
                    word_count=len(response_text.split()),
                    metadata={
                        "sources_count": len(query_result["sources"]),
                        "service_tree": "document-rag-api -> query-processor,response-generator,session-manager",
                        "trace_id": get_current_trace_id(),
                        "w3c_propagation": "enabled",
                        "service_hierarchy": "document-rag-orchestrator -> document-rag-api -> [components]",
                        "external_services": ["openai-api", "qdrant-database"],
                        "processing_pipeline": ["query_processing", "vector_search", "response_generation", "citation_creation"]
                    },
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                
                # Enhanced span attributes
                span.set_attribute("api.processing_time", processing_time)
                span.set_attribute("api.response.sources_count", len(result.sources))
                span.set_attribute("api.response.word_count", result.word_count)
                span.set_attribute("api.response.confidence", result.confidence)
                span.set_attribute("api.status", "success")
                
                # Record enhanced metrics
                record_query_processed(
                    query_type=result.query_type,
                    result_count=len(result.sources),
                    confidence=result.confidence
                )
                
                self.logger.info(f"Query processed: {processing_time:.2f}s, {len(result.sources)} sources, trace_id: {get_current_trace_id()}")
                
                return result
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("api.status", "error")
                span.set_attribute("api.error", str(e))
                span.set_attribute("api.error_type", type(e).__name__)
                self.logger.error(f"Query processing failed: {e}")
                raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    def _create_citations(self, docs: List[Document], format_type: str) -> List[str]:
        """Create enhanced citations with trace correlation"""
        with self.tracer.start_as_current_span("api.create_citations") as span:
            span.set_attribute("citations.format", format_type)
            span.set_attribute("citations.docs_count", len(docs))
            span.set_attribute("service.component", self.service_name)
            
            citations = []
            for i, doc in enumerate(docs):
                with self.tracer.start_as_current_span(f"create_citation_{i}") as citation_span:
                    title = doc.metadata.get("source_file_name", doc.metadata.get("title", f"Document {i+1}"))
                    source = doc.metadata.get("source", "Unknown source")
                    
                    citation = f"[{i+1}] {title} - {source}"
                    citations.append(citation)
                    
                    citation_span.set_attribute("citation.title", title)
                    citation_span.set_attribute("citation.source", source)
            
            span.set_attribute("citations.created_count", len(citations))
            return citations

# Global service instance
enhanced_query_engine = None
orchestrator_tracer = None

# Initialize FastAPI with enhanced hierarchy
app = FastAPI(
    title="Enhanced Document RAG API with Complete W3C Trace Correlation",
    description="Advanced hierarchical query processing with granular service components and full trace propagation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize with enhanced orchestrator context and W3C propagation"""
    global enhanced_query_engine, orchestrator_tracer
    
    # Get orchestrator tracer for parent context
    orchestrator_tracer = get_service_tracer("document-rag-orchestrator")
    
    with orchestrator_tracer.start_as_current_span("api_service.startup") as startup_span:
        startup_span.set_attribute("service.component", "api")
        startup_span.set_attribute("service.parent", "document-rag-orchestrator")
        startup_span.set_attribute("startup.phase", "initialization")
        startup_span.set_attribute("w3c.propagation", "enabled")
        startup_span.set_attribute("service.hierarchy", "document-rag-orchestrator -> document-rag-api")
        
        try:
            enhanced_query_engine = EnhancedQueryEngine()
            startup_span.set_attribute("api.initialization", "success")
            startup_span.set_attribute("api.components_ready", True)
            startup_span.set_attribute("api.vector_store_connected", enhanced_query_engine.vector_store is not None)
        except Exception as e:
            startup_span.record_exception(e)
            startup_span.set_attribute("api.initialization", "failed")
            startup_span.set_attribute("api.error", str(e))
            enhanced_query_engine = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main HTML with enhanced service hierarchy info"""
    with orchestrator_tracer.start_as_current_span("api.serve_root") as span:
        span.set_attribute("endpoint", "/")
        span.set_attribute("service.component", "document-rag-api")
        span.set_attribute("w3c.trace_id", get_current_trace_id())
        
        try:
            index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    content = f.read()
                    span.set_attribute("static.file_served", "index.html")
                    span.set_attribute("static.content_length", len(content))
                    return HTMLResponse(content=content)
            else:
                span.set_attribute("static.file_found", False)
                return HTMLResponse(content=f"""
                <html>
                <head><title>Enhanced Document RAG API</title></head>
                <body>
                    <h1>🔥 Enhanced Document RAG API</h1>
                    <p>Service tree active with full W3C trace propagation</p>
                    <p><strong>Trace ID:</strong> {get_current_trace_id()}</p>
                    <p><strong>Service Hierarchy:</strong> document-rag-orchestrator → document-rag-api</p>
                    <p>Static files not found - API endpoints are still available</p>
                </body>
                </html>
                """)
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("static.error", str(e))
            return HTMLResponse(content=f"<h1>Enhanced Document RAG API</h1><p>Error loading static files: {e}</p>")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process query through enhanced service pipeline with full W3C trace correlation"""
    with orchestrator_tracer.start_as_current_span("api.endpoint_process_query") as span:
        span.set_attribute("endpoint", "/api/query")
        span.set_attribute("request.query", request.query[:100] + "..." if len(request.query) > 100 else request.query)
        span.set_attribute("request.session_id", request.session_id or "new")
        span.set_attribute("request.max_sources", request.max_sources or 5)
        span.set_attribute("w3c.trace_id", get_current_trace_id())
        
        if not enhanced_query_engine:
            span.set_attribute("api.error", "query_engine_not_initialized")
            raise HTTPException(status_code=503, detail="Enhanced query engine not initialized")
        
        # Process query with enhanced timing
        with time_query_processing(session_id=request.session_id or "new", query_type="enhanced_api_request"):
            response = await enhanced_query_engine.process_query(request)
        
        span.set_attribute("api.response.processing_time", response.processing_time)
        span.set_attribute("api.response.sources_count", len(response.sources))
        span.set_attribute("api.response.confidence", response.confidence)
        span.set_attribute("api.response.trace_id", response.metadata.get("trace_id", ""))
        
        return response

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information with enhanced trace correlation"""
    with orchestrator_tracer.start_as_current_span("api.get_session") as span:
        span.set_attribute("endpoint", "/api/sessions/{session_id}")
        span.set_attribute("session.id", session_id)
        span.set_attribute("w3c.trace_id", get_current_trace_id())
        
        if not enhanced_query_engine:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        session_info = enhanced_query_engine.session_manager.get_session_info(session_id)
        
        if not session_info:
            span.set_attribute("session.found", False)
            raise HTTPException(status_code=404, detail="Session not found")
        
        span.set_attribute("session.found", True)
        span.set_attribute("session.message_count", session_info.message_count)
        
        return {
            "session": session_info, 
            "conversation_history": [],
            "trace_context": {
                "trace_id": get_current_trace_id(),
                "service": "document-rag-api",
                "hierarchy": "document-rag-orchestrator -> document-rag-api -> session-manager"
            }
        }

@app.get("/api/health")
async def health_check():
    """Comprehensive health check with full service tree status and W3C trace correlation"""
    with orchestrator_tracer.start_as_current_span("api.health_check") as span:
        span.set_attribute("endpoint", "/api/health")
        span.set_attribute("health.check_type", "comprehensive")
        span.set_attribute("w3c.trace_id", get_current_trace_id())
        
        # Check main API components
        api_healthy = enhanced_query_engine and enhanced_query_engine.vector_store is not None
        
        # Check backend service with enhanced error handling
        backend_status = "unknown"
        backend_details = {}
        if enhanced_query_engine:
            with span:
                backend_result = await enhanced_query_engine.backend_proxy.get_backend_status()
                backend_status = backend_result.get("status", "unknown")
                backend_details = backend_result
        
        # Enhanced component status with trace correlation
        components = {
            "vector_store": {
                "status": api_healthy,
                "service_path": "document-rag-api -> qdrant-database",
                "external": True
            },
            "query_processor": {
                "status": enhanced_query_engine.query_processor is not None if enhanced_query_engine else False,
                "service_path": "document-rag-api -> query-processor"
            },
            "response_generator": {
                "status": enhanced_query_engine.response_generator is not None if enhanced_query_engine else False,
                "service_path": "document-rag-api -> response-generator -> openai-api",
                "external_dependency": "openai-api"
            },
            "session_manager": {
                "status": enhanced_query_engine.session_manager is not None if enhanced_query_engine else False,
                "service_path": "document-rag-api -> session-manager",
                "active_sessions": len(enhanced_query_engine.session_manager.sessions) if enhanced_query_engine else 0
            },
            "backend_proxy": {
                "status": enhanced_query_engine.backend_proxy is not None if enhanced_query_engine else False,
                "service_path": "document-rag-api -> backend-proxy -> document-rag-backend",
                "target_service": "document-rag-backend"
            }
        }
        
        # Determine overall health
        component_health = [comp.get("status", False) for comp in components.values() if not comp.get("external")]
        backend_healthy = backend_status in ["healthy", "running"]
        overall_status = "healthy" if api_healthy and backend_healthy and all(component_health) else "degraded"
        
        health_data = {
            "service": "document-rag-api",
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "components": components,
            "backend_service": {
                "status": backend_status,
                "url": enhanced_query_engine.backend_proxy.backend_url if enhanced_query_engine else "unknown",
                "details": backend_details
            },
            "service_hierarchy": {
                "parent": "document-rag-orchestrator",
                "service": "document-rag-api",
                "children": list(components.keys()),
                "external_dependencies": ["openai-api", "qdrant-database", "document-rag-backend"]
            },
            "trace_context": {
                "trace_id": get_current_trace_id(),
                "w3c_propagation": "enabled",
                "service_chain": "document-rag-orchestrator -> document-rag-api",
                "propagation_formats": ["W3C TraceContext", "B3", "Jaeger"]
            }
        }
        
        # Enhanced span attributes
        span.set_attribute("health.overall_status", overall_status)
        span.set_attribute("health.api_healthy", api_healthy)
        span.set_attribute("health.backend_status", backend_status)
        span.set_attribute("health.components_total", len(components))
        span.set_attribute("health.components_healthy", sum(1 for c in components.values() if c.get("status")))
        
        # Record health metrics
        record_cache_event("health_check", overall_status == "healthy")
        record_cache_event("backend_health_check", backend_healthy)
        
        return health_data

@app.get("/api/backend/status")
async def get_backend_status():
    """Proxy backend status with enhanced W3C trace propagation"""
    with orchestrator_tracer.start_as_current_span("api.get_backend_status") as span:
        span.set_attribute("endpoint", "/api/backend/status")
        span.set_attribute("proxy.target", "document-rag-backend")
        span.set_attribute("w3c.propagation", "enabled")
        
        if not enhanced_query_engine:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        backend_result = await enhanced_query_engine.backend_proxy.get_backend_status()
        
        # Enhanced error handling with trace correlation
        if backend_result.get("status") == "error":
            span.set_attribute("proxy.result", "error")
            span.set_attribute("proxy.error", backend_result.get("error", "Unknown error"))
            raise HTTPException(status_code=502, detail=f"Backend service error: {backend_result.get('error')}")
        elif backend_result.get("status") == "timeout":
            span.set_attribute("proxy.result", "timeout")
            raise HTTPException(status_code=504, detail="Backend service timeout")
        elif backend_result.get("status") == "unreachable":
            span.set_attribute("proxy.result", "unreachable")
            raise HTTPException(status_code=502, detail="Backend service unreachable")
        
        span.set_attribute("proxy.result", "success")
        span.set_attribute("proxy.backend_status", backend_result.get("status", "unknown"))
        
        return backend_result

@app.post("/api/backend/scan")
async def trigger_backend_scan():
    """Trigger backend scan via proxy with enhanced W3C trace propagation"""
    with orchestrator_tracer.start_as_current_span("api.trigger_backend_scan") as span:
        span.set_attribute("endpoint", "/api/backend/scan")
        span.set_attribute("action", "trigger_scan")
        span.set_attribute("proxy.target", "document-rag-backend")
        span.set_attribute("w3c.propagation", "enabled")
        
        if not enhanced_query_engine:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        scan_result = await enhanced_query_engine.backend_proxy.trigger_backend_scan()
        
        # Enhanced scan result processing with trace correlation
        if scan_result.get("status") == "error":
            span.set_attribute("scan.result", "error")
            span.set_attribute("scan.error", scan_result.get("error", "Unknown error"))
            raise HTTPException(status_code=502, detail=f"Backend scan failed: {scan_result.get('error')}")
        elif scan_result.get("status") == "failed":
            span.set_attribute("scan.result", "failed")
            raise HTTPException(status_code=502, detail="Backend scan request failed")
        
        span.set_attribute("scan.result", "success")
        span.set_attribute("scan.triggered", True)
        
        return scan_result

if __name__ == "__main__":
    import uvicorn
    
    # Initialize with orchestrator context for standalone mode
    standalone_tracer = get_service_tracer("document-rag-orchestrator")
    
    with standalone_tracer.start_as_current_span("api_service.standalone_startup") as span:
        span.set_attribute("startup.mode", "standalone")
        span.set_attribute("startup.port", 8000)
        span.set_attribute("w3c.propagation", "enabled")
        
        print("🔥 ENHANCED API SERVICE: Complete W3C Trace Propagation")
        print("=" * 80)
        print("📊 Service Tree: document-rag-orchestrator → document-rag-api")
        print("🧩 Components: query-processor, response-generator, session-manager, backend-proxy")
        print("🌐 External Services: openai-api, qdrant-database, document-rag-backend")
        print("🔗 W3C Propagation: TraceContext + Baggage + B3 + Jaeger")
        print("🚀 Starting on http://0.0.0.0:8000")
        print()
        print("Enhanced API Endpoints:")
        print("  🏠 /                     - Main UI with trace info")
        print("  🔍 /api/query            - Process queries with W3C propagation")
        print("  💊 /api/health           - Comprehensive health check with trace context")
        print("  🗺️  /api/service-map      - Complete service map with W3C details")
        print("  🧩 /api/components       - Detailed component status with trace correlation")
        print("  ⚙️  /api/backend/status   - Backend proxy with enhanced W3C propagation")
        print("  🔄 /api/backend/scan     - Trigger backend scan with trace correlation")
        print()
        print("🆔 Root Trace ID:", get_current_trace_id())
        print("=" * 80)
        
        uvicorn.run(app, host="0.0.0.0", port=8000)