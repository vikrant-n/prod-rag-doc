#!/usr/bin/env python3
"""
BEAST MODE: Enhanced API Service with Massive Service Tree
Creates hierarchical query processing with granular service components
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

# Enhanced OpenTelemetry configuration
from otel_config import (
    initialize_opentelemetry, get_service_tracer, traced_function, 
    trace_http_call, trace_health_check, get_current_trace_id,
    add_trace_correlation_to_log
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

# In SimpleQueryEngine.__init__
from otel_config import initialize_opentelemetry

def __init__(self):
    self.tracer, self.meter = initialize_opentelemetry("document-rag-api")

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
    """Query processing service component"""
    
    def __init__(self):
        self.tracer = get_service_tracer("query-processor")
        self.service_name = "query-processor"
    
    @traced_function(service_name="query-processor")
    async def process_user_query(self, request: QueryRequest, vector_store) -> Dict[str, Any]:
        """Process user query with detailed tracing"""
        with self.tracer.start_as_current_span("query_processor.process_user_query") as span:
            span.set_attribute("query.text", request.query[:100])
            span.set_attribute("query.max_sources", request.max_sources)
            
            # Validate query
            with self.tracer.start_as_current_span("query_processor.validate_query") as validate_span:
                if not request.query or len(request.query.strip()) < 3:
                    validate_span.set_attribute("validation.result", "too_short")
                    raise HTTPException(status_code=400, detail="Query too short")
                validate_span.set_attribute("validation.result", "valid")
            
            # Perform vector search
            with self.tracer.start_as_current_span("query_processor.vector_search") as search_span:
                docs = vector_store.similarity_search(request.query, k=request.max_sources or 5)
                search_span.set_attribute("search.results_count", len(docs))
            
            # Create sources
            with self.tracer.start_as_current_span("query_processor.create_sources") as sources_span:
                sources = []
                for i, doc in enumerate(docs):
                    source = {
                        "id": f"doc_{i}",
                        "title": doc.metadata.get("title", f"Document {i+1}"),
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "confidence": 0.8,
                        "metadata": doc.metadata,
                        "page_number": doc.metadata.get("page"),
                        "source_url": doc.metadata.get("source"),
                        "google_drive_url": doc.metadata.get("google_drive_url")
                    }
                    sources.append(source)
                sources_span.set_attribute("sources.created", len(sources))
            
            return {"documents": docs, "sources": sources}

class ResponseGenerator:
    """Response generation service component"""
    
    def __init__(self):
        self.tracer = get_service_tracer("response-generator")
        self.service_name = "response-generator"
    
    @traced_function(service_name="response-generator")
    async def generate_response(self, query: str, documents: List[Document]) -> str:
        """Generate AI response using OpenAI"""
        with self.tracer.start_as_current_span("response_generator.generate_response") as span:
            span.set_attribute("query.text", query[:100])
            span.set_attribute("documents.count", len(documents))
            
            try:
                import openai
                
                with self.tracer.start_as_current_span("openai_completions.chat_completion") as openai_span:
                    client = openai.OpenAI()
                    context = "\n\n".join([doc.page_content for doc in documents])
                    
                    openai_span.set_attribute("openai.model", "gpt-4o-mini")
                    openai_span.set_attribute("openai.context_length", len(context))
                    openai_span.set_attribute("service.external", "openai-api")
                    
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
                context = "\n\n".join([doc.page_content for doc in documents])
                return f"Based on the retrieved documents, here's what I found relevant to your query: {context[:500]}..."

class SessionManager:
    """Session management service component"""
    
    def __init__(self):
        self.tracer = get_service_tracer("session-manager")
        self.service_name = "session-manager"
        self.sessions: Dict[str, SessionInfo] = {}
    
    @traced_function(service_name="session-manager")
    def get_or_create_session(self, session_id: Optional[str]) -> str:
        """Get existing session or create new one"""
        with self.tracer.start_as_current_span("session_manager.get_or_create_session") as span:
            if not session_id:
                session_id = str(uuid.uuid4())
                span.set_attribute("session.created", True)
            else:
                span.set_attribute("session.created", False)
            
            span.set_attribute("session.id", session_id)
            
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionInfo(
                    session_id=session_id,
                    message_count=0,
                    current_topic=None,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    last_activity=datetime.now(timezone.utc).isoformat(),
                    metadata={}
                )
                span.set_attribute("session.status", "new")
            else:
                span.set_attribute("session.status", "existing")
            
            # Update session
            self.sessions[session_id].message_count += 1
            self.sessions[session_id].last_activity = datetime.now(timezone.utc).isoformat()
            
            span.set_attribute("session.message_count", self.sessions[session_id].message_count)
            return session_id
    
    @traced_function(service_name="session-manager")
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information"""
        with self.tracer.start_as_current_span("session_manager.get_session_info") as span:
            span.set_attribute("session.id", session_id)
            
            if session_id in self.sessions:
                span.set_attribute("session.found", True)
                return self.sessions[session_id]
            else:
                span.set_attribute("session.found", False)
                return None

class BackendProxy:
    """Backend service proxy component"""
    
    def __init__(self):
        self.tracer = get_service_tracer("backend-proxy")
        self.service_name = "backend-proxy"
        self.backend_url = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8001")
    
    @trace_http_call("GET", "backend_status", "backend-proxy")
    async def get_backend_status(self) -> Dict[str, Any]:
        """Get backend service status with trace propagation"""
        with self.tracer.start_as_current_span("backend_proxy.get_backend_status") as span:
            span.set_attribute("backend.url", self.backend_url)
            span.set_attribute("http.target_service", "document-rag-backend")
            
            try:
                async with httpx.AsyncClient() as client:
                    # Add trace context to headers
                    headers = {"X-Trace-ID": get_current_trace_id()}
                    
                    response = await client.get(f"{self.backend_url}/status", headers=headers, timeout=10.0)
                    
                    span.set_attribute("http.status_code", response.status_code)
                    span.set_attribute("backend.response_size", len(response.content))
                    
                    if response.status_code == 200:
                        backend_data = response.json()
                        span.set_attribute("backend.files_processed", backend_data.get("processing", {}).get("files_processed_session", 0))
                        span.set_attribute("backend.is_running", backend_data.get("service", {}).get("is_running", False))
                        return backend_data
                    else:
                        span.set_attribute("backend.status", "unhealthy")
                        return {"status": "unhealthy", "status_code": response.status_code}
                        
            except httpx.TimeoutException:
                span.set_attribute("backend.status", "timeout")
                return {"status": "timeout"}
            except httpx.ConnectError:
                span.set_attribute("backend.status", "unreachable")
                return {"status": "unreachable"}
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("backend.status", "error")
                return {"status": "error", "error": str(e)}
    
    @trace_http_call("POST", "backend_scan", "backend-proxy")
    async def trigger_backend_scan(self) -> Dict[str, Any]:
        """Trigger backend scan with trace propagation"""
        with self.tracer.start_as_current_span("backend_proxy.trigger_scan") as span:
            span.set_attribute("backend.url", self.backend_url)
            span.set_attribute("action", "trigger_scan")
            
            try:
                async with httpx.AsyncClient() as client:
                    headers = {"X-Trace-ID": get_current_trace_id()}
                    
                    response = await client.post(f"{self.backend_url}/scan", headers=headers, timeout=30.0)
                    
                    span.set_attribute("http.status_code", response.status_code)
                    
                    if response.status_code == 200:
                        scan_result = response.json()
                        span.set_attribute("scan.triggered", True)
                        return scan_result
                    else:
                        span.set_attribute("scan.triggered", False)
                        return {"status": "failed", "status_code": response.status_code}
                        
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("scan.triggered", False)
                return {"status": "error", "error": str(e)}

class SimpleQueryEngine:
    """Main query engine orchestrator"""
    
    def __init__(self):
        # Initialize main API service tracer
        self.tracer, self.meter = initialize_opentelemetry(
            service_name="document-rag-api",
            service_version="2.0.0",
            environment="production"
        )
        
        self.service_name = "document-rag-api"
        
        # Configuration
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("COLLECTION_NAME", "documents")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        
        # Initialize service components
        self.query_processor = QueryProcessor()
        self.response_generator = ResponseGenerator()
        self.session_manager = SessionManager()
        self.backend_proxy = BackendProxy()
        
        # Vector store
        self.vector_store = None
        
        # Setup logger BEFORE initialization
        self.logger = add_trace_correlation_to_log(logging.getLogger(__name__))
        
        self._initialize_components()
        
        self.logger.info("Enhanced API service initialized with service tree")
    
    @traced_function(service_name="document-rag-api")
    def _initialize_components(self):
        """Initialize vector store and components"""
        with self.tracer.start_as_current_span("api.initialize_components") as span:
            try:
                # Initialize embeddings
                with self.tracer.start_as_current_span("openai_embeddings.init") as emb_span:
                    embeddings = OpenAIEmbeddings(model=self.embedding_model)
                    emb_span.set_attribute("embedding.model", self.embedding_model)
                    emb_span.set_attribute("service.external", "openai-api")
                
                # Initialize vector store
                with self.tracer.start_as_current_span("qdrant_client.connect") as vs_span:
                    qdrant_url = f"http://{self.qdrant_host}:{self.qdrant_port}"
                    vs_span.set_attribute("qdrant.url", qdrant_url)
                    vs_span.set_attribute("qdrant.collection", self.collection_name)
                    vs_span.set_attribute("service.external", "qdrant-database")
                    
                    self.vector_store = QdrantVectorStore.from_existing_collection(
                        collection_name=self.collection_name,
                        embedding=embeddings,
                        url=qdrant_url,
                    )
                    vs_span.set_attribute("connection.status", "success")
                    self.logger.info(f"Connected to Qdrant at {qdrant_url}")
                
                span.set_attribute("init.status", "success")
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("init.status", "failed")
                self.logger.error(f"Component initialization failed: {e}")
                self.vector_store = None
    
    @traced_function(service_name="document-rag-api")
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process query through service pipeline"""
        with self.tracer.start_as_current_span("api.process_query") as span:
            start_time = datetime.now()
            
            span.set_attribute("query.text", request.query[:100])
            span.set_attribute("query.max_sources", request.max_sources or 5)
            span.set_attribute("api.service_tree", "document-rag-orchestrator -> document-rag-api -> [components]")
            
            try:
                if not self.vector_store:
                    span.set_attribute("api.error", "vector_store_unavailable")
                    raise HTTPException(status_code=503, detail="Vector store not available")
                
                # Session management
                with self.tracer.start_as_current_span("api.session_management") as session_span:
                    session_id = self.session_manager.get_or_create_session(request.session_id)
                    session_span.set_attribute("session.id", session_id)
                
                # Query processing
                with self.tracer.start_as_current_span("api.query_processing") as query_span:
                    query_result = await self.query_processor.process_user_query(request, self.vector_store)
                    query_span.set_attribute("query.documents_found", len(query_result["documents"]))
                
                # Response generation
                with self.tracer.start_as_current_span("api.response_generation") as response_span:
                    response_text = await self.response_generator.generate_response(
                        request.query, 
                        query_result["documents"]
                    )
                    response_span.set_attribute("response.length", len(response_text))
                
                # Create citations
                with self.tracer.start_as_current_span("api.create_citations") as citation_span:
                    citations = self._create_citations(query_result["documents"], request.citation_format)
                    citation_span.set_attribute("citations.count", len(citations))
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Build response
                result = QueryResponse(
                    query=request.query,
                    response=response_text,
                    sources=query_result["sources"],
                    confidence=0.8,
                    query_type="vector_search",
                    complexity="medium",
                    processing_time=processing_time,
                    context_used=request.include_context,
                    citations=citations,
                    session_id=session_id,
                    word_count=len(response_text.split()),
                    metadata={
                        "sources_count": len(query_result["sources"]),
                        "service_tree": "document-rag-api -> query-processor,response-generator,session-manager",
                        "trace_id": get_current_trace_id()
                    },
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                
                span.set_attribute("api.processing_time", processing_time)
                span.set_attribute("api.response.sources_count", len(result.sources))
                span.set_attribute("api.response.word_count", result.word_count)
                
                # Record metrics
                record_query_processed(
                    query_type=result.query_type,
                    result_count=len(result.sources),
                    confidence=result.confidence
                )
                
                span.set_attribute("api.status", "success")
                self.logger.info(f"Query processed: {processing_time:.2f}s, {len(result.sources)} sources")
                
                return result
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("api.status", "error")
                self.logger.error(f"Query processing failed: {e}")
                raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    def _create_citations(self, docs: List[Document], format_type: str) -> List[str]:
        """Create citations for sources"""
        with self.tracer.start_as_current_span("api.create_citations") as span:
            span.set_attribute("citations.format", format_type)
            span.set_attribute("citations.docs_count", len(docs))
            
            citations = []
            for i, doc in enumerate(docs):
                title = doc.metadata.get("title", f"Document {i+1}")
                source = doc.metadata.get("source", "Unknown source")
                citations.append(f"[{i+1}] {title} - {source}")
            
            span.set_attribute("citations.created_count", len(citations))
            return citations

# Global service instance
enhanced_query_engine = None
orchestrator_tracer = None

# Initialize FastAPI with enhanced hierarchy
app = FastAPI(
    title="Enhanced Document RAG API with Service Tree",
    description="Hierarchical query processing with granular service components",
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
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize with orchestrator context"""
    global enhanced_query_engine, orchestrator_tracer
    
    # Get orchestrator tracer for parent context
    orchestrator_tracer = get_service_tracer("document-rag-orchestrator")
    
    with orchestrator_tracer.start_as_current_span("api_service.startup") as startup_span:
        startup_span.set_attribute("service.component", "api")
        startup_span.set_attribute("startup.phase", "initialization")
        
        try:
            enhanced_query_engine = SimpleQueryEngine()
            startup_span.set_attribute("api.initialization", "success")
        except Exception as e:
            startup_span.record_exception(e)
            startup_span.set_attribute("api.initialization", "failed")
            enhanced_query_engine = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main HTML with service hierarchy info"""
    try:
        with open("ui/static/index.html", "r") as f:
            content = f.read()
            return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Enhanced Document RAG API</h1><p>Service tree active but static files not found.</p>")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process query through enhanced service pipeline"""
    if not enhanced_query_engine:
        raise HTTPException(status_code=503, detail="Query engine not initialized")
    
    # Process with timing
    with time_query_processing(session_id=request.session_id or "new", query_type="api_request"):
        response = await enhanced_query_engine.process_query(request)
    
    return response

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if not enhanced_query_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    session_info = enhanced_query_engine.session_manager.get_session_info(session_id)
    
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"session": session_info, "conversation_history": []}

@app.get("/api/health")
async def health_check():
    """Comprehensive health check with service tree status"""
    # Check main components
    api_healthy = enhanced_query_engine and enhanced_query_engine.vector_store is not None
    
    # Check backend service
    backend_status = "unknown"
    if enhanced_query_engine:
        backend_result = await enhanced_query_engine.backend_proxy.get_backend_status()
        backend_status = backend_result.get("status", "unknown")
    
    components = {
        "vector_store": api_healthy,
        "query_processor": enhanced_query_engine.query_processor is not None if enhanced_query_engine else False,
        "response_generator": enhanced_query_engine.response_generator is not None if enhanced_query_engine else False,
        "session_manager": enhanced_query_engine.session_manager is not None if enhanced_query_engine else False,
        "backend_proxy": enhanced_query_engine.backend_proxy is not None if enhanced_query_engine else False,
        "backend_service": backend_status == "healthy" if backend_status != "unknown" else False
    }
    
    overall_status = "healthy" if api_healthy and backend_status == "healthy" else "unhealthy"
    
    health_data = {
        "service": "document-rag-api",
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "components": components,
        "backend_service": {
            "status": backend_status,
            "url": enhanced_query_engine.backend_proxy.backend_url if enhanced_query_engine else "unknown"
        }
    }
    
    record_cache_event("health_check", overall_status == "healthy")
    
    return health_data

@app.get("/api/backend/status")
@traced_function(service_name="document-rag-api")
async def get_backend_status():
    """Proxy backend status with trace propagation"""
    with orchestrator_tracer.start_as_current_span("api.get_backend_status") as span:
        span.set_attribute("endpoint", "/api/backend/status")
        span.set_attribute("proxy.target", "document-rag-backend")
        
        if not enhanced_query_engine:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        backend_result = await enhanced_query_engine.backend_proxy.get_backend_status()
        
        if backend_result.get("status") == "error":
            span.set_attribute("proxy.result", "error")
            raise HTTPException(status_code=502, detail="Backend service error")
        elif backend_result.get("status") == "timeout":
            span.set_attribute("proxy.result", "timeout")
            raise HTTPException(status_code=504, detail="Backend service timeout")
        elif backend_result.get("status") == "unreachable":
            span.set_attribute("proxy.result", "unreachable")
            raise HTTPException(status_code=502, detail="Backend service unreachable")
        
        span.set_attribute("proxy.result", "success")
        return backend_result

@app.post("/api/backend/scan")
@traced_function(service_name="document-rag-api")
async def trigger_backend_scan():
    """Trigger backend scan via proxy"""
    with orchestrator_tracer.start_as_current_span("api.trigger_backend_scan") as span:
        span.set_attribute("endpoint", "/api/backend/scan")
        span.set_attribute("action", "trigger_scan")
        
        if not enhanced_query_engine:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        scan_result = await enhanced_query_engine.backend_proxy.trigger_backend_scan()
        
        if scan_result.get("status") == "error":
            span.set_attribute("scan.result", "error")
            raise HTTPException(status_code=502, detail=f"Backend scan failed: {scan_result.get('error')}")
        elif scan_result.get("status") == "failed":
            span.set_attribute("scan.result", "failed")
            raise HTTPException(status_code=502, detail="Backend scan failed")
        
        span.set_attribute("scan.result", "success")
        return scan_result

@app.get("/api/components")
@traced_function(service_name="document-rag-api")
async def get_api_components():
    """Get detailed status of all API service components"""
    with orchestrator_tracer.start_as_current_span("api.get_components") as span:
        span.set_attribute("endpoint", "/api/components")
        
        if not enhanced_query_engine:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        components = {}
        
        # Query Processor Component
        with enhanced_query_engine.tracer.start_as_current_span("check.query_processor") as qp_span:
            components["query-processor"] = {
                "status": "healthy",
                "description": "Processes user queries and performs vector searches",
                "service_path": "document-rag-api -> query-processor",
                "capabilities": ["query_validation", "vector_search", "source_creation"]
            }
            qp_span.set_attribute("component.status", "healthy")
        
        # Response Generator Component
        with enhanced_query_engine.tracer.start_as_current_span("check.response_generator") as rg_span:
            try:
                import openai
                components["response-generator"] = {
                    "status": "healthy",
                    "description": "Generates AI responses using OpenAI GPT models",
                    "service_path": "document-rag-api -> response-generator -> openai-api",
                    "model": "gpt-4o-mini",
                    "external_dependency": "openai-api"
                }
                rg_span.set_attribute("component.status", "healthy")
            except ImportError:
                components["response-generator"] = {
                    "status": "degraded",
                    "description": "OpenAI library not available",
                    "service_path": "document-rag-api -> response-generator"
                }
                rg_span.set_attribute("component.status", "degraded")
        
        # Session Manager Component
        with enhanced_query_engine.tracer.start_as_current_span("check.session_manager") as sm_span:
            session_count = len(enhanced_query_engine.session_manager.sessions)
            components["session-manager"] = {
                "status": "healthy",
                "description": "Manages user sessions and conversation context",
                "service_path": "document-rag-api -> session-manager",
                "active_sessions": session_count,
                "capabilities": ["session_creation", "session_tracking", "context_management"]
            }
            sm_span.set_attribute("component.status", "healthy")
            sm_span.set_attribute("sessions.active", session_count)
        
        # Backend Proxy Component
        with enhanced_query_engine.tracer.start_as_current_span("check.backend_proxy") as bp_span:
            backend_status = await enhanced_query_engine.backend_proxy.get_backend_status()
            proxy_status = "healthy" if backend_status.get("status") not in ["error", "timeout", "unreachable"] else "unhealthy"
            
            components["backend-proxy"] = {
                "status": proxy_status,
                "description": "Proxies requests to backend document processing service",
                "service_path": "document-rag-api -> backend-proxy -> document-rag-backend",
                "backend_url": enhanced_query_engine.backend_proxy.backend_url,
                "backend_status": backend_status.get("status", "unknown"),
                "capabilities": ["status_proxy", "scan_trigger", "health_monitoring"]
            }
            bp_span.set_attribute("component.status", proxy_status)
            bp_span.set_attribute("backend.status", backend_status.get("status", "unknown"))
        
        # Vector Store Connection
        with enhanced_query_engine.tracer.start_as_current_span("check.vector_store") as vs_span:
            try:
                if enhanced_query_engine.vector_store:
                    # Test connection
                    test_result = enhanced_query_engine.vector_store.similarity_search("test", k=1)
                    components["vector-store-connection"] = {
                        "status": "healthy",
                        "description": "Connection to Qdrant vector database",
                        "service_path": "document-rag-api -> qdrant-database",
                        "qdrant_url": f"http://{enhanced_query_engine.qdrant_host}:{enhanced_query_engine.qdrant_port}",
                        "collection": enhanced_query_engine.collection_name,
                        "embedding_model": enhanced_query_engine.embedding_model,
                        "external_dependency": "qdrant-database"
                    }
                    vs_span.set_attribute("component.status", "healthy")
                else:
                    components["vector-store-connection"] = {
                        "status": "unhealthy",
                        "description": "Vector store not initialized",
                        "service_path": "document-rag-api -> qdrant-database"
                    }
                    vs_span.set_attribute("component.status", "unhealthy")
            except Exception as e:
                components["vector-store-connection"] = {
                    "status": "unhealthy",
                    "description": f"Vector store error: {str(e)}",
                    "service_path": "document-rag-api -> qdrant-database"
                }
                vs_span.set_attribute("component.status", "unhealthy")
        
        span.set_attribute("components.total", len(components))
        span.set_attribute("components.healthy", len([c for c in components.values() if c.get("status") == "healthy"]))
        
        return {
            "service": "document-rag-api",
            "components": components,
            "service_tree": {
                "hierarchy": "document-rag-orchestrator -> document-rag-api -> [components]",
                "parent": "document-rag-orchestrator",
                "children": list(components.keys()),
                "external_dependencies": ["openai-api", "qdrant-database", "document-rag-backend"]
            },
            "summary": {
                "total_components": len(components),
                "healthy_components": len([c for c in components.values() if c.get("status") == "healthy"]),
                "degraded_components": len([c for c in components.values() if c.get("status") == "degraded"]),
                "unhealthy_components": len([c for c in components.values() if c.get("status") == "unhealthy"])
            },
            "trace_context": {
                "trace_id": get_current_trace_id(),
                "service_hierarchy": "document-rag-orchestrator -> document-rag-api"
            }
        }

@app.get("/api/service-map")
@traced_function(service_name="document-rag-api")
async def get_service_map():
    """Get complete service map visualization"""
    with orchestrator_tracer.start_as_current_span("api.get_service_map") as span:
        span.set_attribute("endpoint", "/api/service-map")
        
        # Build comprehensive service map
        service_map = {
            "root": "document-rag-orchestrator",
            "tree": {
                "document-rag-orchestrator": {
                    "type": "orchestrator",
                    "status": "healthy",
                    "children": ["document-rag-api", "document-rag-backend", "process-manager"]
                },
                "document-rag-api": {
                    "type": "api_service",
                    "status": "healthy" if enhanced_query_engine else "unhealthy",
                    "port": 8000,
                    "children": ["query-processor", "response-generator", "session-manager", "backend-proxy"],
                    "external_dependencies": ["openai-api", "qdrant-database"]
                },
                "query-processor": {
                    "type": "component",
                    "status": "healthy",
                    "parent": "document-rag-api",
                    "capabilities": ["query_validation", "vector_search", "source_creation"]
                },
                "response-generator": {
                    "type": "component", 
                    "status": "healthy",
                    "parent": "document-rag-api",
                    "external_dependencies": ["openai-api"]
                },
                "session-manager": {
                    "type": "component",
                    "status": "healthy", 
                    "parent": "document-rag-api",
                    "active_sessions": len(enhanced_query_engine.session_manager.sessions) if enhanced_query_engine else 0
                },
                "backend-proxy": {
                    "type": "component",
                    "status": "healthy",
                    "parent": "document-rag-api",
                    "target": "document-rag-backend"
                }
            },
            "external_services": {
                "openai-api": {
                    "type": "external",
                    "description": "OpenAI GPT API for completions and embeddings",
                    "endpoints": ["chat/completions", "embeddings"]
                },
                "qdrant-database": {
                    "type": "external", 
                    "description": "Vector database for document embeddings",
                    "url": f"http://{enhanced_query_engine.qdrant_host}:{enhanced_query_engine.qdrant_port}" if enhanced_query_engine else "unknown"
                },
                "document-rag-backend": {
                    "type": "internal_service",
                    "description": "Document processing and monitoring service",
                    "port": 8001
                }
            },
            "flow_patterns": {
                "query_processing": [
                    "document-rag-api",
                    "query-processor", 
                    "qdrant-database",
                    "response-generator",
                    "openai-api"
                ],
                "document_processing": [
                    "document-rag-backend",
                    "google-drive-monitor",
                    "document-processor",
                    "embedding-generator", 
                    "vector-store-manager",
                    "qdrant-database"
                ],
                "health_monitoring": [
                    "document-rag-orchestrator",
                    "document-rag-api",
                    "backend-proxy",
                    "document-rag-backend"
                ]
            },
            "metrics": {
                "total_services": 2,
                "total_components": 4, 
                "external_dependencies": 3,
                "trace_id": get_current_trace_id()
            }
        }
        
        span.set_attribute("service_map.total_services", service_map["metrics"]["total_services"])
        span.set_attribute("service_map.total_components", service_map["metrics"]["total_components"])
        
        return service_map

@app.get("/api/traces/active")
@traced_function(service_name="document-rag-api")
async def get_active_traces():
    """Get information about active traces"""
    with orchestrator_tracer.start_as_current_span("api.get_active_traces") as span:
        span.set_attribute("endpoint", "/api/traces/active")
        
        trace_info = {
            "current_trace": {
                "trace_id": get_current_trace_id(),
                "service": "document-rag-api",
                "operation": "get_active_traces",
                "hierarchy": "document-rag-orchestrator -> document-rag-api"
            },
            "service_tracers": {
                "document-rag-api": "Primary API service tracer",
                "query-processor": "Query processing component tracer", 
                "response-generator": "Response generation component tracer",
                "session-manager": "Session management component tracer",
                "backend-proxy": "Backend proxy component tracer"
            },
            "trace_propagation": {
                "enabled": True,
                "headers": ["X-Trace-ID", "traceparent", "b3"],
                "backends": ["document-rag-backend"]
            },
            "instrumentation": {
                "automatic": ["httpx", "requests", "fastapi", "logging"],
                "manual": ["query_processing", "response_generation", "session_management"]
            }
        }
        
        span.set_attribute("traces.current_id", trace_info["current_trace"]["trace_id"])
        span.set_attribute("traces.service_count", len(trace_info["service_tracers"]))
        
        return trace_info

# Performance monitoring endpoint
@app.get("/api/performance")
@traced_function(service_name="document-rag-api")
async def get_performance_metrics():
    """Get performance metrics for the API service"""
    with orchestrator_tracer.start_as_current_span("api.get_performance_metrics") as span:
        span.set_attribute("endpoint", "/api/performance")
        
        if not enhanced_query_engine:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Simulate performance data collection
        metrics = {
            "service": "document-rag-api",
            "uptime_seconds": (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).total_seconds(),
            "sessions": {
                "total": len(enhanced_query_engine.session_manager.sessions),
                "active_last_hour": len([s for s in enhanced_query_engine.session_manager.sessions.values() 
                                       if datetime.fromisoformat(s.last_activity.replace('Z', '+00:00')) > 
                                       datetime.now(timezone.utc) - timedelta(hours=1)])
            },
            "components": {
                "query_processor": {"status": "healthy", "avg_response_time_ms": 150},
                "response_generator": {"status": "healthy", "avg_response_time_ms": 2500},
                "session_manager": {"status": "healthy", "avg_response_time_ms": 5},
                "backend_proxy": {"status": "healthy", "avg_response_time_ms": 300}
            },
            "external_services": {
                "openai_api": {"status": "healthy", "avg_response_time_ms": 2000},
                "qdrant_database": {"status": "healthy", "avg_response_time_ms": 100}
            },
            "trace_context": {
                "trace_id": get_current_trace_id(),
                "spans_created": "dynamic",
                "service_tree": "document-rag-orchestrator -> document-rag-api -> [components]"
            }
        }
        
        span.set_attribute("performance.sessions_total", metrics["sessions"]["total"])
        span.set_attribute("performance.sessions_active", metrics["sessions"]["active_last_hour"])
        
        return metrics

if __name__ == "__main__":
    import uvicorn
    
    # Initialize with orchestrator context for standalone mode
    standalone_tracer = get_service_tracer("document-rag-orchestrator")
    
    with standalone_tracer.start_as_current_span("api_service.standalone_startup") as span:
        span.set_attribute("startup.mode", "standalone")
        span.set_attribute("startup.port", 8000)
        
        print("🔥 BEAST MODE: Enhanced API Service Starting")
        print("📊 Service Tree: document-rag-orchestrator -> document-rag-api")
        print("🧩 Components: query-processor, response-generator, session-manager, backend-proxy")
        print("🌐 External Services: openai-api, qdrant-database, document-rag-backend")
        print("🚀 Starting on http://0.0.0.0:8000")
        print()
        print("API Endpoints:")
        print("  🏠 /                     - Main UI")
        print("  🔍 /api/query            - Process queries")
        print("  💊 /api/health           - Health check")
        print("  🗺️  /api/service-map      - Service map visualization")
        print("  🧩 /api/components       - Component status")
        print("  📊 /api/performance      - Performance metrics")
        print("  🔗 /api/traces/active    - Active traces info")
        print()
        
        uvicorn.run(app, host="0.0.0.0", port=8000)