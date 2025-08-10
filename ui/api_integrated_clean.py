#!/usr/bin/env python3
"""
Integrated FastAPI Backend with Enhanced Retrieval and Complete OpenTelemetry Correlation
Maintains original UI while adding middleware-based W3C trace propagation
"""

import asyncio
import logging
import os
import json
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CRITICAL: Force service name early
os.environ["OTEL_SERVICE_NAME"] = "document-rag-api"

# Import OpenTelemetry configuration - UPDATED for middleware approach
from otel_config import (
    initialize_opentelemetry, get_service_tracer, instrument_fastapi_app,
    inject_trace_context, extract_trace_context, get_current_trace_id,
    TracedHTTPXClient
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenTelemetry with proper hierarchy
tracer, meter = initialize_opentelemetry(
    service_name="document-rag-api",
    service_version="1.0.0",
    environment=os.getenv("OTEL_ENVIRONMENT", "production")
)

# Metrics - preserved from original
query_counter = meter.create_counter("queries_total", description="Total number of queries processed")
query_duration = meter.create_histogram("query_duration_seconds", description="Query processing duration")
response_quality = meter.create_histogram("response_confidence", description="Response confidence scores")
api_errors = meter.create_counter("api_errors_total", description="Total number of API errors")
api_requests = meter.create_counter("api_requests_total", description="Total number of API requests")
external_api_calls = meter.create_counter("external_api_calls_total", description="Total number of external API calls")

# Import enhanced pipeline components - preserved from original
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

# Pydantic models - preserved from original
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

# FastAPI application - same as original
app = FastAPI(
    title="Enhanced Document RAG Query Interface",
    description="Advanced web-based interface with enhanced retrieval and complete service correlation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CRITICAL: Add middleware for trace continuity BEFORE other middleware
app = instrument_fastapi_app(app, "document-rag-api")

# Configure CORS - same as original
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - same as original
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
else:
    # Try ui/static if static doesn't exist
    ui_static_path = "ui/static"
    if os.path.exists(ui_static_path):
        app.mount("/static", StaticFiles(directory=ui_static_path), name="static")

# Global variables for the enhanced query engine components
enhanced_query_engine = None
conversation_contexts: Dict[str, Any] = {}
sessions: Dict[str, SessionInfo] = {}

class EnhancedQueryEngine:
    """Enhanced query engine with complete service correlation and trace propagation."""
    
    def __init__(self):
        self.tracer = get_service_tracer("query-processor")
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
        """Initialize enhanced components with tracing"""
        with self.tracer.start_as_current_span("api_initialize_components") as span:
            span.set_attributes({
                "service.name": "document-rag-api",
                "service.component": "query-processor",
                "operation.name": "initialize_components"
            })
            
            try:
                # Initialize basic vector store
                embeddings = OpenAIEmbeddings(model=self.embedding_model)
                
                with self.tracer.start_as_current_span("connect_qdrant") as qdrant_span:
                    qdrant_span.set_attributes({
                        "service.name": "document-rag-api",
                        "external.service.name": "qdrant",
                        "external.service.type": "vector_db",
                        "db.system": "qdrant"
                    })
                    
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
                span.record_exception(e)
                logger.error(f"❌ Error initializing enhanced query engine: {e}")
                self.vector_store = None
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query with complete service correlation"""
        with self.tracer.start_as_current_span("api_process_query") as span:
            # Set comprehensive correlation attributes
            span.set_attributes({
                "service.name": "document-rag-api",
                "service.component": "query-processor",
                "user.session_id": request.session_id or "anonymous",
                "query.length": len(request.query),
                "query.citation_format": request.citation_format,
                "query.hash": hashlib.md5(request.query.encode()).hexdigest()[:8],
                "w3c.trace_id": get_current_trace_id()
            })
            
            start_time = datetime.now()
            
            try:
                if not self.vector_store:
                    raise HTTPException(status_code=503, detail="Vector store not available")
                
                # Create correlation ID
                correlation_id = get_current_trace_id()[:16]
                logger.info(f"🔗 Processing query with correlation ID: {correlation_id}")
                
                # Get or create conversation context
                session_id = request.session_id or str(uuid.uuid4())
                conversation_context = conversation_contexts.get(session_id)
                
                if ConversationContext and not conversation_context:
                    conversation_context = ConversationContext(session_id=session_id)
                    conversation_contexts[session_id] = conversation_context
                
                # Call backend service for correlation
                with self.tracer.start_as_current_span("call_backend_service") as backend_span:
                    backend_span.set_attributes({
                        "service.name": "document-rag-api",
                        "peer.service": "document-rag-backend",
                        "internal.service.call": True,
                        "correlation.id": correlation_id
                    })
                    
                    try:
                        # Use TracedHTTPXClient for proper context propagation
                        async with TracedHTTPXClient(service_name="backend-proxy") as client:
                            response = await client.get(
                                "http://localhost:8001/status",
                                timeout=5.0
                            )
                            backend_span.set_attribute("backend.status", response.status_code)
                            logger.info(f"✅ Backend service status: {response.status_code}")
                    except Exception as e:
                        backend_span.record_exception(e)
                        logger.warning(f"⚠️ Could not reach backend service: {e}")
                
                # Use enhanced retrieval if available
                if self.context_aware_engine and conversation_context:
                    if hasattr(conversation_context, 'add_message'):
                        conversation_context.add_message(
                            message_type=MessageType.USER,
                            content=request.query,
                            metadata={
                                "timestamp": datetime.now().isoformat(),
                                "correlation_id": correlation_id
                            }
                        )
                    
                    result = await self._process_with_enhanced_engine(request, conversation_context, correlation_id)
                else:
                    result = await self._process_with_basic_engine(request, correlation_id)
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Update result with processing time and correlation
                result.processing_time = processing_time
                result.session_id = session_id
                result.timestamp = datetime.now(timezone.utc).isoformat()
                result.metadata.update({
                    "correlation_id": correlation_id,
                    "service_name": "document-rag-api",
                    "trace_id": get_current_trace_id()
                })
                
                # Add assistant response to context
                if conversation_context and hasattr(conversation_context, 'add_message'):
                    conversation_context.add_message(
                        message_type=MessageType.ASSISTANT,
                        content=result.response,
                        metadata={
                            "sources": result.sources,
                            "confidence": result.confidence,
                            "timestamp": datetime.now().isoformat(),
                            "correlation_id": correlation_id
                        }
                    )
                
                # Record metrics
                query_counter.add(1, {"session_type": "authenticated" if request.session_id else "anonymous"})
                query_duration.record(processing_time)
                response_quality.record(result.confidence)
                
                span.set_attributes({
                    "processing_time": processing_time,
                    "response_confidence": result.confidence,
                    "sources_count": len(result.sources),
                    "correlation.id": correlation_id
                })
                
                logger.info(f"✅ Query processed successfully - Correlation ID: {correlation_id}")
                return result
                
            except Exception as e:
                span.record_exception(e)
                api_errors.add(1, {"operation": "query_processing"})
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    async def _process_with_enhanced_engine(self, request: QueryRequest, context, correlation_id: str) -> QueryResponse:
        """Process query with enhanced components and correlation"""
        with self.tracer.start_as_current_span("enhanced_query_processing") as span:
            span.set_attributes({
                "service.component": "enhanced-retrieval",
                "correlation.id": correlation_id
            })
            
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
                        request.query, query_result.get('documents', []), correlation_id
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
                            "google_drive_url": doc.metadata.get("google_drive_url", None),
                            "correlation_id": correlation_id
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
                        "enhanced_processing": True,
                        "correlation_id": correlation_id
                    },
                    timestamp=""  # Will be updated by caller
                )
                
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Enhanced processing failed, falling back to basic: {e}")
                return await self._process_with_basic_engine(request, correlation_id)
    
    async def _process_with_basic_engine(self, request: QueryRequest, correlation_id: str) -> QueryResponse:
        """Fallback to basic query processing with correlation"""
        with self.tracer.start_as_current_span("basic_query_processing") as span:
            span.set_attributes({
                "service.component": "basic-retrieval",
                "correlation.id": correlation_id
            })
            
            # Perform similarity search with vector database
            with self.tracer.start_as_current_span("qdrant_similarity_search") as search_span:
                search_span.set_attributes({
                    "external.service.name": "qdrant",
                    "external.service.type": "vector_db",
                    "db.system": "qdrant",
                    "db.operation": "similarity_search",
                    "correlation.id": correlation_id
                })
                
                docs = self.vector_store.similarity_search(
                    request.query, k=request.max_sources or 5
                )
                search_span.set_attribute("documents_found", len(docs))
            
            # Track external API call
            external_api_calls.add(1, {"service": "qdrant", "operation": "similarity_search"})
            
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
                    "google_drive_url": doc.metadata.get("google_drive_url", None),
                    "correlation_id": correlation_id
                }
                sources.append(source)
            
            # Generate response using retrieved documents
            response_text = await self._generate_basic_response(request.query, docs, correlation_id)
            
            # Create citations
            citations = self._create_citations(docs, request.citation_format)
            
            return QueryResponse(
                query=request.query,
                response=response_text,
                sources=sources,
                confidence=0.8,
                query_type="basic_retrieval",
                complexity="medium",
                processing_time=0.0,  # Will be updated by caller
                context_used=request.include_context,
                citations=citations,
                session_id="",  # Will be updated by caller
                word_count=len(response_text.split()),
                metadata={
                    "sources_count": len(sources),
                    "enhanced_processing": False,
                    "correlation_id": correlation_id
                },
                timestamp=""  # Will be updated by caller
            )
    
    async def _generate_basic_response(self, query: str, docs: List[Document], correlation_id: str) -> str:
        """Generate response using OpenAI GPT with correlation"""
        with self.tracer.start_as_current_span("openai_completion_call") as span:
            span.set_attributes({
                "external.service.name": "openai",
                "external.service.type": "completion_api",
                "ai.model.name": "gpt-4o-mini",
                "ai.operation": "completion",
                "correlation.id": correlation_id
            })
            
            try:
                import openai
                client = openai.OpenAI()
                
                context = "\n\n".join([doc.page_content for doc in docs])
                prompt = f"""Based on the following context, please answer the user's question accurately and comprehensively.

Context: {context}

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
                
                # Track external API call
                external_api_calls.add(1, {"service": "openai", "operation": "completion"})
                
                span.set_attributes({
                    "response_length": len(response.choices[0].message.content),
                    "model_used": "gpt-4o-mini"
                })
                
                return response.choices[0].message.content
                
            except Exception as e:
                span.record_exception(e)
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
    """Initialize the enhanced query engine with correlation"""
    global enhanced_query_engine
    
    with tracer.start_as_current_span("api_service_startup") as span:
        span.set_attributes({
            "service.name": "document-rag-api",
            "operation.name": "startup"
        })
        
        logger.info("🚀 Starting Enhanced Document RAG API...")
        
        try:
            enhanced_query_engine = EnhancedQueryEngine()
            logger.info("✅ Enhanced query engine initialized successfully")
        except Exception as e:
            span.record_exception(e)
            logger.error(f"❌ Failed to initialize enhanced query engine: {e}")
            enhanced_query_engine = None

# PRESERVED ORIGINAL ENDPOINTS - with added trace context extraction

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page with correlation - SAME AS ORIGINAL"""
    with tracer.start_as_current_span("serve_root_page") as span:
        span.set_attributes({
            "service.name": "document-rag-api",
            "http.method": "GET",
            "http.route": "/",
            "w3c.trace_id": get_current_trace_id()
        })
        
        try:
            # Try to serve from ui/static first, then static
            possible_paths = ["ui/static/index.html", "static/index.html"]
            
            for path in possible_paths:
                try:
                    with open(path, "r") as f:
                        html_content = f.read()
                        # Inject trace information into HTML
                        trace_id = get_current_trace_id()
                        html_content = html_content.replace(
                            "</head>", 
                            f'<meta name="trace-id" content="{trace_id}"></head>'
                        )
                        return HTMLResponse(content=html_content)
                except FileNotFoundError:
                    continue
            
            # Fallback HTML with trace info
            trace_id = get_current_trace_id()
            fallback_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Document RAG Query Interface</title>
                <meta name="trace-id" content="{trace_id}">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .query-box {{ width: 100%; padding: 10px; margin: 20px 0; }}
                    .submit-btn {{ padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }}
                    .result {{ margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🔍 Document RAG Query Interface</h1>
                    <p><strong>Service:</strong> document-rag-api</p>
                    <p><strong>Trace ID:</strong> {trace_id}</p>
                    <p><strong>W3C Propagation:</strong> Enabled</p>
                    
                    <textarea id="queryInput" class="query-box" placeholder="Enter your query here..." rows="3"></textarea>
                    <br>
                    <button class="submit-btn" onclick="submitQuery()">Submit Query</button>
                    
                    <div id="result" class="result" style="display:none;">
                        <h3>Response:</h3>
                        <div id="responseText"></div>
                    </div>
                </div>
                
                <script>
                async function submitQuery() {{
                    const query = document.getElementById('queryInput').value;
                    if (!query.trim()) return;
                    
                    try {{
                        const response = await fetch('/api/query', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ query: query }})
                        }});
                        
                        const result = await response.json();
                        document.getElementById('responseText').innerHTML = result.response || result.answer || 'No response';
                        document.getElementById('result').style.display = 'block';
                    }} catch (error) {{
                        document.getElementById('responseText').innerHTML = 'Error: ' + error.message;
                        document.getElementById('result').style.display = 'block';
                    }}
                }}
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=fallback_html)
            
        except Exception as e:
            span.record_exception(e)
            return HTMLResponse(content=f"<h1>Error loading UI: {str(e)}</h1>", status_code=500)

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query with complete service correlation - ENHANCED FROM ORIGINAL"""
    with tracer.start_as_current_span("api_query_endpoint") as span:
        span.set_attributes({
            "service.name": "document-rag-api",
            "http.method": "POST",
            "http.route": "/api/query",
            "user.query": request.query[:100],  # First 100 chars only
            "w3c.trace_id": get_current_trace_id()
        })
        
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
        
        # Track API request
        api_requests.add(1, {"method": "POST", "endpoint": "/api/query"})
        
        # Process query with enhanced engine
        response = await enhanced_query_engine.process_query(request)
        
        span.set_attributes({
            "response.confidence": response.confidence,
            "response.sources_count": len(response.sources),
            "processing_time": response.processing_time
        })
        
        return response

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information with correlation - SAME AS ORIGINAL"""
    with tracer.start_as_current_span("get_session_info") as span:
        span.set_attributes({
            "service.name": "document-rag-api",
            "session_id": session_id
        })
        
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
        
        span.set_attribute("conversation_history_length", len(conversation_history))
        
        return {
            "session": sessions[session_id],
            "conversation_history": conversation_history,
            "trace_context": {
                "trace_id": get_current_trace_id(),
                "service": "document-rag-api"
            }
        }

@app.get("/api/health")
async def health_check():
    """Health check endpoint with service correlation - ENHANCED FROM ORIGINAL"""
    with tracer.start_as_current_span("health_check") as span:
        span.set_attributes({
            "service.name": "document-rag-api",
            "http.method": "GET",
            "http.route": "/api/health"
        })
        
        status = "healthy" if enhanced_query_engine and enhanced_query_engine.vector_store else "unhealthy"
        
        components = {
            "vector_store": enhanced_query_engine.vector_store is not None if enhanced_query_engine else False,
            "hybrid_search": enhanced_query_engine.hybrid_search is not None if enhanced_query_engine else False,
            "context_aware_engine": enhanced_query_engine.context_aware_engine is not None if enhanced_query_engine else False,
            "response_generator": enhanced_query_engine.response_generator is not None if enhanced_query_engine else False,
            "source_attributor": enhanced_query_engine.source_attributor is not None if enhanced_query_engine else False
        }
        
        span.set_attributes({
            "health.status": status,
            "components.healthy_count": sum(1 for v in components.values() if v)
        })
        
        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "service": "document-rag-api",
            "components": components,
            "trace_context": {
                "trace_id": get_current_trace_id(),
                "w3c_propagation": "enabled",
                "service_hierarchy": "document-rag-orchestrator -> document-rag-api"
            }
        }

# Backend proxy endpoints - with trace propagation
@app.get("/api/backend/status")
async def get_backend_status():
    """Proxy backend status with W3C propagation"""
    with tracer.start_as_current_span("proxy_backend_status") as span:
        span.set_attributes({
            "endpoint": "/api/backend/status",
            "proxy.target": "document-rag-backend"
        })
        
        try:
            async with TracedHTTPXClient(service_name="backend-proxy") as client:
                response = await client.get("http://localhost:8001/status", timeout=10.0)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise HTTPException(status_code=502, detail=f"Backend returned {response.status_code}")
                    
        except Exception as e:
            span.record_exception(e)
            raise HTTPException(status_code=502, detail=f"Backend service error: {str(e)}")

@app.post("/api/backend/scan")
async def trigger_backend_scan():
    """Trigger backend scan with W3C propagation"""
    with tracer.start_as_current_span("trigger_backend_scan") as span:
        try:
            async with TracedHTTPXClient(service_name="backend-proxy") as client:
                response = await client.post("http://localhost:8001/scan", timeout=30.0)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise HTTPException(status_code=502, detail=f"Scan failed: {response.status_code}")
                    
        except Exception as e:
            span.record_exception(e)
            raise HTTPException(status_code=502, detail=f"Scan failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🔥 ENHANCED API SERVICE: Complete W3C Trace Propagation")
    print("=" * 70)
    print("📊 Service Hierarchy: document-rag-orchestrator → document-rag-api")
    print("🧩 Components: query-processor, response-generator, session-manager, backend-proxy")
    print("🔗 W3C Propagation: ENABLED")
    print("🗺️ Middleware-Based Context: ACTIVE")
    print(f"🆔 Root Trace ID: {get_current_trace_id()}")
    print("🚀 Starting on http://0.0.0.0:8000")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
