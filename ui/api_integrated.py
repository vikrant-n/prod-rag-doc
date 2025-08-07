#!/usr/bin/env python3
"""
Integrated FastAPI Backend with Enhanced Retrieval and OpenTelemetry Instrumentation

This module integrates the enhanced pipeline components for superior
document retrieval and query processing with comprehensive EDOT observability.
Enhanced with distributed tracing, metrics, and structured logging correlation.

Demonstrates complete observability WITHOUT LOGSTASH DEPENDENCY - showing how OpenTelemetry 
can handle multiple log types and correlate them with trace IDs and span IDs for ELK visualization.
"""

import asyncio
import logging
import os
import json
import uuid
import time
import socket
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# OpenTelemetry imports - Official EDOT libraries
from opentelemetry import trace, metrics, baggage
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, SERVICE_INSTANCE_ID
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenTelemetry with EDOT configuration - CORRECTED FOR FRESH TRACES
def init_telemetry():
    """Initialize OpenTelemetry with EDOT configuration - CREATES FRESH TRACES per request"""
    
    # Check if we're already using auto-instrumentation
    current_tracer_provider = trace.get_tracer_provider()
    if hasattr(current_tracer_provider, '__class__') and current_tracer_provider.__class__.__name__ != 'ProxyTracerProvider':
        print("✅ Using existing OpenTelemetry auto-instrumentation")
        return trace.get_tracer(__name__), metrics.get_meter(__name__)
    
    # 🎯 REMOVED: Context extraction from orchestrator environment variables
    # This allows each HTTP request to create its own NEW trace ID
    
    # Generate unique service instance ID
    service_instance_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    
    # Resource configuration - Following EDOT best practices
    resource = Resource.create({
        SERVICE_NAME: "document-rag-query-api",
        SERVICE_VERSION: "2.0.0",
        SERVICE_INSTANCE_ID: service_instance_id,
        "service.namespace": "document-rag-system",
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "production"),
        "host.name": socket.gethostname(),
        "process.pid": str(os.getpid()),
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": "1.25.0"
    })
    
    # OTLP endpoint configuration
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    try:
        # Trace provider setup
        trace_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(trace_provider)
        
        # OTLP span exporter
        otlp_span_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,
            headers={}
        )
        
        # Add span processors
        trace_provider.add_span_processor(BatchSpanProcessor(otlp_span_exporter))
        
        # Console exporter for debugging
        if os.getenv("OTEL_DEBUG", "false").lower() == "true":
            trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        
        # Metrics provider setup
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=otlp_endpoint,
            insecure=True,
            headers={}
        )
        
        metric_reader = PeriodicExportingMetricReader(
            exporter=otlp_metric_exporter,
            export_interval_millis=15000,  # 15 seconds
        )
        
        metric_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(metric_provider)
        
        print("✅ OpenTelemetry initialized for query API with fresh trace generation")
        
    except Exception as e:
        print(f"⚠️  OpenTelemetry provider setup error: {e}")
    
    # Set up B3 propagator for distributed tracing
    set_global_textmap(B3MultiFormat())
    
    # Manual instrumentation only if not using auto-instrumentation
    if not os.getenv("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"):
        try:
            LoggingInstrumentor().instrument(set_logging_format=True)
            RequestsInstrumentor().instrument()
            AsyncioInstrumentor().instrument()
        except Exception as e:
            print(f"⚠️  Instrumentation warning: {e}")
    
    return trace.get_tracer(__name__), metrics.get_meter(__name__)

# Initialize telemetry
tracer, meter = init_telemetry()

# Custom metrics for RAG query processing observability
query_processing_counter = meter.create_counter(
    "queries_processed_total",
    description="Total number of queries processed",
    unit="1"
)

query_processing_duration = meter.create_histogram(
    "query_processing_duration_seconds",
    description="Time taken to process queries",
    unit="s"
)

vector_search_counter = meter.create_counter(
    "vector_search_operations_total",
    description="Total number of vector search operations",
    unit="1"
)

llm_generation_counter = meter.create_counter(
    "llm_generation_requests_total",
    description="Total number of LLM generation requests",
    unit="1"
)

session_management_gauge = meter.create_up_down_counter(
    "active_sessions_current",
    description="Current number of active sessions",
    unit="1"
)

api_health_gauge = meter.create_up_down_counter(
    "api_health_status",
    description="API health status (1=healthy, 0=unhealthy)",
    unit="1"
)

# Configure structured logging with OpenTelemetry correlation
class OtelQueryAPIFormatter(logging.Formatter):
    """Custom formatter to include OpenTelemetry trace and span IDs"""
    
    def format(self, record):
        # Get current span context for correlation
        span = trace.get_current_span()
        if span.get_span_context().is_valid:
            trace_id = format(span.get_span_context().trace_id, '032x')
            span_id = format(span.get_span_context().span_id, '016x')
            record.trace_id = trace_id
            record.span_id = span_id
        else:
            record.trace_id = '00000000000000000000000000000000'
            record.span_id = '0000000000000000'
        
        return super().format(record)

# Set up structured logging
formatter = OtelQueryAPIFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler('query_api.log')
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
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

# 🎯 CRITICAL: Middleware to ensure each request gets proper trace correlation
@app.middleware("http") 
async def trace_requests_middleware(request: Request, call_next):
    """Middleware to ensure each request gets proper trace correlation and NEW trace ID"""
    with tracer.start_as_current_span(
        f"http_request_{request.method}_{request.url.path}",
        kind=trace.SpanKind.SERVER
    ) as span:
        # Log the new trace ID for each request
        current_span = trace.get_current_span()
        if current_span.get_span_context().is_valid:
            trace_id = format(current_span.get_span_context().trace_id, '032x')
            logger.info(f"🔍 New user query trace started: {trace_id}")
            span.set_attribute("http.request.trace_id", trace_id)
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            
        response = await call_next(request)
        
        span.set_attribute("http.status_code", response.status_code)
        span.add_event("HTTP request completed")
        
        return response

class EnhancedQueryEngine:
    """Enhanced query engine using advanced pipeline components with comprehensive instrumentation."""

    def __init__(self):
        with tracer.start_as_current_span("enhanced_query_engine_init") as span:
            self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            self.collection_name = os.getenv("COLLECTION_NAME", "documents")
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

            span.set_attribute("qdrant.host", self.qdrant_host)
            span.set_attribute("qdrant.port", self.qdrant_port)
            span.set_attribute("collection.name", self.collection_name)
            span.set_attribute("embedding.model", self.embedding_model)

            # Initialize components
            self.vector_store = None
            self.hybrid_search = None
            self.context_aware_engine = None
            self.response_generator = None
            self.source_attributor = None

            self._initialize_components()
            span.add_event("Query engine initialized")

    def _initialize_components(self):
        """Initialize enhanced components with OpenTelemetry instrumentation"""
        with tracer.start_as_current_span("initialize_components") as span:
            try:
                # Initialize basic vector store
                with tracer.start_as_current_span("initialize_vector_store") as vs_span:
                    embeddings = OpenAIEmbeddings(model=self.embedding_model)
                    vs_span.set_attribute("embedding.model", self.embedding_model)

                    self.vector_store = QdrantVectorStore.from_existing_collection(
                        collection_name=self.collection_name,
                        embedding=embeddings,
                        url=f"http://{self.qdrant_host}:{self.qdrant_port}",
                    )
                    
                    vs_span.set_attribute("vector_store.initialized", True)
                    vs_span.add_event("Vector store connected to Qdrant")

                logger.info(f"✅ Successfully connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")

                # Initialize enhanced components if available
                component_init_results = []
                
                if BM25IndexManager:
                    with tracer.start_as_current_span("initialize_bm25") as bm25_span:
                        self.hybrid_search = BM25IndexManager()
                        bm25_span.set_attribute("component.name", "BM25IndexManager")
                        bm25_span.add_event("BM25 hybrid search initialized")
                        component_init_results.append("BM25")
                        logger.info("✅ Initialized BM25 hybrid search")

                if ContextAwareQueryEngine:
                    with tracer.start_as_current_span("initialize_context_aware_engine") as cae_span:
                        self.context_aware_engine = ContextAwareQueryEngine(
                            vector_store=self.vector_store,
                            embeddings=embeddings
                        )
                        cae_span.set_attribute("component.name", "ContextAwareQueryEngine")
                        cae_span.add_event("Context-aware query engine initialized")
                        component_init_results.append("ContextAware")
                        logger.info("✅ Initialized context-aware query engine")

                if ResponseGenerator:
                    with tracer.start_as_current_span("initialize_response_generator") as rg_span:
                        self.response_generator = ResponseGenerator()
                        rg_span.set_attribute("component.name", "ResponseGenerator")
                        rg_span.add_event("Enhanced response generator initialized")
                        component_init_results.append("ResponseGenerator")
                        logger.info("✅ Initialized enhanced response generator")

                if SourceAttributionProcessor:
                    with tracer.start_as_current_span("initialize_source_attributor") as sa_span:
                        self.source_attributor = SourceAttributionProcessor()
                        sa_span.set_attribute("component.name", "SourceAttributionProcessor")
                        sa_span.add_event("Source attribution processor initialized")
                        component_init_results.append("SourceAttribution")
                        logger.info("✅ Initialized source attribution processor")

                span.set_attribute("components.initialized", component_init_results)
                span.add_event("Component initialization completed", {
                    "components_count": len(component_init_results),
                    "components": component_init_results
                })

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("initialization.status", "failed")
                logger.error(f"❌ Error initializing enhanced query engine: {e}")
                # Fallback to basic functionality
                self.vector_store = None

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query using enhanced retrieval components with comprehensive tracing"""
        with tracer.start_as_current_span("process_user_query") as span:
            start_time = time.time()
            
            span.set_attribute("query.text", request.query[:100])  # Truncate for security
            span.set_attribute("query.session_id", request.session_id or "none")
            span.set_attribute("query.citation_format", request.citation_format)
            span.set_attribute("query.max_sources", request.max_sources)
            span.set_attribute("query.include_context", request.include_context)
            
            # Add baggage for cross-service correlation
            baggage.set_baggage("query.user_id", request.user_id or "anonymous")
            baggage.set_baggage("query.session_id", request.session_id or "none")
            
            query_processing_counter.add(1, {"status": "attempt", "citation_format": request.citation_format})

            try:
                if not self.vector_store:
                    span.set_attribute("error.type", "vector_store_unavailable")
                    query_processing_counter.add(1, {"status": "error", "error_type": "vector_store_unavailable"})
                    raise HTTPException(status_code=503, detail="Vector store not available")

                # Get or create conversation context
                session_id = request.session_id or str(uuid.uuid4())
                span.set_attribute("session.resolved_id", session_id)
                
                with tracer.start_as_current_span("manage_conversation_context") as context_span:
                    conversation_context = conversation_contexts.get(session_id)

                    if ConversationContext and not conversation_context:
                        conversation_context = ConversationContext(session_id=session_id)
                        conversation_contexts[session_id] = conversation_context
                        context_span.set_attribute("context.created", True)
                        context_span.add_event("New conversation context created")
                    else:
                        context_span.set_attribute("context.created", False)

                # Use enhanced retrieval if available
                if self.context_aware_engine and conversation_context:
                    with tracer.start_as_current_span("enhanced_query_processing") as enhanced_span:
                        # Add user message to context
                        if hasattr(conversation_context, 'add_message'):
                            conversation_context.add_message(
                                message_type=MessageType.USER,
                                content=request.query,
                                metadata={"timestamp": datetime.now().isoformat()}
                            )
                            enhanced_span.add_event("User message added to context")

                        # Process with context-aware engine
                        result = await self._process_with_enhanced_engine(request, conversation_context)
                        enhanced_span.set_attribute("processing.type", "enhanced")
                else:
                    with tracer.start_as_current_span("basic_query_processing") as basic_span:
                        # Fallback to basic retrieval
                        result = await self._process_with_basic_engine(request)
                        basic_span.set_attribute("processing.type", "basic")

                # Calculate processing time
                processing_time = time.time() - start_time
                query_processing_duration.record(processing_time, {
                    "query_type": result.query_type,
                    "citation_format": request.citation_format
                })

                # Update result with processing time
                result.processing_time = processing_time
                result.session_id = session_id
                result.timestamp = datetime.now(timezone.utc).isoformat()

                # Add assistant response to context
                if conversation_context and hasattr(conversation_context, 'add_message'):
                    with tracer.start_as_current_span("add_assistant_response") as response_span:
                        conversation_context.add_message(
                            message_type=MessageType.ASSISTANT,
                            content=result.response,
                            metadata={
                                "sources": result.sources,
                                "confidence": result.confidence,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        response_span.add_event("Assistant response added to context")

                # Update metrics
                query_processing_counter.add(1, {"status": "success", "query_type": result.query_type})
                
                span.set_attribute("processing.duration_seconds", processing_time)
                span.set_attribute("response.sources_count", len(result.sources))
                span.set_attribute("response.word_count", result.word_count)
                span.set_attribute("response.confidence", result.confidence)
                
                span.add_event("Query processing completed", {
                    "processing_time": processing_time,
                    "sources_found": len(result.sources),
                    "response_length": len(result.response)
                })

                return result

            except Exception as e:
                processing_time = time.time() - start_time
                span.record_exception(e)
                span.set_attribute("error.processing_time", processing_time)
                
                query_processing_counter.add(1, {"status": "error", "error_type": type(e).__name__})
                
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

    async def _process_with_enhanced_engine(self, request: QueryRequest, context) -> QueryResponse:
        """Process query with enhanced components and comprehensive instrumentation"""
        with tracer.start_as_current_span("enhanced_engine_processing") as span:
            try:
                # Use context-aware query processing
                with tracer.start_as_current_span("context_aware_query_processing") as caq_span:
                    caq_span.set_attribute("context.session_id", context.session_id if hasattr(context, 'session_id') else "unknown")
                    
                    query_result = await self.context_aware_engine.process_query(
                        query=request.query,
                        conversation_context=context,
                        max_sources=request.max_sources
                    )
                    
                    caq_span.set_attribute("query_result.documents_count", len(query_result.get('documents', [])))
                    caq_span.add_event("Context-aware query processing completed")

                # Generate enhanced response
                with tracer.start_as_current_span("generate_enhanced_response") as gen_span:
                    if self.response_generator:
                        gen_span.set_attribute("generator.type", "enhanced")
                        
                        llm_generation_counter.add(1, {"type": "enhanced", "style": request.response_style})
                        
                        response_text = await self.response_generator.generate_response(
                            query=request.query,
                            retrieved_docs=query_result.get('documents', []),
                            context=context,
                            style=request.response_style
                        )
                        gen_span.add_event("Enhanced response generated")
                    else:
                        gen_span.set_attribute("generator.type", "basic_fallback")
                        llm_generation_counter.add(1, {"type": "basic_fallback", "style": request.response_style})
                        
                        response_text = await self._generate_basic_response(
                            request.query,
                            query_result.get('documents', [])
                        )
                        gen_span.add_event("Basic response generated as fallback")

                # Enhanced source attribution
                sources = []
                with tracer.start_as_current_span("process_sources") as source_span:
                    if self.source_attributor and query_result.get('documents'):
                        source_span.set_attribute("attribution.type", "enhanced")
                        
                        sources = self.source_attributor.process_sources(
                            documents=query_result['documents'],
                            query=request.query,
                            citation_format=CitationFormat(request.citation_format)
                        )
                        source_span.add_event("Enhanced source attribution completed")
                    else:
                        source_span.set_attribute("attribution.type", "basic_fallback")
                        
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
                        source_span.add_event("Basic source processing completed")

                # Create citations
                with tracer.start_as_current_span("create_citations") as citation_span:
                    citations = self._create_citations(query_result.get('documents', []), request.citation_format)
                    citation_span.set_attribute("citations.count", len(citations))

                span.set_attribute("enhanced.processing_successful", True)
                span.set_attribute("sources.count", len(sources))
                span.add_event("Enhanced engine processing completed")

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
                span.set_attribute("enhanced.fallback_triggered", True)
                logger.error(f"Enhanced processing failed, falling back to basic: {e}")
                return await self._process_with_basic_engine(request)

    async def _process_with_basic_engine(self, request: QueryRequest) -> QueryResponse:
        """Fallback to basic query processing with instrumentation"""
        with tracer.start_as_current_span("basic_engine_processing") as span:
            span.set_attribute("processing.type", "basic_fallback")
            
            # Perform similarity search
            with tracer.start_as_current_span("vector_similarity_search") as search_span:
                vector_search_counter.add(1, {"type": "similarity", "collection": self.collection_name})
                
                docs = self.vector_store.similarity_search(
                    request.query,
                    k=request.max_sources or 5
                )
                
                search_span.set_attribute("search.query_length", len(request.query))
                search_span.set_attribute("search.results_count", len(docs))
                search_span.set_attribute("search.collection", self.collection_name)
                search_span.add_event("Vector similarity search completed")

            # Create sources from documents
            sources = []
            with tracer.start_as_current_span("process_search_results") as results_span:
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
                
                results_span.set_attribute("results.processed_count", len(sources))

            # Generate response using retrieved documents
            with tracer.start_as_current_span("generate_basic_response") as gen_span:
                llm_generation_counter.add(1, {"type": "basic", "style": request.response_style})
                
                context = "\n\n".join([doc.page_content for doc in docs])
                response_text = await self._generate_basic_response(request.query, docs)
                
                gen_span.set_attribute("context.length", len(context))
                gen_span.set_attribute("response.length", len(response_text))
                gen_span.add_event("Basic response generation completed")

            # Create citations
            with tracer.start_as_current_span("create_basic_citations") as citation_span:
                citations = self._create_citations(docs, request.citation_format)
                citation_span.set_attribute("citations.count", len(citations))

            span.add_event("Basic engine processing completed", {
                "sources_found": len(sources),
                "response_generated": len(response_text) > 0
            })

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
        """Generate response using OpenAI GPT with instrumentation"""
        with tracer.start_as_current_span("openai_response_generation") as span:
            span.set_attribute("llm.model", "gpt-4o-mini")
            span.set_attribute("query.length", len(query))
            span.set_attribute("documents.count", len(docs))
            
            try:
                import openai

                client = openai.OpenAI()
                context = "\n\n".join([doc.page_content for doc in docs])
                
                span.set_attribute("context.length", len(context))

                prompt = f"""Based on the following context, please answer the user's question accurately and comprehensively.

Context:
{context}

Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing."""

                with tracer.start_as_current_span("openai_api_call") as api_span:
                    api_span.set_attribute("api.provider", "openai")
                    api_span.set_attribute("api.model", "gpt-4o-mini")
                    api_span.set_attribute("api.max_tokens", 1000)
                    api_span.set_attribute("api.temperature", 0.1)
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    
                    api_span.set_attribute("api.response.choices", len(response.choices))
                    api_span.add_event("OpenAI API call completed")

                generated_response = response.choices[0].message.content
                span.set_attribute("response.generated_length", len(generated_response))
                span.add_event("Response generation successful")
                
                return generated_response

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("generation.fallback", True)
                
                logger.error(f"Error generating response: {e}")
                context = "\n\n".join([doc.page_content for doc in docs])
                fallback_response = f"Based on the retrieved documents, here's what I found relevant to your query: {context[:500]}..."
                
                span.add_event("Using fallback response", {
                    "fallback_length": len(fallback_response)
                })
                
                return fallback_response

    def _create_citations(self, docs: List[Document], format_type: str) -> List[str]:
        """Create citations for the sources with instrumentation"""
        with tracer.start_as_current_span("create_citations") as span:
            span.set_attribute("citation.format", format_type)
            span.set_attribute("documents.count", len(docs))
            
            citations = []
            for i, doc in enumerate(docs):
                title = doc.metadata.get("title", f"Document {i+1}")
                source = doc.metadata.get("source", "Unknown source")
                citations.append(f"[{i+1}] {title} - {source}")
            
            span.set_attribute("citations.generated", len(citations))
            span.add_event("Citations created")
            
            return citations

# Initialize enhanced query engine
enhanced_query_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced query engine on startup with instrumentation"""
    global enhanced_query_engine
    
    with tracer.start_as_current_span("fastapi_startup") as span:
        logger.info("🚀 Starting Enhanced Document RAG API...")
        span.add_event("FastAPI startup initiated")

        try:
            enhanced_query_engine = EnhancedQueryEngine()
            api_health_gauge.add(1, {"component": "query_engine"})
            
            span.set_attribute("query_engine.initialized", True)
            span.add_event("Enhanced query engine initialized successfully")
            logger.info("✅ Enhanced query engine initialized successfully")
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("query_engine.initialized", False)
            logger.error(f"❌ Failed to initialize enhanced query engine: {e}")
            enhanced_query_engine = None

# Instrument FastAPI after initialization
FastAPIInstrumentor.instrument_app(app)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page with instrumentation"""
    with tracer.start_as_current_span("serve_root_page") as span:
        span.set_attribute("http.endpoint", "/")
        span.set_attribute("content.type", "text/html")
        
        try:
            with open("ui/static/index.html", "r") as f:
                content = f.read()
                span.set_attribute("content.length", len(content))
                span.add_event("Static file served successfully")
                return HTMLResponse(content=content)
        except FileNotFoundError:
            span.set_attribute("content.fallback", True)
            span.add_event("Static file not found, serving fallback")
            return HTMLResponse(content="<h1>Enhanced Document RAG API</h1><p>Static files not found. Please ensure ui/static/index.html exists.</p>")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return enhanced results with comprehensive instrumentation"""
    with tracer.start_as_current_span("api_process_query") as span:
        span.set_attribute("http.endpoint", "/api/query")
        span.set_attribute("http.method", "POST")
        span.set_attribute("query.length", len(request.query))
        
        if not enhanced_query_engine:
            span.set_attribute("error.type", "query_engine_not_initialized")
            span.add_event("Query engine not initialized")
            raise HTTPException(status_code=503, detail="Enhanced query engine not initialized")

        # Create session if not exists
        session_id = request.session_id or str(uuid.uuid4())
        span.set_attribute("session.id", session_id)
        
        with tracer.start_as_current_span("manage_session") as session_span:
            if session_id not in sessions:
                sessions[session_id] = SessionInfo(
                    session_id=session_id,
                    message_count=0,
                    current_topic=None,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    last_activity=datetime.now(timezone.utc).isoformat(),
                    metadata={}
                )
                session_management_gauge.add(1, {"operation": "create"})
                session_span.set_attribute("session.created", True)
                session_span.add_event("New session created")
            else:
                session_span.set_attribute("session.created", False)

            # Update session
            sessions[session_id].message_count += 1
            sessions[session_id].last_activity = datetime.now(timezone.utc).isoformat()
            
            session_span.set_attribute("session.message_count", sessions[session_id].message_count)

        # Process query with enhanced engine
        response = await enhanced_query_engine.process_query(request)
        
        span.set_attribute("response.success", True)
        span.set_attribute("response.processing_time", response.processing_time)
        span.add_event("Query processing completed successfully")

        return response

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information with instrumentation"""
    with tracer.start_as_current_span("api_get_session") as span:
        span.set_attribute("http.endpoint", "/api/sessions/{session_id}")
        span.set_attribute("session.requested_id", session_id)
        
        if session_id not in sessions:
            span.set_attribute("session.found", False)
            span.add_event("Session not found")
            raise HTTPException(status_code=404, detail="Session not found")

        span.set_attribute("session.found", True)
        
        with tracer.start_as_current_span("retrieve_conversation_history") as history_span:
            conversation_context = conversation_contexts.get(session_id)
            conversation_history = []

            if conversation_context and hasattr(conversation_context, 'get_messages'):
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
                history_span.set_attribute("conversation.message_count", len(conversation_history))
            else:
                history_span.set_attribute("conversation.message_count", 0)
            
            history_span.add_event("Conversation history retrieved")

        response_data = {
            "session": sessions[session_id],
            "conversation_history": conversation_history
        }
        
        span.add_event("Session information retrieved")
        return response_data

@app.get("/api/health")
async def health_check():
    """Health check endpoint with comprehensive instrumentation"""
    with tracer.start_as_current_span("api_health_check") as span:
        span.set_attribute("http.endpoint", "/api/health")
        
        status = "healthy" if enhanced_query_engine and enhanced_query_engine.vector_store else "unhealthy"
        span.set_attribute("health.status", status)
        
        components = {}
        if enhanced_query_engine:
            components = {
                "vector_store": enhanced_query_engine.vector_store is not None,
                "hybrid_search": enhanced_query_engine.hybrid_search is not None,
                "context_aware_engine": enhanced_query_engine.context_aware_engine is not None,
                "response_generator": enhanced_query_engine.response_generator is not None,
                "source_attributor": enhanced_query_engine.source_attributor is not None
            }
        
        healthy_components = sum(1 for v in components.values() if v)
        span.set_attribute("components.healthy_count", healthy_components)
        span.set_attribute("components.total_count", len(components))
        
        response_data = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "components": components
        }
        
        span.add_event("Health check completed", {
            "status": status,
            "healthy_components": healthy_components
        })

        return response_data

if __name__ == "__main__":
    import uvicorn
    with tracer.start_as_current_span("uvicorn_direct_run") as span:
        span.set_attribute("server.host", "0.0.0.0")
        span.set_attribute("server.port", 8000)
        logger.info("🌐 Running FastAPI directly with uvicorn")
        uvicorn.run(app, host="0.0.0.0", port=8000)

