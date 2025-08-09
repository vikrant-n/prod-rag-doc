#!/usr/bin/env python3
"""
Integrated FastAPI Backend with Enhanced Retrieval - Clean Version

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

from fastapi import FastAPI, HTTPException
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
    title="Document RAG Query Interface",
    description="Web-based interface for document retrieval and query processing",
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

# Global variables for the query engine components
enhanced_query_engine = None
sessions: Dict[str, SessionInfo] = {}

class SimpleQueryEngine:
    """Simple query engine using basic pipeline components."""
    
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
            
            self._initialize_components()
            span.set_attribute("init.status", "success")
    
    def _initialize_components(self):
        """Initialize components"""
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
                
                span.set_attribute("init.status", "success")
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("init.status", "failed")
                logger.error(f"Error initializing query engine: {e}")
                self.vector_store = None
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query using basic retrieval components"""
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
                
                session_id = request.session_id or str(uuid.uuid4())
                span.set_attribute("session.id", session_id)
                
                # Process with basic retrieval
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
                
                span.set_attribute("query.status", "success")
                logger.info(f"Query processed successfully: {processing_time:.2f}s, {len(result.sources)} sources")
                
                return result
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("query.status", "error")
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    async def _process_with_basic_engine(self, request: QueryRequest) -> QueryResponse:
        """Basic query processing"""
        with tracer.start_as_current_span("query.process_basic") as span:
            span.set_attribute("processing.type", "basic")
            
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
            with tracer.start_as_current_span("sources.processing") as sources_span:
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

# Initialize query engine
enhanced_query_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the query engine on startup"""
    with tracer.start_as_current_span("fastapi.startup") as span:
        global enhanced_query_engine
        logger.info("Starting Document RAG API...")
        
        try:
            enhanced_query_engine = SimpleQueryEngine()
            span.set_attribute("startup.query_engine", "success")
            logger.info("Query engine initialized successfully")
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("startup.query_engine", "failed")
            logger.error(f"Failed to initialize query engine: {e}")
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
            return HTMLResponse(content="<h1>Document RAG API</h1><p>Static files not found. Please ensure ui/static/index.html exists.</p>")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return results"""
    with tracer.start_as_current_span("api.process_query") as span:
        span.set_attribute("endpoint", "/api/query")
        span.set_attribute("request.query", request.query[:100] + "..." if len(request.query) > 100 else request.query)
        span.set_attribute("request.session_id", request.session_id or "new")
        span.set_attribute("request.max_sources", request.max_sources or 5)
        
        if not enhanced_query_engine:
            span.set_attribute("api.error", "query_engine_not_initialized")
            raise HTTPException(status_code=503, detail="Query engine not initialized")
        
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
        
        # Process query
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
        
        session_info = {
            "session": sessions[session_id],
            "conversation_history": []  # Simplified for now
        }
        
        return session_info

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    with tracer.start_as_current_span("api.health_check") as span:
        span.set_attribute("endpoint", "/api/health")
        
        status = "healthy" if enhanced_query_engine and enhanced_query_engine.vector_store else "unhealthy"
        
        # Check backend service health if possible
        backend_status = "unknown"
        try:
            import httpx
            backend_url = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8001")
            
            with tracer.start_as_current_span("backend.health_check") as backend_span:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{backend_url}/status", timeout=5.0)
                    if response.status_code == 200:
                        backend_status = "healthy"
                        backend_span.set_attribute("backend.status", "healthy")
                    else:
                        backend_status = "unhealthy"
                        backend_span.set_attribute("backend.status", "unhealthy")
        except Exception as e:
            backend_status = "unreachable"
        
        components = {
            "vector_store": enhanced_query_engine.vector_store is not None if enhanced_query_engine else False,
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
        
        # Record health check metrics
        record_cache_event("health_check", status == "healthy")
        record_cache_event("backend_health_check", backend_status == "healthy")
        
        return health_data

if __name__ == "__main__":
    import uvicorn
    
    with tracer.start_as_current_span("app.main_startup") as span:
        span.set_attribute("app.mode", "main")
        logger.info("Starting API server directly")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)