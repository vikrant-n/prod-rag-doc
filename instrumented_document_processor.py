#!/usr/bin/env python3
"""
Instrumented Document Processing Pipeline
Comprehensive tracing for the entire document workflow
"""

import os
import time
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import your existing modules (adjust imports as needed)
from otel_config import initialize_instrumentation, instrument_fastapi, get_system_metrics
from trace_correlation import (
    get_document_tracer, get_health_tracer, trace_function,
    start_document_session, start_embedding_operation, start_vector_operation,
    correlated_operation, setup_reduced_logging
)

# Initialize instrumentation
initialize_instrumentation("document-processor", "1.0.0")
setup_reduced_logging()

logger = logging.getLogger(__name__)

class InstrumentedDocumentProcessor:
    """Document processor with comprehensive instrumentation"""
    
    def __init__(self):
        self.tracer = get_document_tracer()
        self.metrics = get_system_metrics()
        self.health_tracer = get_health_tracer()
        self.processed_files = set()  # Track processed files to avoid duplicates
        self.last_scan_time = 0
        self.scan_interval = 30  # Minimum 30 seconds between scans
    
    @trace_function(operation_type="scanner", log_calls=False)
    def scan_google_drive(self, folder_id: str) -> List[Dict[str, Any]]:
        """Scan Google Drive for new documents - reduced logging"""
        current_time = time.time()
        
        # Rate limit scanning
        if current_time - self.last_scan_time < self.scan_interval:
            return []
        
        with correlated_operation(
            "google_drive_scan",
            operation_type="scanner",
            attributes={
                "drive.folder_id": folder_id,
                "scan.interval": self.scan_interval
            }
        ) as span:
            try:
                # Your existing Google Drive scanning logic here
                new_files = self._get_new_files_from_drive(folder_id)
                
                # Only log if new files found or errors
                if new_files:
                    logger.info(f"📁 Found {len(new_files)} new documents in Google Drive")
                    
                    if span:
                        span.set_attributes({
                            "drive.new_files_count": len(new_files),
                            "drive.scan_success": True
                        })
                
                self.last_scan_time = current_time
                return new_files
                
            except Exception as e:
                logger.error(f"❌ Google Drive scan failed: {e}")
                if span:
                    span.set_attribute("drive.scan_success", False)
                raise
    
    @trace_function(operation_type="document", log_calls=True)
    def process_document(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Process a single document with full instrumentation"""
        
        # Check if already processed
        file_hash = self._get_file_hash(file_path)
        if file_hash in self.processed_files:
            logger.debug(f"⏭️ Skipping already processed file: {file_path}")
            return {"status": "skipped", "reason": "already_processed"}
        
        with start_document_session(file_path, file_type) as span:
            try:
                # Extract document content
                content = self._extract_content(file_path, file_type, span)
                
                # Chunk the content
                chunks = self._chunk_content(content, span)
                
                # Generate embeddings
                embeddings = self._generate_embeddings(chunks, span)
                
                # Store in vector database
                stored_count = self._store_vectors(embeddings, chunks, file_path, span)
                
                # Mark as processed
                self.processed_files.add(file_hash)
                
                result = {
                    "status": "success",
                    "file_path": file_path,
                    "chunks_count": len(chunks),
                    "stored_vectors": stored_count,
                    "file_hash": file_hash
                }
                
                if span:
                    span.set_attributes({
                        "document.chunks_created": len(chunks),
                        "document.vectors_stored": stored_count,
                        "document.processing_success": True
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Document processing failed for {file_path}: {e}")
                if span:
                    span.set_attribute("document.processing_success", False)
                raise
    
    def _extract_content(self, file_path: str, file_type: str, parent_span) -> str:
        """Extract content from document"""
        with correlated_operation(
            "extract_content",
            operation_type="extraction",
            attributes={
                "extraction.file_type": file_type,
                "extraction.file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
        ) as span:
            try:
                # Your existing content extraction logic
                content = self._perform_content_extraction(file_path, file_type)
                
                if span:
                    span.set_attributes({
                        "extraction.content_length": len(content),
                        "extraction.success": True
                    })
                
                return content
                
            except Exception as e:
                logger.error(f"❌ Content extraction failed: {e}")
                if span:
                    span.set_attribute("extraction.success", False)
                raise
    
    def _chunk_content(self, content: str, parent_span) -> List[str]:
        """Chunk content into smaller pieces"""
        with correlated_operation(
            "chunk_content",
            operation_type="chunking",
            attributes={
                "chunking.content_length": len(content),
                "chunking.strategy": "recursive"
            }
        ) as span:
            try:
                # Your existing chunking logic
                chunks = self._perform_content_chunking(content)
                
                if span:
                    span.set_attributes({
                        "chunking.chunks_created": len(chunks),
                        "chunking.avg_chunk_size": sum(len(c) for c in chunks) // len(chunks) if chunks else 0,
                        "chunking.success": True
                    })
                
                return chunks
                
            except Exception as e:
                logger.error(f"❌ Content chunking failed: {e}")
                if span:
                    span.set_attribute("chunking.success", False)
                raise
    
    def _generate_embeddings(self, chunks: List[str], parent_span) -> List[List[float]]:
        """Generate embeddings for chunks"""
        with start_embedding_operation(len(chunks), "text-embedding-3-small") as span:
            try:
                # Your existing embedding generation logic
                embeddings = self._perform_embedding_generation(chunks)
                
                if span:
                    span.set_attributes({
                        "embedding.chunks_processed": len(chunks),
                        "embedding.embeddings_created": len(embeddings),
                        "embedding.success": True
                    })
                
                return embeddings
                
            except Exception as e:
                logger.error(f"❌ Embedding generation failed: {e}")
                if span:
                    span.set_attribute("embedding.success", False)
                raise
    
    def _store_vectors(self, embeddings: List[List[float]], chunks: List[str], 
                      file_path: str, parent_span) -> int:
        """Store vectors in Qdrant"""
        with start_vector_operation("upsert", len(embeddings)) as span:
            try:
                # Your existing vector storage logic
                stored_count = self._perform_vector_storage(embeddings, chunks, file_path)
                
                if span:
                    span.set_attributes({
                        "vector.stored_count": stored_count,
                        "vector.collection": "documents",
                        "vector.success": True
                    })
                
                return stored_count
                
            except Exception as e:
                logger.error(f"❌ Vector storage failed: {e}")
                if span:
                    span.set_attribute("vector.success", False)
                raise
    
    def _get_new_files_from_drive(self, folder_id: str) -> List[Dict[str, Any]]:
        """Get new files from Google Drive - implement your logic"""
        # Placeholder - implement your Google Drive API logic
        return []
    
    def _perform_content_extraction(self, file_path: str, file_type: str) -> str:
        """Perform actual content extraction - implement your logic"""
        # Placeholder - implement your document loading logic
        return ""
    
    def _perform_content_chunking(self, content: str) -> List[str]:
        """Perform actual content chunking - implement your logic"""
        # Placeholder - implement your text splitting logic
        return []
    
    def _perform_embedding_generation(self, chunks: List[str]) -> List[List[float]]:
        """Perform actual embedding generation - implement your logic"""
        # Placeholder - implement your OpenAI embedding logic
        return []
    
    def _perform_vector_storage(self, embeddings: List[List[float]], 
                               chunks: List[str], file_path: str) -> int:
        """Perform actual vector storage - implement your logic"""
        # Placeholder - implement your Qdrant storage logic
        return len(embeddings)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get file hash for deduplication"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return hashlib.md5(file_path.encode()).hexdigest()
    
    @trace_function(operation_type="health", log_calls=False)
    def health_check(self) -> Dict[str, Any]:
        """System health check with reduced logging"""
        with self.health_tracer.trace_health_check("document_processor", "comprehensive") as span:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "processed_files": len(self.processed_files),
                "last_scan": self.last_scan_time
            }
            
            # Check dependencies
            try:
                # Check Qdrant
                with correlated_operation("health_check_qdrant", operation_type="health") as qdrant_span:
                    qdrant_healthy = self._check_qdrant_health()
                    health_status["qdrant"] = "healthy" if qdrant_healthy else "unhealthy"
                
                # Check OpenAI
                with correlated_operation("health_check_openai", operation_type="health") as openai_span:
                    openai_healthy = self._check_openai_health()
                    health_status["openai"] = "healthy" if openai_healthy else "unhealthy"
                
                # Check Google Drive
                with correlated_operation("health_check_drive", operation_type="health") as drive_span:
                    drive_healthy = self._check_drive_health()
                    health_status["google_drive"] = "healthy" if drive_healthy else "unhealthy"
                
            except Exception as e:
                health_status["status"] = "unhealthy"
                health_status["error"] = str(e)
                if span:
                    span.set_attribute("health.overall_status", "unhealthy")
            
            return health_status
    
    def _check_qdrant_health(self) -> bool:
        """Check Qdrant health - implement your logic"""
        # Placeholder - implement your Qdrant health check
        return True
    
    def _check_openai_health(self) -> bool:
        """Check OpenAI health - implement your logic"""
        # Placeholder - implement your OpenAI health check
        return True
    
    def _check_drive_health(self) -> bool:
        """Check Google Drive health - implement your logic"""
        # Placeholder - implement your Google Drive health check
        return True

class InstrumentedQueryProcessor:
    """Query processor with comprehensive instrumentation"""
    
    def __init__(self):
        self.query_tracer = get_query_tracer()
        self.metrics = get_system_metrics()
    
    @trace_function(operation_type="query", log_calls=True)
    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process user query with full tracing"""
        
        with self.query_tracer.trace_user_query(query, session_id) as span:
            try:
                # Validate query
                self._validate_query(query, span)
                
                # Generate query embedding
                query_embedding = self._generate_query_embedding(query, span)
                
                # Search vectors
                search_results = self._search_vectors(query_embedding, span)
                
                # Generate response
                response = self._generate_response(query, search_results, span)
                
                result = {
                    "query": query,
                    "response": response,
                    "sources": [r.get("source", "") for r in search_results],
                    "relevance_scores": [r.get("score", 0) for r in search_results]
                }
                
                if span:
                    span.set_attributes({
                        "query.sources_found": len(search_results),
                        "query.response_length": len(response),
                        "query.processing_success": True
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Query processing failed: {e}")
                if span:
                    span.set_attribute("query.processing_success", False)
                raise
    
    def _validate_query(self, query: str, parent_span):
        """Validate query input"""
        with correlated_operation(
            "validate_query",
            operation_type="validation",
            attributes={"query.length": len(query)}
        ) as span:
            if not query or len(query.strip()) < 3:
                raise ValueError("Query too short")
            
            if len(query) > 1000:
                raise ValueError("Query too long")
            
            if span:
                span.set_attribute("validation.success", True)
    
    def _generate_query_embedding(self, query: str, parent_span) -> List[float]:
        """Generate embedding for query"""
        with start_embedding_operation(1, "text-embedding-3-small") as span:
            # Your embedding generation logic
            embedding = self._perform_query_embedding(query)
            
            if span:
                span.set_attributes({
                    "embedding.query_length": len(query),
                    "embedding.vector_size": len(embedding),
                    "embedding.success": True
                })
            
            return embedding
    
    def _search_vectors(self, query_embedding: List[float], parent_span) -> List[Dict[str, Any]]:
        """Search vector database"""
        with self.query_tracer.trace_vector_search(len(query_embedding), top_k=5) as span:
            # Your vector search logic
            results = self._perform_vector_search(query_embedding)
            
            if span:
                span.set_attributes({
                    "search.results_count": len(results),
                    "search.success": True
                })
            
            return results
    
    def _generate_response(self, query: str, search_results: List[Dict[str, Any]], 
                          parent_span) -> str:
        """Generate response using LLM"""
        context = "\n".join([r.get("content", "") for r in search_results])
        prompt_tokens = len(query.split()) + len(context.split())
        
        with self.query_tracer.trace_llm_completion("gpt-3.5-turbo", prompt_tokens) as span:
            # Your LLM completion logic
            response = self._perform_llm_completion(query, context)
            
            if span:
                span.set_attributes({
                    "llm.response_length": len(response),
                    "llm.context_length": len(context),
                    "llm.success": True
                })
            
            return response
    
    def _perform_query_embedding(self, query: str) -> List[float]:
        """Perform query embedding - implement your logic"""
        # Placeholder
        return [0.0] * 1536
    
    def _perform_vector_search(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """Perform vector search - implement your logic"""
        # Placeholder
        return []
    
    def _perform_llm_completion(self, query: str, context: str) -> str:
        """Perform LLM completion - implement your logic"""
        # Placeholder
        return "Generated response"

# Example usage and integration
def create_instrumented_app():
    """Create FastAPI app with instrumentation"""