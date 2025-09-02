#!/usr/bin/env python3
"""
Enhanced Backend Service with Complete OpenTelemetry Instrumentation
Fixed to prevent segmentation faults and memory issues
FIXED: Proper OpenTelemetry initialization order and instrumentation
"""

import os
import sys
import asyncio
import logging
import hashlib
import json
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import shutil
import gc
from contextlib import asynccontextmanager

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

# CRITICAL FIX: Set OpenTelemetry service name early and extract parent context
os.environ["OTEL_SERVICE_NAME"] = "document-rag-backend"

# FIXED: Proper OpenTelemetry initialization with parent context
from opentelemetry import trace, metrics, propagate
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.context import attach, detach

# Import OpenTelemetry configuration EARLY
from otel_config import (
    initialize_opentelemetry, get_service_tracer, instrument_fastapi_app,
    get_current_trace_id, extract_and_activate_context, TracedHTTPXClient,
    get_correlated_logger
)

# FIXED: Extract parent context from environment if available
def extract_parent_context_from_environment():
    """Extract parent trace context from environment variables set by orchestrator"""
    traceparent = os.getenv("OTEL_TRACE_PARENT")
    tracestate = os.getenv("OTEL_TRACE_STATE", "")
    
    if traceparent:
        print(f"📥 Backend extracting parent context from environment:")
        print(f"   traceparent: {traceparent}")
        print(f"   tracestate: {tracestate}")
        
        headers = {
            "traceparent": traceparent,
            "tracestate": tracestate
        }
        
        try:
            parent_context = propagate.extract(headers)
            context_token = attach(parent_context)
            print(f"✅ Parent context extracted and activated")
            return context_token, parent_context
        except Exception as e:
            print(f"❌ Error extracting parent context: {e}")
            return None, None
    else:
        print(f"ℹ️ No parent context found in environment (standalone mode)")
        return None, None

# FIXED: Initialize OpenTelemetry with proper parent context
print("=" * 60)
print("🔧 Backend Service Module Initialization")

# Step 1: Extract parent context FIRST
context_token, parent_context = extract_parent_context_from_environment()

# Step 2: Initialize OpenTelemetry with parent context active
tracer, meter = initialize_opentelemetry(
    service_name="document-rag-backend",
    service_version="2.0.0",
    environment=os.getenv("OTEL_ENVIRONMENT", "production")
)

# Step 3: Create verification span
with tracer.start_as_current_span("backend_service_initialization") as span:
    span_context = span.get_span_context()
    trace_id = format(span_context.trace_id, '032x')
    span_id = format(span_context.span_id, '016x')
    
    span.set_attributes({
        "service.name": "document-rag-backend",
        "service.version": "2.0.0",
        "service.parent": "document-rag-orchestrator",
        "initialization.with_parent": parent_context is not None,
        "initialization.trace_id": trace_id,
        "initialization.span_id": span_id,
        "initialization.parent_trace": os.getenv("OTEL_PARENT_TRACE_ID", "none")
    })
    
    print(f"🆔 Backend Service Telemetry Initialized:")
    print(f"   Current Trace ID: {trace_id}")
    print(f"   Current Span ID: {span_id}")
    print(f"   Parent Trace ID: {os.getenv('OTEL_PARENT_TRACE_ID', 'none')}")
    print(f"   Parent Context: {'✅ Connected' if parent_context else '❌ Standalone'}")

print("=" * 60)

# FastAPI imports for status API
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
import uvicorn
import httpx

# Document processing imports
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError, RequestError

# Loaders
from loaders.master_loaders import load_file
from loaders.google_drive_loader import GoogleDriveMasterLoader

# Text splitting
from text_splitting import split_documents

# Configuration from environment
parent_trace_id = os.getenv("OTEL_PARENT_TRACE_ID")
parent_service = os.getenv("OTEL_SERVICE_PARENT", "document-rag-orchestrator")
orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:8002")

# Use correlated logger - this will now have proper trace context
logger = get_correlated_logger(__name__)

# Log initialization complete with trace context
logger.info_with_context(
    "Backend service module initialization complete",
    extra_attributes={
        "service.name": "document-rag-backend",
        "parent.service": parent_service,
        "parent.trace_id": parent_trace_id,
        "orchestrator.url": orchestrator_url,
        "operation": "module_init"
    }
)

# Metrics - use the initialized meter with proper context
documents_processed = meter.create_counter(
    "documents_processed_total",
    description="Total number of documents processed"
)

scan_duration = meter.create_histogram(
    "scan_duration_seconds",
    description="Time taken for document scanning"
)

files_processed = meter.create_counter(
    "files_processed_total",
    description="Total number of files processed"
)

processing_errors = meter.create_counter(
    "processing_errors_total",
    description="Total number of processing errors"
)

external_api_calls = meter.create_counter(
    "external_api_calls_total",
    description="Total number of external API calls"
)

@dataclass
class ProcessedFile:
    """Metadata for a processed file"""
    file_id: str
    file_name: str
    file_path: str
    file_hash: str
    processed_at: datetime
    document_count: int
    file_size: int
    mime_type: str
    qdrant_point_ids: List[str]  # Track Qdrant point IDs for this file


class FileFingerprintDatabase:
    """SQLite database to track processed files and avoid reprocessing"""
    
    def __init__(self, db_path: str = ".processed_files.db"):
        self.db_path = db_path
        self.tracer = tracer  # Use global tracer
        self.logger = get_correlated_logger(f"{__name__}.FileFingerprintDatabase")
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    processed_at TIMESTAMP NOT NULL,
                    document_count INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    mime_type TEXT,
                    qdrant_point_ids TEXT  -- JSON array of point IDs
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON processed_files(file_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_name ON processed_files(file_name)
            """)
        
        self.logger.info_with_context(
            "File fingerprint database initialized",
            extra_attributes={
                "database.path": self.db_path,
                "operation": "database_init"
            }
        )
    
    def is_file_processed(self, file_path: str, file_hash: str) -> bool:
        """Check if a file has already been processed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM processed_files WHERE file_path = ? AND file_hash = ?",
                (file_path, file_hash)
            )
            result = cursor.fetchone()[0] > 0
            
            self.logger.debug_with_context(
                "Checked file processing status",
                extra_attributes={
                    "file.path": file_path,
                    "file.hash": file_hash[:16],  # Truncated for readability
                    "file.is_processed": result,
                    "operation": "file_check"
                }
            )
            return result
    
    def is_file_in_elasticsearch(self, file_path: str, elasticsearch_client) -> bool:
        """Check if a file already has embeddings in Elasticsearch"""
        try:
            with self.tracer.start_as_current_span("elasticsearch_check_file_exists") as span:
                span.set_attributes({
                    "service.name": "document-rag-backend",
                    "peer.service": "elasticsearch-database",
                    "external.service.name": "elasticsearch-database",
                    "external.service.type": "vector_database",
                    "db.system": "elasticsearch",
                    "db.operation": "search",
                    "db.collection.name": "documents"
                })
                
                search_result = elasticsearch_client.search(
                    index="documents",
                    body={
                        "query": {
                            "term": {
                                "metadata.file_path.keyword": file_path
                            }
                        },
                        "size": 1
                    }
                )
                return search_result['hits']['total']['value'] > 0
        except Exception as e:
            return False
    
    def mark_file_processed(self, processed_file: ProcessedFile):
        """Mark a file as processed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO processed_files 
                (file_id, file_name, file_path, file_hash, processed_at, document_count, file_size, mime_type, qdrant_point_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                processed_file.file_id,
                processed_file.file_name,
                processed_file.file_path,
                processed_file.file_hash,
                processed_file.processed_at,
                processed_file.document_count,
                processed_file.file_size,
                processed_file.mime_type,
                json.dumps(processed_file.qdrant_point_ids)
            ))
        
        self.logger.info_with_context(
            "File marked as processed in database",
            extra_attributes={
                "file.id": processed_file.file_id,
                "file.name": processed_file.file_name,
                "file.document_count": processed_file.document_count,
                "file.size": processed_file.file_size,
                "operation": "file_mark_processed"
            }
        )
    
    def get_processed_files(self, limit: int = 100) -> List[ProcessedFile]:
        """Get list of processed files"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT file_id, file_name, file_path, file_hash, processed_at, 
                       document_count, file_size, mime_type, qdrant_point_ids
                FROM processed_files 
                ORDER BY processed_at DESC 
                LIMIT ?
            """, (limit,))
            
            files = []
            for row in cursor.fetchall():
                files.append(ProcessedFile(
                    file_id=row[0],
                    file_name=row[1],
                    file_path=row[2],
                    file_hash=row[3],
                    processed_at=datetime.fromisoformat(row[4]),
                    document_count=row[5],
                    file_size=row[6],
                    mime_type=row[7],
                    qdrant_point_ids=json.loads(row[8]) if row[8] else []
                ))
            return files
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_files,
                    SUM(document_count) as total_documents,
                    SUM(file_size) as total_size,
                    MAX(processed_at) as last_processed
                FROM processed_files
            """)
            row = cursor.fetchone()
            
            return {
                "total_files": row[0] or 0,
                "total_documents": row[1] or 0,
                "total_size_bytes": row[2] or 0,
                "last_processed": row[3]
            }
    
    def clear_processed_files(self):
        """Clear all processed files from the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM processed_files")
            conn.commit()


class DocumentProcessor:
    """Handles document processing and embedding with Elasticsearch"""
    
    def __init__(self, 
                 elasticsearch_url: str = "https://172.31.23.77:9200",
                 index_name: str = "documents",
                 embedding_model: str = None,
                 elasticsearch_username: str = None,
                 elasticsearch_password: str = None):
        self.elasticsearch_url = elasticsearch_url
        self.index_name = index_name
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.elasticsearch_username = elasticsearch_username or os.getenv("ELASTICSEARCH_USERNAME", "elastic")
        self.elasticsearch_password = elasticsearch_password or os.getenv("ELASTICSEARCH_PASSWORD", "elastic")
        
        self.tracer = tracer  # Use global tracer
        self.logger = get_correlated_logger(f"{__name__}.DocumentProcessor")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        
        # FIXED: Configure Elasticsearch client with authentication and SSL settings + instrumentation
        self.elasticsearch_client = Elasticsearch(
            [self.elasticsearch_url],
            basic_auth=(self.elasticsearch_username, self.elasticsearch_password),
            verify_certs=False,  # Disable SSL certificate verification
            ssl_show_warn=False,  # Suppress SSL warnings
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3
        )
        
        # FIXED: Use manual spans for Elasticsearch operations to avoid instrumentation warnings
        self.elasticsearch_client = Elasticsearch(
            [self.elasticsearch_url],
            basic_auth=(self.elasticsearch_username, self.elasticsearch_password),
            verify_certs=False,
            ssl_show_warn=False,  # Only suppress SSL warnings, not instrumentation
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3,
            # CRITICAL: Add these for proper instrumentation
            headers={
                "User-Agent": "document-rag-backend/2.0.0"
            }
        )
        
        self.vector_store = ElasticsearchStore(
            es_connection=self.elasticsearch_client,
            index_name=self.index_name,
            embedding=self.embeddings,
            vector_query_field="vector",
            query_field="text"
        )
        
        # Initialize index
        self._ensure_index_exists()
        
        self.logger.info_with_context(
            "Document processor initialized",
            extra_attributes={
                "elasticsearch.url": elasticsearch_url,
                "elasticsearch.username": self.elasticsearch_username,
                "index.name": index_name,
                "embedding.model": self.embedding_model,
                "ssl.verify": False,
                "operation": "processor_init"
            }
        )
    
    def _ensure_index_exists(self):
        """Ensure the Elasticsearch index exists"""
        try:
            with self.tracer.start_as_current_span("elasticsearch_ensure_index") as span:
                span.set_attributes({
                    "service.name": "document-rag-backend",
                    "peer.service": "elasticsearch-database",
                    "external.service.name": "elasticsearch-database",
                    "external.service.type": "vector_database",
                    "db.system": "elasticsearch",
                    "db.operation": "check_index",
                    "db.connection_string": self.elasticsearch_url
                })
                
                # Check if index exists
                index_exists = self.elasticsearch_client.indices.exists(index=self.index_name)
                
                vector_size = int(os.getenv("EMBEDDING_VECTOR_SIZE", "3072"))
                
                if not index_exists:
                    with self.tracer.start_as_current_span("elasticsearch_create_index") as create_span:
                        create_span.set_attributes({
                            "service.name": "document-rag-backend",
                            "peer.service": "elasticsearch-database",
                            "external.service.name": "elasticsearch-database",
                            "external.service.type": "vector_database",
                            "db.system": "elasticsearch",
                            "db.operation": "create_index",
                            "db.collection.name": self.index_name,
                            "vector.size": vector_size
                        })
                        
                        self.logger.info_with_context(
                            "Creating new Elasticsearch index",
                            extra_attributes={
                                "index.name": self.index_name,
                                "vector.size": vector_size,
                                "operation": "index_creation"
                            }
                        )
                        
                        # Create index with vector mapping
                        index_mapping = {
                            "mappings": {
                                "properties": {
                                    "text": {"type": "text"},
                                    "vector": {
                                        "type": "dense_vector",
                                        "dims": vector_size,
                                        "index": True,
                                        "similarity": "cosine"
                                    },
                                    "metadata": {
                                        "type": "object",
                                        "properties": {
                                            "file_id": {"type": "keyword"},
                                            "file_name": {"type": "keyword"},
                                            "file_path": {"type": "keyword"},
                                            "processed_at": {"type": "date"},
                                            "file_size": {"type": "long"},
                                            "mime_type": {"type": "keyword"},
                                            "trace_id": {"type": "keyword"},
                                            "processing_service": {"type": "keyword"}
                                        }
                                    }
                                }
                            }
                        }
                        
                        self.elasticsearch_client.indices.create(
                            index=self.index_name,
                            body=index_mapping
                        )
                else:
                    self.logger.info_with_context(
                        "Using existing Elasticsearch index",
                        extra_attributes={
                            "index.name": self.index_name,
                            "operation": "index_connection"
                        }
                    )
                    
                self.vector_store = ElasticsearchStore(
                    es_connection=self.elasticsearch_client,
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    vector_query_field="vector",
                    query_field="text"
                )
            
        except Exception as e:
            self.logger.error_with_context(
                "Failed to initialize index",
                extra_attributes={
                    "elasticsearch.url": self.elasticsearch_url,
                    "index.name": self.index_name,
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "index_init"
                },
                exc_info=True
            )
            raise
    
    def process_documents(self, documents: List[Document], file_info: Dict) -> List[str]:
        """Process documents with PROPER trace context propagation"""
        with self.tracer.start_as_current_span("process_documents_elasticsearch") as main_span:
            try:
                # Set comprehensive span attributes
                main_span.set_attributes({
                    "service.name": "document-rag-backend",
                    "operation.name": "process_documents",
                    "file.name": file_info.get("file_name", "unknown"),
                    "document.count": len(documents),
                    "vector.database": "elasticsearch",
                    "elasticsearch.index": self.index_name
                })
                
                if not documents:
                    main_span.set_attribute("result", "no_documents")
                    return []
                
                # Document splitting with context preservation
                with self.tracer.start_as_current_span("document_splitting") as split_span:
                    chunk_size = int(os.getenv("CHUNK_SIZE", "3000"))
                    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "300"))
                    
                    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    split_span.set_attributes({
                        "chunks.created": len(chunks),
                        "chunk.size": chunk_size,
                        "chunk.overlap": chunk_overlap
                    })
                    
                    del documents  # Memory cleanup
                    gc.collect()
                
                if not chunks:
                    main_span.set_attribute("result", "no_chunks")
                    return []
                
                # CRITICAL: Add trace context to chunk metadata
                current_trace_id = format(main_span.get_span_context().trace_id, '032x')
                for chunk in chunks:
                    chunk.metadata.update({
                        "trace_id": current_trace_id,
                        "processing_service": "document-rag-backend",
                        "file_id": file_info.get("file_id", "unknown"),
                        "file_name": file_info.get("file_name", "unknown"),
                        "processed_at": datetime.now().isoformat()
                    })
                
                # Process in batches with proper context propagation
                point_ids = []
                batch_size = int(os.getenv("BATCH_SIZE", "5"))
                
                with self.tracer.start_as_current_span("elasticsearch_batch_processing") as batch_parent_span:
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        
                        # CRITICAL: Each batch operation gets its own span under parent context
                        with self.tracer.start_as_current_span(f"elasticsearch_batch_{i//batch_size + 1}") as batch_span:
                            batch_span.set_attributes({
                                "batch.number": i//batch_size + 1,
                                "batch.size": len(batch),
                                "elasticsearch.operation": "add_documents",
                                "elasticsearch.index": self.index_name
                            })
                            
                            try:
                                # Generate batch IDs
                                batch_doc_ids = [
                                    hashlib.md5(f"{file_info['file_path']}_{i+j}_{chunk.page_content[:100]}".encode()).hexdigest()
                                    for j, chunk in enumerate(batch)
                                ]
                                
                                # CRITICAL: The instrumented Elasticsearch client will now automatically
                                # create child spans for each operation
                                self.vector_store.add_documents(batch, ids=batch_doc_ids)
                                point_ids.extend(batch_doc_ids)
                                
                                batch_span.set_attributes({
                                    "documents.created": len(batch_doc_ids),
                                    "batch.status": "success"
                                })
                                
                                self.logger.info_with_context(
                                    f"Processed batch {i//batch_size + 1} successfully",
                                    extra_attributes={
                                        "batch.number": i//batch_size + 1,
                                        "documents.created": len(batch_doc_ids),
                                        "elasticsearch.index": self.index_name
                                    }
                                )
                                
                            except Exception as batch_error:
                                batch_span.record_exception(batch_error)
                                batch_span.set_attribute("batch.status", "failed")
                                self.logger.error_with_context(
                                    f"Batch {i//batch_size + 1} failed",
                                    extra_attributes={
                                        "batch.number": i//batch_size + 1,
                                        "error.message": str(batch_error)
                                    },
                                    exc_info=True
                                )
                                continue
                            finally:
                                del batch, batch_doc_ids
                                gc.collect()
                            
                            time.sleep(0.2)  # Rate limiting
                
                # Update metrics and final span attributes
                documents_processed.add(len(chunks), {"database": "elasticsearch"})
                main_span.set_attributes({
                    "documents.processed": len(chunks),
                    "documents.created": len(point_ids),
                    "result": "success"
                })
                
                self.logger.info_with_context(
                    "Document processing completed successfully",
                    extra_attributes={
                        "file.name": file_info.get("file_name"),
                        "chunks.processed": len(chunks),
                        "documents.created": len(point_ids),
                        "elasticsearch.index": self.index_name
                    }
                )
                
                return point_ids
                
            except Exception as e:
                main_span.record_exception(e)
                main_span.set_status(Status(StatusCode.ERROR, str(e)))
                processing_errors.add(1, {"operation": "elasticsearch_processing"})
                
                self.logger.error_with_context(
                    "Document processing failed",
                    extra_attributes={
                        "file.name": file_info.get("file_name"),
                        "error.type": type(e).__name__,
                        "error.message": str(e)
                    },
                    exc_info=True
                )
                raise
            finally:
                gc.collect()


class BackendService:
    """Main backend service with complete trace correlation and memory safety"""
    
    def __init__(self):
        # CRITICAL: Use the module-level tracer and meter, not create new ones
        self.tracer = tracer  # Use the initialized global tracer
        self.meter = meter    # Use the initialized global meter
        self.service_name = "document-rag-backend"
        self.orchestrator_url = orchestrator_url
        
        # Configuration
        self.google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        # Skip local files for now to prevent issues
        self.local_watch_dirs = []
        self.scan_interval = int(os.getenv("SCAN_INTERVAL", "30"))
        
        # Initialize components
        self.fingerprint_db = FileFingerprintDatabase()
        self.processor = DocumentProcessor()
        self.google_drive_loader = None
        
        # Service state
        self.is_running = False
        self.stats = {
            "service_started": datetime.now(),
            "files_processed": 0,
            "documents_created": 0,
            "last_scan": None,
            "errors": []
        }
        
        # Add correlated logger for the service
        self.logger = get_correlated_logger(f"{__name__}.BackendService")
        
        # Initialize Google Drive loader if configured - with safety checks
        if self.google_drive_folder_id:
            try:
                credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
                token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
                
                # Check if credentials exist
                if not os.path.exists(credentials_path):
                    self.logger.warning_with_context(
                        "Google credentials file not found - Google Drive disabled",
                        extra_attributes={
                            "credentials_path": credentials_path,
                            "operation": "service_init"
                        }
                    )
                    self.google_drive_loader = None
                else:
                    self.google_drive_loader = GoogleDriveMasterLoader(
                        folder_id=self.google_drive_folder_id,
                        credentials_path=credentials_path,
                        token_path=token_path,
                        split=False
                    )
                    
                    self.logger.info_with_context(
                        "Google Drive loader initialized",
                        extra_attributes={
                            "google_drive.folder_id": self.google_drive_folder_id,
                            "operation": "service_init"
                        }
                    )
            except Exception as e:
                self.logger.error_with_context(
                    "Google Drive loader initialization failed",
                    extra_attributes={
                        "google_drive.folder_id": self.google_drive_folder_id,
                        "error.type": type(e).__name__,
                        "error.message": str(e),
                        "operation": "service_init"
                    },
                    exc_info=True
                )
                self.google_drive_loader = None
        
        self.logger.info_with_context(
            "Backend service initialized",
            extra_attributes={
                "service.name": self.service_name,
                "orchestrator.url": self.orchestrator_url,
                "scan_interval": self.scan_interval,
                "local_watch_dirs": 0,
                "google_drive.enabled": self.google_drive_loader is not None,
                "operation": "service_init"
            }
        )

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file safely"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error_with_context(
                "Failed to calculate file hash",
                extra_attributes={
                    "file.path": file_path,
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "file_hash"
                }
            )
            return ""

    def cleanup_extracted_images(self, documents: List[Document]):
        """Clean up locally extracted images after processing"""
        try:
            image_paths_to_clean = set()
            
            for doc in documents:
                img_path = doc.metadata.get("image_path")
                if img_path and os.path.exists(img_path):
                    image_paths_to_clean.add(img_path)
                
                related_images = doc.metadata.get("related_images", [])
                for rel_img in related_images:
                    if isinstance(rel_img, str) and os.path.exists(rel_img):
                        image_paths_to_clean.add(rel_img)
            
            cleaned_count = 0
            for img_path in image_paths_to_clean:
                try:
                    if any(dir_name in img_path for dir_name in ['pdf_images', 'pptx_images', 'docx_images']):
                        os.remove(img_path)
                        cleaned_count += 1
                except OSError as e:
                    self.logger.debug_with_context(
                        "Could not remove image file",
                        extra_attributes={
                            "image.path": img_path,
                            "error.message": str(e),
                            "operation": "image_cleanup"
                        }
                    )
            
            if cleaned_count > 0:
                self.logger.info_with_context(
                    "Cleaned up extracted image files",
                    extra_attributes={
                        "images.cleaned": cleaned_count,
                        "operation": "image_cleanup"
                    }
                )
                
        except Exception as e:
            self.logger.warning_with_context(
                "Error during image cleanup",
                extra_attributes={
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "image_cleanup"
                }
            )

    def scan_google_drive(self) -> List[Dict]:
        """Scan Google Drive for new files with memory safety"""
        if not self.google_drive_loader:
            self.logger.debug_with_context(
                "Google Drive loader not available",
                extra_attributes={
                    "operation": "google_drive_scan"
                }
            )
            return []
        
        new_files = []
        try:
            self.logger.info_with_context(
                "Starting Google Drive scan",
                extra_attributes={
                    "google_drive.folder_id": self.google_drive_folder_id,
                    "operation": "google_drive_scan"
                }
            )
            
            # Add timeout and retry logic for Google Drive API
            max_retries = 3
            retry_count = 0
            files = []
            
            while retry_count < max_retries:
                try:
                    files = self.google_drive_loader._list_files(self.google_drive_folder_id)
                    break
                except Exception as api_error:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise api_error
                    
                    self.logger.warning_with_context(
                        f"Google Drive API retry {retry_count}/{max_retries}",
                        extra_attributes={
                            "retry.count": retry_count,
                            "retry.max": max_retries,
                            "error.message": str(api_error),
                            "operation": "google_drive_scan"
                        }
                    )
                    time.sleep(2 ** retry_count)  # Exponential backoff
            
            self.logger.info_with_context(
                "Google Drive files listed",
                extra_attributes={
                    "files.total": len(files),
                    "operation": "google_drive_scan"
                }
            )
            
            for file_info in files:
                file_id = file_info["id"]
                file_name = file_info["name"]
                mime_type = file_info.get("mimeType", "unknown")
                
                # Skip Google Apps files that we can't process directly
                if mime_type.startswith("application/vnd.google-apps.") and mime_type not in [
                    "application/vnd.google-apps.document",
                    "application/vnd.google-apps.spreadsheet", 
                    "application/vnd.google-apps.presentation"
                ]:
                    continue
                
                file_path = f"gdrive://{file_id}/{file_name}"
                file_hash = hashlib.md5(f"{file_id}_{file_name}_{mime_type}".encode()).hexdigest()
                
                if (self.fingerprint_db.is_file_processed(file_path, file_hash) or 
                    self.fingerprint_db.is_file_in_elasticsearch(file_path, self.processor.elasticsearch_client)):
                    self.logger.debug_with_context(
                        "Skipping already processed Google Drive file",
                        extra_attributes={
                            "file.name": file_name,
                            "file.id": file_id,
                            "operation": "google_drive_scan"
                        }
                    )
                    continue
                
                self.logger.info_with_context(
                    "Found new Google Drive file",
                    extra_attributes={
                        "file.name": file_name,
                        "file.type": mime_type,
                        "file.id": file_id,
                        "operation": "google_drive_scan"
                    }
                )
                
                new_files.append({
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "file_size": 0,
                    "mime_type": mime_type,
                    "source": "google_drive",
                    "drive_file_info": file_info
                })
                
        except Exception as e:
            self.logger.error_with_context(
                "Google Drive scan failed",
                extra_attributes={
                    "google_drive.folder_id": self.google_drive_folder_id,
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "google_drive_scan"
                },
                exc_info=True
            )
            self.stats["errors"].append(f"Google Drive scan error: {e}")
        
        return new_files

    def scan_local_directories(self) -> List[Dict]:
        """Scan local directories for new files - currently disabled for safety"""
        # Disabled to prevent segfaults from file system operations
        self.logger.debug_with_context(
            "Local directory scanning disabled for safety",
            extra_attributes={
                "operation": "local_scan"
            }
        )
        return []

    def scan_and_process(self):
        """Enhanced scan and process with memory safety and error recovery"""
        with self.tracer.start_as_current_span("scan_and_process") as span:
            try:
                start_time = time.time()
                
                self.logger.info_with_context(
                    "Starting scan cycle",
                    extra_attributes={
                        "operation": "scan_cycle"
                    }
                )
                
                # Set comprehensive service attributes for correlation
                span.set_attributes({
                    "service.name": "document-rag-backend",
                    "service.version": "2.0.0",
                    "service.namespace": "document-rag",
                    "deployment.environment": os.getenv("OTEL_ENVIRONMENT", "production"),
                    "operation.name": "scan_and_process",
                    "scan.interval": self.scan_interval
                })
                
                # Create trace correlation ID
                trace_id = span.get_span_context().trace_id
                correlation_id = format(trace_id, '032x')[:16]
                
                self.logger.info_with_context(
                    "Scan cycle correlation established",
                    extra_attributes={
                        "correlation.id": correlation_id,
                        "operation": "scan_cycle"
                    }
                )
                
                new_files = []
                
                # Scan Google Drive with enhanced error handling
                if self.google_drive_loader:
                    with self.tracer.start_as_current_span("google_drive_scan") as gdrive_span:
                        gdrive_span.set_attributes({
                            "service.name": "document-rag-backend",
                            "peer.service": "google-drive-api",
                            "external.service.name": "google-drive-api",
                            "external.service.type": "file_storage_api",
                            "operation.name": "scan_files",
                            "correlation.id": correlation_id
                        })
                        
                        try:
                            google_drive_files = self.scan_google_drive()
                            gdrive_span.set_attribute("files_found", len(google_drive_files))
                            new_files.extend(google_drive_files)
                            
                            # Track external API call
                            if google_drive_files:
                                external_api_calls.add(1, {"service": "google-drive-api", "operation": "list_files"})
                                
                        except Exception as gdrive_error:
                            gdrive_span.record_exception(gdrive_error)
                            self.logger.error_with_context(
                                "Google Drive scan failed in cycle",
                                extra_attributes={
                                    "error.type": type(gdrive_error).__name__,
                                    "error.message": str(gdrive_error),
                                    "operation": "scan_cycle"
                                },
                                exc_info=True
                            )
                            # Continue with other operations instead of failing
                
                # Skip local directories scan for safety
                
                self.stats["last_scan"] = datetime.now()
                span.set_attribute("total_files_found", len(new_files))
                
                if not new_files:
                    # Create keepalive spans to maintain service visibility
                    with self.tracer.start_as_current_span("no_files_maintenance") as maintenance_span:
                        maintenance_span.set_attributes({
                            "service.name": "document-rag-backend",
                            "maintenance.type": "periodic_keepalive",
                            "scan.result": "no_new_files",
                            "correlation.id": correlation_id
                        })
                        
                        # Elasticsearch keepalive
                        with self.tracer.start_as_current_span("keepalive_elasticsearch_ping") as es_ping:
                            es_ping.set_attributes({
                                "service.name": "document-rag-backend",
                                "peer.service": "elasticsearch-database",
                                "external.service.name": "elasticsearch-database",
                                "external.service.type": "vector_database",
                                "db.system": "elasticsearch",
                                "db.operation": "health_ping",
                                "ping.purpose": "service_map_visibility"
                            })
                            
                            try:
                                cluster_health = self.processor.elasticsearch_client.cluster.health()
                                es_ping.set_attribute("elasticsearch.responsive", True)
                                es_ping.set_attribute("elasticsearch.status", cluster_health.get("status", "unknown"))
                                es_ping.set_attribute("elasticsearch.indices_count", cluster_health.get("number_of_indices", 0))
                            except Exception as ping_error:
                                es_ping.record_exception(ping_error)
                                es_ping.set_attribute("elasticsearch.responsive", False)

                        
                        # Google Drive keepalive if configured
                        if self.google_drive_loader:
                            with self.tracer.start_as_current_span("keepalive_gdrive_ping") as gdrive_ping:
                                gdrive_ping.set_attributes({
                                    "service.name": "document-rag-backend",
                                    "peer.service": "google-drive-api",
                                    "external.service.name": "google-drive-api",
                                    "external.service.type": "file_storage_api",
                                    "ping.purpose": "service_map_visibility"
                                })
                                
                                self.logger.debug_with_context(
                                    "Google Drive service keepalive ping",
                                    extra_attributes={
                                        "operation": "service_keepalive"
                                    }
                                )
                    
                    # Only log every 10th "no files" scan to reduce noise
                    if hasattr(self, '_no_files_scan_count'):
                        self._no_files_scan_count += 1
                    else:
                        self._no_files_scan_count = 1
                    
                    if self._no_files_scan_count % 10 == 0:
                        self.logger.info_with_context(
                            f"No new files found in {self._no_files_scan_count} scan cycles - service keepalive active",
                            extra_attributes={
                                "correlation.id": correlation_id,
                                "operation": "scan_cycle",
                                "scan.result": "no_files",
                                "scan.cycles_since_files": self._no_files_scan_count,
                                "keepalive.generated": True
                            }
                        )
                    scan_duration.record(time.time() - start_time, {"result": "no_files"})
                    return
                
                self.logger.info_with_context(
                    "New files found for processing",
                    extra_attributes={
                        "files.count": len(new_files),
                        "correlation.id": correlation_id,
                        "operation": "scan_cycle"
                    }
                )
                
                # Process each file with enhanced error handling
                processed_count = 0
                failed_count = 0
                
                for file_info in new_files:
                    if not self.is_running:
                        self.logger.info_with_context(
                            "Service stopping, halting file processing",
                            extra_attributes={
                                "operation": "scan_cycle"
                            }
                        )
                        break
                    
                    with self.tracer.start_as_current_span("process_single_file") as file_span:
                        file_span.set_attributes({
                            "service.name": "document-rag-backend",
                            "service.version": "2.0.0",
                            "file_name": file_info.get("file_name", "unknown"),
                            "file_source": file_info.get("source", "unknown"),
                            "file_type": file_info.get("mime_type", "unknown"),
                            "correlation.id": correlation_id,
                            "operation.name": "process_single_file"
                        })
                        
                        try:
                            self.logger.info_with_context(
                                "Processing file",
                                extra_attributes={
                                    "file.name": file_info['file_name'],
                                    "file.number": f"{processed_count + 1}/{len(new_files)}",
                                    "operation": "file_processing"
                                }
                            )
                            
                            if self.process_file(file_info):
                                processed_count += 1
                                file_span.set_attribute("result", "success")
                                
                                self.logger.info_with_context(
                                    "File processed successfully",
                                    extra_attributes={
                                        "file.name": file_info['file_name'],
                                        "operation": "file_processing",
                                        "status": "success"
                                    }
                                )
                            else:
                                failed_count += 1
                                file_span.set_attribute("result", "failed")
                                
                                self.logger.warning_with_context(
                                    "File processing failed",
                                    extra_attributes={
                                        "file.name": file_info['file_name'],
                                        "operation": "file_processing",
                                        "status": "failed"
                                    }
                                )
                                
                        except Exception as e:
                            failed_count += 1
                            file_span.record_exception(e)
                            file_span.set_status(Status(StatusCode.ERROR, str(e)))
                            processing_errors.add(1, {"operation": "file_processing"})
                            
                            self.logger.error_with_context(
                                "File processing error",
                                extra_attributes={
                                    "file.name": file_info.get('file_name'),
                                    "error.type": type(e).__name__,
                                    "error.message": str(e),
                                    "operation": "file_processing"
                                },
                                exc_info=True
                            )
                            continue
                        finally:
                            # Force cleanup after each file
                            gc.collect()
                        
                        # Longer sleep between files to prevent system overload
                        time.sleep(3)
                
                # Record final metrics and correlation
                duration = time.time() - start_time
                scan_duration.record(duration, {"result": "completed"})
                files_processed.add(processed_count, {"result": "success"})
                files_processed.add(failed_count, {"result": "failed"})
                
                span.set_attributes({
                    "processed_count": processed_count,
                    "failed_count": failed_count,
                    "scan_duration": duration,
                    "correlation.id": correlation_id
                })
                
                self.logger.info_with_context(
                    "Scan cycle completed",
                    extra_attributes={
                        "files.processed": processed_count,
                        "files.failed": failed_count,
                        "scan.duration": duration,
                        "correlation.id": correlation_id,
                        "operation": "scan_cycle"
                    }
                )
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                processing_errors.add(1, {"operation": "scan_cycle"})
                
                self.logger.error_with_context(
                    "Scan cycle error",
                    extra_attributes={
                        "error.type": type(e).__name__,
                        "error.message": str(e),
                        "operation": "scan_cycle"
                    },
                    exc_info=True
                )
                self.stats["errors"].append(f"Scan cycle error: {e}")
            finally:
                # Ensure cleanup
                gc.collect()

    def process_file(self, file_info: Dict) -> bool:
        """Process a single file with memory safety and error recovery"""
        with self.tracer.start_as_current_span("process_document_file") as span:
            span.set_attributes({
                "service.name": "document-rag-backend",
                "service.version": "2.0.0",
                "file_id": file_info.get("file_id", "unknown"),
                "file_name": file_info.get("file_name", "unknown"),
                "file_source": file_info.get("source", "unknown"),
                "file_type": file_info.get("mime_type", "unknown"),
                "operation.name": "process_document_file"
            })

            span.set_attribute("transaction.name", f"process_file_{file_info.get('source', 'unknown')}")
            span.set_attribute("transaction.type", "background")
            
            documents = None
            temp_file = None
            
            try:
                file_path = file_info["file_path"]
                file_name = file_info["file_name"]
                source = file_info["source"]
                
                self.logger.info_with_context(
                    "Starting file processing",
                    extra_attributes={
                        "file.name": file_name,
                        "file.source": source,
                        "operation": "file_processing"
                    }
                )
                
                if source == "local":
                    with self.tracer.start_as_current_span("load_local_file") as load_span:
                        load_span.set_attributes({
                            "service.name": "document-rag-backend",
                            "operation.name": "load_local_file"
                        })
                        documents = load_file(file_path)
                        if documents:
                            self.cleanup_extracted_images(documents)
                        
                elif source == "google_drive":
                    with self.tracer.start_as_current_span("download_google_drive_file") as download_span:
                        download_span.set_attributes({
                            "service.name": "document-rag-backend",
                            "peer.service": "google-drive-api",
                            "external.service.name": "google-drive-api",
                            "external.service.type": "file_download_api",
                            "operation.name": "download_file"
                        })
                        
                        try:
                            drive_file_info = file_info["drive_file_info"]
                            temp_file = self.google_drive_loader._download_file(drive_file_info)
                            download_span.set_attribute("temp_file", os.path.basename(temp_file) if temp_file else "none")
                            
                            if temp_file and os.path.exists(temp_file):
                                documents = load_file(temp_file)
                                if documents:
                                    self.google_drive_loader._process_docs(documents, drive_file_info)
                                file_info["file_size"] = os.path.getsize(temp_file)
                            
                            # Track external API call
                            external_api_calls.add(1, {"service": "google-drive-api", "operation": "download_file"})
                            
                        except Exception as download_error:
                            download_span.record_exception(download_error)
                            self.logger.error_with_context(
                                "Google Drive download failed",
                                extra_attributes={
                                    "file.name": file_name,
                                    "error.type": type(download_error).__name__,
                                    "error.message": str(download_error),
                                    "operation": "file_processing"
                                },
                                exc_info=True
                            )
                            return False
                
                if not documents:
                    self.logger.warning_with_context(
                        "No documents extracted from file",
                        extra_attributes={
                            "file.name": file_name,
                            "operation": "file_processing"
                        }
                    )
                    span.set_status(Status(StatusCode.ERROR, "No documents extracted"))
                    return False
                
                self.logger.info_with_context(
                    "Documents extracted from file",
                    extra_attributes={
                        "file.name": file_name,
                        "documents.count": len(documents),
                        "operation": "file_processing"
                    }
                )
                span.set_attribute("documents_extracted", len(documents))
                
                # Process documents with correlation
                point_ids = self.processor.process_documents(documents, file_info)
                
                # Mark as processed with trace correlation
                with self.tracer.start_as_current_span("mark_file_processed") as mark_span:
                    mark_span.set_attributes({
                        "service.name": "document-rag-backend",
                        "operation.name": "mark_file_processed"
                    })
                    processed_file = ProcessedFile(
                        file_id=file_info["file_id"],
                        file_name=file_name,
                        file_path=file_path,
                        file_hash=file_info["file_hash"],
                        processed_at=datetime.now(),
                        document_count=len(documents),
                        file_size=file_info["file_size"],
                        mime_type=file_info["mime_type"],
                        qdrant_point_ids=point_ids  # Keep this field name for compatibility
                    )
                    
                    self.fingerprint_db.mark_file_processed(processed_file)
                
                # Update stats
                self.stats["files_processed"] += 1
                self.stats["documents_created"] += len(documents)
                
                span.set_attributes({
                    "result": "success",
                    "documents_created": len(point_ids)
                })
                
                self.logger.info_with_context(
                    "File processing completed successfully",
                    extra_attributes={
                        "file.name": file_name,
                        "documents.count": len(documents),
                        "documents.created": len(point_ids),
                        "operation": "file_processing",
                        "status": "success"
                    }
                )
                self.logger.info_with_context(
                    "File processing completed successfully",
                    extra_attributes={
                        "file.name": file_name,
                        "documents.count": len(documents),
                        "documents.created": len(point_ids),
                        "operation": "file_processing",
                        "status": "success"
                    }
                )
                
                return True
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                self.logger.error_with_context(
                    "File processing failed",
                    extra_attributes={
                        "file.name": file_info.get('file_name', 'unknown'),
                        "error.type": type(e).__name__,
                        "error.message": str(e),
                        "operation": "file_processing"
                    },
                    exc_info=True
                )
                self.stats["errors"].append(f"File processing error: {e}")
                return False
            
            finally:
                # Comprehensive cleanup
                try:
                    # Clean up temporary file
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                            self.logger.debug_with_context(
                                "Cleaned up temporary file",
                                extra_attributes={
                                    "temp_file": os.path.basename(temp_file),
                                    "operation": "file_processing"
                                }
                            )
                        except Exception as cleanup_error:
                            self.logger.debug_with_context(
                                "Could not remove temporary file",
                                extra_attributes={
                                    "temp_file": temp_file,
                                    "error.message": str(cleanup_error),
                                    "operation": "file_processing"
                                }
                            )
                    
                    # Clean up extracted images if local processing
                    if documents and file_info.get("source") == "local":
                        self.cleanup_extracted_images(documents)
                    
                    # Force garbage collection
                    if documents:
                        del documents
                    if temp_file:
                        temp_file = None
                    gc.collect()
                    
                except Exception as final_cleanup_error:
                    self.logger.warning_with_context(
                        "Error during final cleanup",
                        extra_attributes={
                            "error.type": type(final_cleanup_error).__name__,
                            "error.message": str(final_cleanup_error),
                            "operation": "file_processing"
                        }
                    )

    async def send_heartbeat(self):
        """Send heartbeat with minimal noise - no spans for successful heartbeats"""
        try:
            async with TracedHTTPXClient(service_name="document-rag-backend") as client:
                response = await client.post(
                    f"{self.orchestrator_url}/heartbeat",
                    json={
                        "service": "document-rag-backend",
                        "status": "healthy",
                        "stats": {
                            "files_processed": self.stats["files_processed"],
                            "documents_created": self.stats["documents_created"]
                        }
                    },
                    timeout=5.0
                )
                # Don't log successful heartbeats to reduce noise
                    
        except Exception as e:
            # Only log heartbeat failures as debug level
            pass  # Failures will be handled by the calling heartbeat_loop

    async def check_orchestrator_health(self):
        """Check orchestrator health with timeout"""
        try:
            async with TracedHTTPXClient(service_name="document-rag-backend") as client:
                with self.tracer.start_as_current_span("check_orchestrator") as span:
                    span.set_attributes({
                        "service.name": "document-rag-backend",
                        "peer.service": "document-rag-orchestrator",
                        "external.service.name": "document-rag-orchestrator",
                        "external.service.type": "orchestrator_api",
                        "operation.name": "health_check"
                    })
                    response = await client.get(
                        f"{self.orchestrator_url}/health",
                        timeout=3.0  # Shorter timeout
                    )
                    
                    is_healthy = response.status_code == 200
                    span.set_attribute("orchestrator.healthy", is_healthy)
                    span.set_attribute("response.status_code", response.status_code)
                    
                    if is_healthy:
                        self.logger.debug_with_context(
                            "Orchestrator health check passed",
                            extra_attributes={
                                "orchestrator.url": self.orchestrator_url,
                                "orchestrator.healthy": is_healthy,
                                "response.status_code": response.status_code,
                                "operation": "health_check"
                            }
                        )
                    
                    return is_healthy
        except Exception as e:
            # Don't log as error since orchestrator might not be running
            self.logger.debug_with_context(
                "Orchestrator not reachable (may not be running)",
                extra_attributes={
                    "orchestrator.url": self.orchestrator_url,
                    "error.type": type(e).__name__,
                    "operation": "health_check"
                }
            )
            return False

    async def start_monitoring(self):
        """Start the continuous monitoring service with enhanced stability"""
        with self.tracer.start_as_current_span("start_monitoring_service") as span:
            self.is_running = True
            span.set_attributes({
                "service.name": "document-rag-backend",
                "service.version": "2.0.0",
                "scan_interval": self.scan_interval,
                "operation.name": "start_monitoring"
            })
            
            self.logger.info_with_context(
                "Starting monitoring loop",
                extra_attributes={
                    "scan_interval": self.scan_interval,
                    "operation": "monitoring_start"
                }
            )
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self.heartbeat_loop())
            
            # Main monitoring loop with enhanced error recovery
            loop_count = 0
            consecutive_errors = 0
            max_consecutive_errors = 3
            
            while self.is_running:
                try:
                    # Create a span for each monitoring cycle
                    with self.tracer.start_as_current_span("monitoring_cycle") as cycle_span:
                        loop_count += 1
                        cycle_span.set_attributes({
                            "service.name": "document-rag-backend",
                            "service.version": "2.0.0",
                            "monitoring.cycle_number": loop_count,
                            "monitoring.uptime_seconds": (datetime.now() - self.stats["service_started"]).total_seconds(),
                            "operation.name": "monitoring_cycle"
                        })
                        
                        try:
                            # Perform the scan and process
                            await asyncio.to_thread(self.scan_and_process)
                            consecutive_errors = 0  # Reset error counter on success
                            cycle_span.set_attribute("cycle.completed", True)
                            
                        except Exception as scan_error:
                            consecutive_errors += 1
                            cycle_span.record_exception(scan_error)
                            
                            self.logger.error_with_context(
                                "Scan error in monitoring cycle",
                                extra_attributes={
                                    "cycle.number": loop_count,
                                    "consecutive_errors": consecutive_errors,
                                    "error.type": type(scan_error).__name__,
                                    "error.message": str(scan_error),
                                    "operation": "monitoring_cycle"
                                },
                                exc_info=True
                            )
                            
                            # If too many consecutive errors, take a longer break
                            if consecutive_errors >= max_consecutive_errors:
                                self.logger.warning_with_context(
                                    "Too many consecutive errors, taking extended break",
                                    extra_attributes={
                                        "consecutive_errors": consecutive_errors,
                                        "break_duration": 60,
                                        "operation": "monitoring_cycle"
                                    }
                                )
                                await asyncio.sleep(60)
                                consecutive_errors = 0  # Reset counter after break
                    
                    # Sleep between cycles with periodic status updates
                    for sleep_iteration in range(self.scan_interval):
                        if not self.is_running:
                            break
                        
                        # Generate periodic keepalive spans
                        if sleep_iteration % 15 == 0:  # Every 15 seconds
                            with self.tracer.start_as_current_span("service_keepalive") as keepalive_span:
                                keepalive_span.set_attributes({
                                    "service.name": "document-rag-backend",
                                    "service.status": "idle_monitoring",
                                    "keepalive.iteration": sleep_iteration,
                                    "next_scan_in_seconds": self.scan_interval - sleep_iteration,
                                    "service.uptime_seconds": (datetime.now() - self.stats["service_started"]).total_seconds()
                                })
                        
                        await asyncio.sleep(1)
                        
                except asyncio.CancelledError:
                    self.logger.info_with_context(
                        "Monitoring loop cancelled",
                        extra_attributes={
                            "operation": "monitoring_loop"
                        }
                    )
                    break
                except Exception as e:
                    # Handle any other unexpected errors
                    with self.tracer.start_as_current_span("monitoring_error") as error_span:
                        error_span.set_attributes({
                            "service.name": "document-rag-backend",
                            "error.in_monitoring": True,
                            "monitoring.cycle_number": loop_count
                        })
                        error_span.record_exception(e)
                        
                        self.logger.error_with_context(
                            "Monitoring error",
                            extra_attributes={
                                "error.type": type(e).__name__,
                                "error.message": str(e),
                                "cycle.number": loop_count,
                                "operation": "monitoring_loop"
                            },
                            exc_info=True
                        )
                        
                        self.stats["errors"].append(f"Monitoring loop error: {e}")
                        processing_errors.add(1, {"operation": "monitoring_loop"})
                        await asyncio.sleep(10)
            
            # Cancel heartbeat task
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Create shutdown span
            with self.tracer.start_as_current_span("service_shutdown") as shutdown_span:
                shutdown_span.set_attributes({
                    "service.name": "document-rag-backend",
                    "shutdown.reason": "monitoring_stopped",
                    "service.total_cycles": loop_count,
                    "service.uptime_seconds": (datetime.now() - self.stats["service_started"]).total_seconds()
                })
                
                self.logger.info_with_context(
                    "Monitoring loop stopped",
                    extra_attributes={
                        "total_cycles": loop_count,
                        "operation": "monitoring_stop"
                    }
                )

    async def heartbeat_loop(self):
        """Send periodic heartbeats to orchestrator with minimal logging"""
        heartbeat_count = 0
        while self.is_running:
            heartbeat_count += 1
            
            # Only create spans for failed heartbeats to reduce noise
            try:
                await self.send_heartbeat()
                
                # Only log every 40 heartbeats (20 minutes) for status
                if heartbeat_count % 40 == 0:
                    with self.tracer.start_as_current_span("heartbeat_status") as status_span:
                        status_span.set_attributes({
                            "service.name": "document-rag-backend",
                            "heartbeat.count": heartbeat_count,
                            "service.uptime_minutes": (datetime.now() - self.stats["service_started"]).total_seconds() / 60,
                            "operation.name": "heartbeat_status"
                        })
                        
                        self.logger.info_with_context(
                            "Backend service heartbeat status",
                            extra_attributes={
                                "heartbeat.count": heartbeat_count,
                                "service.uptime_minutes": (datetime.now() - self.stats["service_started"]).total_seconds() / 60,
                                "operation": "heartbeat_status"
                            }
                        )
                        
            except Exception as e:
                # Only create spans and log for failed heartbeats
                with self.tracer.start_as_current_span("heartbeat_failed") as failed_span:
                    failed_span.set_attributes({
                        "service.name": "document-rag-backend",
                        "heartbeat.number": heartbeat_count,
                        "heartbeat.failed": True,
                        "error.type": type(e).__name__
                    })
                    failed_span.record_exception(e)
            
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break

    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.is_running = False
        
        self.logger.info_with_context(
            "Stopping monitoring",
            extra_attributes={
                "operation": "monitoring_stop"
            }
        )

    def get_status(self) -> Dict:
        """Get service status with trace correlation"""
        db_stats = self.fingerprint_db.get_stats()
        recent_files = self.fingerprint_db.get_processed_files(limit=5)
        
        # Add trace context to status
        current_span = trace.get_current_span()
        trace_info = {}
        if current_span != trace.INVALID_SPAN:
            span_context = current_span.get_span_context()
            trace_info = {
                "trace_id": format(span_context.trace_id, '032x'),
                "span_id": format(span_context.span_id, '016x')
            }
        
        return {
            "service": {
                "name": "document-rag-backend",
                "version": "2.0.0",
                "is_running": self.is_running,
                "started_at": self.stats["service_started"].isoformat(),
                "uptime_seconds": (datetime.now() - self.stats["service_started"]).total_seconds(),
                "last_scan": self.stats["last_scan"].isoformat() if self.stats["last_scan"] else None,
                "scan_interval": self.scan_interval,
                "parent": parent_service,
                "orchestrator_url": self.orchestrator_url
            },
            "processing": {
                "files_processed_session": self.stats["files_processed"],
                "documents_created_session": self.stats["documents_created"],
                "total_files_processed": db_stats["total_files"],
                "total_documents_created": db_stats["total_documents"],
                "total_size_bytes": db_stats["total_size_bytes"],
                "last_processed": db_stats["last_processed"]
            },
            "configuration": {
                "google_drive_folder_id": self.google_drive_folder_id,
                "local_watch_dirs": [],  # Disabled for safety
                "elasticsearch_url": self.processor.elasticsearch_url,
                "index_name": self.processor.index_name,
                "embedding_model": self.processor.embedding_model,
                "chunk_size": int(os.getenv("CHUNK_SIZE", "3000")),
                "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "300")),
                "batch_size": int(os.getenv("BATCH_SIZE", "5"))
            },
            "recent_files": [
                {
                    "file_name": f.file_name,
                    "processed_at": f.processed_at.isoformat(),
                    "document_count": f.document_count,
                    "file_size": f.file_size
                }
                for f in recent_files
            ],
            "errors": self.stats["errors"][-10:],
            "trace_context": trace_info
        }


# Global service instance
service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management with OpenTelemetry correlation"""
    global service
    
    # Startup with immediate trace generation
    with tracer.start_as_current_span("backend_service_startup") as startup_span:
        startup_span.set_attributes({
            "service.name": "document-rag-backend",
            "service.version": "2.0.0",
            "service.namespace": "document-rag-system",
            "lifecycle.phase": "startup",
            "startup.timestamp": datetime.now().isoformat()
        })
        
        logger.info_with_context(
            "Starting Backend Service",
            extra_attributes={
                "operation": "service_startup"
            }
        )
        
        try:
            # Initialize service with trace visibility
            with tracer.start_as_current_span("create_backend_service") as create_span:
                create_span.set_attributes({
                    "service.name": "document-rag-backend",
                    "operation.name": "service_creation"
                })
                
                service = BackendService()
                create_span.set_attribute("service.initialized", True)
            
            # Check orchestrator health with proper spans
            with tracer.start_as_current_span("startup_health_checks") as health_span:
                health_span.set_attributes({
                    "service.name": "document-rag-backend",
                    "operation.name": "startup_health_checks"
                })
                
                try:
                    orchestrator_healthy = await service.check_orchestrator_health()
                    health_span.set_attribute("orchestrator.healthy", orchestrator_healthy)
                    
                    # Test Elasticsearch during startup
                    with tracer.start_as_current_span("startup_elasticsearch_test") as es_test_span:
                        es_test_span.set_attributes({
                            "service.name": "document-rag-backend",
                            "peer.service": "elasticsearch-database",
                            "external.service.name": "elasticsearch-database",
                            "external.service.type": "vector_database",
                            "db.system": "elasticsearch",
                            "test.phase": "startup"
                        })
                        
                        try:
                            cluster_health = service.processor.elasticsearch_client.cluster.health()
                            es_test_span.set_attribute("elasticsearch.available", True)
                            es_test_span.set_attribute("elasticsearch.status", cluster_health.get("status", "unknown"))
                            es_test_span.set_attribute("elasticsearch.indices_count", cluster_health.get("number_of_indices", 0))
                            
                            logger.info_with_context(
                                "Elasticsearch connection verified at startup",
                                extra_attributes={
                                    "elasticsearch.status": cluster_health.get("status"),
                                    "elasticsearch.indices": cluster_health.get("number_of_indices", 0),
                                    "operation": "startup_health_check"
                                }
                            )
                        except Exception as es_error:
                            es_test_span.record_exception(es_error)
                            es_test_span.set_attribute("elasticsearch.available", False)
                            logger.warning_with_context(
                                "Elasticsearch connection issue at startup",
                                extra_attributes={
                                    "error.message": str(es_error),
                                    "operation": "startup_health_check"
                                }
                            )
                    
                    # Test Google Drive during startup if available
                    if service.google_drive_loader:
                        with tracer.start_as_current_span("startup_gdrive_test") as gdrive_test_span:
                            gdrive_test_span.set_attributes({
                                "service.name": "document-rag-backend",
                                "peer.service": "google-drive-api",
                                "external.service.name": "google-drive-api",
                                "external.service.type": "file_storage_api",
                                "test.phase": "startup"
                            })
                            
                            logger.info_with_context(
                                "Google Drive loader ready",
                                extra_attributes={
                                    "google_drive.folder_id": service.google_drive_folder_id,
                                    "operation": "startup_health_check"
                                }
                            )
                
                except Exception as health_error:
                    health_span.record_exception(health_error)
                    logger.warning_with_context(
                        "Some health checks failed during startup",
                        extra_attributes={
                            "error.type": type(health_error).__name__,
                            "error.message": str(health_error),
                            "operation": "startup_health_check"
                        }
                    )
            
            # Start monitoring task
            with tracer.start_as_current_span("start_monitoring_task") as monitor_span:
                monitor_span.set_attributes({
                    "service.name": "document-rag-backend",
                    "operation.name": "start_monitoring_task"
                })
                
                monitoring_task = asyncio.create_task(service.start_monitoring())
                monitor_span.set_attribute("monitoring_task.started", True)
                
                logger.info_with_context(
                    "Backend service startup completed",
                    extra_attributes={
                        "operation": "service_startup",
                        "status": "completed"
                    }
                )
        
        except Exception as startup_error:
            startup_span.record_exception(startup_error)
            startup_span.set_attribute("startup.failed", True)
            logger.error_with_context(
                "Backend service startup failed",
                extra_attributes={
                    "error.type": type(startup_error).__name__,
                    "error.message": str(startup_error),
                    "operation": "service_startup",
                    "status": "failed"
                },
                exc_info=True
            )
            raise
    
    yield
    
    # Shutdown with proper trace correlation
    with tracer.start_as_current_span("backend_service_shutdown") as shutdown_span:
        shutdown_span.set_attributes({
            "service.name": "document-rag-backend",
            "lifecycle.phase": "shutdown",
            "shutdown.timestamp": datetime.now().isoformat()
        })
        
        logger.info_with_context(
            "Shutting down Backend Service",
            extra_attributes={
                "operation": "service_shutdown"
            }
        )
        
        if service:
            with tracer.start_as_current_span("stop_monitoring") as stop_span:
                stop_span.set_attributes({
                    "service.name": "document-rag-backend",
                    "operation.name": "stop_monitoring"
                })
                
                service.stop_monitoring()
                stop_span.set_attribute("monitoring.stopped", True)
        
        if 'monitoring_task' in locals() and not monitoring_task.done():
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        
        shutdown_span.set_attribute("shutdown.completed", True)
        logger.info_with_context(
            "Backend service shutdown completed",
            extra_attributes={
                "operation": "service_shutdown",
                "status": "completed"
            }
        )

# FastAPI app for status monitoring
app = FastAPI(
    title="Document Processing Backend Service",
    description="Continuous monitoring and processing of documents with complete OpenTelemetry correlation",
    version="2.0.0",
    lifespan=lifespan
)

# CRITICAL FIX: Instrument FastAPI with correlation and middleware
app = instrument_fastapi_app(app, "document-rag-backend")

# FIXED: Add correlation middleware with proper context handling
@app.middleware("http")
async def add_correlation_headers(request: Request, call_next):
    """Add correlation headers to all requests with proper W3C context extraction"""
    # Extract trace context from incoming headers
    carrier = dict(request.headers)
    
    # CRITICAL FIX: Use propagate.extract directly for better compatibility
    extracted_context = propagate.extract(carrier)
    context_token = attach(extracted_context)
    
    try:
        # Process request with extracted context
        response = await call_next(request)
        
        # Add correlation headers to response
        current_span = trace.get_current_span()
        if current_span != trace.INVALID_SPAN:
            span_context = current_span.get_span_context()
            response.headers["X-Trace-ID"] = format(span_context.trace_id, '032x')
            response.headers["X-Span-ID"] = format(span_context.span_id, '016x')
            response.headers["X-Service-Name"] = "document-rag-backend"
            response.headers["X-Service-Version"] = "2.0.0"
            
            # CRITICAL FIX: Add W3C headers for downstream services
            w3c_headers = {}
            propagate.inject(w3c_headers, context=extracted_context)
            for header, value in w3c_headers.items():
                response.headers[f"X-{header}"] = value
        
        return response
    finally:
        # CRITICAL FIX: Always detach context
        detach(context_token)

@app.get("/")
async def root(request: Request):
    """Root endpoint with correlation"""
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("backend_root_endpoint") as span:
        span.set_attributes({
            "service.name": "document-rag-backend",
            "http.method": "GET",
            "http.route": "/"
        })
        
        logger.debug_with_context(
            "Root endpoint accessed",
            extra_attributes={
                "operation": "root_endpoint"
            }
        )
        
        return {
            "service": "document-rag-backend", 
            "version": "2.0.0",
            "status": "running",
            "orchestrator": orchestrator_url
        }

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("health_check") as span:
        logger.debug_with_context(
            "Health check requested",
            extra_attributes={
                "operation": "health_check"
            }
        )
        
        return {
            "status": "healthy",
            "service": "document-rag-backend",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/status")
async def get_status(request: Request):
    """Get service status with correlation"""
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("backend_get_status") as span:
        span.set_attributes({
            "service.name": "document-rag-backend",
            "http.method": "GET",
            "http.route": "/status"
        })

        span.set_attribute("transaction.name", "GET /status")
        span.set_attribute("transaction.type", "request")
        
        if not service:
            logger.error_with_context(
                "Status requested but service not initialized",
                extra_attributes={
                    "operation": "status_check"
                }
            )
            span.set_status(Status(StatusCode.ERROR, "Service not initialized"))
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        status = service.get_status()
        span.set_attributes({
            "is_running": status["service"]["is_running"],
            "files_processed": status["processing"]["files_processed_session"],
            "total_files": status["processing"]["total_files_processed"]
        })
        
        logger.debug_with_context(
            "Status information provided",
            extra_attributes={
                "service.is_running": status["service"]["is_running"],
                "files.processed": status["processing"]["files_processed_session"],
                "operation": "status_check"
            }
        )
        
        return status

@app.post("/scan")
async def trigger_scan(request: Request):
    """Manually trigger a scan with correlation"""
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("backend_trigger_manual_scan") as span:
        span.set_attributes({
            "service.name": "document-rag-backend",
            "http.method": "POST",
            "http.route": "/scan"
        })
        
        if not service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not service.is_running:
            logger.error_with_context(
                "Scan requested but service not running",
                extra_attributes={
                    "service.initialized": service is not None,
                    "service.is_running": service.is_running if service else False,
                    "operation": "manual_scan"
                }
            )
            raise HTTPException(status_code=503, detail="Service not running")
        
        logger.info_with_context(
            "Manual scan triggered via API",
            extra_attributes={
                "operation": "manual_scan"
            }
        )
        
        asyncio.create_task(asyncio.to_thread(service.scan_and_process))
        
        return {"message": "Scan triggered successfully"}

@app.post("/stop")
async def stop_service(request: Request):
    """Stop the monitoring service"""
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("backend_stop_service"):
        if not service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        service.stop_monitoring()
        
        logger.info_with_context(
            "Service stopped via API",
            extra_attributes={
                "operation": "service_stop"
            }
        )
        
        return {"message": "Service stopped"}

@app.post("/start")
async def start_service(request: Request):
    """Start the monitoring service"""
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("backend_start_service"):
        if not service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if service.is_running:
            return {"message": "Service already running"}
        
        asyncio.create_task(service.start_monitoring())
        
        logger.info_with_context(
            "Service started via API",
            extra_attributes={
                "operation": "service_start"
            }
        )
        
        return {"message": "Service started"}

@app.post("/reset")
async def reset_processed_files(request: Request):
    """Clear the processed files database"""
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("backend_reset_processed_files"):
        if not service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            service.fingerprint_db.clear_processed_files()
            
            logger.info_with_context(
                "Processed files database cleared",
                extra_attributes={
                    "operation": "database_reset"
                }
            )
            
            return {"message": "Processed files database cleared successfully"}
        except Exception as e:
            logger.error_with_context(
                "Failed to clear processed files database",
                extra_attributes={
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "database_reset"
                },
                exc_info=True
            )
            raise HTTPException(status_code=500, detail=f"Failed to clear database: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Processing Backend Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()

    # The tracer is already initialized at module level with parent context
    # Just create the logger here
    logger = get_correlated_logger("backend_startup")
    
    # Create startup span with existing tracer
    with tracer.start_as_current_span("backend_main_startup") as main_span:
        main_span.set_attributes({
            "service.name": "document-rag-backend",
            "service.version": "2.0.0",
            "startup.mode": "standalone",
            "startup.host": args.host,
            "startup.port": args.port,
            "startup.timestamp": datetime.now().isoformat()
        })
        
        # Get trace info for display
        span_context = main_span.get_span_context()
        trace_id = format(span_context.trace_id, '032x')
        span_id = format(span_context.span_id, '016x')
        
        logger.info_with_context(
            "Backend service starting up",
            extra_attributes={
                "host": args.host,
                "port": args.port,
                "orchestrator_url": orchestrator_url,
                "startup.mode": "standalone",
                "operation": "service_startup"
            }
        )
        
        print(f"🚀 Backend Service")
        print(f"📡 Orchestrator: {orchestrator_url}")
        print(f"🌐 Starting server on {args.host}:{args.port}")
        print(f"🆔 Startup Trace ID: {trace_id}")
        print(f"🆔 Startup Span ID: {span_id}")
        print(f"📤 OTEL Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
        print(f"🔗 Parent Trace: {os.getenv('OTEL_TRACE_PARENT', 'None')}")
        print("=" * 60)
        
        # Start uvicorn server
        uvicorn.run(
            "backend_service:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )