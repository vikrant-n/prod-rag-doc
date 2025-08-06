#!/usr/bin/env python3
"""
Continuous Backend Service for Document Processing

This service continuously monitors for new files and processes only those that haven't been embedded yet.
It tracks processed files to avoid reprocessing and provides a REST API for status monitoring.
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
from contextlib import asynccontextmanager

# FastAPI imports for status API
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

# Document processing imports
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

# Loaders
from loaders.master_loaders import load_file
from loaders.google_drive_loader import GoogleDriveMasterLoader

# Text splitting
from text_splitting import split_documents

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenTelemetry early
from otel_config import initialize_opentelemetry, shutdown_opentelemetry, trace_function, traced_operation, add_trace_context_to_logs
from opentelemetry import trace, metrics, baggage
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize OpenTelemetry for the backend service
tracer, meter = initialize_opentelemetry(
    service_name="document-rag-backend",
    service_version="1.0.0"
)

# Configure logging with trace context
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend_service.log')
    ]
)
logger = logging.getLogger(__name__)

# Add trace context to logs
add_trace_context_to_logs()

# Create metrics for backend service monitoring
file_processing_counter = meter.create_counter(
    "documents_processed_total",
    description="Total number of documents processed",
    unit="1"
)

file_processing_duration = meter.create_histogram(
    "document_processing_duration_seconds",
    description="Time taken to process documents",
    unit="s"
)

embedding_operations = meter.create_counter(
    "embedding_operations_total",
    description="Total number of embedding operations",
    unit="1"
)

qdrant_operations = meter.create_counter(
    "qdrant_operations_total", 
    description="Total number of Qdrant operations",
    unit="1"
)

google_drive_operations = meter.create_counter(
    "google_drive_operations_total",
    description="Total number of Google Drive operations", 
    unit="1"
)

active_processing_gauge = meter.create_up_down_counter(
    "active_document_processing",
    description="Number of documents currently being processed",
    unit="1"
)

database_operations = meter.create_counter(
    "database_operations_total",
    description="Total number of database operations",
    unit="1"
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
    
    @trace_function("database.is_file_processed", {"component": "database", "operation": "query"})
    def is_file_processed(self, file_path: str, file_hash: str) -> bool:
        """Check if a file has already been processed"""
        with traced_operation("file_processed_check") as span:
            span.set_attribute("file.path", file_path)
            span.set_attribute("file.hash", file_hash[:16] + "...")  # Truncate hash for security
            span.set_attribute("db.operation", "SELECT")
            span.set_attribute("db.table", "processed_files")
            
            database_operations.add(1, {"operation": "select", "table": "processed_files"})
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM processed_files WHERE file_path = ? AND file_hash = ?",
                    (file_path, file_hash)
                )
                count = cursor.fetchone()[0]
                is_processed = count > 0
                
                span.set_attribute("query.result.count", count)
                span.set_attribute("file.is_processed", is_processed)
                span.add_event("file_processed_check_complete", {
                    "is_processed": is_processed,
                    "count": count
                })
                
                return is_processed
    
    @trace_function("qdrant.is_file_in_qdrant", {"component": "qdrant", "operation": "search"})
    def is_file_in_qdrant(self, file_path: str, qdrant_client) -> bool:
        """Check if a file already has embeddings in Qdrant"""
        with traced_operation("qdrant_file_check") as span:
            span.set_attribute("file.path", file_path)
            span.set_attribute("qdrant.collection", "documents")
            span.set_attribute("qdrant.operation", "scroll")
            
            qdrant_operations.add(1, {"operation": "scroll", "collection": "documents"})
            
            try:
                # Search for any points with this file_path in metadata
                search_result = qdrant_client.scroll(
                    collection_name="documents",
                    scroll_filter={
                        "must": [
                            {
                                "key": "file_path",
                                "match": {"value": file_path}
                            }
                        ]
                    },
                    limit=1,
                    with_payload=True
                )
                
                points_found = len(search_result[0])
                file_exists = points_found > 0
                
                span.set_attribute("qdrant.search.points_found", points_found)
                span.set_attribute("file.exists_in_qdrant", file_exists)
                span.add_event("qdrant_file_check_complete", {
                    "file_exists": file_exists,
                    "points_found": points_found
                })
                
                return file_exists
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("qdrant.error.type", type(e).__name__)
                span.set_attribute("qdrant.error.message", str(e))
                span.add_event("qdrant_file_check_error", {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "assumption": "file_not_processed"
                })
                # If there's an error (like collection doesn't exist), assume not processed
                return False
    
    @trace_function("database.mark_file_processed", {"component": "database", "operation": "insert"})
    def mark_file_processed(self, processed_file: ProcessedFile):
        """Mark a file as processed"""
        with traced_operation("mark_file_processed") as span:
            span.set_attribute("file.id", processed_file.file_id)
            span.set_attribute("file.name", processed_file.file_name)
            span.set_attribute("file.path", processed_file.file_path)
            span.set_attribute("file.hash", processed_file.file_hash[:16] + "...")
            span.set_attribute("file.document_count", processed_file.document_count)
            span.set_attribute("file.size", processed_file.file_size)
            span.set_attribute("file.mime_type", processed_file.mime_type)
            span.set_attribute("file.qdrant_points", len(processed_file.qdrant_point_ids))
            span.set_attribute("db.operation", "INSERT OR REPLACE")
            span.set_attribute("db.table", "processed_files")
            
            database_operations.add(1, {"operation": "insert", "table": "processed_files"})
            
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
                
                span.add_event("file_marked_processed", {
                    "file_id": processed_file.file_id,
                    "document_count": processed_file.document_count,
                    "qdrant_points": len(processed_file.qdrant_point_ids)
                })
    
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
    """Handles document processing and embedding with OpenTelemetry instrumentation"""
    
    @trace_function("document_processor.init", {"component": "document_processor", "operation": "initialization"})
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "documents",
                 embedding_model: str = "text-embedding-3-large"):
        with traced_operation("document_processor_initialization") as span:
            self.qdrant_url = qdrant_url
            self.collection_name = collection_name
            self.embedding_model = embedding_model
            
            span.set_attribute("qdrant.url", qdrant_url)
            span.set_attribute("qdrant.collection", collection_name)
            span.set_attribute("embedding.model", embedding_model)
            span.add_event("initializing_components")
            
            # Initialize components
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
            self.qdrant_client = QdrantClient(url=qdrant_url)
            self.vector_store = None
            
            span.add_event("components_initialized")
            
            # Initialize collection
            self._ensure_collection_exists()
            
            span.add_event("document_processor_ready")
            logger.info(f"✅ Document processor initialized with collection: {collection_name}")
    
    @trace_function("qdrant.ensure_collection_exists", {"component": "qdrant", "operation": "collection_setup"})
    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists"""
        with traced_operation("qdrant_collection_setup") as span:
            span.set_attribute("qdrant.collection", self.collection_name)
            span.set_attribute("qdrant.vector_size", 3072)
            span.set_attribute("qdrant.distance_metric", "COSINE")
            
            try:
                qdrant_operations.add(1, {"operation": "get_collections", "collection": self.collection_name})
                collections = self.qdrant_client.get_collections()
                collection_names = [col.name for col in collections.collections]
                
                span.set_attribute("qdrant.existing_collections.count", len(collection_names))
                span.add_event("collections_retrieved", {"count": len(collection_names)})
                
                if self.collection_name not in collection_names:
                    span.set_attribute("qdrant.collection.exists", False)
                    span.add_event("creating_collection", {"collection": self.collection_name})
                    logger.info(f"📦 Creating collection: {self.collection_name}")
                    
                    qdrant_operations.add(1, {"operation": "create_collection", "collection": self.collection_name})
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)  # text-embedding-3-large
                    )
                    
                    span.add_event("collection_created", {"collection": self.collection_name})
                    logger.info(f"✅ Collection '{self.collection_name}' created")
                else:
                    span.set_attribute("qdrant.collection.exists", True)
                    span.add_event("collection_already_exists", {"collection": self.collection_name})
                    logger.info(f"✅ Collection '{self.collection_name}' already exists")
                    
                # Initialize vector store
                span.add_event("initializing_vector_store")
                self.vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings
                )
                span.add_event("vector_store_initialized")
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("qdrant.error.type", type(e).__name__)
                span.set_attribute("qdrant.error.message", str(e))
                span.add_event("collection_setup_failed", {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
                logger.error(f"❌ Failed to initialize collection: {e}")
                raise
    
    @trace_function("document_processor.process_documents", {"component": "document_processor", "operation": "embedding"})
    def process_documents(self, documents: List[Document], file_info: Dict) -> List[str]:
        """Process documents and add to Qdrant, return point IDs"""
        processing_start_time = time.time()
        
        with traced_operation("document_processing") as span:
            span.set_attribute("file.name", file_info.get("file_name", "unknown"))
            span.set_attribute("file.path", file_info.get("file_path", "unknown"))
            span.set_attribute("file.size", file_info.get("file_size", 0))
            span.set_attribute("file.mime_type", file_info.get("mime_type", "unknown"))
            span.set_attribute("documents.input_count", len(documents))
            
            active_processing_gauge.add(1, {"file_type": file_info.get("mime_type", "unknown")})
            
            try:
                if not documents:
                    span.set_attribute("processing.result", "skipped_empty")
                    span.add_event("processing_skipped", {"reason": "no_documents"})
                    return []
                
                # Split documents into chunks
                span.add_event("chunking_documents")
                chunks = split_documents(documents, chunk_size=1000, chunk_overlap=120)
                
                span.set_attribute("documents.chunks_count", len(chunks))
                span.set_attribute("chunking.chunk_size", 1000)
                span.set_attribute("chunking.overlap", 120)
                span.add_event("documents_chunked", {"chunks_count": len(chunks)})
                logger.info(f"📄 Split into {len(chunks)} chunks")
                
                if not chunks:
                    span.set_attribute("processing.result", "skipped_no_chunks")
                    span.add_event("processing_skipped", {"reason": "no_chunks_after_splitting"})
                    return []
                
                # Add metadata
                span.add_event("adding_metadata")
                for chunk in chunks:
                    chunk.metadata.update({
                        "file_id": file_info.get("file_id", "unknown"),
                        "file_name": file_info.get("file_name", "unknown"),
                        "file_path": file_info.get("file_path", "unknown"),
                        "processed_at": datetime.now().isoformat(),
                        "file_size": file_info.get("file_size", 0),
                        "mime_type": file_info.get("mime_type", "unknown")
                    })
                span.add_event("metadata_added")
                
                # Generate embeddings and add to Qdrant
                point_ids = []
                batch_size = 10
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                span.set_attribute("embedding.batch_size", batch_size)
                span.set_attribute("embedding.total_batches", total_batches)
                span.add_event("embedding_processing_start", {
                    "total_chunks": len(chunks),
                    "batch_size": batch_size,
                    "total_batches": total_batches
                })
                
                for i in range(0, len(chunks), batch_size):
                    batch_num = i // batch_size + 1
                    batch = chunks[i:i + batch_size]
                    
                    with traced_operation(f"embedding_batch_{batch_num}") as batch_span:
                        batch_span.set_attribute("batch.number", batch_num)
                        batch_span.set_attribute("batch.size", len(batch))
                        batch_span.set_attribute("batch.total_batches", total_batches)
                        
                        # Generate unique point IDs
                        batch_point_ids = [
                            hashlib.md5(f"{file_info['file_path']}_{i+j}_{chunk.page_content[:100]}".encode()).hexdigest()
                            for j, chunk in enumerate(batch)
                        ]
                        
                        batch_span.set_attribute("batch.point_ids_count", len(batch_point_ids))
                        batch_span.add_event("point_ids_generated", {"count": len(batch_point_ids)})
                        
                        # Add to vector store
                        embedding_operations.add(1, {"operation": "add_documents", "batch_size": len(batch)})
                        qdrant_operations.add(1, {"operation": "add_documents", "collection": self.collection_name})
                        
                        self.vector_store.add_documents(batch, ids=batch_point_ids)
                        point_ids.extend(batch_point_ids)
                        
                        batch_span.add_event("batch_processed", {
                            "batch_number": batch_num,
                            "documents_added": len(batch),
                            "point_ids_added": len(batch_point_ids)
                        })
                        
                        logger.info(f"✅ Processed batch {batch_num}/{total_batches}")
                        
                        # Small delay to avoid overwhelming the system
                        time.sleep(0.1)
                
                processing_duration = time.time() - processing_start_time
                file_processing_duration.record(processing_duration, {
                    "file_type": file_info.get("mime_type", "unknown"),
                    "status": "success"
                })
                file_processing_counter.add(1, {
                    "file_type": file_info.get("mime_type", "unknown"),
                    "status": "success"
                })
                
                span.set_attribute("processing.result", "success")
                span.set_attribute("processing.duration_seconds", processing_duration)
                span.set_attribute("processing.chunks_processed", len(chunks))
                span.set_attribute("processing.points_created", len(point_ids))
                span.add_event("document_processing_complete", {
                    "chunks_processed": len(chunks),
                    "points_created": len(point_ids),
                    "duration_seconds": processing_duration
                })
                
                logger.info(f"✅ Successfully processed {len(chunks)} chunks with {len(point_ids)} points")
                return point_ids
                
            except Exception as e:
                processing_duration = time.time() - processing_start_time
                file_processing_duration.record(processing_duration, {
                    "file_type": file_info.get("mime_type", "unknown"),
                    "status": "error"
                })
                file_processing_counter.add(1, {
                    "file_type": file_info.get("mime_type", "unknown"),
                    "status": "error"
                })
                
                span.record_exception(e)
                span.set_attribute("processing.result", "error")
                span.set_attribute("processing.error.type", type(e).__name__)
                span.set_attribute("processing.error.message", str(e))
                span.add_event("document_processing_failed", {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "duration_seconds": processing_duration
                })
                
                logger.error(f"❌ Failed to process documents: {e}")
                raise
            finally:
                active_processing_gauge.add(-1, {"file_type": file_info.get("mime_type", "unknown")})

class BackendService:
    """Main backend service for continuous file monitoring and processing"""
    
    @trace_function("backend_service.init", {"component": "backend_service", "operation": "initialization"})
    def __init__(self):
        with traced_operation("backend_service_initialization") as span:
            # Configuration
            self.google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
            self.local_watch_dirs = os.getenv("LOCAL_WATCH_DIRS", "").split(",") if os.getenv("LOCAL_WATCH_DIRS") else []
            self.scan_interval = int(os.getenv("SCAN_INTERVAL", "60"))  # seconds
            
            span.set_attribute("config.google_drive_folder_id", bool(self.google_drive_folder_id))
            span.set_attribute("config.local_watch_dirs_count", len(self.local_watch_dirs))
            span.set_attribute("config.scan_interval", self.scan_interval)
            
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
            
            span.add_event("backend_service_initialized", {
                "google_drive_enabled": bool(self.google_drive_folder_id),
                "local_dirs_count": len(self.local_watch_dirs),
                "scan_interval": self.scan_interval
            })
        
        # Initialize Google Drive loader if configured
        if self.google_drive_folder_id:
            try:
                # Get credentials paths from environment
                credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
                token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
                
                self.google_drive_loader = GoogleDriveMasterLoader(
                    folder_id=self.google_drive_folder_id,
                    credentials_path=credentials_path,
                    token_path=token_path,
                    split=False  # We'll handle splitting ourselves
                )
                logger.info(f"✅ Google Drive loader initialized for folder: {self.google_drive_folder_id}")
                logger.info(f"📋 Using credentials: {credentials_path}")
                logger.info(f"🔑 Using token: {token_path}")
            except Exception as e:
                logger.error(f"❌ Could not initialize Google Drive loader: {e}")
                logger.error(f"   Make sure credentials are set up properly at: {os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')}")
                self.google_drive_loader = None
        
        logger.info(f"🚀 Backend service initialized")
        logger.info(f"📁 Google Drive folder: {self.google_drive_folder_id}")
        logger.info(f"📂 Local watch directories: {self.local_watch_dirs}")
        logger.info(f"⏱️ Scan interval: {self.scan_interval} seconds")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"❌ Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def scan_local_directories(self) -> List[Dict]:
        """Scan local directories for new files"""
        new_files = []
        
        for watch_dir in self.local_watch_dirs:
            if not os.path.exists(watch_dir):
                logger.warning(f"⚠️ Watch directory does not exist: {watch_dir}")
                continue
            
            try:
                for root, dirs, files in os.walk(watch_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Skip hidden files and non-supported formats
                        if file.startswith('.'):
                            continue
                        
                        # Check if file is supported
                        ext = os.path.splitext(file)[1].lower()
                        supported_exts = {'.pdf', '.docx', '.pptx', '.txt', '.csv', '.xlsx', '.xls', 
                                        '.png', '.jpg', '.jpeg', '.gif', '.tiff', '.tif', '.bmp', '.webp'}
                        if ext not in supported_exts:
                            continue
                        
                        # Calculate file hash
                        file_hash = self.calculate_file_hash(file_path)
                        if not file_hash:
                            continue
                        
                        # Check if already processed (both in local DB and Qdrant)
                        if (self.fingerprint_db.is_file_processed(file_path, file_hash) or 
                            self.fingerprint_db.is_file_in_qdrant(file_path, self.processor.qdrant_client)):
                            continue
                        
                        # Get file info
                        stat = os.stat(file_path)
                        new_files.append({
                            "file_id": file_hash,  # Use hash as ID for local files
                            "file_name": file,
                            "file_path": file_path,
                            "file_hash": file_hash,
                            "file_size": stat.st_size,
                            "mime_type": f"application/{ext[1:]}" if ext else "application/octet-stream",
                            "source": "local"
                        })
                        
            except Exception as e:
                logger.error(f"❌ Error scanning directory {watch_dir}: {e}")
                self.stats["errors"].append(f"Directory scan error: {e}")
        
        return new_files
    
    def scan_google_drive(self) -> List[Dict]:
        """Scan Google Drive for new files"""
        if not self.google_drive_loader:
            logger.warning("⚠️ Google Drive loader not available")
            return []
        
        new_files = []
        try:
            logger.info("🔍 Scanning Google Drive for files...")
            
            # Get all files from Google Drive (recursively from all subfolders)
            files = self.google_drive_loader._list_files(self.google_drive_folder_id)
            logger.info(f"📁 Found {len(files)} total files in Google Drive")
            
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
                
                # For Google Drive files, we use file_id + name as the path
                file_path = f"gdrive://{file_id}/{file_name}"
                
                # Use file_id + name as hash since we can't get actual file hash without downloading
                file_hash = hashlib.md5(f"{file_id}_{file_name}_{mime_type}".encode()).hexdigest()
                
                # Check if already processed (both in local DB and Qdrant)
                if (self.fingerprint_db.is_file_processed(file_path, file_hash) or 
                    self.fingerprint_db.is_file_in_qdrant(file_path, self.processor.qdrant_client)):
                    logger.debug(f"⏭️  Skipping already processed file: {file_name}")
                    continue
                
                logger.info(f"🆕 Found new file: {file_name} (Type: {mime_type})")
                
                new_files.append({
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "file_size": 0,  # We don't have size info without downloading
                    "mime_type": mime_type,
                    "source": "google_drive",
                    "drive_file_info": file_info
                })
                
        except Exception as e:
            logger.error(f"❌ Error scanning Google Drive: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.stats["errors"].append(f"Google Drive scan error: {e}")
        
        return new_files
    
    def cleanup_extracted_images(self, documents: List[Document]):
        """Clean up locally extracted images after processing"""
        try:
            image_paths_to_clean = set()
            
            # Collect all image paths from documents
            for doc in documents:
                # Check for direct image paths
                img_path = doc.metadata.get("image_path")
                if img_path and os.path.exists(img_path):
                    image_paths_to_clean.add(img_path)
                
                # Check for related images
                related_images = doc.metadata.get("related_images", [])
                for rel_img in related_images:
                    if isinstance(rel_img, str) and os.path.exists(rel_img):
                        image_paths_to_clean.add(rel_img)
            
            # Clean up the image files
            cleaned_count = 0
            for img_path in image_paths_to_clean:
                try:
                    # Only clean up if it's in one of our extraction directories
                    if any(dir_name in img_path for dir_name in ['pdf_images', 'pptx_images', 'docx_images']):
                        os.remove(img_path)
                        cleaned_count += 1
                        logger.debug(f"🧹 Cleaned up extracted image: {os.path.basename(img_path)}")
                except OSError as e:
                    logger.debug(f"⚠️ Could not remove image {img_path}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_count} extracted image files")
                
        except Exception as e:
            logger.warning(f"⚠️ Error during image cleanup: {e}")

    def process_file(self, file_info: Dict) -> bool:
        """Process a single file"""
        try:
            file_path = file_info["file_path"]
            file_name = file_info["file_name"]
            source = file_info["source"]
            
            logger.info(f"🔄 Processing {source} file: {file_name}")
            
            # Load documents
            documents = []
            temp_file = None
            
            if source == "local":
                # Load local file directly
                documents = load_file(file_path)
                # For local files, we need to clean up extracted images manually
                self.cleanup_extracted_images(documents)
            elif source == "google_drive":
                # Download from Google Drive first
                drive_file_info = file_info["drive_file_info"]
                temp_file = self.google_drive_loader._download_file(drive_file_info)
                documents = load_file(temp_file)
                
                # Process documents with Google Drive loader to handle image upload and cleanup
                self.google_drive_loader._process_docs(documents, drive_file_info)
                
                # Update file size with actual downloaded size
                if temp_file and os.path.exists(temp_file):
                    file_info["file_size"] = os.path.getsize(temp_file)
            
            if not documents:
                logger.warning(f"⚠️ No documents extracted from {file_name}")
                return False
            
            logger.info(f"📄 Extracted {len(documents)} documents from {file_name}")
            
            # Process documents and get point IDs
            point_ids = self.processor.process_documents(documents, file_info)
            
            # Mark as processed
            processed_file = ProcessedFile(
                file_id=file_info["file_id"],
                file_name=file_name,
                file_path=file_path,
                file_hash=file_info["file_hash"],
                processed_at=datetime.now(),
                document_count=len(documents),
                file_size=file_info["file_size"],
                mime_type=file_info["mime_type"],
                qdrant_point_ids=point_ids
            )
            
            self.fingerprint_db.mark_file_processed(processed_file)
            
            # Update stats
            self.stats["files_processed"] += 1
            self.stats["documents_created"] += len(documents)
            
            logger.info(f"✅ Successfully processed {file_name} -> {len(documents)} documents, {len(point_ids)} points")
            
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f"🧹 Cleaned up temporary file: {os.path.basename(temp_file)}")
                except Exception as e:
                    logger.debug(f"⚠️ Could not remove temporary file {temp_file}: {e}")
            
            # Final cleanup: remove any remaining extracted images that might have been missed
            if source == "local":
                self.cleanup_extracted_images(documents)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process file {file_info.get('file_name', 'unknown')}: {e}")
            self.stats["errors"].append(f"File processing error: {e}")
            return False
    
    def scan_and_process(self):
        """Scan for new files and process them"""
        try:
            logger.info("🔍 Starting scan cycle...")
            
            # Prioritize Google Drive scanning
            new_files = []
            
            # Scan Google Drive first (main focus)
            google_drive_files = self.scan_google_drive()
            new_files.extend(google_drive_files)
            
            # Scan local directories if configured
            if self.local_watch_dirs:
                local_files = self.scan_local_directories()
                new_files.extend(local_files)
            
            self.stats["last_scan"] = datetime.now()
            
            if not new_files:
                logger.info("✅ No new files found")
                return
            
            logger.info(f"📁 Found {len(new_files)} new files to process:")
            for file_info in new_files:
                logger.info(f"   - {file_info['file_name']} ({file_info['source']})")
            
            # Process each file
            processed_count = 0
            failed_count = 0
            
            for file_info in new_files:
                if not self.is_running:
                    logger.info("🛑 Service stopping, halting file processing")
                    break
                
                try:
                    logger.info(f"🔄 Processing file {processed_count + 1}/{len(new_files)}: {file_info['file_name']}")
                    
                    if self.process_file(file_info):
                        processed_count += 1
                        logger.info(f"✅ Successfully processed: {file_info['file_name']}")
                    else:
                        failed_count += 1
                        logger.warning(f"⚠️ Failed to process: {file_info['file_name']}")
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ Error processing file {file_info.get('file_name')}: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    continue
                
                # Small delay between files to avoid overwhelming the system
                time.sleep(2)
            
            logger.info(f"✅ Scan and process cycle completed: {processed_count} processed, {failed_count} failed")
            
        except Exception as e:
            logger.error(f"❌ Error in scan and process cycle: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.stats["errors"].append(f"Scan cycle error: {e}")
    
    @trace_function("backend_service.start_monitoring", {"component": "backend_service", "operation": "monitoring_loop"})
    async def start_monitoring(self):
        """Start the continuous monitoring service"""
        with traced_operation("monitoring_service_lifecycle") as span:
            self.is_running = True
            span.set_attribute("service.is_running", self.is_running)
            span.set_attribute("service.scan_interval", self.scan_interval)
            span.add_event("monitoring_service_started", {"scan_interval": self.scan_interval})
            
            logger.info(f"🚀 Starting continuous monitoring service...")
            logger.info(f"⏱️ Scan interval: {self.scan_interval} seconds")
            
            scan_count = 0
            error_count = 0
            
            while self.is_running:
                try:
                    with traced_operation(f"monitoring_scan_{scan_count}") as scan_span:
                        scan_span.set_attribute("scan.number", scan_count)
                        scan_span.set_attribute("scan.interval", self.scan_interval)
                        
                        self.scan_and_process()
                        scan_count += 1
                        
                        scan_span.add_event("scan_completed", {"scan_number": scan_count})
                    
                    # Wait for next scan
                    for _ in range(self.scan_interval):
                        if not self.is_running:
                            break
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    error_count += 1
                    span.record_exception(e)
                    span.set_attribute("service.error_count", error_count)
                    span.add_event("monitoring_error", {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "scan_number": scan_count,
                        "error_count": error_count
                    })
                    
                    logger.error(f"❌ Error in monitoring loop: {e}")
                    self.stats["errors"].append(f"Monitoring loop error: {e}")
                    await asyncio.sleep(10)  # Wait before retrying
            
            span.set_attribute("service.total_scans", scan_count)
            span.set_attribute("service.total_errors", error_count)
            span.add_event("monitoring_service_stopped", {
                "total_scans": scan_count,
                "total_errors": error_count
            })
            
            logger.info("🛑 Monitoring service stopped")
    
    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.is_running = False
        logger.info("🛑 Stopping monitoring service...")
    
    def get_status(self) -> Dict:
        """Get service status"""
        db_stats = self.fingerprint_db.get_stats()
        recent_files = self.fingerprint_db.get_processed_files(limit=5)
        
        return {
            "service": {
                "is_running": self.is_running,
                "started_at": self.stats["service_started"].isoformat(),
                "uptime_seconds": (datetime.now() - self.stats["service_started"]).total_seconds(),
                "last_scan": self.stats["last_scan"].isoformat() if self.stats["last_scan"] else None,
                "scan_interval": self.scan_interval
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
                "local_watch_dirs": self.local_watch_dirs,
                "qdrant_url": self.processor.qdrant_url,
                "collection_name": self.processor.collection_name,
                "embedding_model": self.processor.embedding_model
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
            "errors": self.stats["errors"][-10:]  # Last 10 errors
        }

# Global service instance
service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management"""
    global service
    
    # Startup
    logger.info("🚀 Starting Backend Service...")
    service = BackendService()
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(service.start_monitoring())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Backend Service...")
    if service:
        service.stop_monitoring()
    monitoring_task.cancel()

# FastAPI app for status monitoring
app = FastAPI(
    title="Document Processing Backend Service",
    description="Continuous monitoring and processing of documents",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Document Processing Backend Service", "status": "running"}

@app.get("/status")
@trace_function("api.get_status", {"component": "backend_api", "operation": "status_check"})
async def get_status():
    """Get service status"""
    with traced_operation("status_check") as span:
        if not service:
            span.set_attribute("status.service_initialized", False)
            span.add_event("service_not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        span.set_attribute("status.service_initialized", True)
        status = service.get_status()
        span.set_attribute("status.is_running", status.get("service", {}).get("is_running", False))
        span.set_attribute("status.files_processed", status.get("processing", {}).get("files_processed_session", 0))
        span.add_event("status_retrieved", {
            "is_running": status.get("service", {}).get("is_running", False),
            "files_processed": status.get("processing", {}).get("files_processed_session", 0)
        })
        
        return status

@app.post("/scan")
@trace_function("api.trigger_scan", {"component": "backend_api", "operation": "manual_scan"})
async def trigger_scan():
    """Manually trigger a scan for new files"""
    with traced_operation("manual_scan_trigger") as span:
        if not service:
            span.set_attribute("scan.service_initialized", False)
            span.add_event("service_not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not service.is_running:
            span.set_attribute("scan.service_running", False)
            span.add_event("service_not_running")
            raise HTTPException(status_code=503, detail="Service not running")
        
        span.set_attribute("scan.service_initialized", True)
        span.set_attribute("scan.service_running", True)
        span.add_event("manual_scan_triggered")
        
        # Trigger scan in background
        asyncio.create_task(asyncio.to_thread(service.scan_and_process))
        
        return {"message": "Scan triggered successfully"}

@app.post("/stop")
async def stop_service():
    """Stop the monitoring service"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    service.stop_monitoring()
    return {"message": "Service stopped"}

@app.post("/start")
async def start_service():
    """Start the monitoring service"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if service.is_running:
        return {"message": "Service already running"}
    
    # Start monitoring in background
    asyncio.create_task(service.start_monitoring())
    return {"message": "Service started"}

@app.post("/reset")
async def reset_processed_files():
    """Clear the processed files database to force reprocessing"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        service.fingerprint_db.clear_processed_files()
        return {"message": "Processed files database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Processing Backend Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"🌐 Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "backend_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )