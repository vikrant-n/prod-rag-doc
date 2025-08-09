#!/usr/bin/env python3
"""
Continuous Backend Service for Document Processing - Instrumented with OpenTelemetry

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

# Initialize OpenTelemetry
from otel_config import initialize_opentelemetry, get_tracer, add_trace_correlation_to_log
from metrics import (
    rag_metrics, time_document_processing, time_query_processing, 
    record_document_processed, record_cache_event
)

# Initialize OpenTelemetry for backend service
tracer, meter = initialize_opentelemetry(
    service_name="document-rag-backend",
    service_version="1.0.0",
    environment="development"
)

# Configure logging with trace correlation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend_service.log')
    ]
)
logger = add_trace_correlation_to_log(logging.getLogger(__name__))

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
        with tracer.start_as_current_span("database.init_schema") as span:
            span.set_attribute("database.path", self.db_path)
            
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
                span.set_attribute("database.init.status", "success")
                logger.info("Database schema initialized successfully")
    
    def is_file_processed(self, file_path: str, file_hash: str) -> bool:
        """Check if a file has already been processed"""
        with tracer.start_as_current_span("database.check_file_processed") as span:
            span.set_attribute("file.path", file_path)
            span.set_attribute("file.hash", file_hash[:16] + "...")  # Truncated for security
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM processed_files WHERE file_path = ? AND file_hash = ?",
                    (file_path, file_hash)
                )
                count = cursor.fetchone()[0]
                is_processed = count > 0
                
                span.set_attribute("file.is_processed", is_processed)
                record_cache_event("file_check", is_processed)
                return is_processed
    
    def is_file_in_qdrant(self, file_path: str, qdrant_client) -> bool:
        """Check if a file already has embeddings in Qdrant"""
        with tracer.start_as_current_span("qdrant.check_file_exists") as span:
            span.set_attribute("file.path", file_path)
            
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
                exists = len(search_result[0]) > 0
                span.set_attribute("qdrant.file_exists", exists)
                span.set_attribute("qdrant.check.status", "success")
                record_cache_event("qdrant_check", exists)
                return exists
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("qdrant.check.status", "error")
                # If there's an error (like collection doesn't exist), assume not processed
                return False
    
    def mark_file_processed(self, processed_file: ProcessedFile):
        """Mark a file as processed"""
        with tracer.start_as_current_span("database.mark_file_processed") as span:
            span.set_attribute("file.name", processed_file.file_name)
            span.set_attribute("file.document_count", processed_file.document_count)
            span.set_attribute("file.size", processed_file.file_size)
            
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
                span.set_attribute("database.operation", "insert_or_replace")
                logger.info(f"Marked file as processed: {processed_file.file_name}")
    
    def get_processed_files(self, limit: int = 100) -> List[ProcessedFile]:
        """Get list of processed files"""
        with tracer.start_as_current_span("database.get_processed_files") as span:
            span.set_attribute("query.limit", limit)
            
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
                
                span.set_attribute("query.result_count", len(files))
                return files
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with tracer.start_as_current_span("database.get_stats") as span:
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
                
                stats = {
                    "total_files": row[0] or 0,
                    "total_documents": row[1] or 0,
                    "total_size_bytes": row[2] or 0,
                    "last_processed": row[3]
                }
                
                span.set_attribute("stats.total_files", stats["total_files"])
                span.set_attribute("stats.total_documents", stats["total_documents"])
                span.set_attribute("stats.total_size_bytes", stats["total_size_bytes"])
                
                return stats
    
    def clear_processed_files(self):
        """Clear all processed files from the database"""
        with tracer.start_as_current_span("database.clear_processed_files") as span:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM processed_files")
                count_before = cursor.fetchone()[0]
                
                conn.execute("DELETE FROM processed_files")
                conn.commit()
                
                span.set_attribute("database.cleared_count", count_before)
                logger.info(f"Cleared {count_before} processed files from database")

class DocumentProcessor:
    """Handles document processing and embedding"""
    
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "documents",
                 embedding_model: str = "text-embedding-3-large"):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        with tracer.start_as_current_span("document_processor.init") as span:
            span.set_attribute("qdrant.url", qdrant_url)
            span.set_attribute("qdrant.collection", collection_name)
            span.set_attribute("embedding.model", embedding_model)
            
            # Initialize components
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
            self.qdrant_client = QdrantClient(url=qdrant_url)
            self.vector_store = None
            
            # Initialize collection
            self._ensure_collection_exists()
            
            span.set_attribute("init.status", "success")
            logger.info(f"Document processor initialized with collection: {collection_name}")
    
    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists"""
        with tracer.start_as_current_span("qdrant.ensure_collection") as span:
            span.set_attribute("collection.name", self.collection_name)
            
            try:
                collections = self.qdrant_client.get_collections()
                collection_names = [col.name for col in collections.collections]
                
                if self.collection_name not in collection_names:
                    logger.info(f"Creating collection: {self.collection_name}")
                    span.add_event("creating_collection")
                    
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)  # text-embedding-3-large
                    )
                    span.set_attribute("collection.created", True)
                    logger.info(f"Collection '{self.collection_name}' created")
                else:
                    span.set_attribute("collection.created", False)
                    logger.info(f"Collection '{self.collection_name}' already exists")
                    
                # Initialize vector store
                self.vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings
                )
                span.set_attribute("vector_store.initialized", True)
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("collection.init.status", "failed")
                logger.error(f"Failed to initialize collection: {e}")
                raise
    
    def process_documents(self, documents: List[Document], file_info: Dict) -> List[str]:
        """Process documents and add to Qdrant, return point IDs"""
        with tracer.start_as_current_span("document_processor.process_documents") as span:
            span.set_attribute("documents.count", len(documents))
            span.set_attribute("file.name", file_info.get("file_name", "unknown"))
            span.set_attribute("file.size", file_info.get("file_size", 0))
            span.set_attribute("file.type", file_info.get("mime_type", "unknown"))
            
            try:
                if not documents:
                    span.set_attribute("processing.status", "no_documents")
                    return []
                
                # Split documents into chunks
                with tracer.start_as_current_span("text_splitting.split_documents") as split_span:
                    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=120)
                    split_span.set_attribute("chunks.created", len(chunks))
                    logger.info(f"Split into {len(chunks)} chunks")
                
                if not chunks:
                    span.set_attribute("processing.status", "no_chunks")
                    return []
                
                # Add metadata
                with tracer.start_as_current_span("document_processor.add_metadata") as meta_span:
                    for chunk in chunks:
                        chunk.metadata.update({
                            "file_id": file_info.get("file_id", "unknown"),
                            "file_name": file_info.get("file_name", "unknown"),
                            "file_path": file_info.get("file_path", "unknown"),
                            "processed_at": datetime.now().isoformat(),
                            "file_size": file_info.get("file_size", 0),
                            "mime_type": file_info.get("mime_type", "unknown")
                        })
                    meta_span.set_attribute("chunks.metadata_added", len(chunks))
                
                # Generate embeddings and add to Qdrant
                point_ids = []
                batch_size = 10
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                span.set_attribute("processing.batch_size", batch_size)
                span.set_attribute("processing.total_batches", total_batches)
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    
                    with tracer.start_as_current_span(f"qdrant.add_batch_{batch_num}") as batch_span:
                        batch_span.set_attribute("batch.number", batch_num)
                        batch_span.set_attribute("batch.size", len(batch))
                        
                        # Generate unique point IDs
                        batch_point_ids = [
                            hashlib.md5(f"{file_info['file_path']}_{i+j}_{chunk.page_content[:100]}".encode()).hexdigest()
                            for j, chunk in enumerate(batch)
                        ]
                        
                        # Add to vector store
                        self.vector_store.add_documents(batch, ids=batch_point_ids)
                        point_ids.extend(batch_point_ids)
                        
                        batch_span.set_attribute("batch.point_ids", len(batch_point_ids))
                        logger.info(f"Processed batch {batch_num}/{total_batches}")
                        
                        # Small delay to avoid overwhelming the system
                        time.sleep(0.1)
                
                span.set_attribute("processing.total_chunks", len(chunks))
                span.set_attribute("processing.total_points", len(point_ids))
                span.set_attribute("processing.status", "success")
                
                # Record metrics
                record_document_processed(
                    file_type=file_info.get("mime_type", "unknown"),
                    file_size=file_info.get("file_size", 0),
                    chunk_count=len(chunks)
                )
                
                logger.info(f"Successfully processed {len(chunks)} chunks with {len(point_ids)} points")
                return point_ids
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("processing.status", "error")
                logger.error(f"Failed to process documents: {e}")
                raise

class BackendService:
    """Main backend service for continuous file monitoring and processing"""
    
    def __init__(self):
        with tracer.start_as_current_span("backend_service.init") as span:
            # Configuration
            self.google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
            self.local_watch_dirs = os.getenv("LOCAL_WATCH_DIRS", "").split(",") if os.getenv("LOCAL_WATCH_DIRS") else []
            self.scan_interval = int(os.getenv("SCAN_INTERVAL", "60"))  # seconds
            
            span.set_attribute("config.google_drive_folder_id", self.google_drive_folder_id or "not_set")
            span.set_attribute("config.local_watch_dirs", len(self.local_watch_dirs))
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
            
            # Initialize Google Drive loader if configured
            if self.google_drive_folder_id:
                try:
                    with tracer.start_as_current_span("google_drive.init") as gdrive_span:
                        # Get credentials paths from environment
                        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
                        token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
                        
                        gdrive_span.set_attribute("credentials.path", credentials_path)
                        gdrive_span.set_attribute("token.path", token_path)
                        
                        self.google_drive_loader = GoogleDriveMasterLoader(
                            folder_id=self.google_drive_folder_id,
                            credentials_path=credentials_path,
                            token_path=token_path,
                            split=False  # We'll handle splitting ourselves
                        )
                        gdrive_span.set_attribute("google_drive.init.status", "success")
                        logger.info(f"Google Drive loader initialized for folder: {self.google_drive_folder_id}")
                except Exception as e:
                    span.record_exception(e)
                    logger.error(f"Could not initialize Google Drive loader: {e}")
                    self.google_drive_loader = None
            
            span.set_attribute("init.status", "success")
            logger.info("Backend service initialized")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        with tracer.start_as_current_span("file.calculate_hash") as span:
            span.set_attribute("file.path", file_path)
            
            hash_sha256 = hashlib.sha256()
            try:
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_sha256.update(chunk)
                file_hash = hash_sha256.hexdigest()
                span.set_attribute("file.hash", file_hash[:16] + "...")  # Truncated
                span.set_attribute("hash.calculation.status", "success")
                return file_hash
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("hash.calculation.status", "error")
                logger.error(f"Failed to calculate hash for {file_path}: {e}")
                return ""
    
    def scan_local_directories(self) -> List[Dict]:
        """Scan local directories for new files"""
        with tracer.start_as_current_span("scan.local_directories") as span:
            span.set_attribute("directories.count", len(self.local_watch_dirs))
            new_files = []
            
            for watch_dir in self.local_watch_dirs:
                with tracer.start_as_current_span(f"scan.directory.{os.path.basename(watch_dir)}") as dir_span:
                    dir_span.set_attribute("directory.path", watch_dir)
                    
                    if not os.path.exists(watch_dir):
                        dir_span.set_attribute("directory.exists", False)
                        logger.warning(f"Watch directory does not exist: {watch_dir}")
                        continue
                    
                    dir_span.set_attribute("directory.exists", True)
                    files_found = 0
                    files_processed = 0
                    
                    try:
                        for root, dirs, files in os.walk(watch_dir):
                            for file in files:
                                files_found += 1
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
                                files_processed += 1
                                
                    except Exception as e:
                        dir_span.record_exception(e)
                        logger.error(f"Error scanning directory {watch_dir}: {e}")
                        self.stats["errors"].append(f"Directory scan error: {e}")
                    
                    dir_span.set_attribute("scan.files_found", files_found)
                    dir_span.set_attribute("scan.files_new", files_processed)
            
            span.set_attribute("scan.total_new_files", len(new_files))
            return new_files
    
    def scan_google_drive(self) -> List[Dict]:
        """Scan Google Drive for new files"""
        with tracer.start_as_current_span("scan.google_drive") as span:
            if not self.google_drive_loader:
                span.set_attribute("google_drive.available", False)
                logger.warning("Google Drive loader not available")
                return []
            
            span.set_attribute("google_drive.available", True)
            span.set_attribute("google_drive.folder_id", self.google_drive_folder_id)
            
            new_files = []
            try:
                logger.info("Scanning Google Drive for files...")
                
                # Get all files from Google Drive (recursively from all subfolders)
                with tracer.start_as_current_span("google_drive.list_files") as list_span:
                    files = self.google_drive_loader._list_files(self.google_drive_folder_id)
                    list_span.set_attribute("files.total_found", len(files))
                    logger.info(f"Found {len(files)} total files in Google Drive")
                
                files_processed = 0
                files_skipped = 0
                
                for file_info in files:
                    with tracer.start_as_current_span("google_drive.process_file") as file_span:
                        file_id = file_info["id"]
                        file_name = file_info["name"]
                        mime_type = file_info.get("mimeType", "unknown")
                        
                        file_span.set_attribute("file.id", file_id)
                        file_span.set_attribute("file.name", file_name)
                        file_span.set_attribute("file.mime_type", mime_type)
                        
                        # Skip Google Apps files that we can't process directly
                        if mime_type.startswith("application/vnd.google-apps.") and mime_type not in [
                            "application/vnd.google-apps.document",
                            "application/vnd.google-apps.spreadsheet", 
                            "application/vnd.google-apps.presentation"
                        ]:
                            file_span.set_attribute("file.skipped", "unsupported_google_app")
                            files_skipped += 1
                            continue
                        
                        # For Google Drive files, we use file_id + name as the path
                        file_path = f"gdrive://{file_id}/{file_name}"
                        
                        # Use file_id + name as hash since we can't get actual file hash without downloading
                        file_hash = hashlib.md5(f"{file_id}_{file_name}_{mime_type}".encode()).hexdigest()
                        
                        # Check if already processed (both in local DB and Qdrant)
                        if (self.fingerprint_db.is_file_processed(file_path, file_hash) or 
                            self.fingerprint_db.is_file_in_qdrant(file_path, self.processor.qdrant_client)):
                            file_span.set_attribute("file.skipped", "already_processed")
                            files_skipped += 1
                            continue
                        
                        file_span.set_attribute("file.status", "new")
                        logger.info(f"Found new file: {file_name} (Type: {mime_type})")
                        
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
                        files_processed += 1
                
                span.set_attribute("scan.files_new", files_processed)
                span.set_attribute("scan.files_skipped", files_skipped)
                span.set_attribute("scan.status", "success")
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("scan.status", "error")
                logger.error(f"Error scanning Google Drive: {e}")
                self.stats["errors"].append(f"Google Drive scan error: {e}")
            
            return new_files
    
    def cleanup_extracted_images(self, documents: List[Document]):
        """Clean up locally extracted images after processing"""
        with tracer.start_as_current_span("cleanup.extracted_images") as span:
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
                    except OSError as e:
                        logger.debug(f"Could not remove image {img_path}: {e}")
                
                span.set_attribute("cleanup.images_cleaned", cleaned_count)
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} extracted image files")
                    
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Error during image cleanup: {e}")

    def process_file(self, file_info: Dict) -> bool:
        """Process a single file"""
        with tracer.start_as_current_span("file.process") as span:
            file_path = file_info["file_path"]
            file_name = file_info["file_name"]
            source = file_info["source"]
            
            span.set_attribute("file.name", file_name)
            span.set_attribute("file.path", file_path)
            span.set_attribute("file.source", source)
            span.set_attribute("file.size", file_info.get("file_size", 0))
            span.set_attribute("file.mime_type", file_info.get("mime_type", "unknown"))
            
            try:
                logger.info(f"Processing {source} file: {file_name}")
                
                # Load documents
                documents = []
                temp_file = None
                
                with time_document_processing(source=source, file_type=file_info.get("mime_type", "unknown")):
                    if source == "local":
                        # Load local file directly
                        with tracer.start_as_current_span("file.load_local") as load_span:
                            documents = load_file(file_path)
                            load_span.set_attribute("documents.loaded", len(documents))
                            # For local files, we need to clean up extracted images manually
                            self.cleanup_extracted_images(documents)
                    elif source == "google_drive":
                        # Download from Google Drive first
                        with tracer.start_as_current_span("google_drive.download") as download_span:
                            drive_file_info = file_info["drive_file_info"]
                            temp_file = self.google_drive_loader._download_file(drive_file_info)
                            download_span.set_attribute("temp_file.path", temp_file)
                            
                            # Update file size with actual downloaded size
                            if temp_file and os.path.exists(temp_file):
                                file_info["file_size"] = os.path.getsize(temp_file)
                                download_span.set_attribute("file.downloaded_size", file_info["file_size"])
                        
                        with tracer.start_as_current_span("file.load_google_drive") as load_span:
                            documents = load_file(temp_file)
                            load_span.set_attribute("documents.loaded", len(documents))
                            
                            # Process documents with Google Drive loader to handle image upload and cleanup
                            self.google_drive_loader._process_docs(documents, drive_file_info)
                
                if not documents:
                    span.set_attribute("processing.status", "no_documents")
                    logger.warning(f"No documents extracted from {file_name}")
                    return False
                
                span.set_attribute("documents.extracted", len(documents))
                logger.info(f"Extracted {len(documents)} documents from {file_name}")
                
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
                
                span.set_attribute("processing.status", "success")
                span.set_attribute("qdrant.points_created", len(point_ids))
                
                logger.info(f"Successfully processed {file_name} -> {len(documents)} documents, {len(point_ids)} points")
                
                # Clean up temporary file
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        logger.debug(f"Cleaned up temporary file: {os.path.basename(temp_file)}")
                    except Exception as e:
                        logger.debug(f"Could not remove temporary file {temp_file}: {e}")
                
                # Final cleanup: remove any remaining extracted images that might have been missed
                if source == "local":
                    self.cleanup_extracted_images(documents)
                
                return True
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("processing.status", "error")
                logger.error(f"Failed to process file {file_info.get('file_name', 'unknown')}: {e}")
                self.stats["errors"].append(f"File processing error: {e}")
                return False
    
    def scan_and_process(self):
        """Scan for new files and process them"""
        with tracer.start_as_current_span("service.scan_and_process") as span:
            try:
                logger.info("Starting scan cycle...")
                
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
                span.set_attribute("scan.files_found", len(new_files))
                
                if not new_files:
                    span.set_attribute("scan.result", "no_new_files")
                    logger.info("No new files found")
                    return
                
                logger.info(f"Found {len(new_files)} new files to process:")
                for file_info in new_files:
                    logger.info(f"   - {file_info['file_name']} ({file_info['source']})")
                
                # Process each file
                processed_count = 0
                failed_count = 0
                
                for file_info in new_files:
                    if not self.is_running:
                        span.add_event("service_stopping")
                        logger.info("Service stopping, halting file processing")
                        break
                    
                    try:
                        logger.info(f"Processing file {processed_count + 1}/{len(new_files)}: {file_info['file_name']}")
                        
                        if self.process_file(file_info):
                            processed_count += 1
                            logger.info(f"Successfully processed: {file_info['file_name']}")
                        else:
                            failed_count += 1
                            logger.warning(f"Failed to process: {file_info['file_name']}")
                            
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Error processing file {file_info.get('file_name')}: {e}")
                        continue
                    
                    # Small delay between files to avoid overwhelming the system
                    time.sleep(2)
                
                span.set_attribute("scan.files_processed", processed_count)
                span.set_attribute("scan.files_failed", failed_count)
                span.set_attribute("scan.result", "completed")
                
                logger.info(f"Scan and process cycle completed: {processed_count} processed, {failed_count} failed")
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("scan.result", "error")
                logger.error(f"Error in scan and process cycle: {e}")
                self.stats["errors"].append(f"Scan cycle error: {e}")
    
    async def start_monitoring(self):
        """Start the continuous monitoring service"""
        with tracer.start_as_current_span("service.start_monitoring") as span:
            self.is_running = True
            span.set_attribute("monitoring.scan_interval", self.scan_interval)
            span.set_attribute("monitoring.status", "started")
            
            logger.info(f"Starting continuous monitoring service...")
            logger.info(f"Scan interval: {self.scan_interval} seconds")
            
            while self.is_running:
                try:
                    self.scan_and_process()
                    
                    # Wait for next scan
                    for _ in range(self.scan_interval):
                        if not self.is_running:
                            break
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    self.stats["errors"].append(f"Monitoring loop error: {e}")
                    await asyncio.sleep(10)  # Wait before retrying
            
            span.set_attribute("monitoring.status", "stopped")
            logger.info("Monitoring service stopped")
    
    def stop_monitoring(self):
        """Stop the monitoring service"""
        with tracer.start_as_current_span("service.stop_monitoring") as span:
            self.is_running = False
            span.set_attribute("monitoring.action", "stop_requested")
            logger.info("Stopping monitoring service...")
    
    def get_status(self) -> Dict:
        """Get service status"""
        with tracer.start_as_current_span("service.get_status") as span:
            db_stats = self.fingerprint_db.get_stats()
            recent_files = self.fingerprint_db.get_processed_files(limit=5)
            
            status = {
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
            
            span.set_attribute("status.files_processed", status["processing"]["files_processed_session"])
            span.set_attribute("status.is_running", status["service"]["is_running"])
            
            return status

# Global service instance
service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management"""
    global service
    
    with tracer.start_as_current_span("fastapi.lifespan") as span:
        # Startup
        logger.info("Starting Backend Service...")
        service = BackendService()
        
        # Start monitoring in background
        monitoring_task = asyncio.create_task(service.start_monitoring())
        span.add_event("monitoring_started")
        
        yield
        
        # Shutdown
        span.add_event("shutdown_initiated")
        logger.info("Shutting down Backend Service...")
        if service:
            service.stop_monitoring()
        monitoring_task.cancel()
        span.set_attribute("shutdown.status", "completed")

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
    with tracer.start_as_current_span("api.root") as span:
        span.set_attribute("endpoint", "/")
        return {"message": "Document Processing Backend Service", "status": "running"}

@app.get("/status")
async def get_status():
    """Get service status"""
    with tracer.start_as_current_span("api.get_status") as span:
        span.set_attribute("endpoint", "/status")
        
        if not service:
            span.set_attribute("api.error", "service_not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        status = service.get_status()
        span.set_attribute("api.response.files_processed", status["processing"]["files_processed_session"])
        
        return status

@app.post("/scan")
async def trigger_scan():
    """Manually trigger a scan for new files"""
    with tracer.start_as_current_span("api.trigger_scan") as span:
        span.set_attribute("endpoint", "/scan")
        
        if not service:
            span.set_attribute("api.error", "service_not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not service.is_running:
            span.set_attribute("api.error", "service_not_running")
            raise HTTPException(status_code=503, detail="Service not running")
        
        # Trigger scan in background
        asyncio.create_task(asyncio.to_thread(service.scan_and_process))
        
        span.set_attribute("api.action", "scan_triggered")
        logger.info("Manual scan triggered via API")
        
        return {"message": "Scan triggered successfully"}

@app.post("/stop")
async def stop_service():
    """Stop the monitoring service"""
    with tracer.start_as_current_span("api.stop_service") as span:
        span.set_attribute("endpoint", "/stop")
        
        if not service:
            span.set_attribute("api.error", "service_not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        service.stop_monitoring()
        span.set_attribute("api.action", "service_stopped")
        logger.info("Service stopped via API")
        
        return {"message": "Service stopped"}

@app.post("/start")
async def start_service():
    """Start the monitoring service"""
    with tracer.start_as_current_span("api.start_service") as span:
        span.set_attribute("endpoint", "/start")
        
        if not service:
            span.set_attribute("api.error", "service_not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if service.is_running:
            span.set_attribute("api.warning", "service_already_running")
            return {"message": "Service already running"}
        
        # Start monitoring in background
        asyncio.create_task(service.start_monitoring())
        span.set_attribute("api.action", "service_started")
        logger.info("Service started via API")
        
        return {"message": "Service started"}

@app.post("/reset")
async def reset_processed_files():
    """Clear the processed files database to force reprocessing"""
    with tracer.start_as_current_span("api.reset_processed_files") as span:
        span.set_attribute("endpoint", "/reset")
        
        if not service:
            span.set_attribute("api.error", "service_not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            service.fingerprint_db.clear_processed_files()
            span.set_attribute("api.action", "database_cleared")
            logger.info("Processed files database cleared via API")
            
            return {"message": "Processed files database cleared successfully"}
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("api.error", "database_clear_failed")
            raise HTTPException(status_code=500, detail=f"Failed to clear database: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Processing Backend Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "backend_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )