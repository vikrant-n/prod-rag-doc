#!/usr/bin/env python3
"""
Enhanced Backend Service with Granular Service Components
Creates a hierarchical service tree with dedicated components and proper W3C propagation
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

# Force service name early
os.environ["OTEL_SERVICE_NAME"] = "document-rag-backend"

# FastAPI imports for status API
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import httpx

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

# Enhanced OpenTelemetry configuration with W3C propagation
from otel_config import (
    initialize_opentelemetry, get_service_tracer, traced_function, 
    trace_http_call, trace_health_check, get_current_trace_id,
    add_trace_correlation_to_log, inject_trace_context, extract_trace_context,
    SERVICE_HIERARCHY
)
from metrics import (
    rag_metrics, time_document_processing, time_query_processing, 
    record_document_processed, record_cache_event
)

# Initialize OpenTelemetry with proper hierarchy
tracer, meter = initialize_opentelemetry("document-rag-backend")

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
    qdrant_point_ids: List[str]

class FileFingerprintDatabase:
    """File tracking database component with enhanced tracing"""
    
    def __init__(self, db_path: str = ".processed_files.db"):
        self.db_path = db_path
        self.tracer = get_service_tracer("file-fingerprint-db")
        self.service_name = "file-fingerprint-db"
        self._init_db()
    
    @traced_function(service_name="file-fingerprint-db")
    def _init_db(self):
        """Initialize the database schema with proper tracing"""
        with self.tracer.start_as_current_span("database.init_schema") as span:
            span.set_attribute("database.path", self.db_path)
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-backend")
            
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
                        qdrant_point_ids TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON processed_files(file_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_file_name ON processed_files(file_name)")
                span.set_attribute("database.init.status", "success")
    
    @traced_function(service_name="file-fingerprint-db")
    def is_file_processed(self, file_path: str, file_hash: str) -> bool:
        """Check if file already processed with enhanced tracing"""
        with self.tracer.start_as_current_span("database.check_file_processed") as span:
            span.set_attribute("file.path", file_path)
            span.set_attribute("file.hash", file_hash[:16])
            span.set_attribute("service.component", self.service_name)
            
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
    
    @traced_function(service_name="file-fingerprint-db")
    def mark_file_processed(self, processed_file: ProcessedFile):
        """Mark file as processed with enhanced tracing"""
        with self.tracer.start_as_current_span("database.mark_file_processed") as span:
            span.set_attribute("file.name", processed_file.file_name)
            span.set_attribute("file.document_count", processed_file.document_count)
            span.set_attribute("service.component", self.service_name)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO processed_files 
                    (file_id, file_name, file_path, file_hash, processed_at, 
                     document_count, file_size, mime_type, qdrant_point_ids)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    processed_file.file_id, processed_file.file_name, processed_file.file_path,
                    processed_file.file_hash, processed_file.processed_at, processed_file.document_count,
                    processed_file.file_size, processed_file.mime_type,
                    json.dumps(processed_file.qdrant_point_ids)
                ))
                span.set_attribute("database.operation", "file_marked_processed")
    
    @traced_function(service_name="file-fingerprint-db")
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics with enhanced tracing"""
        with self.tracer.start_as_current_span("database.get_stats") as span:
            span.set_attribute("service.component", self.service_name)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_files,
                        SUM(document_count) as total_documents,
                        SUM(file_size) as total_size_bytes,
                        MAX(processed_at) as last_processed
                    FROM processed_files
                """)
                result = cursor.fetchone()
                
                stats = {
                    "total_files": result[0] or 0,
                    "total_documents": result[1] or 0,
                    "total_size_bytes": result[2] or 0,
                    "last_processed": result[3]
                }
                
                span.set_attribute("stats.total_files", stats["total_files"])
                span.set_attribute("stats.total_documents", stats["total_documents"])
                
                return stats

class GoogleDriveMonitor:
    """Google Drive monitoring component with enhanced tracing"""
    
    def __init__(self, folder_id: str):
        self.folder_id = folder_id
        self.tracer = get_service_tracer("google-drive-monitor")
        self.service_name = "google-drive-monitor"
        self.loader = None
        self._init_loader()
    
    @traced_function(service_name="google-drive-monitor")
    def _init_loader(self):
        """Initialize Google Drive loader with proper tracing"""
        with self.tracer.start_as_current_span("google_drive.init_loader") as span:
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-backend")
            
            try:
                credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
                token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
                
                span.set_attribute("credentials.path", credentials_path)
                span.set_attribute("folder.id", self.folder_id)
                
                self.loader = GoogleDriveMasterLoader(
                    folder_id=self.folder_id,
                    credentials_path=credentials_path,
                    token_path=token_path,
                    split=False
                )
                span.set_attribute("loader.init.status", "success")
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("loader.init.status", "failed")
                raise
    
    @traced_function(service_name="google-drive-monitor")
    def scan_for_files(self) -> List[Dict]:
        """Scan Google Drive for new files with enhanced tracing"""
        with self.tracer.start_as_current_span("google_drive.scan_files") as span:
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("folder.id", self.folder_id)
            
            if not self.loader:
                span.set_attribute("scan.status", "no_loader")
                return []
            
            try:
                files = self.loader._list_files(self.folder_id)
                span.set_attribute("scan.files_found", len(files))
                
                new_files = []
                for file_info in files:
                    with self.tracer.start_as_current_span("google_drive.process_file_info") as file_span:
                        file_id = file_info["id"]
                        file_name = file_info["name"]
                        mime_type = file_info.get("mimeType", "unknown")
                        
                        file_span.set_attribute("file.id", file_id)
                        file_span.set_attribute("file.name", file_name)
                        file_span.set_attribute("file.mime_type", mime_type)
                        file_span.set_attribute("service.component", self.service_name)
                        
                        # Skip unsupported Google Apps files
                        if mime_type.startswith("application/vnd.google-apps.") and mime_type not in [
                            "application/vnd.google-apps.document",
                            "application/vnd.google-apps.spreadsheet", 
                            "application/vnd.google-apps.presentation"
                        ]:
                            file_span.set_attribute("file.skipped", "unsupported")
                            continue
                        
                        file_path = f"gdrive://{file_id}/{file_name}"
                        file_hash = hashlib.md5(f"{file_id}_{file_name}_{mime_type}".encode()).hexdigest()
                        
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
                
                span.set_attribute("scan.new_files", len(new_files))
                return new_files
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("scan.status", "error")
                return []

class LocalFileScanner:
    """Local file system scanner component with enhanced tracing"""
    
    def __init__(self, watch_dirs: List[str]):
        self.watch_dirs = watch_dirs
        self.tracer = get_service_tracer("local-file-scanner")
        self.service_name = "local-file-scanner"
        self.supported_exts = {'.pdf', '.docx', '.pptx', '.txt', '.csv', '.xlsx', '.xls', 
                              '.png', '.jpg', '.jpeg', '.gif', '.tiff', '.tif', '.bmp', '.webp'}
    
    @traced_function(service_name="local-file-scanner")
    def scan_directories(self) -> List[Dict]:
        """Scan local directories for files with enhanced tracing"""
        with self.tracer.start_as_current_span("local_scanner.scan_directories") as span:
            span.set_attribute("directories.count", len(self.watch_dirs))
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-backend")
            
            new_files = []
            for watch_dir in self.watch_dirs:
                with self.tracer.start_as_current_span(f"local_scanner.scan_dir") as dir_span:
                    dir_span.set_attribute("directory.path", watch_dir)
                    dir_span.set_attribute("service.component", self.service_name)
                    
                    if not os.path.exists(watch_dir):
                        dir_span.set_attribute("directory.exists", False)
                        continue
                    
                    files_found = self._scan_directory(watch_dir, new_files)
                    dir_span.set_attribute("files.found", files_found)
            
            span.set_attribute("scan.total_files", len(new_files))
            return new_files
    
    def _scan_directory(self, directory: str, new_files: List[Dict]) -> int:
        """Scan single directory with proper tracing"""
        files_found = 0
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.startswith('.'):
                        continue
                    
                    ext = os.path.splitext(file)[1].lower()
                    if ext not in self.supported_exts:
                        continue
                    
                    file_path = os.path.join(root, file)
                    file_hash = self._calculate_file_hash(file_path)
                    if not file_hash:
                        continue
                    
                    stat = os.stat(file_path)
                    new_files.append({
                        "file_id": file_hash,
                        "file_name": file,
                        "file_path": file_path,
                        "file_hash": file_hash,
                        "file_size": stat.st_size,
                        "mime_type": f"application/{ext[1:]}",
                        "source": "local"
                    })
                    files_found += 1
        except Exception as e:
            logging.error(f"Error scanning {directory}: {e}")
        
        return files_found
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash with tracing"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""

class DocumentProcessor:
    """Document processing component with sub-components and enhanced tracing"""
    
    def __init__(self):
        self.tracer = get_service_tracer("document-processor")
        self.service_name = "document-processor"
        
        # Sub-component tracers with proper hierarchy
        self.pdf_tracer = get_service_tracer("pdf-loader")
        self.docx_tracer = get_service_tracer("docx-loader")
        self.pptx_tracer = get_service_tracer("pptx-loader")
        self.image_tracer = get_service_tracer("image-processor")
    
    @traced_function(service_name="document-processor")
    def process_file(self, file_path: str, file_info: Dict) -> List[Document]:
        """Process file with appropriate loader and enhanced tracing"""
        with self.tracer.start_as_current_span("document_processor.process_file") as span:
            file_type = file_info.get("mime_type", "")
            span.set_attribute("file.type", file_type)
            span.set_attribute("file.path", file_path)
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-backend")
            
            try:
                # Route to appropriate sub-component with proper context propagation
                if "pdf" in file_type:
                    documents = self._process_pdf(file_path)
                elif "word" in file_type or file_path.endswith('.docx'):
                    documents = self._process_docx(file_path)
                elif "presentation" in file_type or file_path.endswith('.pptx'):
                    documents = self._process_pptx(file_path)
                elif any(ext in file_path.lower() for ext in ['.png', '.jpg', '.jpeg']):
                    documents = self._process_image(file_path)
                else:
                    documents = self._process_generic(file_path)
                
                span.set_attribute("documents.extracted", len(documents))
                span.set_attribute("processing.status", "success")
                return documents
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("processing.status", "error")
                return []
    
    @traced_function(service_name="pdf-loader")
    def _process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF file with enhanced tracing"""
        with self.pdf_tracer.start_as_current_span("pdf_loader.process") as span:
            span.set_attribute("file.path", file_path)
            span.set_attribute("service.component", "pdf-loader")
            span.set_attribute("service.parent", "document-processor")
            span.set_attribute("service.hierarchy.level", 3)
            
            documents = load_file(file_path)
            span.set_attribute("pdf.documents_extracted", len(documents))
            return documents
    
    @traced_function(service_name="docx-loader")
    def _process_docx(self, file_path: str) -> List[Document]:
        """Process DOCX file with enhanced tracing"""
        with self.docx_tracer.start_as_current_span("docx_loader.process") as span:
            span.set_attribute("file.path", file_path)
            span.set_attribute("service.component", "docx-loader")
            span.set_attribute("service.parent", "document-processor")
            span.set_attribute("service.hierarchy.level", 3)
            
            documents = load_file(file_path)
            span.set_attribute("docx.documents_extracted", len(documents))
            return documents
    
    @traced_function(service_name="pptx-loader")
    def _process_pptx(self, file_path: str) -> List[Document]:
        """Process PPTX file with enhanced tracing"""
        with self.pptx_tracer.start_as_current_span("pptx_loader.process") as span:
            span.set_attribute("file.path", file_path)
            span.set_attribute("service.component", "pptx-loader")
            span.set_attribute("service.parent", "document-processor")
            span.set_attribute("service.hierarchy.level", 3)
            
            documents = load_file(file_path)
            span.set_attribute("pptx.documents_extracted", len(documents))
            return documents
    
    @traced_function(service_name="image-processor")
    def _process_image(self, file_path: str) -> List[Document]:
        """Process image file with enhanced tracing"""
        with self.image_tracer.start_as_current_span("image_processor.process") as span:
            span.set_attribute("file.path", file_path)
            span.set_attribute("service.component", "image-processor")
            span.set_attribute("service.parent", "document-processor")
            span.set_attribute("service.hierarchy.level", 3)
            
            documents = load_file(file_path)
            span.set_attribute("image.documents_extracted", len(documents))
            return documents
    
    def _process_generic(self, file_path: str) -> List[Document]:
        """Process generic file with basic tracing"""
        return load_file(file_path)

class TextSplitter:
    """Text splitting component with enhanced tracing"""
    
    def __init__(self):
        self.tracer = get_service_tracer("text-splitter")
        self.service_name = "text-splitter"
    
    @traced_function(service_name="text-splitter")
    def split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 120) -> List[Document]:
        """Split documents into chunks with enhanced tracing"""
        with self.tracer.start_as_current_span("text_splitter.split_documents") as span:
            span.set_attribute("documents.input_count", len(documents))
            span.set_attribute("split.chunk_size", chunk_size)
            span.set_attribute("split.chunk_overlap", chunk_overlap)
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-backend")
            
            chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            span.set_attribute("documents.output_count", len(chunks))
            span.set_attribute("splitting.status", "success")
            return chunks

class EmbeddingGenerator:
    """Embedding generation component with enhanced tracing"""
    
    def __init__(self, model: str = "text-embedding-3-large"):
        self.model = model
        self.tracer = get_service_tracer("embedding-generator")
        self.service_name = "embedding-generator"
        self.embeddings = OpenAIEmbeddings(model=model)
    
    @traced_function(service_name="embedding-generator")
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts with enhanced tracing"""
        with self.tracer.start_as_current_span("embedding_generator.generate") as span:
            span.set_attribute("texts.count", len(texts))
            span.set_attribute("embedding.model", self.model)
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-backend")
            span.set_attribute("service.external", "openai-api")
            
            try:
                embeddings = self.embeddings.embed_documents(texts)
                span.set_attribute("embeddings.generated", len(embeddings))
                span.set_attribute("embedding.dimension", len(embeddings[0]) if embeddings else 0)
                span.set_attribute("embedding.status", "success")
                return embeddings
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("embedding.status", "failed")
                raise

class VectorStoreManager:
    """Vector store management component with enhanced tracing"""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "documents"):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.tracer = get_service_tracer("vector-store-manager")
        self.service_name = "vector-store-manager"
        
        self.client = QdrantClient(url=qdrant_url)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self._ensure_collection()
    
    @traced_function(service_name="vector-store-manager")
    def _ensure_collection(self):
        """Ensure collection exists with enhanced tracing"""
        with self.tracer.start_as_current_span("vector_store.ensure_collection") as span:
            span.set_attribute("collection.name", self.collection_name)
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-backend")
            span.set_attribute("service.external", "qdrant-database")
            
            try:
                collections = self.client.get_collections()
                collection_names = [col.name for col in collections.collections]
                
                if self.collection_name not in collection_names:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
                    )
                    span.set_attribute("collection.created", True)
                else:
                    span.set_attribute("collection.created", False)
                
                self.vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings
                )
                
                span.set_attribute("vector_store.initialized", True)
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("vector_store.initialized", False)
                raise
    
    @traced_function(service_name="vector-store-manager")
    def add_documents(self, documents: List[Document], file_info: Dict) -> List[str]:
        """Add documents to vector store with enhanced tracing"""
        with self.tracer.start_as_current_span("vector_store.add_documents") as span:
            span.set_attribute("documents.count", len(documents))
            span.set_attribute("file.name", file_info.get("file_name", "unknown"))
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.external", "qdrant-database")
            
            try:
                # Add metadata with trace context
                trace_id = get_current_trace_id()
                for doc in documents:
                    doc.metadata.update({
                        "file_id": file_info.get("file_id"),
                        "file_name": file_info.get("file_name"),
                        "file_path": file_info.get("file_path"),
                        "processed_at": datetime.now().isoformat(),
                        "file_size": file_info.get("file_size", 0),
                        "mime_type": file_info.get("mime_type"),
                        "trace_id": trace_id,
                        "processed_by_service": "document-rag-backend"
                    })
                
                # Generate point IDs
                point_ids = [
                    hashlib.md5(f"{file_info['file_path']}_{i}_{doc.page_content[:100]}".encode()).hexdigest()
                    for i, doc in enumerate(documents)
                ]
                
                # Add to vector store
                self.vector_store.add_documents(documents, ids=point_ids)
                
                span.set_attribute("vector_store.points_added", len(point_ids))
                span.set_attribute("vector_store.status", "success")
                return point_ids
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("vector_store.status", "failed")
                raise

class BackendService:
    """Main backend service orchestrator with enhanced W3C propagation"""
    
    def __init__(self):
        # Initialize main service tracer with proper hierarchy
        self.tracer, self.meter = initialize_opentelemetry(
            service_name="document-rag-backend",
            service_version="1.0.0",
            environment="production"
        )
        
        self.service_name = "document-rag-backend"
        
        # Configuration
        self.google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        self.local_watch_dirs = os.getenv("LOCAL_WATCH_DIRS", "").split(",") if os.getenv("LOCAL_WATCH_DIRS") else []
        self.scan_interval = int(os.getenv("SCAN_INTERVAL", "30"))
        
        # Initialize components with proper hierarchy
        self.fingerprint_db = FileFingerprintDatabase()
        self.document_processor = DocumentProcessor()
        self.text_splitter = TextSplitter()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store_manager = VectorStoreManager()
        
        # Optional components
        self.google_drive_monitor = None
        if self.google_drive_folder_id:
            self.google_drive_monitor = GoogleDriveMonitor(self.google_drive_folder_id)
        
        self.local_scanner = None
        if self.local_watch_dirs:
            self.local_scanner = LocalFileScanner(self.local_watch_dirs)
        
        # Service state
        self.is_running = False
        self.stats = {
            "service_started": datetime.now(),
            "files_processed": 0,
            "documents_created": 0,
            "last_scan": None,
            "errors": []
        }
        
        # Setup logger
        self.logger = add_trace_correlation_to_log(logging.getLogger(__name__))
        self.logger.info("Backend service initialized with enhanced W3C tracing")
    
    @traced_function(service_name="document-rag-backend")
    def scan_for_new_files(self) -> List[Dict]:
        """Scan all sources for new files with enhanced tracing"""
        with self.tracer.start_as_current_span("backend.scan_for_new_files") as span:
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.hierarchy", "document-rag-orchestrator -> document-rag-backend")
            
            all_files = []
            
            # Scan Google Drive with context propagation
            if self.google_drive_monitor:
                with self.tracer.start_as_current_span("backend.scan_google_drive") as gdrive_span:
                    gdrive_files = self.google_drive_monitor.scan_for_files()
                    all_files.extend(gdrive_files)
                    gdrive_span.set_attribute("google_drive.files_found", len(gdrive_files))
                    gdrive_span.set_attribute("service.external", "google-drive-api")
            
            # Scan local directories with context propagation
            if self.local_scanner:
                with self.tracer.start_as_current_span("backend.scan_local") as local_span:
                    local_files = self.local_scanner.scan_directories()
                    all_files.extend(local_files)
                    local_span.set_attribute("local.files_found", len(local_files))
            
            # Filter already processed files
            new_files = []
            for file_info in all_files:
                if not self.fingerprint_db.is_file_processed(file_info["file_path"], file_info["file_hash"]):
                    new_files.append(file_info)
            
            span.set_attribute("scan.total_files", len(all_files))
            span.set_attribute("scan.new_files", len(new_files))
            
            return new_files
    
    @traced_function(service_name="document-rag-backend")
    def process_single_file(self, file_info: Dict) -> bool:
        """Process a single file through the pipeline with enhanced tracing"""
        with self.tracer.start_as_current_span("backend.process_single_file") as span:
            file_name = file_info["file_name"]
            source = file_info["source"]
            
            span.set_attribute("file.name", file_name)
            span.set_attribute("file.source", source)
            span.set_attribute("file.size", file_info.get("file_size", 0))
            span.set_attribute("service.component", self.service_name)
            
            try:
                # Download/prepare file
                temp_file = None
                if source == "google_drive":
                    with self.tracer.start_as_current_span("backend.download_google_drive_file") as download_span:
                        temp_file = self.google_drive_monitor.loader._download_file(file_info["drive_file_info"])
                        file_path = temp_file
                        download_span.set_attribute("temp_file.path", temp_file)
                        download_span.set_attribute("service.external", "google-drive-api")
                else:
                    file_path = file_info["file_path"]
                
                # Process document with proper context propagation
                with self.tracer.start_as_current_span("backend.document_processing") as doc_span:
                    documents = self.document_processor.process_file(file_path, file_info)
                    doc_span.set_attribute("documents.extracted", len(documents))
                
                if not documents:
                    span.set_attribute("processing.status", "no_documents")
                    return False
                
                # Split documents
                with self.tracer.start_as_current_span("backend.text_splitting") as split_span:
                    chunks = self.text_splitter.split_documents(documents)
                    split_span.set_attribute("chunks.created", len(chunks))
                
                # Add to vector store
                with self.tracer.start_as_current_span("backend.vector_storage") as vector_span:
                    point_ids = self.vector_store_manager.add_documents(chunks, file_info)
                    vector_span.set_attribute("points.stored", len(point_ids))
                
                # Mark as processed
                processed_file = ProcessedFile(
                    file_id=file_info["file_id"],
                    file_name=file_name,
                    file_path=file_info["file_path"],
                    file_hash=file_info["file_hash"],
                    processed_at=datetime.now(),
                    document_count=len(documents),
                    file_size=file_info["file_size"],
                    mime_type=file_info["mime_type"],
                    qdrant_point_ids=point_ids
                )
                
                self.fingerprint_db.mark_file_processed(processed_file)
                
                # Clean up temp file
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
                
                # Update stats
                self.stats["files_processed"] += 1
                self.stats["documents_created"] += len(documents)
                
                span.set_attribute("processing.status", "success")
                span.set_attribute("processing.documents_created", len(documents))
                span.set_attribute("processing.chunks_created", len(chunks))
                
                return True
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("processing.status", "error")
                self.logger.error(f"Failed to process {file_name}: {e}")
                return False
    
    @traced_function(service_name="document-rag-backend")
    async def scan_and_process_cycle(self):
        """Complete scan and process cycle with enhanced tracing"""
        with self.tracer.start_as_current_span("backend.scan_and_process_cycle") as span:
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("trace.id", get_current_trace_id())
            
            try:
                self.logger.info("Starting scan and process cycle")
                
                # Scan for new files
                new_files = self.scan_for_new_files()
                self.stats["last_scan"] = datetime.now()
                
                span.set_attribute("cycle.files_found", len(new_files))
                
                if not new_files:
                    span.set_attribute("cycle.result", "no_new_files")
                    self.logger.info("No new files found")
                    return
                
                self.logger.info(f"Found {len(new_files)} new files to process")
                
                # Process each file
                processed_count = 0
                failed_count = 0
                
                for file_info in new_files:
                    if not self.is_running:
                        break
                    
                    with self.tracer.start_as_current_span("backend.process_file_iteration") as file_span:
                        file_span.set_attribute("file.name", file_info["file_name"])
                        file_span.set_attribute("service.component", self.service_name)
                        
                        if self.process_single_file(file_info):
                            processed_count += 1
                            file_span.set_attribute("file.processing_result", "success")
                        else:
                            failed_count += 1
                            file_span.set_attribute("file.processing_result", "failed")
                        
                        # Small delay between files
                        await asyncio.sleep(2)
                
                span.set_attribute("cycle.files_processed", processed_count)
                span.set_attribute("cycle.files_failed", failed_count)
                span.set_attribute("cycle.result", "completed")
                
                self.logger.info(f"Cycle completed: {processed_count} processed, {failed_count} failed")
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("cycle.result", "error")
                self.logger.error(f"Error in scan cycle: {e}")
                self.stats["errors"].append(f"Scan cycle error: {e}")
    
    @traced_function(service_name="document-rag-backend")
    async def start_monitoring(self):
        """Start continuous monitoring service with enhanced tracing"""
        with self.tracer.start_as_current_span("backend.start_monitoring") as span:
            self.is_running = True
            span.set_attribute("monitoring.scan_interval", self.scan_interval)
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.status", "running")
            
            self.logger.info(f"Starting continuous monitoring (interval: {self.scan_interval}s)")
            
            while self.is_running:
                try:
                    await self.scan_and_process_cycle()
                    
                    # Wait for next scan
                    for _ in range(self.scan_interval):
                        if not self.is_running:
                            break
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(10)
            
            span.set_attribute("monitoring.status", "stopped")
            self.logger.info("Monitoring service stopped")
    
    @traced_function(service_name="document-rag-backend")
    def stop_monitoring(self):
        """Stop monitoring service with tracing"""
        with self.tracer.start_as_current_span("backend.stop_monitoring") as span:
            self.is_running = False
            span.set_attribute("action", "stop_requested")
            span.set_attribute("service.component", self.service_name)
            self.logger.info("Stopping monitoring service")
    
    def get_service_status(self) -> Dict:
        """Get comprehensive service status with enhanced tracing"""
        with self.tracer.start_as_current_span("backend.get_service_status") as span:
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("status.request_trace_id", get_current_trace_id())
            
            # Get database stats
            db_stats = self.fingerprint_db.get_stats()
            
            # Build component status
            components_status = {}
            
            # Check each component health
            with self.tracer.start_as_current_span("backend.check_components_health") as health_span:
                # Vector store health
                try:
                    collections = self.vector_store_manager.client.get_collections()
                    components_status["vector_store"] = "healthy"
                    health_span.set_attribute("vector_store.status", "healthy")
                except Exception as e:
                    components_status["vector_store"] = f"unhealthy: {str(e)}"
                    health_span.set_attribute("vector_store.status", "unhealthy")
                
                # Google Drive health
                if self.google_drive_monitor:
                    try:
                        if self.google_drive_monitor.loader:
                            components_status["google_drive"] = "healthy"
                        else:
                            components_status["google_drive"] = "not_initialized"
                        health_span.set_attribute("google_drive.status", components_status["google_drive"])
                    except Exception as e:
                        components_status["google_drive"] = f"unhealthy: {str(e)}"
                        health_span.set_attribute("google_drive.status", "unhealthy")
                else:
                    components_status["google_drive"] = "not_configured"
                
                # Local scanner health
                if self.local_scanner:
                    components_status["local_scanner"] = "healthy"
                    health_span.set_attribute("local_scanner.status", "healthy")
                else:
                    components_status["local_scanner"] = "not_configured"
            
            status = {
                "service": {
                    "name": self.service_name,
                    "is_running": self.is_running,
                    "started_at": self.stats["service_started"].isoformat(),
                    "uptime_seconds": (datetime.now() - self.stats["service_started"]).total_seconds(),
                    "last_scan": self.stats["last_scan"].isoformat() if self.stats["last_scan"] else None,
                    "scan_interval": self.scan_interval
                },
                "processing": {
                    "files_processed_session": self.stats["files_processed"],
                    "documents_created_session": self.stats["documents_created"],
                    "total_files_processed": db_stats.get("total_files", 0),
                    "total_documents_created": db_stats.get("total_documents", 0),
                    "total_size_bytes": db_stats.get("total_size_bytes", 0),
                    "last_processed": db_stats.get("last_processed")
                },
                "components": components_status,
                "configuration": {
                    "google_drive_folder_id": self.google_drive_folder_id,
                    "local_watch_dirs": self.local_watch_dirs,
                    "vector_store_url": self.vector_store_manager.qdrant_url,
                    "collection_name": self.vector_store_manager.collection_name
                },
                "health": {
                    "overall_status": "healthy" if all(
                        status != "unhealthy" for status in components_status.values()
                        if not status.startswith("not_")
                    ) else "unhealthy",
                    "components_count": len(components_status),
                    "healthy_components": len([s for s in components_status.values() if s == "healthy"])
                },
                "errors": self.stats["errors"][-10:],  # Last 10 errors
                "trace_context": {
                    "trace_id": get_current_trace_id(),
                    "service_hierarchy": "document-rag-orchestrator -> document-rag-backend",
                    "w3c_propagation": "enabled"
                }
            }
            
            span.set_attribute("status.is_running", self.is_running)
            span.set_attribute("status.files_processed", self.stats["files_processed"])
            span.set_attribute("status.components_healthy", status["health"]["healthy_components"])
            
            return status

# Enhanced FastAPI application with hierarchical tracing and W3C propagation
service_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced FastAPI lifespan with service hierarchy and W3C context"""
    global service_instance
    
    # Initialize with parent service context
    orchestrator_tracer = get_service_tracer("document-rag-orchestrator")
    
    with orchestrator_tracer.start_as_current_span("backend_service.startup") as startup_span:
        startup_span.set_attribute("service.component", "backend")
        startup_span.set_attribute("startup.phase", "initialization")
        startup_span.set_attribute("w3c.propagation", "enabled")
        
        try:
            service_instance = BackendService()
            startup_span.set_attribute("backend.initialization", "success")
            
            # Start monitoring
            monitoring_task = asyncio.create_task(service_instance.start_monitoring())
            startup_span.set_attribute("backend.monitoring", "started")
            
            yield
            
        except Exception as e:
            startup_span.record_exception(e)
            startup_span.set_attribute("backend.initialization", "failed")
            raise
        finally:
            # Shutdown
            startup_span.add_event("shutdown_initiated")
            if service_instance:
                service_instance.stop_monitoring()
            if 'monitoring_task' in locals():
                monitoring_task.cancel()

# Initialize FastAPI with enhanced tracing
app = FastAPI(
    title="Enhanced Document Processing Backend Service",
    description="Hierarchical document processing with granular service components and W3C propagation",
    version="2.0.0",
    lifespan=lifespan
)

# Health check endpoint (REMOVED @traced_function decorator)
@app.get("/health")
async def health_check():
    """Simple health check endpoint with enhanced tracing"""
    with tracer.start_as_current_span("api.health_check") as span:
        span.set_attribute("endpoint", "/health")
        span.set_attribute("service.component", "document-rag-backend")
        
        if not service_instance:
            span.set_attribute("service.status", "not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        span.set_attribute("service.status", "healthy")
        
        return {
            "status": "healthy",
            "service": "document-rag-backend",
            "timestamp": datetime.now().isoformat(),
            "trace_id": get_current_trace_id()
        }

@app.get("/")
async def root():
    """Root endpoint with service identification and W3C context"""
    with tracer.start_as_current_span("api.root") as span:
        span.set_attribute("endpoint", "/")
        span.set_attribute("service.component", "document-rag-backend")
        
        return {
            "service": "document-rag-backend",
            "version": "2.0.0",
            "components": [
                "google-drive-monitor",
                "local-file-scanner", 
                "document-processor",
                "text-splitter",
                "embedding-generator",
                "vector-store-manager",
                "file-fingerprint-db"
            ],
            "sub_components": {
                "document-processor": ["pdf-loader", "docx-loader", "pptx-loader", "image-processor"]
            },
            "status": "running",
            "hierarchy": "document-rag-orchestrator -> document-rag-backend",
            "w3c_propagation": "enabled",
            "trace_id": get_current_trace_id()
        }

@app.get("/status")
async def get_status():
    """Get detailed service status with W3C context propagation"""
    with tracer.start_as_current_span("api.get_status") as span:
        span.set_attribute("endpoint", "/status")
        span.set_attribute("service.component", "document-rag-backend")
        
        if not service_instance:
            span.set_attribute("service.status", "not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        status = service_instance.get_service_status()
        span.set_attribute("status.response_size", len(json.dumps(status)))
        
        return status

@app.post("/scan")
async def trigger_manual_scan():
    """Manually trigger scan cycle with W3C context propagation"""
    with tracer.start_as_current_span("api.trigger_manual_scan") as span:
        span.set_attribute("endpoint", "/scan")
        span.set_attribute("trigger.source", "manual_api")
        span.set_attribute("service.component", "document-rag-backend")
        
        if not service_instance:
            span.set_attribute("service.status", "not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not service_instance.is_running:
            span.set_attribute("service.status", "not_running")
            raise HTTPException(status_code=503, detail="Service not running")
        
        # Trigger scan in background with trace context
        span.set_attribute("scan.triggered", True)
        asyncio.create_task(service_instance.scan_and_process_cycle())
        
        return {
            "message": "Manual scan triggered", 
            "trace_id": get_current_trace_id(),
            "service": "document-rag-backend"
        }

@app.post("/stop")
async def stop_service():
    """Stop the monitoring service with enhanced tracing"""
    with tracer.start_as_current_span("api.stop_service") as span:
        span.set_attribute("endpoint", "/stop")
        span.set_attribute("action", "stop_service")
        
        if not service_instance:
            span.set_attribute("service.status", "not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        service_instance.stop_monitoring()
        span.set_attribute("service.stopped", True)
        
        return {
            "message": "Service stopped", 
            "trace_id": get_current_trace_id()
        }

@app.post("/start")
async def start_service():
    """Start the monitoring service with enhanced tracing"""
    with tracer.start_as_current_span("api.start_service") as span:
        span.set_attribute("endpoint", "/start")
        span.set_attribute("action", "start_service")
        
        if not service_instance:
            span.set_attribute("service.status", "not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if service_instance.is_running:
            span.set_attribute("service.already_running", True)
            return {
                "message": "Service already running", 
                "trace_id": get_current_trace_id()
            }
        
        asyncio.create_task(service_instance.start_monitoring())
        span.set_attribute("service.started", True)
        
        return {
            "message": "Service started", 
            "trace_id": get_current_trace_id()
        }

@app.get("/components")
async def get_components_status():
    """Get detailed status of all service components with W3C context"""
    with tracer.start_as_current_span("api.get_components_status") as span:
        span.set_attribute("endpoint", "/components")
        span.set_attribute("service.component", "document-rag-backend")
        
        if not service_instance:
            span.set_attribute("service.status", "not_initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        components = {}
        
        # Google Drive Monitor
        if service_instance.google_drive_monitor:
            with tracer.start_as_current_span("check.google_drive_monitor") as gdrive_span:
                try:
                    gdrive_status = "healthy" if service_instance.google_drive_monitor.loader else "not_initialized"
                    components["google-drive-monitor"] = {
                        "status": gdrive_status,
                        "folder_id": service_instance.google_drive_folder_id,
                        "service_path": "document-rag-backend -> google-drive-monitor",
                        "external_service": "google-drive-api"
                    }
                    gdrive_span.set_attribute("component.status", gdrive_status)
                except Exception as e:
                    components["google-drive-monitor"] = {"status": "error", "error": str(e)}
        else:
            components["google-drive-monitor"] = {"status": "not_configured"}
        
        # Local File Scanner
        if service_instance.local_scanner:
            components["local-file-scanner"] = {
                "status": "healthy",
                "watch_dirs": service_instance.local_watch_dirs,
                "service_path": "document-rag-backend -> local-file-scanner"
            }
        else:
            components["local-file-scanner"] = {"status": "not_configured"}
        
        # Document Processor
        components["document-processor"] = {
            "status": "healthy",
            "sub_components": ["pdf-loader", "docx-loader", "pptx-loader", "image-processor"],
            "service_path": "document-rag-backend -> document-processor"
        }
        
        # Text Splitter
        components["text-splitter"] = {
            "status": "healthy",
            "service_path": "document-rag-backend -> text-splitter"
        }
        
        # Embedding Generator
        components["embedding-generator"] = {
            "status": "healthy",
            "model": service_instance.embedding_generator.model,
            "service_path": "document-rag-backend -> embedding-generator",
            "external_service": "openai-api"
        }
        
        # Vector Store Manager
        try:
            collections = service_instance.vector_store_manager.client.get_collections()
            components["vector-store-manager"] = {
                "status": "healthy",
                "url": service_instance.vector_store_manager.qdrant_url,
                "collection": service_instance.vector_store_manager.collection_name,
                "collections_count": len(collections.collections),
                "service_path": "document-rag-backend -> vector-store-manager",
                "external_service": "qdrant-database"
            }
        except Exception as e:
            components["vector-store-manager"] = {"status": "unhealthy", "error": str(e)}
        
        # File Fingerprint DB
        components["file-fingerprint-db"] = {
            "status": "healthy",
            "database_path": service_instance.fingerprint_db.db_path,
            "service_path": "document-rag-backend -> file-fingerprint-db"
        }
        
        span.set_attribute("components.total", len(components))
        span.set_attribute("components.healthy", len([c for c in components.values() if c.get("status") == "healthy"]))
        
        return {
            "service": "document-rag-backend",
            "components": components,
            "hierarchy": "document-rag-orchestrator -> document-rag-backend -> [sub-components]",
            "w3c_propagation": "enabled",
            "trace_id": get_current_trace_id()
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Document Processing Backend Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Initialize with orchestrator context
    orchestrator_tracer = get_service_tracer("document-rag-orchestrator")
    
    with orchestrator_tracer.start_as_current_span("backend_service.main_startup") as span:
        span.set_attribute("startup.host", args.host)
        span.set_attribute("startup.port", args.port)
        span.set_attribute("startup.mode", "standalone")
        span.set_attribute("w3c.propagation", "enabled")
        
        print(f"🚀 Starting Enhanced Backend Service on {args.host}:{args.port}")
        print(f"📊 Service Hierarchy: document-rag-orchestrator -> document-rag-backend")
        print(f"🔧 Components: google-drive-monitor, local-file-scanner, document-processor, text-splitter, embedding-generator, vector-store-manager, file-fingerprint-db")
        print(f"🌐 W3C Trace Context: ENABLED")
        print(f"🔗 Propagation: W3C TraceContext + Baggage + B3 + Jaeger")
        
        uvicorn.run(
            "backend_service:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )