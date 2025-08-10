#!/usr/bin/env python3
"""
Enhanced Backend Service with W3C Context Propagation
Reduced lines while maintaining full functionality
"""

import os
import sys
import asyncio
import logging
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from contextlib import asynccontextmanager

# Force service name and extract parent context
os.environ["OTEL_SERVICE_NAME"] = "document-rag-backend"

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# Document processing imports
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Loaders and text splitting
from loaders.master_loaders import load_file
from loaders.google_drive_loader import GoogleDriveMasterLoader
from text_splitting import split_documents

from dotenv import load_dotenv
load_dotenv()

# OpenTelemetry imports - CRITICAL
from otel_config import (
    initialize_opentelemetry, get_service_tracer, instrument_fastapi_app,
    get_current_trace_id, extract_and_activate_context
)

# CRITICAL: Extract parent context from environment
parent_trace_id = os.getenv("OTEL_PARENT_TRACE_ID")
parent_service = os.getenv("OTEL_SERVICE_PARENT", "document-rag-orchestrator")

# Initialize with proper parent context
if parent_trace_id:
    # Create context headers to simulate parent
    context_headers = {
        "traceparent": f"00-{parent_trace_id}-{'0' * 16}-01",
        "x-parent-service": parent_service
    }
    # Extract and activate parent context
    context_token = extract_and_activate_context(context_headers)

# Initialize OpenTelemetry with proper hierarchy
tracer, meter = initialize_opentelemetry("document-rag-backend", "2.0.0", "production")

@dataclass
class ProcessedFile:
    file_id: str
    file_name: str
    file_path: str
    file_hash: str
    processed_at: datetime
    document_count: int
    file_size: int
    mime_type: str

class FileFingerprintDatabase:
    """File tracking with simplified tracing"""
    
    def __init__(self, db_path: str = ".processed_files.db"):
        self.db_path = db_path
        self.tracer = get_service_tracer("file-fingerprint-db")
        self._init_db()
    
    def _init_db(self):
        with self.tracer.start_as_current_span("database.init") as span:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processed_files (
                        file_id TEXT PRIMARY KEY, file_name TEXT NOT NULL,
                        file_path TEXT NOT NULL, file_hash TEXT NOT NULL,
                        processed_at TIMESTAMP NOT NULL, document_count INTEGER NOT NULL,
                        file_size INTEGER NOT NULL, mime_type TEXT
                    )
                """)
            span.set_attribute("database.initialized", True)
    
    def is_file_processed(self, file_path: str, file_hash: str) -> bool:
        with self.tracer.start_as_current_span("check_file_processed") as span:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM processed_files WHERE file_path = ? AND file_hash = ?",
                    (file_path, file_hash)
                )
                is_processed = cursor.fetchone()[0] > 0
                span.set_attribute("file.processed", is_processed)
                return is_processed
    
    def mark_file_processed(self, processed_file: ProcessedFile):
        with self.tracer.start_as_current_span("mark_file_processed"):
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO processed_files 
                    (file_id, file_name, file_path, file_hash, processed_at, 
                     document_count, file_size, mime_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    processed_file.file_id, processed_file.file_name, processed_file.file_path,
                    processed_file.file_hash, processed_file.processed_at, 
                    processed_file.document_count, processed_file.file_size, processed_file.mime_type
                ))

class DocumentProcessor:
    """Simplified document processor"""
    
    def __init__(self):
        self.tracer = get_service_tracer("document-processor")
    
    def process_file(self, file_path: str, file_info: Dict) -> List[Document]:
        with self.tracer.start_as_current_span("process_file") as span:
            span.set_attributes({
                "file.name": file_info.get("file_name"),
                "file.type": file_info.get("mime_type"),
                "service.component": "document-processor"
            })
            
            try:
                documents = load_file(file_path)
                span.set_attribute("documents.extracted", len(documents))
                return documents
            except Exception as e:
                span.record_exception(e)
                return []

class VectorStoreManager:
    """Simplified vector store management"""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "documents"):
        self.tracer = get_service_tracer("vector-store-manager")
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        
        # Initialize clients
        self.client = QdrantClient(url=qdrant_url)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self._ensure_collection()
    
    def _ensure_collection(self):
        with self.tracer.start_as_current_span("ensure_collection") as span:
            try:
                collections = self.client.get_collections()
                if self.collection_name not in [col.name for col in collections.collections]:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
                    )
                
                self.vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings
                )
                span.set_attribute("collection.ready", True)
            except Exception as e:
                span.record_exception(e)
                raise
    
    def add_documents(self, documents: List[Document], file_info: Dict) -> List[str]:
        with self.tracer.start_as_current_span("add_documents") as span:
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "file_id": file_info.get("file_id"),
                    "file_name": file_info.get("file_name"),
                    "processed_at": datetime.now().isoformat(),
                    "trace_id": get_current_trace_id()
                })
            
            # Generate IDs and add to vector store
            point_ids = [
                hashlib.md5(f"{file_info['file_path']}_{i}_{doc.page_content[:50]}".encode()).hexdigest()
                for i, doc in enumerate(documents)
            ]
            
            self.vector_store.add_documents(documents, ids=point_ids)
            span.set_attribute("points.added", len(point_ids))
            return point_ids

class BackendService:
    """Main backend service with proper context propagation"""
    
    def __init__(self):
        self.tracer = tracer
        self.service_name = "document-rag-backend"
        
        # Configuration
        self.google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        self.local_watch_dirs = os.getenv("LOCAL_WATCH_DIRS", "").split(",") if os.getenv("LOCAL_WATCH_DIRS") else []
        self.scan_interval = int(os.getenv("SCAN_INTERVAL", "30"))
        
        # Initialize components
        self.fingerprint_db = FileFingerprintDatabase()
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        
        # Service state
        self.is_running = False
        self.stats = {
            "service_started": datetime.now(),
            "files_processed": 0,
            "documents_created": 0,
            "last_scan": None
        }
        
        # Initialize Google Drive if configured
        self.google_drive_loader = None
        if self.google_drive_folder_id:
            try:
                self.google_drive_loader = GoogleDriveMasterLoader(
                    folder_id=self.google_drive_folder_id,
                    credentials_path=os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json"),
                    token_path=os.getenv("GOOGLE_TOKEN_PATH", "token.json"),
                    split=False
                )
            except Exception as e:
                logging.warning(f"Google Drive not configured: {e}")
    
    def scan_for_new_files(self) -> List[Dict]:
        """Scan for new files with proper tracing"""
        with self.tracer.start_as_current_span("scan_for_new_files") as span:
            all_files = []
            
            # Scan Google Drive
            if self.google_drive_loader:
                with self.tracer.start_as_current_span("scan_google_drive"):
                    try:
                        files = self.google_drive_loader._list_files(self.google_drive_folder_id)
                        for file_info in files:
                            file_path = f"gdrive://{file_info['id']}/{file_info['name']}"
                            file_hash = hashlib.md5(f"{file_info['id']}_{file_info['name']}".encode()).hexdigest()
                            all_files.append({
                                "file_id": file_info["id"],
                                "file_name": file_info["name"],
                                "file_path": file_path,
                                "file_hash": file_hash,
                                "file_size": 0,
                                "mime_type": file_info.get("mimeType", "unknown"),
                                "source": "google_drive",
                                "drive_file_info": file_info
                            })
                    except Exception as e:
                        logging.error(f"Google Drive scan failed: {e}")
            
            # Scan local directories
            for watch_dir in self.local_watch_dirs:
                if os.path.exists(watch_dir):
                    with self.tracer.start_as_current_span("scan_local_directory"):
                        for root, _, files in os.walk(watch_dir):
                            for file in files:
                                if file.startswith('.'):
                                    continue
                                
                                file_path = os.path.join(root, file)
                                try:
                                    file_hash = hashlib.sha256(open(file_path, 'rb').read()).hexdigest()
                                    stat = os.stat(file_path)
                                    all_files.append({
                                        "file_id": file_hash,
                                        "file_name": file,
                                        "file_path": file_path,
                                        "file_hash": file_hash,
                                        "file_size": stat.st_size,
                                        "mime_type": f"application/{os.path.splitext(file)[1][1:]}",
                                        "source": "local"
                                    })
                                except Exception:
                                    continue
            
            # Filter already processed files
            new_files = [
                f for f in all_files 
                if not self.fingerprint_db.is_file_processed(f["file_path"], f["file_hash"])
            ]
            
            span.set_attributes({
                "scan.total_files": len(all_files),
                "scan.new_files": len(new_files)
            })
            
            return new_files
    
    def process_single_file(self, file_info: Dict) -> bool:
        """Process single file with proper tracing"""
        with self.tracer.start_as_current_span("process_single_file") as span:
            span.set_attributes({
                "file.name": file_info["file_name"],
                "file.source": file_info["source"],
                "service.component": self.service_name
            })
            
            try:
                # Download/prepare file
                if file_info["source"] == "google_drive":
                    temp_file = self.google_drive_loader._download_file(file_info["drive_file_info"])
                    file_path = temp_file
                else:
                    file_path = file_info["file_path"]
                
                # Process document
                documents = self.document_processor.process_file(file_path, file_info)
                if not documents:
                    return False
                
                # Split documents
                chunks = split_documents(documents, chunk_size=1000, chunk_overlap=120)
                
                # Add to vector store
                self.vector_store_manager.add_documents(chunks, file_info)
                
                # Mark as processed
                processed_file = ProcessedFile(
                    file_id=file_info["file_id"],
                    file_name=file_info["file_name"],
                    file_path=file_info["file_path"],
                    file_hash=file_info["file_hash"],
                    processed_at=datetime.now(),
                    document_count=len(documents),
                    file_size=file_info["file_size"],
                    mime_type=file_info["mime_type"]
                )
                
                self.fingerprint_db.mark_file_processed(processed_file)
                
                # Clean up temp file
                if file_info["source"] == "google_drive" and os.path.exists(file_path):
                    os.remove(file_path)
                
                # Update stats
                self.stats["files_processed"] += 1
                self.stats["documents_created"] += len(documents)
                
                span.set_attribute("processing.success", True)
                return True
                
            except Exception as e:
                span.record_exception(e)
                return False
    
    async def scan_and_process_cycle(self):
        """Scan and process cycle"""
        with self.tracer.start_as_current_span("scan_and_process_cycle") as span:
            try:
                new_files = self.scan_for_new_files()
                self.stats["last_scan"] = datetime.now()
                
                if not new_files:
                    return
                
                processed_count = 0
                for file_info in new_files:
                    if not self.is_running:
                        break
                    
                    if self.process_single_file(file_info):
                        processed_count += 1
                    
                    await asyncio.sleep(1)  # Brief delay
                
                span.set_attribute("cycle.processed", processed_count)
                
            except Exception as e:
                span.record_exception(e)
                logging.error(f"Scan cycle error: {e}")
    
    async def start_monitoring(self):
        """Start monitoring service"""
        with self.tracer.start_as_current_span("start_monitoring"):
            self.is_running = True
            
            while self.is_running:
                try:
                    await self.scan_and_process_cycle()
                    
                    # Wait for next scan
                    for _ in range(self.scan_interval):
                        if not self.is_running:
                            break
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logging.error(f"Monitoring error: {e}")
                    await asyncio.sleep(10)
    
    def stop_monitoring(self):
        """Stop monitoring service"""
        self.is_running = False
    
    def get_service_status(self) -> Dict:
        """Get service status"""
        with self.tracer.start_as_current_span("get_service_status") as span:
            span.set_attribute("service.component", self.service_name)
            
            return {
                "service": {
                    "name": self.service_name,
                    "is_running": self.is_running,
                    "parent": parent_service,
                    "started_at": self.stats["service_started"].isoformat(),
                    "uptime_seconds": (datetime.now() - self.stats["service_started"]).total_seconds()
                },
                "processing": {
                    "files_processed": self.stats["files_processed"],
                    "documents_created": self.stats["documents_created"],
                    "last_scan": self.stats["last_scan"].isoformat() if self.stats["last_scan"] else None
                },
                "trace_context": {
                    "trace_id": get_current_trace_id(),
                    "parent_trace_id": parent_trace_id,
                    "service_hierarchy": f"{parent_service} -> document-rag-backend"
                },
                "components": {
                    "document-processor": "healthy",
                    "vector-store-manager": "healthy", 
                    "file-fingerprint-db": "healthy"
                }
            }

# Global service instance
service_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Service lifespan with proper context propagation"""
    global service_instance
    
    # CRITICAL: Use parent tracer context
    parent_tracer = get_service_tracer(parent_service)
    
    with parent_tracer.start_as_current_span("backend_service.startup") as span:
        span.set_attributes({
            "service.component": "document-rag-backend",
            "service.parent": parent_service,
            "parent.trace_id": parent_trace_id
        })
        
        try:
            service_instance = BackendService()
            # Start monitoring in background
            monitoring_task = asyncio.create_task(service_instance.start_monitoring())
            
            yield
            
        finally:
            if service_instance:
                service_instance.stop_monitoring()
            if 'monitoring_task' in locals():
                monitoring_task.cancel()

# Initialize FastAPI with middleware
app = FastAPI(
    title="Document Processing Backend Service",
    description="Backend processing with W3C trace propagation",
    version="2.0.0",
    lifespan=lifespan
)

# CRITICAL: Add middleware for trace continuity
app = instrument_fastapi_app(app, "document-rag-backend")

@app.get("/health")
async def health_check(request: Request):
    """Health check with context extraction"""
    # CRITICAL: Extract context from incoming request
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("health_check") as span:
        span.set_attributes({
            "endpoint": "/health",
            "service.component": "document-rag-backend",
            "service.parent": parent_service
        })
        
        return {
            "status": "healthy",
            "service": "document-rag-backend",
            "parent": parent_service,
            "trace_id": get_current_trace_id(),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/status")
async def get_status(request: Request):
    """Status endpoint with context extraction"""
    # CRITICAL: Extract context from incoming request
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("get_status") as span:
        if not service_instance:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        status = service_instance.get_service_status()
        span.set_attribute("status.retrieved", True)
        return status

@app.post("/scan")
async def trigger_scan(request: Request):
    """Manual scan trigger with context extraction"""
    # CRITICAL: Extract context from incoming request
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("trigger_scan") as span:
        if not service_instance or not service_instance.is_running:
            raise HTTPException(status_code=503, detail="Service not running")
        
        asyncio.create_task(service_instance.scan_and_process_cycle())
        
        return {
            "message": "Scan triggered",
            "trace_id": get_current_trace_id(),
            "parent": parent_service
        }

@app.get("/")
async def root(request: Request):
    """Root endpoint with context extraction"""
    # CRITICAL: Extract context from incoming request
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("root_endpoint") as span:
        return {
            "service": "document-rag-backend",
            "version": "2.0.0",
            "parent": parent_service,
            "hierarchy": f"{parent_service} -> document-rag-backend",
            "components": ["document-processor", "vector-store-manager", "file-fingerprint-db"],
            "trace_id": get_current_trace_id()
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    
    print(f"🚀 Enhanced Backend Service: {parent_service} -> document-rag-backend")
    print(f"🆔 Parent Trace ID: {parent_trace_id}")
    print(f"🔗 W3C Context Propagation: ENABLED")
    
    uvicorn.run(app, host=args.host, port=args.port)
