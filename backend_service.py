#!/usr/bin/env python3
"""
Enhanced Backend Service with Orchestrator Communication
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

os.environ["OTEL_SERVICE_NAME"] = "document-rag-backend"

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import httpx

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from loaders.master_loaders import load_file
from loaders.google_drive_loader import GoogleDriveMasterLoader
from text_splitting import split_documents

from dotenv import load_dotenv
load_dotenv()

from otel_config import (
    initialize_opentelemetry, get_service_tracer, instrument_fastapi_app,
    get_current_trace_id, extract_and_activate_context, TracedHTTPXClient
)

parent_trace_id = os.getenv("OTEL_PARENT_TRACE_ID")
parent_service = os.getenv("OTEL_SERVICE_PARENT", "document-rag-orchestrator")
orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:8002")

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
    def __init__(self, db_path: str = ".processed_files.db"):
        self.db_path = db_path
        self.tracer = get_service_tracer("file-fingerprint-db")
        self._init_db()
    
    def _init_db(self):
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
    
    def is_file_processed(self, file_path: str, file_hash: str) -> bool:
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM processed_files WHERE file_path = ? AND file_hash = ?",
                (file_path, file_hash)
            )
            return cursor.fetchone()[0] > 0
    
    def mark_file_processed(self, processed_file: ProcessedFile):
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
    def __init__(self):
        self.tracer = get_service_tracer("document-processor")
    
    def process_file(self, file_path: str, file_info: Dict) -> List[Document]:
        with self.tracer.start_as_current_span("process_file") as span:
            span.set_attribute("file.name", file_info.get("file_name"))
            try:
                documents = load_file(file_path)
                span.set_attribute("documents.extracted", len(documents))
                return documents
            except Exception as e:
                span.record_exception(e)
                return []

class VectorStoreManager:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "documents"):
        self.tracer = get_service_tracer("vector-store-manager")
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        
        self.client = QdrantClient(url=qdrant_url)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self._ensure_collection()
    
    def _ensure_collection(self):
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
        except Exception as e:
            raise
    
    def add_documents(self, documents: List[Document], file_info: Dict) -> List[str]:
        for doc in documents:
            doc.metadata.update({
                "file_id": file_info.get("file_id"),
                "file_name": file_info.get("file_name"),
                "processed_at": datetime.now().isoformat()
            })
        
        point_ids = [
            hashlib.md5(f"{file_info['file_path']}_{i}_{doc.page_content[:50]}".encode()).hexdigest()
            for i, doc in enumerate(documents)
        ]
        
        self.vector_store.add_documents(documents, ids=point_ids)
        return point_ids

class BackendService:
    def __init__(self):
        self.tracer = tracer
        self.service_name = "document-rag-backend"
        self.orchestrator_url = orchestrator_url
        
        self.google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        self.local_watch_dirs = os.getenv("LOCAL_WATCH_DIRS", "").split(",") if os.getenv("LOCAL_WATCH_DIRS") else []
        self.scan_interval = int(os.getenv("SCAN_INTERVAL", "30"))
        
        self.fingerprint_db = FileFingerprintDatabase()
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        
        self.is_running = False
        self.stats = {
            "service_started": datetime.now(),
            "files_processed": 0,
            "documents_created": 0,
            "last_scan": None
        }
        
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
    
    async def send_heartbeat(self):
        """Send heartbeat to orchestrator"""
        try:
            async with TracedHTTPXClient(service_name="document-rag-backend") as client:
                with self.tracer.start_as_current_span("send_heartbeat") as span:
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
                    span.set_attribute("heartbeat.sent", True)
        except Exception as e:
            logging.warning(f"Heartbeat failed: {e}")
    
    async def check_orchestrator_health(self):
        """Check orchestrator health"""
        try:
            async with TracedHTTPXClient(service_name="document-rag-backend") as client:
                with self.tracer.start_as_current_span("check_orchestrator") as span:
                    response = await client.get(
                        f"{self.orchestrator_url}/health",
                        timeout=5.0
                    )
                    span.set_attribute("orchestrator.healthy", response.status_code == 200)
                    return response.status_code == 200
        except Exception:
            return False
    
    def scan_for_new_files(self) -> List[Dict]:
        with self.tracer.start_as_current_span("scan_for_new_files") as span:
            all_files = []
            
            if self.google_drive_loader:
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
            
            for watch_dir in self.local_watch_dirs:
                if os.path.exists(watch_dir):
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
        with self.tracer.start_as_current_span("process_single_file") as span:
            span.set_attribute("file.name", file_info["file_name"])
            
            try:
                if file_info["source"] == "google_drive":
                    temp_file = self.google_drive_loader._download_file(file_info["drive_file_info"])
                    file_path = temp_file
                else:
                    file_path = file_info["file_path"]
                
                documents = self.document_processor.process_file(file_path, file_info)
                if not documents:
                    return False
                
                chunks = split_documents(documents, chunk_size=1000, chunk_overlap=120)
                self.vector_store_manager.add_documents(chunks, file_info)
                
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
                
                if file_info["source"] == "google_drive" and os.path.exists(file_path):
                    os.remove(file_path)
                
                self.stats["files_processed"] += 1
                self.stats["documents_created"] += len(documents)
                
                return True
                
            except Exception as e:
                span.record_exception(e)
                return False
    
    async def scan_and_process_cycle(self):
        try:
            new_files = self.scan_for_new_files()
            self.stats["last_scan"] = datetime.now()
            
            if not new_files:
                return
            
            for file_info in new_files:
                if not self.is_running:
                    break
                
                self.process_single_file(file_info)
                await asyncio.sleep(1)
                
        except Exception as e:
            logging.error(f"Scan cycle error: {e}")
    
    async def start_monitoring(self):
        self.is_running = True
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        
        while self.is_running:
            try:
                await self.scan_and_process_cycle()
                
                for _ in range(self.scan_interval):
                    if not self.is_running:
                        break
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
        
        heartbeat_task.cancel()
    
    async def heartbeat_loop(self):
        """Send periodic heartbeats to orchestrator"""
        while self.is_running:
            await self.send_heartbeat()
            await asyncio.sleep(30)
    
    def stop_monitoring(self):
        self.is_running = False
    
    def get_service_status(self) -> Dict:
        return {
            "service": {
                "name": self.service_name,
                "is_running": self.is_running,
                "parent": parent_service,
                "orchestrator_url": self.orchestrator_url,
                "started_at": self.stats["service_started"].isoformat(),
                "uptime_seconds": (datetime.now() - self.stats["service_started"]).total_seconds()
            },
            "processing": {
                "files_processed": self.stats["files_processed"],
                "documents_created": self.stats["documents_created"],
                "last_scan": self.stats["last_scan"].isoformat() if self.stats["last_scan"] else None
            }
        }

service_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global service_instance
    
    try:
        service_instance = BackendService()
        
        # Check orchestrator on startup
        await service_instance.check_orchestrator_health()
        
        monitoring_task = asyncio.create_task(service_instance.start_monitoring())
        
        yield
        
    finally:
        if service_instance:
            service_instance.stop_monitoring()
        if 'monitoring_task' in locals():
            monitoring_task.cancel()

app = FastAPI(
    title="Document Processing Backend Service",
    version="2.0.0",
    lifespan=lifespan
)

app = instrument_fastapi_app(app, "document-rag-backend")

@app.get("/health")
async def health_check(request: Request):
    context = extract_and_activate_context(dict(request.headers))
    
    with tracer.start_as_current_span("health_check") as span:
        return {
            "status": "healthy",
            "service": "document-rag-backend",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/status")
async def get_status(request: Request):
    context = extract_and_activate_context(dict(request.headers))
    
    if not service_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return service_instance.get_service_status()

@app.post("/scan")
async def trigger_scan(request: Request):
    context = extract_and_activate_context(dict(request.headers))
    
    if not service_instance or not service_instance.is_running:
        raise HTTPException(status_code=503, detail="Service not running")
    
    asyncio.create_task(service_instance.scan_and_process_cycle())
    
    return {"message": "Scan triggered"}

@app.get("/")
async def root(request: Request):
    context = extract_and_activate_context(dict(request.headers))
    
    return {
        "service": "document-rag-backend",
        "version": "2.0.0",
        "orchestrator": orchestrator_url
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    
    print(f"🚀 Backend Service")
    print(f"📡 Orchestrator: {orchestrator_url}")
    
    uvicorn.run(app, host=args.host, port=args.port)