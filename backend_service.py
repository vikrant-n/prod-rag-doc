#!/usr/bin/env python3
"""
Enhanced Backend Service with Orchestrator Communication and Correlated Logging
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

# Add correlated logging import
from otel_config import (
    initialize_opentelemetry, get_service_tracer, instrument_fastapi_app,
    get_current_trace_id, extract_and_activate_context, TracedHTTPXClient,
    get_correlated_logger  # NEW: Import correlated logger
)

parent_trace_id = os.getenv("OTEL_PARENT_TRACE_ID")
parent_service = os.getenv("OTEL_SERVICE_PARENT", "document-rag-orchestrator")
orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:8002")

tracer, meter = initialize_opentelemetry("document-rag-backend", "2.0.0", "production")

# Replace existing logger with correlated logger
logger = get_correlated_logger(__name__)

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
        self.logger = get_correlated_logger(f"{__name__}.FileFingerprintDatabase")
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
        
        self.logger.info_with_context(
            "File fingerprint database initialized",
            extra_attributes={
                "database.path": self.db_path,
                "operation": "database_init"
            }
        )
    
    def is_file_processed(self, file_path: str, file_hash: str) -> bool:
        import sqlite3
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

class DocumentProcessor:
    def __init__(self):
        self.tracer = get_service_tracer("document-processor")
        self.logger = get_correlated_logger(f"{__name__}.DocumentProcessor")
    
    def process_file(self, file_path: str, file_info: Dict) -> List[Document]:
        with self.tracer.start_as_current_span("process_file") as span:
            span.set_attribute("file.name", file_info.get("file_name"))
            
            self.logger.info_with_context(
                "Starting document processing",
                extra_attributes={
                    "file.name": file_info.get("file_name"),
                    "file.path": file_path,
                    "file.source": file_info.get("source", "unknown"),
                    "operation": "document_processing"
                }
            )
            
            try:
                documents = load_file(file_path)
                span.set_attribute("documents.extracted", len(documents))
                
                self.logger.info_with_context(
                    "Document processing completed",
                    extra_attributes={
                        "file.name": file_info.get("file_name"),
                        "documents.extracted": len(documents),
                        "processing.status": "success",
                        "operation": "document_processing"
                    }
                )
                
                return documents
            except Exception as e:
                span.record_exception(e)
                
                self.logger.error_with_context(
                    "Document processing failed",
                    extra_attributes={
                        "file.name": file_info.get("file_name"),
                        "file.path": file_path,
                        "error.type": type(e).__name__,
                        "error.message": str(e),
                        "processing.status": "failed",
                        "operation": "document_processing"
                    },
                    exc_info=True
                )
                return []

class VectorStoreManager:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "documents"):
        self.tracer = get_service_tracer("vector-store-manager")
        self.logger = get_correlated_logger(f"{__name__}.VectorStoreManager")
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        
        self.client = QdrantClient(url=qdrant_url)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self._ensure_collection()
    
    def _ensure_collection(self):
        try:
            collections = self.client.get_collections()
            collection_exists = self.collection_name in [col.name for col in collections.collections]
            
            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
                )
                
                self.logger.info_with_context(
                    "Created new Qdrant collection",
                    extra_attributes={
                        "collection.name": self.collection_name,
                        "collection.vector_size": 3072,
                        "collection.distance": "COSINE",
                        "operation": "collection_creation"
                    }
                )
            else:
                self.logger.info_with_context(
                    "Using existing Qdrant collection",
                    extra_attributes={
                        "collection.name": self.collection_name,
                        "operation": "collection_connection"
                    }
                )
            
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )
        except Exception as e:
            self.logger.error_with_context(
                "Failed to initialize vector store",
                extra_attributes={
                    "qdrant.url": self.qdrant_url,
                    "collection.name": self.collection_name,
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "vector_store_init"
                },
                exc_info=True
            )
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
        
        self.logger.info_with_context(
            "Adding documents to vector store",
            extra_attributes={
                "file.id": file_info.get("file_id"),
                "file.name": file_info.get("file_name"),
                "documents.count": len(documents),
                "vector_ids.generated": len(point_ids),
                "operation": "vector_store_add"
            }
        )
        
        try:
            self.vector_store.add_documents(documents, ids=point_ids)
            
            self.logger.info_with_context(
                "Documents successfully added to vector store",
                extra_attributes={
                    "file.name": file_info.get("file_name"),
                    "documents.added": len(documents),
                    "operation": "vector_store_add",
                    "status": "success"
                }
            )
        except Exception as e:
            self.logger.error_with_context(
                "Failed to add documents to vector store",
                extra_attributes={
                    "file.name": file_info.get("file_name"),
                    "documents.count": len(documents),
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "vector_store_add",
                    "status": "failed"
                },
                exc_info=True
            )
            raise
        
        return point_ids

class BackendService:
    def __init__(self):
        self.tracer = tracer
        self.service_name = "document-rag-backend"
        self.orchestrator_url = orchestrator_url
        
        # Add correlated logger for the service
        self.logger = get_correlated_logger(f"{__name__}.BackendService")
        
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
                
                self.logger.info_with_context(
                    "Google Drive loader initialized",
                    extra_attributes={
                        "google_drive.folder_id": self.google_drive_folder_id,
                        "operation": "service_init"
                    }
                )
            except Exception as e:
                self.logger.warning_with_context(
                    "Google Drive loader initialization failed",
                    extra_attributes={
                        "google_drive.folder_id": self.google_drive_folder_id,
                        "error.type": type(e).__name__,
                        "error.message": str(e),
                        "operation": "service_init"
                    }
                )
        
        self.logger.info_with_context(
            "Backend service initialized",
            extra_attributes={
                "service.name": self.service_name,
                "orchestrator.url": self.orchestrator_url,
                "scan_interval": self.scan_interval,
                "local_watch_dirs": len(self.local_watch_dirs),
                "google_drive.enabled": self.google_drive_loader is not None,
                "operation": "service_init"
            }
        )
    
    async def send_heartbeat(self):
        """Send heartbeat with logging"""
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
                    
                    self.logger.debug_with_context(
                        "Heartbeat sent to orchestrator",
                        extra_attributes={
                            "orchestrator.url": self.orchestrator_url,
                            "heartbeat.status": "sent",
                            "response.status_code": response.status_code,
                            "stats.files_processed": self.stats["files_processed"],
                            "stats.documents_created": self.stats["documents_created"],
                            "operation": "heartbeat"
                        }
                    )
                    
        except Exception as e:
            self.logger.error_with_context(
                "Heartbeat failed",
                extra_attributes={
                    "orchestrator.url": self.orchestrator_url,
                    "heartbeat.status": "failed",
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "heartbeat"
                },
                exc_info=True
            )
    
    async def check_orchestrator_health(self):
        """Check orchestrator health"""
        try:
            async with TracedHTTPXClient(service_name="document-rag-backend") as client:
                with self.tracer.start_as_current_span("check_orchestrator") as span:
                    response = await client.get(
                        f"{self.orchestrator_url}/health",
                        timeout=5.0
                    )
                    
                    is_healthy = response.status_code == 200
                    span.set_attribute("orchestrator.healthy", is_healthy)
                    
                    self.logger.info_with_context(
                        "Orchestrator health check completed",
                        extra_attributes={
                            "orchestrator.url": self.orchestrator_url,
                            "orchestrator.healthy": is_healthy,
                            "response.status_code": response.status_code,
                            "operation": "health_check"
                        }
                    )
                    
                    return is_healthy
        except Exception as e:
            self.logger.error_with_context(
                "Orchestrator health check failed",
                extra_attributes={
                    "orchestrator.url": self.orchestrator_url,
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "health_check"
                },
                exc_info=True
            )
            return False
    
    def scan_for_new_files(self) -> List[Dict]:
        with self.tracer.start_as_current_span("scan_for_new_files") as span:
            all_files = []
            
            self.logger.info_with_context(
                "Starting file scan",
                extra_attributes={
                    "google_drive.enabled": self.google_drive_loader is not None,
                    "local_dirs.count": len(self.local_watch_dirs),
                    "operation": "file_scan"
                }
            )
            
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
                    
                    self.logger.info_with_context(
                        "Google Drive scan completed",
                        extra_attributes={
                            "google_drive.files_found": len(files),
                            "operation": "file_scan"
                        }
                    )
                except Exception as e:
                    self.logger.error_with_context(
                        "Google Drive scan failed",
                        extra_attributes={
                            "google_drive.folder_id": self.google_drive_folder_id,
                            "error.type": type(e).__name__,
                            "error.message": str(e),
                            "operation": "file_scan"
                        },
                        exc_info=True
                    )
            
            local_files_count = 0
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
                                local_files_count += 1
                            except Exception:
                                continue
            
            if local_files_count > 0:
                self.logger.info_with_context(
                    "Local directory scan completed",
                    extra_attributes={
                        "local.files_found": local_files_count,
                        "operation": "file_scan"
                    }
                )
            
            new_files = [
                f for f in all_files 
                if not self.fingerprint_db.is_file_processed(f["file_path"], f["file_hash"])
            ]
            
            span.set_attributes({
                "scan.total_files": len(all_files),
                "scan.new_files": len(new_files)
            })
            
            self.logger.info_with_context(
                "File scan completed",
                extra_attributes={
                    "scan.total_files": len(all_files),
                    "scan.new_files": len(new_files),
                    "scan.already_processed": len(all_files) - len(new_files),
                    "operation": "file_scan"
                }
            )
            
            return new_files
    
    def process_single_file(self, file_info: Dict) -> bool:
        with self.tracer.start_as_current_span("process_single_file") as span:
            # Add structured logging with automatic correlation
            self.logger.info_with_context(
                "Starting file processing",
                extra_attributes={
                    "file.name": file_info["file_name"],
                    "file.size": file_info["file_size"],
                    "file.source": file_info["source"],
                    "operation": "file_processing"
                }
            )
            
            try:
                if file_info["source"] == "google_drive":
                    temp_file = self.google_drive_loader._download_file(file_info["drive_file_info"])
                    file_path = temp_file
                    
                    self.logger.debug_with_context(
                        "Downloaded file from Google Drive",
                        extra_attributes={
                            "temp_file_path": temp_file,
                            "drive_file_id": file_info["drive_file_info"]["id"],
                            "operation": "file_processing"
                        }
                    )
                else:
                    file_path = file_info["file_path"]
                
                documents = self.document_processor.process_file(file_path, file_info)
                if not documents:
                    self.logger.warning_with_context(
                        "No documents extracted from file",
                        extra_attributes={
                            "file.name": file_info["file_name"],
                            "file.path": file_path,
                            "operation": "file_processing"
                        }
                    )
                    return False
                
                chunks = split_documents(documents, chunk_size=1000, chunk_overlap=120)
                self.vector_store_manager.add_documents(chunks, file_info)
                
                # Log success with metrics
                self.logger.info_with_context(
                    "File processing completed successfully",
                    extra_attributes={
                        "file.name": file_info["file_name"],
                        "documents.count": len(documents),
                        "chunks.count": len(chunks),
                        "processing.status": "success",
                        "operation": "file_processing"
                    }
                )
                
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
                    self.logger.debug_with_context(
                        "Cleaned up temporary Google Drive file",
                        extra_attributes={
                            "temp_file_path": file_path,
                            "operation": "file_processing"
                        }
                    )
                
                # Update stats and mark as processed
                self.stats["files_processed"] += 1
                self.stats["documents_created"] += len(documents)
                
                return True
                
            except Exception as e:
                span.record_exception(e)
                
                # Structured error logging
                self.logger.error_with_context(
                    "File processing failed",
                    extra_attributes={
                        "file.name": file_info["file_name"],
                        "file.path": file_info.get("file_path", "unknown"),
                        "error.type": type(e).__name__,
                        "error.message": str(e),
                        "operation": "file_processing",
                        "processing.status": "failed"
                    },
                    exc_info=True  # Include full stack trace
                )
                return False
    
    async def scan_and_process_cycle(self):
        self.logger.info_with_context(
            "Starting scan and process cycle",
            extra_attributes={
                "operation": "scan_cycle"
            }
        )
        
        try:
            new_files = self.scan_for_new_files()
            self.stats["last_scan"] = datetime.now()
            
            if not new_files:
                self.logger.debug_with_context(
                    "No new files found in scan cycle",
                    extra_attributes={
                        "operation": "scan_cycle"
                    }
                )
                return
            
            self.logger.info_with_context(
                "Processing new files found in scan",
                extra_attributes={
                    "files.new_count": len(new_files),
                    "operation": "scan_cycle"
                }
            )
            
            processed_count = 0
            failed_count = 0
            
            for file_info in new_files:
                if not self.is_running:
                    break
                
                success = self.process_single_file(file_info)
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                
                await asyncio.sleep(1)
            
            self.logger.info_with_context(
                "Scan and process cycle completed",
                extra_attributes={
                    "files.processed": processed_count,
                    "files.failed": failed_count,
                    "files.total": len(new_files),
                    "operation": "scan_cycle"
                }
            )
                
        except Exception as e:
            self.logger.error_with_context(
                "Scan cycle error",
                extra_attributes={
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "scan_cycle"
                },
                exc_info=True
            )
    
    async def start_monitoring(self):
        self.is_running = True
        
        self.logger.info_with_context(
            "Starting monitoring loop",
            extra_attributes={
                "scan_interval": self.scan_interval,
                "operation": "monitoring_start"
            }
        )
        
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
                self.logger.error_with_context(
                    "Monitoring error",
                    extra_attributes={
                        "error.type": type(e).__name__,
                        "error.message": str(e),
                        "operation": "monitoring_loop"
                    },
                    exc_info=True
                )
                await asyncio.sleep(10)
        
        heartbeat_task.cancel()
        
        self.logger.info_with_context(
            "Monitoring loop stopped",
            extra_attributes={
                "operation": "monitoring_stop"
            }
        )
    
    async def heartbeat_loop(self):
        """Send periodic heartbeats to orchestrator"""
        while self.is_running:
            await self.send_heartbeat()
            await asyncio.sleep(30)
    
    def stop_monitoring(self):
        self.is_running = False
        
        self.logger.info_with_context(
            "Stopping monitoring",
            extra_attributes={
                "operation": "monitoring_stop"
            }
        )
    
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
    context = extract_and_activate_context(dict(request.headers))
    
    if not service_instance:
        logger.error_with_context(
            "Status requested but service not initialized",
            extra_attributes={
                "operation": "status_check"
            }
        )
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    status = service_instance.get_service_status()
    
    logger.debug_with_context(
        "Status information provided",
        extra_attributes={
            "service.is_running": status["service"]["is_running"],
            "files.processed": status["processing"]["files_processed"],
            "operation": "status_check"
        }
    )
    
    return status

@app.post("/scan")
async def trigger_scan(request: Request):
    context = extract_and_activate_context(dict(request.headers))
    
    if not service_instance or not service_instance.is_running:
        logger.error_with_context(
            "Scan requested but service not running",
            extra_attributes={
                "service.initialized": service_instance is not None,
                "service.is_running": service_instance.is_running if service_instance else False,
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
    
    # Log service startup
    startup_logger = get_correlated_logger("startup")
    startup_logger.info_with_context(
        "Backend service starting up",
        extra_attributes={
            "host": args.host,
            "port": args.port,
            "orchestrator_url": orchestrator_url,
            "operation": "service_startup"
        }
    )
    
    uvicorn.run(app, host=args.host, port=args.port)
