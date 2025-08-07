#!/usr/bin/env python3
"""
Continuous Backend Service for Document Processing with OpenTelemetry Instrumentation

This service continuously monitors for new files and processes only those that haven't been embedded yet.
It tracks processed files to avoid reprocessing and provides a REST API for status monitoring.
Enhanced with comprehensive OpenTelemetry observability following EDOT best practices.
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
import socket
import uuid

# OpenTelemetry imports - Official EDOT libraries
from opentelemetry import trace, metrics, baggage
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, SERVICE_INSTANCE_ID
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.propagate import set_global_textmap, extract
from opentelemetry.propagators.b3 import B3MultiFormat

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

# Initialize OpenTelemetry with EDOT configuration
def init_telemetry():
    """Initialize OpenTelemetry with EDOT configuration and context extraction"""

    # Check if we're already using auto-instrumentation
    current_tracer_provider = trace.get_tracer_provider()
    if hasattr(current_tracer_provider, '__class__') and current_tracer_provider.__class__.__name__ != 'ProxyTracerProvider':
        print("✅ Using existing OpenTelemetry auto-instrumentation")
        return trace.get_tracer(__name__), metrics.get_meter(__name__)

    # Extract trace context from environment variables set by orchestrator
    carrier = {}
    for key, value in os.environ.items():
        if key.startswith('OTEL_PROPAGATED_'):
            header_key = key.replace('OTEL_PROPAGATED_', '').replace('_', '-').lower()
            carrier[header_key] = value

    # Extract and set context from orchestrator
    extracted_context = extract(carrier) if carrier else None
    if extracted_context:
        print("✅ Extracted trace context from orchestrator")

    # Generate unique service instance ID
    service_instance_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

    # Resource configuration - Following EDOT best practices
    resource = Resource.create({
        SERVICE_NAME: "document-rag-backend",
        SERVICE_VERSION: "1.0.0",
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

        print("✅ OpenTelemetry initialized for backend service")

    except Exception as e:
        print(f"⚠️  OpenTelemetry provider setup error: {e}")

    # Set up B3 propagator for distributed tracing
    set_global_textmap(B3MultiFormat())

    # Manual instrumentation only if not using auto-instrumentation
    if not os.getenv("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"):
        try:
            LoggingInstrumentor().instrument(set_logging_format=True)
            RequestsInstrumentor().instrument()
            SQLite3Instrumentor().instrument()
            AsyncioInstrumentor().instrument()
        except Exception as e:
            print(f"⚠️  Instrumentation warning: {e}")

    return trace.get_tracer(__name__), metrics.get_meter(__name__)

# Initialize telemetry
tracer, meter = init_telemetry()

# Custom metrics for document processing observability
document_processing_counter = meter.create_counter(
    "documents_processed_total",
    description="Total number of documents processed",
    unit="1"
)

document_processing_duration = meter.create_histogram(
    "document_processing_duration_seconds",
    description="Time taken to process documents",
    unit="s"
)

file_processing_counter = meter.create_counter(
    "files_processed_total",
    description="Total number of files processed",
    unit="1"
)

qdrant_operations_counter = meter.create_counter(
    "qdrant_operations_total",
    description="Total number of Qdrant operations",
    unit="1"
)

database_operations_counter = meter.create_counter(
    "database_operations_total",
    description="Total number of database operations",
    unit="1"
)

service_health_gauge = meter.create_up_down_counter(
    "service_health_status",
    description="Backend service health status (1=healthy, 0=unhealthy)",
    unit="1"
)

google_drive_scan_counter = meter.create_counter(
    "google_drive_scans_total",
    description="Total number of Google Drive scans",
    unit="1"
)

# Configure structured logging with OpenTelemetry correlation
class OtelBackendFormatter(logging.Formatter):
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
formatter = OtelBackendFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler('backend_service.log')
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

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
    """SQLite database to track processed files and avoid reprocessing with OpenTelemetry instrumentation"""

    def __init__(self, db_path: str = ".processed_files.db"):
        self.db_path = db_path
        with tracer.start_as_current_span("fingerprint_db_init") as span:
            span.set_attribute("db.system", "sqlite")
            span.set_attribute("db.name", db_path)
            self._init_db()
            span.add_event("Database initialized")

    def _init_db(self):
        """Initialize the database schema"""
        with tracer.start_as_current_span("db_schema_init") as span:
            database_operations_counter.add(1, {"operation": "schema_init", "db_type": "sqlite"})

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

                span.set_attribute("db.operation", "create_table")
                span.set_attribute("table.name", "processed_files")
                span.add_event("Schema creation completed")

    def is_file_processed(self, file_path: str, file_hash: str) -> bool:
        """Check if a file has already been processed"""
        with tracer.start_as_current_span("check_file_processed") as span:
            span.set_attribute("db.operation", "select")
            span.set_attribute("file.path", file_path[:100])  # Truncate for security
            span.set_attribute("file.hash", file_hash)

            database_operations_counter.add(1, {"operation": "select", "table": "processed_files"})

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM processed_files WHERE file_path = ? AND file_hash = ?",
                    (file_path, file_hash)
                )
                result = cursor.fetchone()[0] > 0

                span.set_attribute("file.already_processed", result)
                span.add_event("File processing check completed", {"result": result})

                return result

    def is_file_in_qdrant(self, file_path: str, qdrant_client) -> bool:
        """Check if a file already has embeddings in Qdrant"""
        with tracer.start_as_current_span("check_file_in_qdrant") as span:
            span.set_attribute("db.system", "qdrant")
            span.set_attribute("file.path", file_path[:100])

            qdrant_operations_counter.add(1, {"operation": "scroll", "collection": "documents"})

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

                result = len(search_result[0]) > 0
                span.set_attribute("qdrant.points_found", len(search_result[0]))
                span.set_attribute("file.in_qdrant", result)
                span.add_event("Qdrant check completed", {"points_found": len(search_result[0])})

                return result

            except Exception as e:
                # If there's an error (like collection doesn't exist), assume not processed
                span.record_exception(e)
                span.set_attribute("file.in_qdrant", False)
                span.add_event("Qdrant check failed, assuming not processed")
                return False

    def mark_file_processed(self, processed_file: ProcessedFile):
        """Mark a file as processed"""
        with tracer.start_as_current_span("mark_file_processed") as span:
            span.set_attribute("db.operation", "insert")
            span.set_attribute("file.id", processed_file.file_id)
            span.set_attribute("file.name", processed_file.file_name)
            span.set_attribute("document.count", processed_file.document_count)
            span.set_attribute("file.size", processed_file.file_size)

            database_operations_counter.add(1, {"operation": "insert", "table": "processed_files"})

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

                span.add_event("File marked as processed", {
                    "file_name": processed_file.file_name,
                    "document_count": processed_file.document_count,
                    "qdrant_points": len(processed_file.qdrant_point_ids)
                })

    def get_processed_files(self, limit: int = 100) -> List[ProcessedFile]:
        """Get list of processed files"""
        with tracer.start_as_current_span("get_processed_files") as span:
            span.set_attribute("db.operation", "select")
            span.set_attribute("query.limit", limit)

            database_operations_counter.add(1, {"operation": "select", "table": "processed_files"})

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

                span.set_attribute("files.retrieved", len(files))
                span.add_event("Files retrieved", {"count": len(files)})
                return files

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with tracer.start_as_current_span("get_db_stats") as span:
            span.set_attribute("db.operation", "aggregate")

            database_operations_counter.add(1, {"operation": "aggregate", "table": "processed_files"})

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

                span.set_attributes({
                    "stats.total_files": stats["total_files"],
                    "stats.total_documents": stats["total_documents"],
                    "stats.total_size_bytes": stats["total_size_bytes"]
                })

                return stats

    def clear_processed_files(self):
        """Clear all processed files from the database"""
        with tracer.start_as_current_span("clear_processed_files") as span:
            span.set_attribute("db.operation", "delete")

            database_operations_counter.add(1, {"operation": "delete", "table": "processed_files"})

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM processed_files")
                conn.commit()

                span.add_event("Processed files database cleared")

class DocumentProcessor:
    """Handles document processing and embedding with comprehensive OpenTelemetry instrumentation"""

    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "documents",
                 embedding_model: str = "text-embedding-3-large"):

        with tracer.start_as_current_span("document_processor_init") as span:
            self.qdrant_url = qdrant_url
            self.collection_name = collection_name
            self.embedding_model = embedding_model

            span.set_attribute("qdrant.url", qdrant_url)
            span.set_attribute("qdrant.collection", collection_name)
            span.set_attribute("embedding.model", embedding_model)

            # Initialize components
            span.add_event("Initializing OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(model=embedding_model)

            span.add_event("Initializing Qdrant client")
            self.qdrant_client = QdrantClient(url=qdrant_url)
            self.vector_store = None

            # Initialize collection
            self._ensure_collection_exists()

            span.add_event("Document processor initialized")
            logger.info(f"✅ Document processor initialized with collection: {collection_name}")

    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists"""
        with tracer.start_as_current_span("ensure_qdrant_collection") as span:
            span.set_attribute("qdrant.collection", self.collection_name)
            span.set_attribute("qdrant.operation", "ensure_collection")

            qdrant_operations_counter.add(1, {"operation": "get_collections"})

            try:
                collections = self.qdrant_client.get_collections()
                collection_names = [col.name for col in collections.collections]

                span.set_attribute("existing_collections.count", len(collection_names))

                if self.collection_name not in collection_names:
                    logger.info(f"📦 Creating collection: {self.collection_name}")
                    span.add_event("Creating new collection")

                    qdrant_operations_counter.add(1, {"operation": "create_collection"})

                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)  # text-embedding-3-large
                    )

                    span.set_attribute("collection.created", True)
                    span.add_event("Collection created successfully")
                    logger.info(f"✅ Collection '{self.collection_name}' created")
                else:
                    span.set_attribute("collection.created", False)
                    span.add_event("Collection already exists")
                    logger.info(f"✅ Collection '{self.collection_name}' already exists")

                # Initialize vector store
                span.add_event("Initializing vector store")
                self.vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings
                )

                span.set_attribute("vector_store.initialized", True)

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("collection.initialization", "failed")
                logger.error(f"❌ Failed to initialize collection: {e}")
                raise

    def process_documents(self, documents: List[Document], file_info: Dict) -> List[str]:
        """Process documents and add to Qdrant, return point IDs"""
        with tracer.start_as_current_span("process_documents") as span:
            start_time = time.time()

            span.set_attribute("file.name", file_info.get("file_name", "unknown"))
            span.set_attribute("file.source", file_info.get("source", "unknown"))
            span.set_attribute("documents.input_count", len(documents))

            try:
                if not documents:
                    span.set_attribute("documents.processed", 0)
                    span.add_event("No documents to process")
                    return []

                # Split documents into chunks
                with tracer.start_as_current_span("split_documents") as split_span:
                    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=120)
                    split_span.set_attribute("chunks.count", len(chunks))
                    split_span.set_attribute("chunk_size", 1000)
                    split_span.set_attribute("chunk_overlap", 120)

                    logger.info(f"📄 Split into {len(chunks)} chunks")

                if not chunks:
                    span.set_attribute("documents.processed", 0)
                    span.add_event("No chunks created after splitting")
                    return []

                span.set_attribute("chunks.count", len(chunks))

                # Add metadata
                with tracer.start_as_current_span("add_metadata") as meta_span:
                    for chunk in chunks:
                        chunk.metadata.update({
                            "file_id": file_info.get("file_id", "unknown"),
                            "file_name": file_info.get("file_name", "unknown"),
                            "file_path": file_info.get("file_path", "unknown"),
                            "processed_at": datetime.now().isoformat(),
                            "file_size": file_info.get("file_size", 0),
                            "mime_type": file_info.get("mime_type", "unknown")
                        })

                    meta_span.set_attribute("metadata.updated_chunks", len(chunks))
                    meta_span.add_event("Metadata added to all chunks")

                # Generate embeddings and add to Qdrant
                point_ids = []
                batch_size = 10

                with tracer.start_as_current_span("batch_processing") as batch_span:
                    batch_count = (len(chunks) + batch_size - 1) // batch_size
                    batch_span.set_attribute("batch.size", batch_size)
                    batch_span.set_attribute("batch.total_count", batch_count)

                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        batch_number = i // batch_size + 1

                        with tracer.start_as_current_span(f"process_batch_{batch_number}") as chunk_span:
                            chunk_span.set_attribute("batch.number", batch_number)
                            chunk_span.set_attribute("batch.chunk_count", len(batch))

                            # Generate unique point IDs
                            batch_point_ids = [
                                hashlib.md5(f"{file_info['file_path']}_{i+j}_{chunk.page_content[:100]}".encode()).hexdigest()
                                for j, chunk in enumerate(batch)
                            ]

                            # Add to vector store
                            qdrant_operations_counter.add(len(batch), {"operation": "add_documents", "collection": self.collection_name})

                            self.vector_store.add_documents(batch, ids=batch_point_ids)
                            point_ids.extend(batch_point_ids)

                            chunk_span.set_attribute("batch.point_ids_generated", len(batch_point_ids))
                            chunk_span.add_event("Batch processed successfully", {
                                "batch_number": batch_number,
                                "chunks_processed": len(batch),
                                "point_ids": batch_point_ids[:3]  # Log first 3 for reference
                            })

                            logger.info(f"✅ Processed batch {batch_number}/{batch_count}")

                            # Small delay to avoid overwhelming the system
                            time.sleep(0.1)

                # Update metrics
                processing_duration = time.time() - start_time
                document_processing_duration.record(processing_duration, {
                    "file_type": file_info.get("mime_type", "unknown"),
                    "source": file_info.get("source", "unknown")
                })

                document_processing_counter.add(len(documents), {
                    "source": file_info.get("source", "unknown"),
                    "status": "success"
                })

                span.set_attribute("processing.duration_seconds", processing_duration)
                span.set_attribute("documents.processed", len(chunks))
                span.set_attribute("qdrant.point_ids_generated", len(point_ids))

                span.add_event("Document processing completed", {
                    "chunks_processed": len(chunks),
                    "point_ids_generated": len(point_ids),
                    "duration_seconds": processing_duration
                })

                logger.info(f"✅ Successfully processed {len(chunks)} chunks with {len(point_ids)} points")
                return point_ids

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("processing.status", "failed")

                document_processing_counter.add(1, {
                    "source": file_info.get("source", "unknown"),
                    "status": "error"
                })

                logger.error(f"❌ Failed to process documents: {e}")
                raise

class BackendService:
    """Main backend service for continuous file monitoring and processing with OpenTelemetry"""

    def __init__(self):
        with tracer.start_as_current_span("backend_service_init") as span:
            # Configuration
            self.google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
            self.local_watch_dirs = os.getenv("LOCAL_WATCH_DIRS", "").split(",") if os.getenv("LOCAL_WATCH_DIRS") else []
            self.scan_interval = int(os.getenv("SCAN_INTERVAL", "60"))  # seconds

            span.set_attribute("google_drive.folder_id", self.google_drive_folder_id or "not_configured")
            span.set_attribute("local_watch_dirs.count", len(self.local_watch_dirs))
            span.set_attribute("scan_interval", self.scan_interval)

            # Initialize components
            span.add_event("Initializing fingerprint database")
            self.fingerprint_db = FileFingerprintDatabase()

            span.add_event("Initializing document processor")
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
                with tracer.start_as_current_span("google_drive_loader_init") as drive_span:
                    drive_span.set_attribute("folder_id", self.google_drive_folder_id)

                    try:
                        # Get credentials paths from environment
                        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
                        token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")

                        drive_span.set_attribute("credentials_path", credentials_path)
                        drive_span.set_attribute("token_path", token_path)

                        self.google_drive_loader = GoogleDriveMasterLoader(
                            folder_id=self.google_drive_folder_id,
                            credentials_path=credentials_path,
                            token_path=token_path,
                            split=False  # We'll handle splitting ourselves
                        )

                        drive_span.set_attribute("google_drive.initialized", True)
                        drive_span.add_event("Google Drive loader initialized successfully")

                        logger.info(f"✅ Google Drive loader initialized for folder: {self.google_drive_folder_id}")
                        logger.info(f"📋 Using credentials: {credentials_path}")
                        logger.info(f"🔑 Using token: {token_path}")

                    except Exception as e:
                        drive_span.record_exception(e)
                        drive_span.set_attribute("google_drive.initialized", False)

                        logger.error(f"❌ Could not initialize Google Drive loader: {e}")
                        logger.error(f"   Make sure credentials are set up properly at: {os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')}")
                        self.google_drive_loader = None

            # Set initial health status
            service_health_gauge.add(1, {"service": "backend"})

            span.add_event("Backend service initialization completed")

            logger.info(f"🚀 Backend service initialized")
            logger.info(f"📁 Google Drive folder: {self.google_drive_folder_id}")
            logger.info(f"📂 Local watch directories: {self.local_watch_dirs}")
            logger.info(f"⏱️ Scan interval: {self.scan_interval} seconds")

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        with tracer.start_as_current_span("calculate_file_hash") as span:
            span.set_attribute("file.path", file_path[:100])  # Truncate for security

            hash_sha256 = hashlib.sha256()
            try:
                with open(file_path, "rb") as f:
                    bytes_read = 0
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_sha256.update(chunk)
                        bytes_read += len(chunk)

                file_hash = hash_sha256.hexdigest()

                span.set_attribute("file.size_bytes", bytes_read)
                span.set_attribute("file.hash", file_hash)
                span.add_event("File hash calculated", {"size_bytes": bytes_read})

                return file_hash

            except Exception as e:
                span.record_exception(e)
                logger.error(f"❌ Failed to calculate hash for {file_path}: {e}")
                return ""

    def scan_local_directories(self) -> List[Dict]:
        """Scan local directories for new files"""
        with tracer.start_as_current_span("scan_local_directories") as span:
            span.set_attribute("scan.type", "local")
            span.set_attribute("watch_dirs.count", len(self.local_watch_dirs))

            new_files = []
            directories_scanned = 0
            files_found = 0

            for watch_dir in self.local_watch_dirs:
                with tracer.start_as_current_span(f"scan_directory") as dir_span:
                    dir_span.set_attribute("directory.path", watch_dir)

                    if not os.path.exists(watch_dir):
                        dir_span.set_attribute("directory.exists", False)
                        logger.warning(f"⚠️ Watch directory does not exist: {watch_dir}")
                        continue

                    directories_scanned += 1
                    dir_span.set_attribute("directory.exists", True)

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

                        dir_span.set_attribute("files.found_in_directory", len([f for f in new_files if watch_dir in f["file_path"]]))
                        dir_span.add_event("Directory scan completed")

                    except Exception as e:
                        dir_span.record_exception(e)
                        logger.error(f"❌ Error scanning directory {watch_dir}: {e}")
                        self.stats["errors"].append(f"Directory scan error: {e}")

            span.set_attribute("directories.scanned", directories_scanned)
            span.set_attribute("files.total_found", files_found)
            span.set_attribute("files.new_found", len(new_files))

            span.add_event("Local directory scan completed", {
                "directories_scanned": directories_scanned,
                "new_files_found": len(new_files)
            })

            return new_files

    def scan_google_drive(self) -> List[Dict]:
        """Scan Google Drive for new files"""
        with tracer.start_as_current_span("scan_google_drive") as span:
            span.set_attribute("scan.type", "google_drive")

            google_drive_scan_counter.add(1, {"status": "attempt"})

            if not self.google_drive_loader:
                span.set_attribute("google_drive.available", False)
                span.add_event("Google Drive loader not available")
                logger.warning("⚠️ Google Drive loader not available")
                return []

            span.set_attribute("google_drive.available", True)
            span.set_attribute("folder_id", self.google_drive_folder_id)

            new_files = []
            try:
                logger.info("🔍 Scanning Google Drive for files...")
                span.add_event("Starting Google Drive scan")

                # Get all files from Google Drive (recursively from all subfolders)
                with tracer.start_as_current_span("list_drive_files") as list_span:
                    files = self.google_drive_loader._list_files(self.google_drive_folder_id)
                    list_span.set_attribute("files.total_found", len(files))

                    logger.info(f"📁 Found {len(files)} total files in Google Drive")

                span.set_attribute("drive_files.total", len(files))
                processable_files = 0

                for file_info in files:
                    with tracer.start_as_current_span("process_drive_file_info") as file_span:
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
                            file_span.set_attribute("file.processable", False)
                            file_span.add_event("Skipped Google Apps file")
                            continue

                        processable_files += 1
                        file_span.set_attribute("file.processable", True)

                        # For Google Drive files, we use file_id + name as the path
                        file_path = f"gdrive://{file_id}/{file_name}"

                        # Use file_id + name as hash since we can't get actual file hash without downloading
                        file_hash = hashlib.md5(f"{file_id}_{file_name}_{mime_type}".encode()).hexdigest()

                        file_span.set_attribute("file.path", file_path)
                        file_span.set_attribute("file.hash", file_hash)

                        # Check if already processed (both in local DB and Qdrant)
                        with tracer.start_as_current_span("check_file_processed") as check_span:
                            already_processed = (
                                self.fingerprint_db.is_file_processed(file_path, file_hash) or
                                self.fingerprint_db.is_file_in_qdrant(file_path, self.processor.qdrant_client)
                            )

                            check_span.set_attribute("file.already_processed", already_processed)

                            if already_processed:
                                logger.debug(f"⏭️  Skipping already processed file: {file_name}")
                                file_span.add_event("File already processed")
                                continue

                        logger.info(f"🆕 Found new file: {file_name} (Type: {mime_type})")
                        file_span.add_event("New file found")

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

                span.set_attribute("drive_files.processable", processable_files)
                span.set_attribute("drive_files.new", len(new_files))

                google_drive_scan_counter.add(1, {"status": "success"})

                span.add_event("Google Drive scan completed", {
                    "total_files": len(files),
                    "processable_files": processable_files,
                    "new_files": len(new_files)
                })

            except Exception as e:
                span.record_exception(e)
                google_drive_scan_counter.add(1, {"status": "error"})

                logger.error(f"❌ Error scanning Google Drive: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.stats["errors"].append(f"Google Drive scan error: {e}")

            return new_files

    def cleanup_extracted_images(self, documents: List[Document]):
        """Clean up locally extracted images after processing"""
        with tracer.start_as_current_span("cleanup_extracted_images") as span:
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

                span.set_attribute("images.to_clean", len(image_paths_to_clean))

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

                span.set_attribute("images.cleaned", cleaned_count)

                if cleaned_count > 0:
                    span.add_event("Image cleanup completed", {"cleaned_count": cleaned_count})
                    logger.info(f"🧹 Cleaned up {cleaned_count} extracted image files")

            except Exception as e:
                span.record_exception(e)
                logger.warning(f"⚠️ Error during image cleanup: {e}")

    def process_file(self, file_info: Dict) -> bool:
        """Process a single file"""
        with tracer.start_as_current_span("process_single_file") as span:
            file_path = file_info["file_path"]
            file_name = file_info["file_name"]
            source = file_info["source"]

            span.set_attribute("file.name", file_name)
            span.set_attribute("file.source", source)
            span.set_attribute("file.path", file_path[:100])  # Truncate for security
            span.set_attribute("file.size", file_info.get("file_size", 0))
            span.set_attribute("file.mime_type", file_info.get("mime_type", "unknown"))

            file_processing_counter.add(1, {"source": source, "status": "attempt"})

            try:
                logger.info(f"🔄 Processing {source} file: {file_name}")
                span.add_event("Starting file processing")

                # Load documents
                documents = []
                temp_file = None

                if source == "local":
                    with tracer.start_as_current_span("load_local_file") as load_span:
                        load_span.set_attribute("file.type", "local")
                        documents = load_file(file_path)
                        load_span.set_attribute("documents.loaded", len(documents))

                        # For local files, we need to clean up extracted images manually
                        self.cleanup_extracted_images(documents)

                elif source == "google_drive":
                    with tracer.start_as_current_span("load_google_drive_file") as drive_load_span:
                        drive_load_span.set_attribute("file.type", "google_drive")
                        drive_file_info = file_info["drive_file_info"]

                        # Download from Google Drive first
                        temp_file = self.google_drive_loader._download_file(drive_file_info)
                        drive_load_span.set_attribute("temp_file", temp_file if temp_file else "none")

                        documents = load_file(temp_file)
                        drive_load_span.set_attribute("documents.loaded", len(documents))

                        # Process documents with Google Drive loader to handle image upload and cleanup
                        self.google_drive_loader._process_docs(documents, drive_file_info)

                        # Update file size with actual downloaded size
                        if temp_file and os.path.exists(temp_file):
                            file_info["file_size"] = os.path.getsize(temp_file)
                            drive_load_span.set_attribute("file.actual_size", file_info["file_size"])

                if not documents:
                    span.set_attribute("documents.count", 0)
                    span.add_event("No documents extracted")

                    file_processing_counter.add(1, {"source": source, "status": "no_documents"})

                    logger.warning(f"⚠️ No documents extracted from {file_name}")
                    return False

                span.set_attribute("documents.count", len(documents))
                logger.info(f"📄 Extracted {len(documents)} documents from {file_name}")

                # Process documents and get point IDs
                with tracer.start_as_current_span("process_and_embed_documents") as process_span:
                    process_span.set_attribute("documents.to_process", len(documents))
                    point_ids = self.processor.process_documents(documents, file_info)
                    process_span.set_attribute("qdrant.point_ids_generated", len(point_ids))

                # Mark as processed
                with tracer.start_as_current_span("mark_file_processed") as mark_span:
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
                    mark_span.set_attribute("processed_file.document_count", len(documents))

                # Update stats
                self.stats["files_processed"] += 1
                self.stats["documents_created"] += len(documents)

                # Clean up temporary file
                if temp_file and os.path.exists(temp_file):
                    with tracer.start_as_current_span("cleanup_temp_file") as cleanup_span:
                        try:
                            os.remove(temp_file)
                            cleanup_span.set_attribute("cleanup.success", True)
                            logger.debug(f"🧹 Cleaned up temporary file: {os.path.basename(temp_file)}")
                        except Exception as e:
                            cleanup_span.record_exception(e)
                            logger.debug(f"⚠️ Could not remove temporary file {temp_file}: {e}")

                # Final cleanup: remove any remaining extracted images that might have been missed
                if source == "local":
                    self.cleanup_extracted_images(documents)

                file_processing_counter.add(1, {"source": source, "status": "success"})

                span.set_attribute("processing.success", True)
                span.add_event("File processing completed successfully", {
                    "documents_processed": len(documents),
                    "point_ids_generated": len(point_ids)
                })

                logger.info(f"✅ Successfully processed {file_name} -> {len(documents)} documents, {len(point_ids)} points")
                return True

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("processing.success", False)

                file_processing_counter.add(1, {"source": source, "status": "error"})

                logger.error(f"❌ Failed to process file {file_info.get('file_name', 'unknown')}: {e}")
                self.stats["errors"].append(f"File processing error: {e}")
                return False

    def scan_and_process(self):
        """Scan for new files and process them - each file gets its own trace"""
        with tracer.start_as_current_span("scan_and_process_cycle") as span:
            try:
                logger.info("🔍 Starting scan cycle...")
                span.add_event("Scan cycle started")

                # Prioritize Google Drive scanning
                new_files = []

                # Scan Google Drive first (main focus)
                with tracer.start_as_current_span("google_drive_scan") as drive_scan:
                    google_drive_files = self.scan_google_drive()
                    new_files.extend(google_drive_files)
                    drive_scan.set_attribute("files.found", len(google_drive_files))

                # Scan local directories if configured
                if self.local_watch_dirs:
                    with tracer.start_as_current_span("local_directories_scan") as local_scan:
                        local_files = self.scan_local_directories()
                        new_files.extend(local_files)
                        local_scan.set_attribute("files.found", len(local_files))

                self.stats["last_scan"] = datetime.now()
                span.set_attribute("scan.timestamp", self.stats["last_scan"].isoformat())
                span.set_attribute("files.total_found", len(new_files))

                if not new_files:
                    span.add_event("No new files found")
                    logger.info("✅ No new files found")
                    return

                logger.info(f"📁 Found {len(new_files)} new files to process:")
                for file_info in new_files:
                    logger.info(f"   - {file_info['file_name']} ({file_info['source']})")

                # Process each file with NEW TRACE for each file
                processed_count = 0
                failed_count = 0

                for i, file_info in enumerate(new_files):
                    if not self.is_running:
                        span.add_event("Service stopping, halting file processing")
                        logger.info("🛑 Service stopping, halting file processing")
                        break

                    # 🎯 CREATE NEW TRACE FOR EACH FILE - This is the key change!
                    with tracer.start_as_current_span(
                        "process_new_file", 
                        kind=trace.SpanKind.SERVER  # Root span for new trace
                    ) as file_span:
                        # Generate new trace context for this file
                        file_span.set_attribute("file.name", file_info['file_name'])
                        file_span.set_attribute("file.source", file_info['source'])
                        file_span.set_attribute("file.index", i + 1)
                        file_span.set_attribute("file.total", len(new_files))
                        file_span.set_attribute("file.size", file_info.get('file_size', 0))
                        file_span.set_attribute("file.mime_type", file_info.get('mime_type', 'unknown'))
                        
                        try:
                            logger.info(f"🔄 Processing file {processed_count + 1}/{len(new_files)}: {file_info['file_name']}")

                            # All subsequent spans inherit this NEW trace ID
                            if self.process_file(file_info):
                                processed_count += 1
                                file_span.set_attribute("processing.result", "success")
                                file_span.add_event("File processed successfully")
                                logger.info(f"✅ Successfully processed: {file_info['file_name']}")
                            else:
                                failed_count += 1
                                file_span.set_attribute("processing.result", "failed")
                                file_span.add_event("File processing failed")
                                logger.warning(f"⚠️ Failed to process: {file_info['file_name']}")

                        except Exception as e:
                            failed_count += 1
                            file_span.record_exception(e)
                            file_span.set_attribute("processing.result", "error")
                            file_span.add_event("File processing error", {
                                "error_message": str(e)
                            })
                            logger.error(f"❌ Error processing file {file_info.get('file_name')}: {e}")
                            import traceback
                            logger.error(f"Full traceback: {traceback.format_exc()}")
                            continue

                        # Small delay between files to avoid overwhelming the system
                        time.sleep(2)

                span.set_attribute("processing.successful", processed_count)
                span.set_attribute("processing.failed", failed_count)

                span.add_event("Scan and process cycle completed", {
                    "files_processed": processed_count,
                    "files_failed": failed_count,
                    "total_files": len(new_files)
                })

                logger.info(f"✅ Scan and process cycle completed: {processed_count} processed, {failed_count} failed")

            except Exception as e:
                span.record_exception(e)
                logger.error(f"❌ Error in scan and process cycle: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.stats["errors"].append(f"Scan cycle error: {e}")

    async def start_monitoring(self):
        """Start the continuous monitoring service"""
        with tracer.start_as_current_span("start_monitoring_service") as span:
            self.is_running = True
            span.set_attribute("monitoring.status", "started")
            span.set_attribute("scan_interval", self.scan_interval)

            logger.info(f"🚀 Starting continuous monitoring service...")
            logger.info(f"⏱️ Scan interval: {self.scan_interval} seconds")

            scan_count = 0

            while self.is_running:
                try:
                    scan_count += 1

                    with tracer.start_as_current_span(f"monitoring_scan_{scan_count}") as scan_span:
                        scan_span.set_attribute("scan.number", scan_count)
                        scan_span.set_attribute("service.running", self.is_running)

                        self.scan_and_process()
                        scan_span.add_event("Scan completed")

                    # Wait for next scan
                    for _ in range(self.scan_interval):
                        if not self.is_running:
                            break
                        await asyncio.sleep(1)

                except Exception as e:
                    with tracer.start_as_current_span("monitoring_error") as error_span:
                        error_span.record_exception(e)
                        error_span.set_attribute("scan.number", scan_count)

                        logger.error(f"❌ Error in monitoring loop: {e}")
                        self.stats["errors"].append(f"Monitoring loop error: {e}")
                        await asyncio.sleep(10)  # Wait before retrying

            span.add_event("Monitoring service stopped", {"total_scans": scan_count})
            logger.info("🛑 Monitoring service stopped")

    def stop_monitoring(self):
        """Stop the monitoring service"""
        with tracer.start_as_current_span("stop_monitoring_service") as span:
            self.is_running = False
            service_health_gauge.add(-1, {"service": "backend"})

            span.set_attribute("monitoring.status", "stopped")
            span.add_event("Monitoring service stop requested")

            logger.info("🛑 Stopping monitoring service...")

    def get_status(self) -> Dict:
        """Get service status"""
        with tracer.start_as_current_span("get_service_status") as span:
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

            span.set_attribute("service.is_running", self.is_running)
            span.set_attribute("processing.total_files", db_stats["total_files"])
            span.set_attribute("processing.total_documents", db_stats["total_documents"])

            return status

# Global service instance
service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management with OpenTelemetry instrumentation"""
    global service

    with tracer.start_as_current_span("fastapi_lifespan") as span:
        # Startup
        logger.info("🚀 Starting Backend Service...")
        span.add_event("FastAPI startup initiated")

        service = BackendService()
        span.set_attribute("service.initialized", True)

        # Start monitoring in background
        monitoring_task = asyncio.create_task(service.start_monitoring())
        span.add_event("Background monitoring task started")

        yield

        # Shutdown
        logger.info("🛑 Shutting down Backend Service...")
        span.add_event("FastAPI shutdown initiated")

        if service:
            service.stop_monitoring()

        monitoring_task.cancel()
        span.add_event("Background monitoring task cancelled")

# FastAPI app for status monitoring
app = FastAPI(
    title="Document Processing Backend Service",
    description="Continuous monitoring and processing of documents with OpenTelemetry observability",
    version="1.0.0",
    lifespan=lifespan
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

@app.get("/")
async def root():
    """Root endpoint"""
    with tracer.start_as_current_span("api_root") as span:
        span.set_attribute("http.endpoint", "/")
        return {"message": "Document Processing Backend Service", "status": "running"}

@app.get("/status")
async def get_status():
    """Get service status"""
    with tracer.start_as_current_span("api_get_status") as span:
        span.set_attribute("http.endpoint", "/status")

        if not service:
            span.set_attribute("service.available", False)
            raise HTTPException(status_code=503, detail="Service not initialized")

        span.set_attribute("service.available", True)
        status = service.get_status()
        span.add_event("Status retrieved")

        return status

@app.post("/scan")
async def trigger_scan():
    """Manually trigger a scan for new files"""
    with tracer.start_as_current_span("api_trigger_scan") as span:
        span.set_attribute("http.endpoint", "/scan")

        if not service:
            span.set_attribute("service.available", False)
            raise HTTPException(status_code=503, detail="Service not initialized")

        if not service.is_running:
            span.set_attribute("service.running", False)
            raise HTTPException(status_code=503, detail="Service not running")

        span.set_attribute("service.running", True)

        # Trigger scan in background
        asyncio.create_task(asyncio.to_thread(service.scan_and_process))
        span.add_event("Manual scan triggered")

        return {"message": "Scan triggered successfully"}

@app.post("/stop")
async def stop_service():
    """Stop the monitoring service"""
    with tracer.start_as_current_span("api_stop_service") as span:
        span.set_attribute("http.endpoint", "/stop")

        if not service:
            span.set_attribute("service.available", False)
            raise HTTPException(status_code=503, detail="Service not initialized")

        service.stop_monitoring()
        span.add_event("Service stopped")

        return {"message": "Service stopped"}

@app.post("/start")
async def start_service():
    """Start the monitoring service"""
    with tracer.start_as_current_span("api_start_service") as span:
        span.set_attribute("http.endpoint", "/start")

        if not service:
            span.set_attribute("service.available", False)
            raise HTTPException(status_code=503, detail="Service not initialized")

        if service.is_running:
            span.set_attribute("service.already_running", True)
            return {"message": "Service already running"}

        # Start monitoring in background
        asyncio.create_task(service.start_monitoring())
        span.add_event("Service started")

        return {"message": "Service started"}

@app.post("/reset")
async def reset_processed_files():
    """Clear the processed files database to force reprocessing"""
    with tracer.start_as_current_span("api_reset_database") as span:
        span.set_attribute("http.endpoint", "/reset")

        if not service:
            span.set_attribute("service.available", False)
            raise HTTPException(status_code=503, detail="Service not initialized")

        try:
            service.fingerprint_db.clear_processed_files()
            span.add_event("Database reset completed")
            return {"message": "Processed files database cleared successfully"}
        except Exception as e:
            span.record_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to clear database: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Document Processing Backend Service with OpenTelemetry")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    with tracer.start_as_current_span("uvicorn_server_start") as span:
        span.set_attribute("server.host", args.host)
        span.set_attribute("server.port", args.port)
        span.set_attribute("server.reload", args.reload)

        logger.info(f"🌐 Starting server on {args.host}:{args.port}")

        uvicorn.run(
            "backend_service:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )

