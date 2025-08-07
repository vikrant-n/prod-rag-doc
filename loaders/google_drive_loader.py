#!/usr/bin/env python3
"""
Google Drive Loader with OpenTelemetry Instrumentation

Enhanced with comprehensive OpenTelemetry observability following EDOT best practices
for distributed tracing, metrics, and structured logging correlation.

Demonstrates complete Google Drive integration observability WITHOUT LOGSTASH DEPENDENCY - showing how OpenTelemetry 
can handle multiple log types and correlate them with trace IDs and span IDs for ELK visualization.
"""

import io
import os
import json
import tempfile
import time
import socket
import uuid
import logging
from pathlib import Path
from typing import List, Dict

# OpenTelemetry imports - Official EDOT libraries
from opentelemetry import trace, metrics, baggage
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, SERVICE_INSTANCE_ID
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.propagate import set_global_textmap, extract
from opentelemetry.propagators.b3 import B3MultiFormat

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from langchain_core.documents import Document
from .master_loaders import load_file

# Remove circular import: import these only inside methods
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 100

SCOPES = ["https://www.googleapis.com/auth/drive"]

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
        SERVICE_NAME: "document-rag-google-drive",
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
        
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(meter_provider)
        
        print("✅ OpenTelemetry initialized for Google Drive loader")
        
    except Exception as e:
        print(f"⚠️  OpenTelemetry provider setup error: {e}")
    
    # Set up B3 propagator for distributed tracing
    set_global_textmap(B3MultiFormat())
    
    # Manual instrumentation only if not using auto-instrumentation
    if not os.getenv("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"):
        try:
            LoggingInstrumentor().instrument(set_logging_format=True)
        except Exception as e:
            print(f"⚠️  Instrumentation warning: {e}")
    
    return trace.get_tracer(__name__), metrics.get_meter(__name__)

# Initialize telemetry
tracer, meter = init_telemetry()

# Custom metrics for Google Drive operations observability
google_api_calls_counter = meter.create_counter(
    "google_api_calls_total",
    description="Total number of Google API calls",
    unit="1"
)

file_download_duration = meter.create_histogram(
    "file_download_duration_seconds",
    description="Time taken to download files from Google Drive",
    unit="s"
)

file_upload_counter = meter.create_counter(
    "file_uploads_total",
    description="Total number of files uploaded to Google Drive",
    unit="1"
)

authentication_counter = meter.create_counter(
    "authentication_operations_total",
    description="Total number of authentication operations",
    unit="1"
)

# Configure structured logging with OpenTelemetry correlation
class OtelGoogleDriveFormatter(logging.Formatter):
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
formatter = OtelGoogleDriveFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler('google_drive_loader.log')
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

class GoogleDriveMasterLoader:
    """Recursively fetch files from a Google Drive folder and load them with comprehensive OpenTelemetry instrumentation.

    Parameters
    ----------
    folder_id : str
        Google Drive folder ID to traverse.
    credentials_path : str, optional
        Path to OAuth 2.0 credentials JSON, by default ``"credentials.json"``.
    token_path : str, optional
        Path to the token JSON file, by default ``"token.json"``.
    chunk_size : int, optional
        Desired chunk size when ``split`` is ``True``.
    chunk_overlap : int, optional
        Desired chunk overlap when ``split`` is ``True``.
    split : bool, optional
        Whether to split loaded documents using :func:`split_documents`,
        by default ``True``.
    """

    def __init__(
        self,
        folder_id: str,
        credentials_path: str = "credentials.json",
        token_path: str = "token.json",
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        split: bool = True,
    ):
        with tracer.start_as_current_span("google_drive_loader_init") as span:
            self.folder_id = folder_id
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.split = split
            
            span.set_attribute("folder_id", folder_id)
            span.set_attribute("chunk_size", chunk_size)
            span.set_attribute("chunk_overlap", chunk_overlap)
            span.set_attribute("split", split)
            span.set_attribute("credentials_path", credentials_path)
            span.set_attribute("token_path", token_path)
            
            # Add baggage for cross-service correlation
            baggage.set_baggage("google_drive.folder_id", folder_id)
            
            creds = None
            
            # Check for existing token
            with tracer.start_as_current_span("load_existing_token") as token_span:
                if os.path.exists(token_path):
                    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
                    token_span.set_attribute("token.loaded", True)
                    token_span.add_event("Existing token loaded")
                    logger.info("🔑 Loaded existing Google Drive token")
                else:
                    token_span.set_attribute("token.loaded", False)
                    token_span.add_event("No existing token found")

            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                with tracer.start_as_current_span("authenticate_google_drive") as auth_span:
                    authentication_counter.add(1, {"operation": "attempt"})
                    
                    if creds and creds.expired and creds.refresh_token:
                        try:
                            with tracer.start_as_current_span("refresh_token") as refresh_span:
                                creds.refresh(Request())
                                refresh_span.set_attribute("refresh.successful", True)
                                refresh_span.add_event("Token refreshed successfully")
                                authentication_counter.add(1, {"operation": "refresh_success"})
                                logger.info("🔄 Google Drive token refreshed")
                        except Exception as e:
                            refresh_span.record_exception(e)
                            refresh_span.set_attribute("refresh.successful", False)
                            authentication_counter.add(1, {"operation": "refresh_failed"})
                            logger.warning(f"⚠️ Token refresh failed: {e}")
                            creds = None
                    
                    if not creds or not creds.valid:
                        with tracer.start_as_current_span("oauth_flow") as oauth_span:
                            oauth_span.set_attribute("credentials_path", credentials_path)
                            
                            flow = InstalledAppFlow.from_client_secrets_file(
                                credentials_path, SCOPES
                            )
                            creds = flow.run_local_server(port=0)
                            
                            oauth_span.add_event("OAuth flow completed")
                            authentication_counter.add(1, {"operation": "oauth_success"})
                            logger.info("🔐 Google Drive OAuth authentication completed")
                    
                    # Save the credentials for the next run
                    with tracer.start_as_current_span("save_token") as save_span:
                        with open(token_path, "w") as token:
                            token.write(creds.to_json())
                        save_span.add_event("Token saved")
                        logger.info("💾 Google Drive token saved")
                    
                    auth_span.set_attribute("authentication.successful", True)

            # Build the service
            with tracer.start_as_current_span("build_drive_service") as service_span:
                self.service = build("drive", "v3", credentials=creds)
                service_span.add_event("Google Drive service built")
                
            # Initialize image folder
            with tracer.start_as_current_span("setup_image_folder") as img_folder_span:
                self.image_folder_id = self._get_or_create_image_folder()
                img_folder_span.set_attribute("image_folder_id", self.image_folder_id)
                img_folder_span.add_event("Image folder setup completed")
                
            span.add_event("Google Drive loader initialized successfully")
            logger.info("✅ Google Drive loader initialized successfully")

    def _get_or_create_image_folder(self) -> str:
        """Ensure a subfolder for extracted images exists and return its ID with instrumentation."""
        with tracer.start_as_current_span("get_or_create_image_folder") as span:
            google_api_calls_counter.add(1, {"operation": "list_files", "purpose": "image_folder_check"})
            
            query = (
                f"'{self.folder_id}' in parents and "
                "mimeType='application/vnd.google-apps.folder' and "
                "name='extracted_images' and trashed=false"
            )
            
            span.set_attribute("query", query)
            
            resp = (
                self.service.files()
                .list(q=query, fields="files(id, name)")
                .execute()
            )
            files = resp.get("files", [])
            
            span.set_attribute("existing_folders_found", len(files))
            
            if files:
                folder_id = files[0]["id"]
                span.set_attribute("folder.exists", True)
                span.set_attribute("folder.id", folder_id)
                span.add_event("Existing image folder found")
                logger.info("📁 Found existing extracted_images folder")
                return folder_id

            # Create new folder
            with tracer.start_as_current_span("create_image_folder") as create_span:
                google_api_calls_counter.add(1, {"operation": "create_folder", "purpose": "image_storage"})
                
                metadata = {
                    "name": "extracted_images",
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [self.folder_id],
                }
                
                folder = (
                    self.service.files()
                    .create(body=metadata, fields="id")
                    .execute()
                )
                
                folder_id = folder["id"]
                create_span.set_attribute("folder.id", folder_id)
                create_span.add_event("Image folder created")
                
            span.set_attribute("folder.exists", False)
            span.set_attribute("folder.created", True)
            span.set_attribute("folder.id", folder_id)
            
            logger.info("📁 Created new extracted_images folder")
            return folder_id

    def _upload_file_to_drive(self, local_path: str, filename: str) -> str:
        """Upload a file to the images folder and return the file ID with instrumentation."""
        with tracer.start_as_current_span("upload_file_to_drive") as span:
            start_time = time.time()
            
            span.set_attribute("file.local_path", local_path[:100])  # Truncate for security
            span.set_attribute("file.name", filename)
            span.set_attribute("destination.folder_id", self.image_folder_id)
            
            file_upload_counter.add(1, {"status": "attempt"})
            google_api_calls_counter.add(1, {"operation": "upload_file", "purpose": "image_upload"})
            
            try:
                # Get file size for metrics
                file_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                span.set_attribute("file.size_bytes", file_size)
                
                media = MediaFileUpload(local_path, resumable=True)
                body = {"name": filename, "parents": [self.image_folder_id]}
                
                uploaded = (
                    self.service.files()
                    .create(body=body, media_body=media, fields="id")
                    .execute()
                )
                
                upload_duration = time.time() - start_time
                file_id = uploaded["id"]
                
                span.set_attribute("upload.duration_seconds", upload_duration)
                span.set_attribute("upload.file_id", file_id)
                span.set_attribute("upload.successful", True)
                
                span.add_event("File upload completed", {
                    "file_id": file_id,
                    "duration_seconds": upload_duration,
                    "file_size_bytes": file_size
                })
                
                file_upload_counter.add(1, {"status": "success"})
                
                logger.info(f"⬆️ Uploaded {filename} to Google Drive in {upload_duration:.2f}s")
                return file_id
                
            except Exception as e:
                upload_duration = time.time() - start_time
                span.record_exception(e)
                span.set_attribute("upload.duration_seconds", upload_duration)
                span.set_attribute("upload.successful", False)
                
                file_upload_counter.add(1, {"status": "error"})
                
                logger.error(f"❌ Failed to upload {filename}: {e}")
                raise

    def _process_docs(self, docs: List[Document], file_meta: Dict) -> None:
        """Update metadata and upload extracted images with instrumentation."""
        with tracer.start_as_current_span("process_document_metadata") as span:
            span.set_attribute("documents.count", len(docs))
            span.set_attribute("file.id", file_meta["id"])
            span.set_attribute("file.name", file_meta["name"])
            
            drive_link = f"https://drive.google.com/file/d/{file_meta['id']}/view?usp=drive_link"
            path_map: Dict[str, str] = {}
            
            # Update metadata for all documents
            with tracer.start_as_current_span("update_document_metadata") as meta_span:
                for doc in docs:
                    doc.metadata["source"] = drive_link
                    doc.metadata["drive_file_id"] = file_meta["id"]
                    doc.metadata["drive_file_name"] = file_meta["name"]
                
                meta_span.add_event("Document metadata updated")

            # Process and upload images
            image_count = 0
            with tracer.start_as_current_span("process_extracted_images") as img_span:
                for doc in docs:
                    img_path = doc.metadata.get("image_path")
                    if img_path:
                        image_count += 1
                        
                        with tracer.start_as_current_span(f"upload_image_{image_count}") as upload_span:
                            upload_span.set_attribute("image.local_path", img_path[:100])
                            
                            img_id = self._upload_file_to_drive(img_path, os.path.basename(img_path))
                            drive_img_link = f"https://drive.google.com/file/d/{img_id}/view?usp=drive_link"
                            path_map[img_path] = drive_img_link
                            doc.metadata["image_path"] = drive_img_link
                            doc.metadata["image_file_name"] = os.path.basename(img_path)
                            
                            upload_span.set_attribute("image.drive_id", img_id)
                            upload_span.add_event("Image uploaded and metadata updated")
                            
                            # Clean up local file
                            try:
                                os.remove(img_path)
                                upload_span.add_event("Local image file cleaned up")
                            except OSError:
                                upload_span.add_event("Failed to clean up local image file")

                img_span.set_attribute("images.processed", image_count)
                img_span.add_event("Image processing completed")

            # Update related images metadata
            with tracer.start_as_current_span("update_related_images") as related_span:
                for doc in docs:
                    if "related_images" in doc.metadata:
                        doc.metadata["related_images"] = [path_map.get(p, p) for p in doc.metadata["related_images"]]
                
                related_span.add_event("Related images metadata updated")

            span.add_event("Document processing completed", {
                "documents_processed": len(docs),
                "images_uploaded": image_count
            })
            
            logger.info(f"🔄 Processed {len(docs)} documents with {image_count} images")

    def _list_files(self, folder_id: str) -> List[dict]:
        """Yield file metadata for all files in the folder (recursively) with instrumentation."""
        with tracer.start_as_current_span("list_files_recursive") as span:
            span.set_attribute("folder_id", folder_id)
            
            # Skip the image folder to avoid processing our own uploads
            if folder_id == self.image_folder_id:
                span.add_event("Skipping image folder")
                return []

            google_api_calls_counter.add(1, {"operation": "list_files", "purpose": "file_discovery"})
            
            query = f"'{folder_id}' in parents and trashed=false"
            page_token = None
            files = []
            page_count = 0

            while True:
                page_count += 1
                with tracer.start_as_current_span(f"list_files_page_{page_count}") as page_span:
                    page_span.set_attribute("page.number", page_count)
                    page_span.set_attribute("page.token", page_token or "first_page")
                    
                    response = (
                        self.service.files()
                        .list(q=query, fields="nextPageToken, files(id, name, mimeType)", pageToken=page_token)
                        .execute()
                    )
                    
                    page_files = response.get("files", [])
                    files.extend(page_files)
                    page_token = response.get("nextPageToken")
                    
                    page_span.set_attribute("page.files_found", len(page_files))
                    page_span.add_event("Page processed")
                    
                    if not page_token:
                        break

            span.set_attribute("files.found", len(files))
            span.set_attribute("pages.processed", page_count)

            # Process files and recurse into subfolders
            result = []
            folder_count = 0
            
            for f in files:
                if f["id"] == self.image_folder_id:
                    continue
                    
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    folder_count += 1
                    with tracer.start_as_current_span(f"recurse_folder_{folder_count}") as recurse_span:
                        recurse_span.set_attribute("folder.name", f["name"])
                        recurse_span.set_attribute("folder.id", f["id"])
                        
                        subfolder_files = self._list_files(f["id"])
                        result.extend(subfolder_files)
                        
                        recurse_span.set_attribute("subfolder.files_found", len(subfolder_files))
                        recurse_span.add_event("Subfolder processed")
                else:
                    result.append(f)

            span.set_attribute("subfolders.processed", folder_count)
            span.set_attribute("files.total_result", len(result))
            
            span.add_event("File listing completed", {
                "direct_files": len(files) - folder_count,
                "subfolders": folder_count,
                "total_files": len(result)
            })
            
            logger.info(f"📋 Listed {len(result)} files from folder (processed {folder_count} subfolders)")
            return result

    def _download_file(self, file_info: dict) -> str:
        """Download a Drive file and return the local path with comprehensive instrumentation."""
        with tracer.start_as_current_span("download_file_from_drive") as span:
            start_time = time.time()
            
            mime = file_info["mimeType"]
            file_id = file_info["id"]
            name = file_info["name"]
            
            span.set_attribute("file.id", file_id)
            span.set_attribute("file.name", name)
            span.set_attribute("file.mime_type", mime)
            
            google_api_calls_counter.add(1, {"operation": "download_file", "mime_type": mime})
            
            try:
                # Handle Google Docs types via export
                with tracer.start_as_current_span("prepare_download_request") as prep_span:
                    if mime == "application/vnd.google-apps.document":
                        request = self.service.files().export_media(
                            fileId=file_id,
                            mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )
                        suffix = ".docx"
                        prep_span.set_attribute("export.format", "docx")
                    elif mime == "application/vnd.google-apps.spreadsheet":
                        request = self.service.files().export_media(
                            fileId=file_id,
                            mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                        suffix = ".xlsx"
                        prep_span.set_attribute("export.format", "xlsx")
                    elif mime == "application/vnd.google-apps.presentation":
                        request = self.service.files().export_media(
                            fileId=file_id,
                            mimeType="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        )
                        suffix = ".pptx"
                        prep_span.set_attribute("export.format", "pptx")
                    else:
                        request = self.service.files().get_media(fileId=file_id)
                        _, suffix = os.path.splitext(name)
                        prep_span.set_attribute("download.type", "direct")
                    
                    prep_span.set_attribute("file.suffix", suffix)

                # Download the file
                with tracer.start_as_current_span("download_file_content") as download_span:
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    chunks_downloaded = 0
                    
                    while not done:
                        status, done = downloader.next_chunk()
                        chunks_downloaded += 1
                        if status:
                            download_span.set_attribute("download.progress", status.progress())
                    
                    download_span.set_attribute("download.chunks", chunks_downloaded)
                    download_span.add_event("File download completed")

                # Save to temporary file
                with tracer.start_as_current_span("save_temp_file") as save_span:
                    fh.seek(0)
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(fh.read())
                    tmp.close()
                    
                    file_size = os.path.getsize(tmp.name)
                    download_duration = time.time() - start_time
                    
                    save_span.set_attribute("temp_file.path", tmp.name)
                    save_span.set_attribute("temp_file.size_bytes", file_size)
                    save_span.add_event("Temporary file created")

                # Record metrics
                file_download_duration.record(download_duration, {
                    "mime_type": mime,
                    "file_size_mb": str(file_size // (1024 * 1024))
                })

                span.set_attribute("download.duration_seconds", download_duration)
                span.set_attribute("download.file_size_bytes", file_size)
                span.set_attribute("download.temp_path", tmp.name)
                span.set_attribute("download.successful", True)
                
                span.add_event("File download completed successfully", {
                    "file_name": name,
                    "file_size_bytes": file_size,
                    "duration_seconds": download_duration,
                    "chunks_downloaded": chunks_downloaded
                })
                
                logger.info(f"⬇️ Downloaded {name} ({file_size / 1024 / 1024:.1f} MB) in {download_duration:.2f}s")
                return tmp.name
                
            except Exception as e:
                download_duration = time.time() - start_time
                span.record_exception(e)
                span.set_attribute("download.duration_seconds", download_duration)
                span.set_attribute("download.successful", False)
                
                logger.error(f"❌ Failed to download {name}: {e}")
                raise

    def load(self) -> List[Document]:
        """Load all documents from Google Drive with comprehensive instrumentation."""
        with tracer.start_as_current_span("load_google_drive_documents") as span:
            start_time = time.time()
            
            span.set_attribute("folder_id", self.folder_id)
            span.set_attribute("split_documents", self.split)
            span.set_attribute("chunk_size", self.chunk_size)
            span.set_attribute("chunk_overlap", self.chunk_overlap)
            
            docs: List[Document] = []
            
            try:
                # Get all files from the folder
                with tracer.start_as_current_span("discover_all_files") as discovery_span:
                    all_files = self._list_files(self.folder_id)
                    discovery_span.set_attribute("files.discovered", len(all_files))
                    discovery_span.add_event("File discovery completed")
                    
                    logger.info(f"📂 Discovered {len(all_files)} files in Google Drive")

                # Process each file
                processed_files = 0
                failed_files = 0
                
                for file_meta in all_files:
                    with tracer.start_as_current_span("process_single_drive_file") as file_span:
                        file_span.set_attribute("file.id", file_meta["id"])
                        file_span.set_attribute("file.name", file_meta["name"])
                        file_span.set_attribute("file.mime_type", file_meta["mimeType"])
                        
                        local_path = None
                        try:
                            # Download the file
                            local_path = self._download_file(file_meta)
                            
                            # Load the document
                            with tracer.start_as_current_span("load_downloaded_file") as load_span:
                                loaded_docs = load_file(local_path)
                                load_span.set_attribute("documents.loaded", len(loaded_docs))
                                
                            # Process metadata and images
                            self._process_docs(loaded_docs, file_meta)
                            
                            # Split documents if requested
                            if self.split:
                                with tracer.start_as_current_span("split_documents") as split_span:
                                    from text_splitting import split_documents
                                    loaded_docs = split_documents(
                                        loaded_docs,
                                        chunk_size=self.chunk_size,
                                        chunk_overlap=self.chunk_overlap,
                                    )
                                    split_span.set_attribute("chunks.created", len(loaded_docs))
                                    split_span.add_event("Documents split successfully")
                            
                            docs.extend(loaded_docs)
                            processed_files += 1
                            
                            file_span.set_attribute("processing.successful", True)
                            file_span.set_attribute("documents.final_count", len(loaded_docs))
                            file_span.add_event("File processed successfully")
                            
                            logger.info(f"✅ Processed {file_meta['name']} → {len(loaded_docs)} documents")
                            
                        except Exception as e:
                            failed_files += 1
                            file_span.record_exception(e)
                            file_span.set_attribute("processing.successful", False)
                            
                            logger.error(f"❌ Failed to process {file_meta['name']}: {e}")
                            
                        finally:
                            # Clean up downloaded file
                            if local_path and os.path.exists(local_path):
                                try:
                                    os.remove(local_path)
                                    file_span.add_event("Temporary file cleaned up")
                                except OSError:
                                    file_span.add_event("Failed to clean up temporary file")

                loading_duration = time.time() - start_time
                
                span.set_attribute("loading.duration_seconds", loading_duration)
                span.set_attribute("files.processed", processed_files)
                span.set_attribute("files.failed", failed_files)
                span.set_attribute("documents.total", len(docs))
                span.set_attribute("loading.successful", True)
                
                span.add_event("Google Drive loading completed successfully", {
                    "files_processed": processed_files,
                    "files_failed": failed_files,
                    "total_documents": len(docs),
                    "duration_seconds": loading_duration
                })
                
                logger.info(f"🎉 Google Drive loading completed: {processed_files} files → {len(docs)} documents in {loading_duration:.2f}s")
                return docs
                
            except Exception as e:
                loading_duration = time.time() - start_time
                span.record_exception(e)
                span.set_attribute("loading.duration_seconds", loading_duration)
                span.set_attribute("loading.successful", False)
                
                logger.error(f"❌ Google Drive loading failed: {e}")
                raise

def is_markdown(content: str) -> bool:
    """Check if content appears to be markdown formatted."""
    md_indicators = ["#", "-", "*", "`", ">", "[", "!", "```"]
    lines = content.strip().splitlines()
    return any(line.strip().startswith(tuple(md_indicators)) for line in lines[:5])  # Check first few lines

def main():
    """Main function with OpenTelemetry instrumentation for CLI usage"""
    import argparse

    with tracer.start_as_current_span("google_drive_loader_main") as span:
        parser = argparse.ArgumentParser(description="Load documents from a Google Drive folder")
        parser.add_argument("folder_id", help="ID of the Google Drive folder")
        parser.add_argument(
            "--credentials",
            default="credentials.json",
            help="Path to OAuth 2.0 credentials JSON",
        )
        parser.add_argument("--token", default="token.json", help="Path to token JSON")
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=DEFAULT_CHUNK_SIZE,
            help="Chunk size in characters",
        )
        parser.add_argument(
            "--chunk-overlap",
            type=int,
            default=DEFAULT_CHUNK_OVERLAP,
            help="Chunk overlap in characters",
        )
        parser.add_argument(
            "--output-file",
            default="output_documents.md",
            help="Write all chunks to this file",
        )
        parser.add_argument(
            "--print-chunks",
            action="store_true",
            help="Print each chunk to stdout",
        )
        parser.add_argument(
            "--no-split",
            action="store_true",
            help="Return raw documents without splitting",
        )
        args = parser.parse_args()

        span.set_attribute("cli.folder_id", args.folder_id)
        span.set_attribute("cli.chunk_size", args.chunk_size)
        span.set_attribute("cli.no_split", args.no_split)
        span.set_attribute("execution.mode", "cli")

        loader = GoogleDriveMasterLoader(
            args.folder_id,
            credentials_path=args.credentials,
            token_path=args.token,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            split=not args.no_split,
        )
        documents = loader.load()

        logger.info(f"\n✅ Loaded {len(documents)} documents from Google Drive\n")

        # Generate markdown output
        with tracer.start_as_current_span("generate_output") as output_span:
            markdown_lines = []

            for i, doc in enumerate(documents, start=1):
                markdown_lines.append(f"## 📄 Document {i}\n")

                markdown_lines.append(
                    f"**Metadata:**\n```json\n{json.dumps(doc.metadata, indent=2)}\n```"
                )

                markdown_lines.append("**Content:**")
                content = doc.page_content.strip()

                if is_markdown(content):
                    markdown_lines.append(content + "\n")
                else:
                    markdown_lines.append(f"```\n{content}\n```")

                if args.print_chunks:
                    print("---")
                    print(json.dumps(doc.metadata, indent=2))
                    print(content)

            output_path = Path(args.output_file)
            output_path.write_text("\n".join(markdown_lines), encoding="utf-8")
            
            output_span.set_attribute("output.file", args.output_file)
            output_span.set_attribute("output.lines", len(markdown_lines))
            output_span.add_event("Output file generated")

        span.add_event("CLI execution completed", {
            "documents_processed": len(documents),
            "output_file": args.output_file
        })
        
        logger.info(f"📝 Output saved to {output_path}")

if __name__ == "__main__":
    with tracer.start_as_current_span("google_drive_loader_standalone") as span:
        span.set_attribute("execution.context", "standalone")
        logger.info("🌐 Running Google Drive Loader in standalone mode")
        main()

