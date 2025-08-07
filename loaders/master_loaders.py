#!/usr/bin/env python3
"""
Master loader that delegates file loading to specific loaders based on file extension
Enhanced with comprehensive OpenTelemetry instrumentation following EDOT best practices
for distributed tracing, metrics, and structured logging correlation.
"""

import os
import time
import socket
import uuid
from typing import List

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

from langchain_core.documents import Document

from .pdf_loader import load_pdf_as_langchain_docs
from .docx_loader import load_docx_as_langchain_docs
from .pptx_loader import load_pptx_as_langchain_docs
from .csv_loader import CSVLoader
from .excel_loader import ExcelLoader
from .raw_image_loader import RawImageLoader
from .text_loader import TextLoader

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
        SERVICE_NAME: "document-rag-loaders",
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
        
        print("✅ OpenTelemetry initialized for document loaders")
        
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

# Custom metrics for document loading observability
document_loading_counter = meter.create_counter(
    "documents_loaded_total",
    description="Total number of documents loaded by type",
    unit="1"
)

document_loading_duration = meter.create_histogram(
    "document_loading_duration_seconds",
    description="Time taken to load documents by type",
    unit="s"
)

file_size_histogram = meter.create_histogram(
    "document_file_size_bytes",
    description="Size of loaded document files",
    unit="bytes"
)

loader_selection_counter = meter.create_counter(
    "loader_selection_total",
    description="Total number of loader selections by file extension",
    unit="1"
)

document_content_gauge = meter.create_histogram(
    "document_content_length_chars",
    description="Length of document content in characters",
    unit="chars"
)

# Configure structured logging with OpenTelemetry correlation
import logging

class OtelDocumentLoaderFormatter(logging.Formatter):
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
formatter = OtelDocumentLoaderFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler('document_loaders.log')
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

def get_file_info(path: str) -> dict:
    """Get comprehensive file information for instrumentation"""
    with tracer.start_as_current_span("get_file_info") as span:
        try:
            stat = os.stat(path)
            ext = os.path.splitext(path)[1].lower()
            
            file_info = {
                "size_bytes": stat.st_size,
                "extension": ext,
                "filename": os.path.basename(path),
                "directory": os.path.dirname(path),
                "absolute_path": os.path.abspath(path)
            }
            
            span.set_attribute("file.size_bytes", file_info["size_bytes"])
            span.set_attribute("file.extension", file_info["extension"])
            span.set_attribute("file.name", file_info["filename"])
            span.add_event("File information collected")
            
            return file_info
            
        except Exception as e:
            span.record_exception(e)
            logger.error(f"❌ Error getting file info for {path}: {e}")
            return {"error": str(e)}

def select_loader(extension: str) -> tuple[str, callable]:
    """Select appropriate loader based on file extension with instrumentation"""
    with tracer.start_as_current_span("select_document_loader") as span:
        span.set_attribute("file.extension", extension)
        
        loader_selection_counter.add(1, {"extension": extension, "status": "attempt"})
        
        loader_mapping = {
            ".pdf": ("PDF", load_pdf_as_langchain_docs),
            ".docx": ("DOCX", load_docx_as_langchain_docs),
            ".pptx": ("PPTX", load_pptx_as_langchain_docs),
            ".csv": ("CSV", lambda path: CSVLoader(path).load()),
            ".tsv": ("TSV", lambda path: CSVLoader(path).load()),
            ".xlsx": ("Excel", lambda path: ExcelLoader(path).load()),
            ".xls": ("Excel", lambda path: ExcelLoader(path).load()),
            ".png": ("Image", lambda path: RawImageLoader(path).load()),
            ".jpg": ("Image", lambda path: RawImageLoader(path).load()),
            ".jpeg": ("Image", lambda path: RawImageLoader(path).load()),
            ".gif": ("Image", lambda path: RawImageLoader(path).load()),
            ".tiff": ("Image", lambda path: RawImageLoader(path).load()),
            ".tif": ("Image", lambda path: RawImageLoader(path).load()),
            ".bmp": ("Image", lambda path: RawImageLoader(path).load()),
            ".webp": ("Image", lambda path: RawImageLoader(path).load())
        }
        
        if extension in loader_mapping:
            loader_type, loader_func = loader_mapping[extension]
            span.set_attribute("loader.type", loader_type)
            span.set_attribute("loader.selected", True)
            
            loader_selection_counter.add(1, {"extension": extension, "status": "success", "loader_type": loader_type})
            
            span.add_event("Specific loader selected", {
                "loader_type": loader_type,
                "extension": extension
            })
            
            logger.info(f"✅ Selected {loader_type} loader for extension: {extension}")
            return loader_type, loader_func
        else:
            # Default to generic text loader
            loader_type = "Text"
            loader_func = lambda path: TextLoader(path).load()
            
            span.set_attribute("loader.type", loader_type)
            span.set_attribute("loader.selected", True)
            span.set_attribute("loader.fallback", True)
            
            loader_selection_counter.add(1, {"extension": extension, "status": "fallback", "loader_type": loader_type})
            
            span.add_event("Fallback to text loader", {
                "extension": extension,
                "reason": "no_specific_loader"
            })
            
            logger.info(f"⚠️  Using fallback Text loader for extension: {extension}")
            return loader_type, loader_func

def load_file(path: str) -> List[Document]:
    """Load a single file using the appropriate loader based on extension with comprehensive instrumentation"""
    with tracer.start_as_current_span("load_document_file") as span:
        start_time = time.time()
        
        # Set basic span attributes
        span.set_attribute("file.path", path[:100])  # Truncate for security
        span.set_attribute("operation.type", "document_loading")
        
        # Add baggage for cross-service correlation
        baggage.set_baggage("file.path", os.path.basename(path))
        
        # File existence validation
        with tracer.start_as_current_span("validate_file_existence") as validation_span:
            if not os.path.exists(path):
                validation_span.set_attribute("file.exists", False)
                validation_span.add_event("File not found")
                
                document_loading_counter.add(1, {"status": "error", "error_type": "file_not_found"})
                
                error_msg = f"File not found: {path}"
                logger.error(f"❌ {error_msg}")
                raise FileNotFoundError(path)
            
            validation_span.set_attribute("file.exists", True)
            validation_span.add_event("File existence validated")

        # Get file information
        with tracer.start_as_current_span("collect_file_metadata") as metadata_span:
            file_info = get_file_info(path)
            
            if "error" in file_info:
                metadata_span.set_attribute("metadata.collection", "failed")
                document_loading_counter.add(1, {"status": "error", "error_type": "metadata_error"})
                raise Exception(f"Failed to get file metadata: {file_info['error']}")
            
            # Update span with file metadata
            span.set_attribute("file.size_bytes", file_info["size_bytes"])
            span.set_attribute("file.extension", file_info["extension"])
            span.set_attribute("file.name", file_info["filename"])
            
            metadata_span.set_attribute("metadata.collection", "success")
            metadata_span.add_event("File metadata collected")
            
            # Record file size metric
            file_size_histogram.record(file_info["size_bytes"], {
                "extension": file_info["extension"],
                "loader_type": "unknown"  # Will be updated after loader selection
            })

        # Select appropriate loader
        with tracer.start_as_current_span("select_and_initialize_loader") as loader_span:
            loader_type, loader_func = select_loader(file_info["extension"])
            
            loader_span.set_attribute("loader.type", loader_type)
            loader_span.set_attribute("loader.extension", file_info["extension"])

        # Load document using selected loader
        try:
            with tracer.start_as_current_span(f"load_with_{loader_type.lower()}_loader") as loading_span:
                loading_span.set_attribute("loader.type", loader_type)
                loading_span.set_attribute("file.size_bytes", file_info["size_bytes"])
                loading_span.set_attribute("file.extension", file_info["extension"])
                
                logger.info(f"🔄 Loading {file_info['filename']} using {loader_type} loader...")
                
                # Execute the actual loading
                documents = loader_func(path)
                
                # Validate documents were loaded
                if not documents:
                    loading_span.set_attribute("documents.loaded_count", 0)
                    loading_span.add_event("No documents extracted")
                    
                    document_loading_counter.add(1, {
                        "status": "empty", 
                        "loader_type": loader_type,
                        "extension": file_info["extension"]
                    })
                    
                    logger.warning(f"⚠️  No documents extracted from {file_info['filename']}")
                    return documents
                
                # Process document metadata and metrics
                total_content_length = 0
                
                with tracer.start_as_current_span("process_loaded_documents") as process_span:
                    for i, doc in enumerate(documents):
                        # Add file metadata to each document
                        doc.metadata.update({
                            "source_file": file_info["filename"],
                            "file_path": path,
                            "file_size": file_info["size_bytes"],
                            "file_extension": file_info["extension"],
                            "loader_type": loader_type,
                            "document_index": i,
                            "total_documents": len(documents),
                            "loading_timestamp": time.time()
                        })
                        
                        # Measure content length
                        content_length = len(doc.page_content)
                        total_content_length += content_length
                        
                        # Record individual document content length
                        document_content_gauge.record(content_length, {
                            "loader_type": loader_type,
                            "extension": file_info["extension"],
                            "document_index": str(i)
                        })
                    
                    process_span.set_attribute("documents.processed_count", len(documents))
                    process_span.set_attribute("content.total_length", total_content_length)
                    process_span.add_event("Document metadata enriched")

                # Calculate loading duration and record metrics
                loading_duration = time.time() - start_time
                
                document_loading_duration.record(loading_duration, {
                    "loader_type": loader_type,
                    "extension": file_info["extension"],
                    "status": "success"
                })
                
                document_loading_counter.add(1, {
                    "status": "success",
                    "loader_type": loader_type,
                    "extension": file_info["extension"]
                })
                
                # Update span with success metrics
                loading_span.set_attribute("documents.loaded_count", len(documents))
                loading_span.set_attribute("content.total_length", total_content_length)
                loading_span.set_attribute("loading.duration_seconds", loading_duration)
                loading_span.set_attribute("loading.success", True)
                
                loading_span.add_event("Document loading completed", {
                    "documents_loaded": len(documents),
                    "total_content_length": total_content_length,
                    "loading_duration": loading_duration
                })

        except Exception as e:
            loading_duration = time.time() - start_time
            
            span.record_exception(e)
            span.set_attribute("loading.success", False)
            span.set_attribute("loading.duration_seconds", loading_duration)
            span.set_attribute("error.type", type(e).__name__)
            
            document_loading_counter.add(1, {
                "status": "error",
                "loader_type": loader_type,
                "extension": file_info["extension"],
                "error_type": type(e).__name__
            })
            
            document_loading_duration.record(loading_duration, {
                "loader_type": loader_type,
                "extension": file_info["extension"],
                "status": "error"
            })
            
            logger.error(f"❌ Error loading {file_info['filename']} with {loader_type} loader: {e}")
            raise

        # Final span updates
        span.set_attribute("loading.duration_seconds", loading_duration)
        span.set_attribute("loading.success", True)
        span.set_attribute("documents.final_count", len(documents))
        span.set_attribute("loader.type_used", loader_type)
        
        span.add_event("File loading process completed", {
            "file_name": file_info["filename"],
            "loader_type": loader_type,
            "documents_count": len(documents),
            "total_duration": loading_duration,
            "content_length": total_content_length
        })

        logger.info(f"✅ Successfully loaded {len(documents)} documents from {file_info['filename']} using {loader_type} loader (Duration: {loading_duration:.2f}s)")
        
        return documents

def main():
    """Main function with OpenTelemetry instrumentation for CLI usage"""
    import argparse
    
    with tracer.start_as_current_span("master_loader_main") as span:
        parser = argparse.ArgumentParser(description="Load a document using the appropriate loader")
        parser.add_argument("file", help="Path to the file to load")
        args = parser.parse_args()
        
        span.set_attribute("cli.file_argument", args.file)
        span.set_attribute("execution.mode", "cli")
        
        logger.info(f"🚀 Starting document loading for: {args.file}")
        
        try:
            docs = load_file(args.file)
            
            span.set_attribute("execution.success", True)
            span.set_attribute("documents.loaded", len(docs))
            
            # Display results
            with tracer.start_as_current_span("display_results") as display_span:
                print(f"Loaded {len(docs)} documents")
                
                for i, doc in enumerate(docs[:3]):  # Show first 3 documents
                    print("---")
                    print("Metadata:", doc.metadata)
                    print("Content preview:", doc.page_content[:200])
                
                if len(docs) > 3:
                    print(f"... and {len(docs) - 3} more documents")
                
                display_span.set_attribute("documents.displayed", min(len(docs), 3))
                display_span.add_event("Results displayed to user")
            
            span.add_event("CLI execution completed successfully", {
                "documents_loaded": len(docs),
                "file_processed": args.file
            })
            
            logger.info(f"✅ CLI execution completed: {len(docs)} documents loaded")
            
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("execution.success", False)
            span.set_attribute("error.type", type(e).__name__)
            
            logger.error(f"❌ CLI execution failed: {e}")
            print(f"Error loading file: {e}")
            exit(1)

if __name__ == "__main__":
    main()

