#!/usr/bin/env python3
"""
Universal Document Chunker with OpenTelemetry Instrumentation

Enhanced with comprehensive OpenTelemetry observability following EDOT best practices
for distributed tracing, metrics, and structured logging correlation.

Demonstrates complete text splitting observability WITHOUT LOGSTASH DEPENDENCY - showing how OpenTelemetry 
can handle multiple log types and correlate them with trace IDs and span IDs for ELK visualization.
"""

import sys
import os
import time
import socket
import uuid
import logging
from typing import List, Dict, Any
from collections import defaultdict
from dotenv import load_dotenv

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

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders.google_drive_loader import GoogleDriveMasterLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        SERVICE_NAME: "document-rag-text-splitter",
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
        
        print("✅ OpenTelemetry initialized for text splitter")
        
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

# Custom metrics for text splitting observability
document_splitting_counter = meter.create_counter(
    "documents_split_total",
    description="Total number of documents processed for splitting",
    unit="1"
)

chunk_creation_counter = meter.create_counter(
    "chunks_created_total",
    description="Total number of text chunks created",
    unit="1"
)

splitting_duration = meter.create_histogram(
    "splitting_duration_seconds",
    description="Time taken to split documents",
    unit="s"
)

chunk_size_histogram = meter.create_histogram(
    "chunk_size_characters",
    description="Distribution of chunk sizes in characters",
    unit="chars"
)

document_type_counter = meter.create_counter(
    "documents_by_type_total",
    description="Total number of documents processed by type",
    unit="1"
)

# Configure structured logging with OpenTelemetry correlation
class OtelTextSplitterFormatter(logging.Formatter):
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
formatter = OtelTextSplitterFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler('text_splitter.log')
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

def split_documents(documents, chunk_size=1000, chunk_overlap=120):
    """
    Universal chunker with hierarchical and semantic splitting with comprehensive OpenTelemetry instrumentation:
    - PPTX: group all content from a slide (title + bullets) into one chunk, split further if >chunk_size
    - DOCX/PDF: group by section/heading, split further if >chunk_size
    - Excel: group by row, include column headers, split further if needed
    - TXT: uses RecursiveCharacterTextSplitter
    - Images: returns as-is (with metadata)
    - Enriches metadata for all chunks
    - Filters out empty/boilerplate chunks
    """
    with tracer.start_as_current_span("split_documents") as span:
        start_time = time.time()
        
        # Set span attributes
        span.set_attribute("splitting.chunk_size", chunk_size)
        span.set_attribute("splitting.chunk_overlap", chunk_overlap)
        span.set_attribute("documents.input_count", len(documents) if documents else 0)
        
        # Add baggage for cross-service correlation
        baggage.set_baggage("splitting.chunk_size", str(chunk_size))
        baggage.set_baggage("splitting.chunk_overlap", str(chunk_overlap))
        
        document_splitting_counter.add(1, {"status": "attempt"})
        
        try:
            # Validate input parameters
            with tracer.start_as_current_span("validate_parameters") as validate_span:
                if not isinstance(chunk_size, int) or chunk_size <= 0:
                    chunk_size = 1000
                    validate_span.add_event("Chunk size corrected to default", {"default_size": 1000})
                    
                if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
                    chunk_overlap = 120
                    validate_span.add_event("Chunk overlap corrected to default", {"default_overlap": 120})

                # Ensure overlap is not greater than chunk_size
                if chunk_overlap >= chunk_size:
                    chunk_overlap = chunk_size // 4  # Use 25% overlap as fallback
                    validate_span.add_event("Chunk overlap reduced to prevent overflow", {"adjusted_overlap": chunk_overlap})

                validate_span.set_attribute("validated.chunk_size", chunk_size)
                validate_span.set_attribute("validated.chunk_overlap", chunk_overlap)

            # Validate documents input
            if not documents:
                span.set_attribute("documents.empty_input", True)
                span.add_event("No documents to process")
                return []

            # Ensure documents is a list
            if not isinstance(documents, list):
                documents = [documents]
                span.add_event("Single document converted to list")

            chunks = []
            pptx_slides = defaultdict(list)
            docx_sections = defaultdict(list)
            pdf_sections = defaultdict(list)
            excel_sheets = defaultdict(list)
            other_docs = []

            # Document categorization
            with tracer.start_as_current_span("categorize_documents") as categorize_span:
                document_type_counts = defaultdict(int)
                valid_doc_count = 0
                
                for doc in documents:
                    # Skip invalid documents
                    if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
                        continue
                        
                    valid_doc_count += 1
                    file_type = doc.metadata.get("file_type")
                    document_type_counts[file_type] += 1
                    
                    if file_type == "pptx" and doc.metadata.get("content_type") == "slide_content":
                        slide_number = doc.metadata.get("slide_number")
                        pptx_slides[slide_number].append(doc)
                    elif file_type == "docx" and doc.metadata.get("content_type") == "section":
                        section = doc.metadata.get("section_title", "Unknown Section")
                        docx_sections[section].append(doc)
                    elif file_type == "pdf" and doc.metadata.get("content_type") == "section":
                        section = doc.metadata.get("section_title", "Unknown Section")
                        pdf_sections[section].append(doc)
                    elif file_type == "excel" and doc.metadata.get("content_type") == "row":
                        sheet = doc.metadata.get("sheet_name", "Sheet1")
                        excel_sheets[sheet].append(doc)
                    else:
                        other_docs.append(doc)

                categorize_span.set_attribute("documents.valid_count", valid_doc_count)
                categorize_span.set_attribute("categories.pptx_slides", len(pptx_slides))
                categorize_span.set_attribute("categories.docx_sections", len(docx_sections))
                categorize_span.set_attribute("categories.pdf_sections", len(pdf_sections))
                categorize_span.set_attribute("categories.excel_sheets", len(excel_sheets))
                categorize_span.set_attribute("categories.other_docs", len(other_docs))

                # Record document type metrics
                for doc_type, count in document_type_counts.items():
                    document_type_counter.add(count, {"document_type": doc_type or "unknown"})

                categorize_span.add_event("Document categorization completed", {
                    "valid_documents": valid_doc_count,
                    "pptx_slides": len(pptx_slides),
                    "docx_sections": len(docx_sections),
                    "pdf_sections": len(pdf_sections),
                    "excel_sheets": len(excel_sheets),
                    "other_docs": len(other_docs)
                })

            # PPTX: Chunk by slide, split further if needed
            if pptx_slides:
                with tracer.start_as_current_span("process_pptx_documents") as pptx_span:
                    pptx_chunks_created = 0
                    
                    for slide_number, slide_docs in pptx_slides.items():
                        with tracer.start_as_current_span(f"process_pptx_slide_{slide_number}") as slide_span:
                            slide_span.set_attribute("slide.number", slide_number)
                            slide_span.set_attribute("slide.documents_count", len(slide_docs))
                            
                            # Since each slide now has one comprehensive document, just use it directly
                            for slide_doc in slide_docs:
                                slide_text = slide_doc.page_content.strip()
                                if not slide_text or len(slide_text.strip()) < 10:
                                    continue
                                    
                                meta = dict(slide_doc.metadata)
                                meta["document_type"] = meta.get("file_type", "pptx")
                                meta["file_name"] = meta.get("drive_file_name", "unknown")
                                meta["slide_number"] = slide_number
                                meta["heading"] = meta.get("slide_title", "Unknown")
                                meta["content_type"] = "slide_content"
                                
                                # Split if too large
                                try:
                                    if len(slide_text) > chunk_size:
                                        with tracer.start_as_current_span("split_large_slide") as split_span:
                                            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                                            text_chunks = splitter.split_text(slide_text)
                                            
                                            split_span.set_attribute("chunks.created", len(text_chunks))
                                            split_span.add_event("Large slide split into chunks")
                                            
                                            for chunk in text_chunks:
                                                if chunk and chunk.strip():
                                                    chunks.append(Document(page_content=chunk, metadata=meta))
                                                    pptx_chunks_created += 1
                                                    chunk_size_histogram.record(len(chunk), {"document_type": "pptx"})
                                    else:
                                        chunks.append(Document(page_content=slide_text, metadata=meta))
                                        pptx_chunks_created += 1
                                        chunk_size_histogram.record(len(slide_text), {"document_type": "pptx"})
                                        
                                except Exception as e:
                                    slide_span.record_exception(e)
                                    # If splitting fails, add the original document
                                    chunks.append(Document(page_content=slide_text, metadata=meta))
                                    pptx_chunks_created += 1

                    pptx_span.set_attribute("pptx.chunks_created", pptx_chunks_created)
                    pptx_span.add_event("PPTX processing completed", {"chunks_created": pptx_chunks_created})
                    logger.info(f"✅ Processed {len(pptx_slides)} PPTX slides, created {pptx_chunks_created} chunks")

            # DOCX: Chunk by section, split further if needed
            if docx_sections:
                with tracer.start_as_current_span("process_docx_documents") as docx_span:
                    docx_chunks_created = 0
                    
                    for section, docs_in_section in docx_sections.items():
                        with tracer.start_as_current_span(f"process_docx_section") as section_span:
                            section_span.set_attribute("section.name", section[:100])  # Truncate for security
                            section_span.set_attribute("section.documents_count", len(docs_in_section))
                            
                            section_text = "\n".join(d.page_content.strip() for d in docs_in_section if d.page_content and d.page_content.strip())
                            if not section_text or len(section_text.strip()) < 10:
                                section_span.add_event("Section skipped - insufficient content")
                                continue
                                
                            meta = dict(docs_in_section[0].metadata)
                            meta["document_type"] = meta.get("file_type", "docx")
                            meta["file_name"] = meta.get("file_name", "unknown")
                            meta["section_title"] = section
                            meta["heading"] = meta.get("section_title", "Unknown")
                            meta["content_type"] = "section"
                            
                            if len(section_text) > chunk_size:
                                with tracer.start_as_current_span("split_large_docx_section") as split_span:
                                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                                    text_chunks = splitter.split_text(section_text)
                                    
                                    split_span.set_attribute("chunks.created", len(text_chunks))
                                    
                                    for chunk in text_chunks:
                                        chunks.append(Document(page_content=chunk, metadata=meta))
                                        docx_chunks_created += 1
                                        chunk_size_histogram.record(len(chunk), {"document_type": "docx"})
                            else:
                                chunks.append(Document(page_content=section_text, metadata=meta))
                                docx_chunks_created += 1
                                chunk_size_histogram.record(len(section_text), {"document_type": "docx"})

                    docx_span.set_attribute("docx.chunks_created", docx_chunks_created)
                    docx_span.add_event("DOCX processing completed", {"chunks_created": docx_chunks_created})
                    logger.info(f"✅ Processed {len(docx_sections)} DOCX sections, created {docx_chunks_created} chunks")

            # PDF: Chunk by section, split further if needed
            if pdf_sections:
                with tracer.start_as_current_span("process_pdf_documents") as pdf_span:
                    pdf_chunks_created = 0
                    
                    for section, docs_in_section in pdf_sections.items():
                        with tracer.start_as_current_span(f"process_pdf_section") as section_span:
                            section_span.set_attribute("section.name", section[:100])  # Truncate for security
                            section_span.set_attribute("section.documents_count", len(docs_in_section))
                            
                            section_text = "\n".join(d.page_content.strip() for d in docs_in_section if d.page_content and d.page_content.strip())
                            if not section_text or len(section_text.strip()) < 10:
                                section_span.add_event("Section skipped - insufficient content")
                                continue
                                
                            meta = dict(docs_in_section[0].metadata)
                            meta["document_type"] = meta.get("file_type", "pdf")
                            meta["file_name"] = meta.get("file_name", "unknown")
                            meta["section_title"] = section
                            meta["heading"] = meta.get("section_title", "Unknown")
                            meta["content_type"] = "section"
                            
                            if len(section_text) > chunk_size:
                                with tracer.start_as_current_span("split_large_pdf_section") as split_span:
                                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                                    text_chunks = splitter.split_text(section_text)
                                    
                                    split_span.set_attribute("chunks.created", len(text_chunks))
                                    
                                    for chunk in text_chunks:
                                        chunks.append(Document(page_content=chunk, metadata=meta))
                                        pdf_chunks_created += 1
                                        chunk_size_histogram.record(len(chunk), {"document_type": "pdf"})
                            else:
                                chunks.append(Document(page_content=section_text, metadata=meta))
                                pdf_chunks_created += 1
                                chunk_size_histogram.record(len(section_text), {"document_type": "pdf"})

                    pdf_span.set_attribute("pdf.chunks_created", pdf_chunks_created)
                    pdf_span.add_event("PDF processing completed", {"chunks_created": pdf_chunks_created})
                    logger.info(f"✅ Processed {len(pdf_sections)} PDF sections, created {pdf_chunks_created} chunks")

            # Excel: Chunk by row, include column headers
            if excel_sheets:
                with tracer.start_as_current_span("process_excel_documents") as excel_span:
                    excel_chunks_created = 0
                    
                    for sheet, rows in excel_sheets.items():
                        with tracer.start_as_current_span(f"process_excel_sheet") as sheet_span:
                            sheet_span.set_attribute("sheet.name", sheet)
                            sheet_span.set_attribute("sheet.rows_count", len(rows))
                            
                            for row_doc in rows:
                                if row_doc.page_content and len(row_doc.page_content.strip()) >= 10:
                                    meta = dict(row_doc.metadata)
                                    meta["document_type"] = meta.get("file_type", "excel")
                                    meta["file_name"] = meta.get("file_name", "unknown")
                                    meta["sheet_name"] = sheet
                                    meta["heading"] = meta.get("sheet_name", "Unknown")
                                    meta["content_type"] = "row"
                                    
                                    # Optionally, add column headers if available
                                    if "column_headers" in row_doc.metadata:
                                        meta["column_headers"] = row_doc.metadata["column_headers"]
                                    
                                    # Split if row is very long
                                    if len(row_doc.page_content) > chunk_size:
                                        with tracer.start_as_current_span("split_large_excel_row") as split_span:
                                            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                                            text_chunks = splitter.split_text(row_doc.page_content)
                                            
                                            for chunk in text_chunks:
                                                chunks.append(Document(page_content=chunk, metadata=meta))
                                                excel_chunks_created += 1
                                                chunk_size_histogram.record(len(chunk), {"document_type": "excel"})
                                    else:
                                        chunks.append(Document(page_content=row_doc.page_content, metadata=meta))
                                        excel_chunks_created += 1
                                        chunk_size_histogram.record(len(row_doc.page_content), {"document_type": "excel"})

                    excel_span.set_attribute("excel.chunks_created", excel_chunks_created)
                    excel_span.add_event("Excel processing completed", {"chunks_created": excel_chunks_created})
                    logger.info(f"✅ Processed {len(excel_sheets)} Excel sheets, created {excel_chunks_created} chunks")

            # Other docs (TXT, images, etc.)
            if other_docs:
                with tracer.start_as_current_span("process_other_documents") as other_span:
                    other_chunks_created = 0
                    
                    for doc in other_docs:
                        if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
                            if not doc.page_content or len(doc.page_content.strip()) < 10:
                                continue
                                
                            meta = dict(doc.metadata)
                            meta["document_type"] = meta.get("file_type", "unknown")
                            meta["file_name"] = meta.get("file_name", "unknown")
                            meta["heading"] = meta.get("heading", "Unknown")
                            meta["content_type"] = meta.get("content_type", "unknown")
                            
                            # For images, just attach metadata
                            if doc.metadata.get("file_type") == "image":
                                chunks.append(Document(page_content=doc.page_content, metadata=meta))
                                other_chunks_created += 1
                                chunk_size_histogram.record(len(doc.page_content), {"document_type": "image"})
                            else:
                                with tracer.start_as_current_span("split_text_document") as text_span:
                                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                                    text_chunks = splitter.split_text(doc.page_content)
                                    
                                    text_span.set_attribute("chunks.created", len(text_chunks))
                                    
                                    for chunk_text in text_chunks:
                                        if chunk_text and len(chunk_text.strip()) >= 10:
                                            chunk_meta = dict(meta)
                                            chunk_meta["document_type"] = chunk_meta.get("file_type", meta["document_type"])
                                            chunk_meta["file_name"] = chunk_meta.get("file_name", meta["file_name"])
                                            chunk_meta["heading"] = chunk_meta.get("heading", meta["heading"])
                                            chunk_meta["content_type"] = chunk_meta.get("content_type", meta["content_type"])
                                            chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))
                                            other_chunks_created += 1
                                            chunk_size_histogram.record(len(chunk_text), {"document_type": meta["document_type"]})
                        else:
                            chunks.append(doc)
                            other_chunks_created += 1

                    other_span.set_attribute("other.chunks_created", other_chunks_created)
                    other_span.add_event("Other documents processing completed", {"chunks_created": other_chunks_created})
                    logger.info(f"✅ Processed {len(other_docs)} other documents, created {other_chunks_created} chunks")

            # Calculate final metrics
            processing_duration = time.time() - start_time
            total_chunks = len(chunks)
            
            splitting_duration.record(processing_duration, {
                "chunk_size": str(chunk_size),
                "documents_count": str(len(documents))
            })
            
            chunk_creation_counter.add(total_chunks, {"status": "success"})
            document_splitting_counter.add(1, {"status": "success"})

            # Update span with final results
            span.set_attribute("splitting.processing_duration_seconds", processing_duration)
            span.set_attribute("splitting.total_chunks_created", total_chunks)
            span.set_attribute("splitting.success", True)
            
            span.add_event("Document splitting completed successfully", {
                "input_documents": len(documents),
                "output_chunks": total_chunks,
                "processing_duration": processing_duration,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            })

            logger.info(f"✅ Document splitting completed: {len(documents)} documents → {total_chunks} chunks in {processing_duration:.2f}s")
            
            return chunks

        except Exception as e:
            processing_duration = time.time() - start_time
            
            span.record_exception(e)
            span.set_attribute("splitting.success", False)
            span.set_attribute("splitting.processing_duration_seconds", processing_duration)
            span.set_attribute("error.type", type(e).__name__)
            
            document_splitting_counter.add(1, {"status": "error", "error_type": type(e).__name__})
            
            logger.error(f"❌ Document splitting failed: {e}")
            raise

# --- CONFIGURATION FROM ENVIRONMENT ---
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
VECTOR_SIZE = 3072  # For text-embedding-3-large

# Default chunking constants for export
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

def main():
    """Main function to process documents from Google Drive with OpenTelemetry instrumentation"""
    with tracer.start_as_current_span("text_splitter_main") as span:
        span.set_attribute("execution.mode", "main")
        
        # Validate required environment variables
        if not FOLDER_ID:
            span.set_attribute("error.type", "missing_folder_id")
            raise ValueError("GOOGLE_DRIVE_FOLDER_ID environment variable is required")

        span.set_attribute("google_drive.folder_id", FOLDER_ID)
        span.set_attribute("qdrant.host", QDRANT_HOST)
        span.set_attribute("qdrant.collection", COLLECTION_NAME)
        
        logger.info("🚀 Starting Google Drive document processing with text splitting")

        # --- 1. Load all documents from Google Drive ---
        with tracer.start_as_current_span("load_google_drive_documents") as load_span:
            loader = GoogleDriveMasterLoader(
                folder_id=FOLDER_ID,
                credentials_path=CREDENTIALS_PATH,
                token_path=TOKEN_PATH,
                split=False  # We'll chunk manually
            )
            docs = loader.load()
            
            load_span.set_attribute("documents.loaded", len(docs))
            load_span.add_event("Documents loaded from Google Drive")
            
            logger.info(f"📄 Loaded {len(docs)} raw documents from Google Drive.")

        # --- 2. Chunk all documents using universal chunker ---
        chunks = split_documents(docs, chunk_size=1024, chunk_overlap=100)
        logger.info(f"✂️ Created {len(chunks)} chunks from all Google Drive docs.")

        # --- 3. Connect to Qdrant and (re)create collection ---
        with tracer.start_as_current_span("setup_qdrant_collection") as qdrant_span:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
            
            if client.collection_exists(COLLECTION_NAME):
                logger.info("🗑️ Deleting existing collection...")
                client.delete_collection(collection_name=COLLECTION_NAME)
                
            logger.info("📦 Creating new collection...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"}
            )
            
            qdrant_span.set_attribute("collection.name", COLLECTION_NAME)
            qdrant_span.set_attribute("collection.vector_size", VECTOR_SIZE)
            qdrant_span.add_event("Qdrant collection setup completed")

        # --- 4. Embed and upload chunks to Qdrant ---
        with tracer.start_as_current_span("upload_chunks_to_qdrant") as upload_span:
            embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embedding_function
            )
            
            logger.info("⬆️ Uploading chunks to Qdrant...")
            BATCH_SIZE = 50
            
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i+BATCH_SIZE]
                
                with tracer.start_as_current_span(f"upload_batch_{i//BATCH_SIZE + 1}") as batch_span:
                    batch_span.set_attribute("batch.size", len(batch))
                    batch_span.set_attribute("batch.start_index", i)
                    
                    vectorstore.add_documents(batch)
                    batch_span.add_event("Batch uploaded successfully")
                    
            upload_span.set_attribute("chunks.uploaded", len(chunks))
            upload_span.set_attribute("batches.total", (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE)
            upload_span.add_event("All chunks uploaded to Qdrant")

        logger.info("✅ Done! All chunks embedded and uploaded to Qdrant.")

        # --- (Optional) Print a few chunks for verification ---
        with tracer.start_as_current_span("display_sample_chunks") as display_span:
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"--- Chunk {i} ---")
                print("Metadata:", chunk.metadata)
                print("Content:", chunk.page_content[:200], "...")
                
            display_span.set_attribute("chunks.displayed", min(len(chunks), 3))
            display_span.add_event("Sample chunks displayed")

        span.add_event("Main processing completed successfully", {
            "documents_loaded": len(docs),
            "chunks_created": len(chunks),
            "collection_name": COLLECTION_NAME
        })

if __name__ == "__main__":
    with tracer.start_as_current_span("text_splitter_standalone") as span:
        span.set_attribute("execution.context", "standalone")
        logger.info("🎯 Running Text Splitter in standalone mode")
        main()

