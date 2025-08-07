#!/usr/bin/env python3
"""
PDF Loader with Enhanced Multi-Modal Extraction and OpenTelemetry Instrumentation

Enhanced with comprehensive OpenTelemetry observability following EDOT best practices
for distributed tracing, metrics, and structured logging correlation.

Demonstrates complete PDF processing observability WITHOUT LOGSTASH DEPENDENCY - showing how OpenTelemetry 
can handle multiple log types and correlate them with trace IDs and span IDs for ELK visualization.
"""

import os
import time
import socket
import uuid
import logging
from typing import List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
from loaders.image_extractor import extract_images_from_pdf
from tools.vision_tool import analyze_images_with_vision_model, VISION_PROMPT

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

# OCR imports
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: easyocr not installed. OCR text extraction will be skipped.")

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
        SERVICE_NAME: "document-rag-pdf-loader",
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
        
        print("✅ OpenTelemetry initialized for PDF loader")
        
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

# Custom metrics for PDF processing observability
pdf_processing_counter = meter.create_counter(
    "pdf_documents_processed_total",
    description="Total number of PDF documents processed",
    unit="1"
)

pdf_processing_duration = meter.create_histogram(
    "pdf_processing_duration_seconds",
    description="Time taken to process PDF documents",
    unit="s"
)

pdf_pages_counter = meter.create_counter(
    "pdf_pages_processed_total",
    description="Total number of PDF pages processed",
    unit="1"
)

pdf_images_counter = meter.create_counter(
    "pdf_images_extracted_total",
    description="Total number of images extracted from PDFs",
    unit="1"
)

ocr_operations_counter = meter.create_counter(
    "ocr_operations_total",
    description="Total number of OCR operations performed",
    unit="1"
)

vision_analysis_counter = meter.create_counter(
    "vision_analysis_operations_total",
    description="Total number of vision AI analysis operations",
    unit="1"
)

# Configure structured logging with OpenTelemetry correlation
class OtelPDFLoaderFormatter(logging.Formatter):
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
formatter = OtelPDFLoaderFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler('pdf_loader.log')
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

PDF_IMAGE_DIR = "pdf_images"
os.makedirs(PDF_IMAGE_DIR, exist_ok=True)

def extract_text_with_ocr(image_path: str) -> str:
    """Extract text from a PDF image using OCR with comprehensive instrumentation"""
    with tracer.start_as_current_span("extract_text_with_ocr") as span:
        span.set_attribute("image.path", image_path[:100])  # Truncate for security
        span.set_attribute("ocr.available", OCR_AVAILABLE)
        
        ocr_operations_counter.add(1, {"status": "attempt"})
        
        if not OCR_AVAILABLE:
            span.set_attribute("ocr.skipped", True)
            span.add_event("OCR not available, skipping")
            return ""

        try:
            with tracer.start_as_current_span("easyocr_initialization") as init_span:
                reader = easyocr.Reader(['en'])
                init_span.set_attribute("ocr.language", "en")
                init_span.add_event("EasyOCR reader initialized")

            with tracer.start_as_current_span("easyocr_text_extraction") as extract_span:
                results = reader.readtext(image_path)
                extract_span.set_attribute("ocr.results_count", len(results))

                # Combine all detected text
                ocr_text = []
                high_confidence_count = 0
                
                for (bbox, text, confidence) in results:
                    if confidence > 0.5:  # Only include high-confidence text
                        ocr_text.append(text)
                        high_confidence_count += 1

                final_text = "\n".join(ocr_text)
                
                extract_span.set_attribute("ocr.high_confidence_results", high_confidence_count)
                extract_span.set_attribute("ocr.extracted_text_length", len(final_text))
                extract_span.add_event("OCR text extraction completed", {
                    "total_results": len(results),
                    "high_confidence_results": high_confidence_count,
                    "extracted_text_length": len(final_text)
                })

            ocr_operations_counter.add(1, {"status": "success"})
            
            span.set_attribute("ocr.extraction_successful", True)
            span.set_attribute("ocr.text_length", len(final_text))
            
            logger.info(f"✅ OCR extracted {len(final_text)} characters from {os.path.basename(image_path)}")
            
            return final_text

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("ocr.extraction_successful", False)
            
            ocr_operations_counter.add(1, {"status": "error"})
            
            logger.error(f"❌ OCR extraction failed for {image_path}: {e}")
            return ""

def load_pdf_as_langchain_docs(file_path: str) -> List[Document]:
    """
    Load a PDF file and return a list of LangChain Document objects with comprehensive text extraction and instrumentation.

    Uses multiple extraction methods:
    1. Traditional text extraction via PDFPlumber
    2. OCR on extracted images
    3. Vision AI analysis for context

    Args:
        file_path (str): Path to the PDF file to process.

    Returns:
        List[Document]: List of LangChain Document objects containing all extractable content.
    """
    with tracer.start_as_current_span("load_pdf_as_langchain_docs") as span:
        start_time = time.time()
        
        span.set_attribute("pdf.file_path", file_path[:100])  # Truncate for security
        span.set_attribute("pdf.filename", os.path.basename(file_path))
        
        # Add baggage for cross-service correlation
        baggage.set_baggage("pdf.filename", os.path.basename(file_path))
        
        pdf_processing_counter.add(1, {"status": "attempt"})
        
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                span.set_attribute("pdf.file_exists", False)
                pdf_processing_counter.add(1, {"status": "error", "error_type": "file_not_found"})
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            span.set_attribute("pdf.file_exists", True)
            span.set_attribute("pdf.file_size", os.path.getsize(file_path))
            
            docs = []

            # 1. Traditional text and tables extraction
            with tracer.start_as_current_span("traditional_text_extraction") as text_span:
                text_span.set_attribute("extraction.method", "pdfplumber")
                
                logger.info(f"🔄 Starting PDF text extraction for {os.path.basename(file_path)}")
                
                loader = PDFPlumberLoader(file_path)
                text_docs = loader.load()
                
                text_span.set_attribute("text_docs.count", len(text_docs))
                text_span.add_event("PDFPlumber extraction completed", {
                    "documents_extracted": len(text_docs)
                })
                
                # Count pages from text documents
                page_count = len(text_docs)
                pdf_pages_counter.add(page_count, {"source": "text_extraction"})
                
                logger.info(f"📄 Extracted {len(text_docs)} text documents from PDF")

            # 2. Extract images from PDF
            with tracer.start_as_current_span("pdf_image_extraction") as img_span:
                img_span.set_attribute("extraction.method", "image_extraction")
                img_span.set_attribute("output.directory", PDF_IMAGE_DIR)
                
                logger.info(f"🖼️ Extracting images from PDF...")
                
                images = extract_images_from_pdf(file_path, PDF_IMAGE_DIR)
                
                img_span.set_attribute("images.extracted_count", len(images))
                img_span.add_event("Image extraction completed", {
                    "images_extracted": len(images)
                })
                
                pdf_images_counter.add(len(images), {"source": "pdf_extraction"})
                
                logger.info(f"🖼️ Extracted {len(images)} images from PDF")

            # 3. Enhanced Vision AI analysis with comprehensive prompt
            vision_prompt = """
            Analyze this PDF page/image and extract:
            1. ALL visible text (exact transcription)
            2. Key concepts and information
            3. Context and meaning
            4. Any structured data (lists, tables, diagrams, charts)
            5. Technical diagrams or flowcharts
            6. Any text that might be in images or graphics

            If this appears to be a document page, transcribe all text exactly as it appears.
            Focus on extracting actionable information that would be useful for answering questions.
            Include any text that might be embedded in images, charts, or diagrams.
            """

            image_docs = []
            if images:
                with tracer.start_as_current_span("vision_ai_analysis") as vision_span:
                    vision_span.set_attribute("analysis.method", "vision_ai")
                    vision_span.set_attribute("images.to_analyze", len(images))
                    vision_span.set_attribute("vision.prompt_length", len(vision_prompt))
                    
                    vision_analysis_counter.add(1, {"status": "attempt", "images_count": len(images)})
                    
                    logger.info(f"🤖 Analyzing {len(images)} images with Vision AI...")
                    
                    image_docs = analyze_images_with_vision_model(images, prompt=vision_prompt)
                    
                    vision_span.set_attribute("vision_docs.generated", len(image_docs))
                    vision_span.add_event("Vision AI analysis completed", {
                        "images_analyzed": len(images),
                        "documents_generated": len(image_docs)
                    })
                    
                    vision_analysis_counter.add(1, {"status": "success", "docs_generated": len(image_docs)})
                    
                    logger.info(f"🤖 Vision AI generated {len(image_docs)} document analyses")

            # 4. Add OCR text extraction to image documents
            with tracer.start_as_current_span("ocr_text_enhancement") as ocr_span:
                ocr_enhanced_count = 0
                total_ocr_text_length = 0
                
                for img_doc in image_docs:
                    img_path = img_doc.metadata.get("image_path")
                    if img_path and os.path.exists(img_path):
                        with tracer.start_as_current_span(f"ocr_process_image") as img_ocr_span:
                            img_ocr_span.set_attribute("image.path", os.path.basename(img_path))
                            
                            ocr_text = extract_text_with_ocr(img_path)
                            if ocr_text:
                                # Combine vision AI analysis with OCR text
                                combined_content = f"{img_doc.page_content}\n\n[OCR Extracted Text]\n{ocr_text}"
                                img_doc.page_content = combined_content
                                img_doc.metadata["extraction_methods"] = "vision_ai,ocr"
                                ocr_enhanced_count += 1
                                total_ocr_text_length += len(ocr_text)
                                
                                img_ocr_span.set_attribute("ocr.text_added", True)
                                img_ocr_span.set_attribute("ocr.text_length", len(ocr_text))
                            else:
                                img_doc.metadata["extraction_methods"] = "vision_ai"
                                img_ocr_span.set_attribute("ocr.text_added", False)

                ocr_span.set_attribute("ocr.enhanced_documents", ocr_enhanced_count)
                ocr_span.set_attribute("ocr.total_text_length", total_ocr_text_length)
                ocr_span.add_event("OCR enhancement completed", {
                    "documents_enhanced": ocr_enhanced_count,
                    "total_ocr_text_length": total_ocr_text_length
                })

            # 5. Enhance text documents with comprehensive metadata
            with tracer.start_as_current_span("metadata_enhancement") as meta_span:
                for doc in text_docs:
                    doc.metadata["file_type"] = "pdf"
                    doc.metadata["extraction_methods"] = "pdfplumber,vision_ai,ocr"
                    doc.metadata["content_type"] = "text"
                    doc.metadata["loader_version"] = "1.0.0"
                    doc.metadata["processing_timestamp"] = time.time()

                meta_span.set_attribute("text_docs.enhanced", len(text_docs))
                meta_span.add_event("Text document metadata enhanced")

            # 6. Link images to text chunks by page number
            with tracer.start_as_current_span("associate_images_to_pages") as assoc_span:
                page_to_images = {}
                for img_doc in image_docs:
                    page = img_doc.metadata.get("page", img_doc.metadata.get("page_number"))
                    if page is not None:
                        page_to_images.setdefault(page, []).append(img_doc.metadata.get("image_path"))

                associated_pages = 0
                for doc in text_docs:
                    page = doc.metadata.get("page", doc.metadata.get("page_number"))
                    if page is not None and page in page_to_images:
                        doc.metadata["related_images"] = page_to_images[page]
                        associated_pages += 1

                assoc_span.set_attribute("pages.with_images", len(page_to_images))
                assoc_span.set_attribute("docs.with_associated_images", associated_pages)
                assoc_span.add_event("Image-to-page association completed")

            # Calculate processing metrics
            processing_duration = time.time() - start_time
            total_docs = len(text_docs) + len(image_docs)
            
            pdf_processing_duration.record(processing_duration, {
                "pdf_size": "large" if os.path.getsize(file_path) > 10_000_000 else "small",
                "status": "success"
            })
            
            pdf_processing_counter.add(1, {"status": "success"})

            # Update span with final metrics
            span.set_attribute("pdf.processing_duration_seconds", processing_duration)
            span.set_attribute("pdf.total_documents_generated", total_docs)
            span.set_attribute("pdf.text_documents", len(text_docs))
            span.set_attribute("pdf.image_documents", len(image_docs))
            span.set_attribute("pdf.images_extracted", len(images))
            span.set_attribute("pdf.processing_successful", True)
            
            span.add_event("PDF processing completed successfully", {
                "total_documents": total_docs,
                "text_documents": len(text_docs),
                "image_documents": len(image_docs),
                "images_extracted": len(images),
                "processing_duration": processing_duration
            })

            logger.info(f"✅ PDF processing completed: {total_docs} total documents ({len(text_docs)} text, {len(image_docs)} image) in {processing_duration:.2f}s")

            # 7. Return all documents
            return text_docs + image_docs

        except Exception as e:
            processing_duration = time.time() - start_time
            
            span.record_exception(e)
            span.set_attribute("pdf.processing_successful", False)
            span.set_attribute("pdf.processing_duration_seconds", processing_duration)
            span.set_attribute("error.type", type(e).__name__)
            
            pdf_processing_counter.add(1, {"status": "error", "error_type": type(e).__name__})
            
            pdf_processing_duration.record(processing_duration, {
                "pdf_size": "unknown",
                "status": "error"
            })
            
            logger.error(f"❌ PDF processing failed for {file_path}: {e}")
            raise

def test_pdf_loader():
    """Test the PDF loader with OpenTelemetry instrumentation"""
    with tracer.start_as_current_span("test_pdf_loader") as span:
        sample_path = "sample.pdf"  # Place a sample PDF in the project root for testing
        
        span.set_attribute("test.sample_path", sample_path)
        
        logger.info(f"🧪 Testing comprehensive PDF loader with multi-modal extraction: {sample_path}")
        
        if not os.path.isfile(sample_path):
            span.set_attribute("test.sample_file_exists", False)
            span.add_event("Test skipped - sample file not found")
            logger.warning("[Test Skipped] sample.pdf not found.")
            return
        
        span.set_attribute("test.sample_file_exists", True)
        
        try:
            docs = load_pdf_as_langchain_docs(sample_path)
            
            span.set_attribute("test.documents_loaded", len(docs))
            span.add_event("Test completed successfully", {
                "documents_loaded": len(docs)
            })
            
            logger.info(f"📊 Test Results: Loaded {len(docs)} Document objects.")
            
            # Display sample results
            for i, doc in enumerate(docs[:3]):
                logger.info("---")
                logger.info(f"Document {i+1} Metadata: {doc.metadata}")
                logger.info(f"Content (first 300 chars): {doc.page_content[:300]}")
            
            if len(docs) > 3:
                logger.info(f"... and {len(docs) - 3} more documents")
                
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("test.successful", False)
            logger.error(f"❌ Test failed: {e}")
            raise

if __name__ == "__main__":
    with tracer.start_as_current_span("pdf_loader_main") as span:
        span.set_attribute("execution.mode", "standalone")
        logger.info("🚀 Running PDF Loader in standalone mode")
        test_pdf_loader()

