#!/usr/bin/env python3
"""
Startup script for the Document RAG FastAPI server with OpenTelemetry Instrumentation

Enhanced with comprehensive OpenTelemetry observability following EDOT best practices
for distributed tracing, metrics, and structured logging correlation.
"""

import os
import sys
import time
import socket
import uuid
import logging
import uvicorn
from dotenv import load_dotenv

# OpenTelemetry imports - Official EDOT libraries
from opentelemetry import trace, metrics, baggage
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
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
        SERVICE_NAME: "document-rag-api-ui",
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
        
        print("✅ OpenTelemetry initialized for API server")
        
    except Exception as e:
        print(f"⚠️  OpenTelemetry provider setup error: {e}")
    
    # Set up B3 propagator for distributed tracing
    set_global_textmap(B3MultiFormat())
    
    # Manual instrumentation only if not using auto-instrumentation
    if not os.getenv("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"):
        try:
            LoggingInstrumentor().instrument(set_logging_format=True)
            RequestsInstrumentor().instrument()
        except Exception as e:
            print(f"⚠️  Instrumentation warning: {e}")
    
    return trace.get_tracer(__name__), metrics.get_meter(__name__)

# Initialize telemetry
tracer, meter = init_telemetry()

# Custom metrics for API server observability
server_startup_counter = meter.create_counter(
    "server_startup_total",
    description="Total number of server startup attempts",
    unit="1"
)

server_health_gauge = meter.create_up_down_counter(
    "server_health_status",
    description="API server health status (1=healthy, 0=unhealthy)",
    unit="1"
)

environment_validation_counter = meter.create_counter(
    "environment_validation_total",
    description="Total environment validation attempts",
    unit="1"
)

server_uptime_gauge = meter.create_up_down_counter(
    "server_uptime_seconds",
    description="Server uptime in seconds",
    unit="s"
)

# Configure structured logging with OpenTelemetry correlation
class OtelServerFormatter(logging.Formatter):
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
formatter = OtelServerFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler('api_server.log')
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

def validate_environment() -> bool:
    """Validate environment variables with comprehensive tracing"""
    with tracer.start_as_current_span("validate_environment") as span:
        span.set_attribute("validation.type", "environment_variables")
        
        environment_validation_counter.add(1, {"status": "attempt"})
        
        # Check required environment variables
        required_vars = ["OPENAI_API_KEY"]
        optional_vars = ["COHERE_API_KEY", "QDRANT_HOST", "QDRANT_PORT"]
        
        span.set_attribute("required_vars.count", len(required_vars))
        span.set_attribute("optional_vars.count", len(optional_vars))

        missing_vars = []
        present_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
            else:
                present_vars.append(var)

        span.set_attribute("missing_vars.count", len(missing_vars))
        span.set_attribute("present_vars.count", len(present_vars))

        if missing_vars:
            span.set_attribute("validation.result", "failed")
            span.add_event("Missing required environment variables", {
                "missing_vars": missing_vars
            })
            
            environment_validation_counter.add(1, {"status": "failed"})
            
            logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            logger.info("Please set them in your .env file or environment")
            return False

        # Set default values for optional variables and track them
        defaults_set = []
        
        with tracer.start_as_current_span("set_default_values") as defaults_span:
            default_qdrant_host = os.environ.setdefault("QDRANT_HOST", "localhost")
            default_qdrant_port = os.environ.setdefault("QDRANT_PORT", "6333")
            
            if default_qdrant_host == "localhost":
                defaults_set.append("QDRANT_HOST")
            if default_qdrant_port == "6333":
                defaults_set.append("QDRANT_PORT")
            
            defaults_span.set_attribute("defaults_set.count", len(defaults_set))
            defaults_span.set_attribute("qdrant.host", default_qdrant_host)
            defaults_span.set_attribute("qdrant.port", default_qdrant_port)
            
            if defaults_set:
                defaults_span.add_event("Default values set", {
                    "variables": defaults_set
                })

        # Check optional variables presence
        optional_present = []
        optional_missing = []
        
        for var in optional_vars:
            if os.getenv(var):
                optional_present.append(var)
            else:
                optional_missing.append(var)

        span.set_attribute("optional_present.count", len(optional_present))
        span.set_attribute("optional_missing.count", len(optional_missing))
        
        if optional_missing:
            span.add_event("Optional variables not set", {
                "missing_optional": optional_missing
            })
            logger.info(f"ℹ️  Optional variables not set: {', '.join(optional_missing)}")

        span.set_attribute("validation.result", "passed")
        span.add_event("Environment validation completed", {
            "required_vars_present": present_vars,
            "optional_vars_present": optional_present,
            "defaults_set": defaults_set
        })
        
        environment_validation_counter.add(1, {"status": "success"})
        
        logger.info("✅ Environment validation successful")
        return True

def display_server_info():
    """Display server information with OpenTelemetry correlation"""
    with tracer.start_as_current_span("display_server_info") as span:
        span.set_attribute("server.host", "0.0.0.0")
        span.set_attribute("server.port", 8000)
        span.set_attribute("qdrant.host", os.getenv('QDRANT_HOST'))
        span.set_attribute("qdrant.port", os.getenv('QDRANT_PORT'))
        
        logger.info("🚀 Starting Document RAG FastAPI server...")
        logger.info(f"📍 Server will be available at: http://localhost:8000")
        logger.info(f"📊 API documentation at: http://localhost:8000/docs")
        logger.info(f"🔍 Qdrant connection: {os.getenv('QDRANT_HOST')}:{os.getenv('QDRANT_PORT')}")
        logger.info(f"🔍 OpenTelemetry Configuration:")
        logger.info(f"   📡 OTLP Endpoint:     {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4317')}")
        logger.info(f"   🏷️  Service Name:      document-rag-api-ui")
        
        # Get current trace information for logging
        current_span = trace.get_current_span()
        if current_span.get_span_context().is_valid:
            trace_id = format(current_span.get_span_context().trace_id, '032x')
            logger.info(f"   📊 Trace ID:          {trace_id}")
        
        span.add_event("Server information displayed", {
            "api_url": "http://localhost:8000",
            "docs_url": "http://localhost:8000/docs",
            "qdrant_connection": f"{os.getenv('QDRANT_HOST')}:{os.getenv('QDRANT_PORT')}"
        })

def start_uvicorn_server():
    """Start the Uvicorn server with comprehensive instrumentation"""
    with tracer.start_as_current_span("start_uvicorn_server") as span:
        span.set_attribute("server.type", "uvicorn")
        span.set_attribute("server.host", "0.0.0.0")
        span.set_attribute("server.port", 8000)
        span.set_attribute("server.reload", False)
        span.set_attribute("server.log_level", "info")
        span.set_attribute("app.module", "ui.api_integrated:app")
        
        server_startup_counter.add(1, {"server_type": "uvicorn", "status": "attempt"})
        
        # Set server health to healthy before starting
        server_health_gauge.add(1, {"server": "api_ui"})
        
        start_time = time.time()
        
        try:
            span.add_event("Starting Uvicorn server")
            logger.info("🚀 Launching Uvicorn server with FastAPI application...")
            
            # Pre-instrument the FastAPI app if possible
            try:
                # Import and instrument the FastAPI app before starting
                from ui.api_integrated import app
                FastAPIInstrumentor.instrument_app(app)
                span.add_event("FastAPI app pre-instrumented")
                logger.info("✅ FastAPI application pre-instrumented with OpenTelemetry")
            except Exception as e:
                span.add_event("FastAPI pre-instrumentation skipped", {"reason": str(e)})
                logger.info("ℹ️  FastAPI pre-instrumentation skipped, will use auto-instrumentation")
            
            uvicorn.run(
                "ui.api_integrated:app",
                host="0.0.0.0",
                port=8000,
                reload=False,  # Set to True for development
                log_level="info"
            )
            
            # This line should not be reached in normal operation
            server_uptime = time.time() - start_time
            span.set_attribute("server.uptime_seconds", server_uptime)
            
            server_startup_counter.add(1, {"server_type": "uvicorn", "status": "stopped"})
            span.add_event("Server stopped normally")
            
            logger.info("🛑 Server stopped normally")
            
        except KeyboardInterrupt:
            server_uptime = time.time() - start_time
            span.set_attribute("server.uptime_seconds", server_uptime)
            span.set_attribute("shutdown.reason", "keyboard_interrupt")
            
            server_startup_counter.add(1, {"server_type": "uvicorn", "status": "keyboard_interrupt"})
            server_health_gauge.add(-1, {"server": "api_ui"})
            
            span.add_event("Server stopped by user", {"uptime_seconds": server_uptime})
            logger.info("\n🛑 Server stopped by user")
            
        except Exception as e:
            server_uptime = time.time() - start_time
            span.record_exception(e)
            span.set_attribute("server.uptime_seconds", server_uptime)
            span.set_attribute("shutdown.reason", "error")
            
            server_startup_counter.add(1, {"server_type": "uvicorn", "status": "error"})
            server_health_gauge.add(-1, {"server": "api_ui"})
            
            span.add_event("Server error", {
                "error_message": str(e),
                "uptime_seconds": server_uptime
            })
            
            logger.error(f"❌ Error starting server: {e}")
            
            # Add detailed error debugging
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            sys.exit(1)

@tracer.start_as_current_span("main_server_startup")
def main():
    """Start the FastAPI server with comprehensive OpenTelemetry instrumentation"""
    current_span = trace.get_current_span()
    current_span.set_attribute("service.name", "document-rag-api-ui")
    current_span.set_attribute("service.type", "api_server")
    current_span.set_attribute("startup.version", "1.0.0")
    
    # Add baggage for cross-service correlation
    baggage.set_baggage("system.component", "api_server")
    baggage.set_baggage("startup.session", str(uuid.uuid4()))
    
    logger.info("🎯 Starting Document RAG API Server with OpenTelemetry")
    logger.info("=" * 70)
    
    # Environment validation
    with tracer.start_as_current_span("environment_validation") as env_span:
        if not validate_environment():
            env_span.set_attribute("validation.result", "failed")
            current_span.set_attribute("startup.result", "failed")
            logger.error("❌ Environment validation failed, exiting")
            sys.exit(1)
        
        env_span.set_attribute("validation.result", "passed")
        current_span.add_event("Environment validation passed")

    # Display server information
    with tracer.start_as_current_span("server_info_display") as info_span:
        display_server_info()
        info_span.add_event("Server information displayed")

    logger.info("=" * 70)
    
    # Start the server
    with tracer.start_as_current_span("server_startup") as startup_span:
        current_span.set_attribute("startup.result", "success")
        current_span.add_event("Server startup initiated")
        
        startup_span.add_event("Uvicorn server starting")
        start_uvicorn_server()

if __name__ == "__main__":
    main()

