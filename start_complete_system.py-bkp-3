#!/usr/bin/env python3
"""
Complete System Startup Script with OpenTelemetry Instrumentation

Starts both the backend document processing service and the API+UI server
with distributed tracing and observability
"""

import os
import sys
import subprocess
import time
import signal
import threading
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
import socket
import uuid

# OpenTelemetry imports - corrected versions
from opentelemetry import trace, metrics, baggage
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, SERVICE_INSTANCE_ID
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat

load_dotenv()

# Initialize OpenTelemetry - avoid conflicts with auto-instrumentation
def init_telemetry():
    """Initialize OpenTelemetry with EDOT configuration"""
    
    # Check if we're already using auto-instrumentation
    current_tracer_provider = trace.get_tracer_provider()
    if hasattr(current_tracer_provider, '__class__') and current_tracer_provider.__class__.__name__ != 'ProxyTracerProvider':
        print("✅ Using existing OpenTelemetry auto-instrumentation")
        return trace.get_tracer(__name__), metrics.get_meter(__name__)
    
    # Generate unique service instance ID
    service_instance_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    
    # Resource configuration
    resource = Resource.create({
        SERVICE_NAME: "document-rag-orchestrator",
        SERVICE_VERSION: "1.0.0",
        SERVICE_INSTANCE_ID: service_instance_id,
        "service.namespace": "document-rag-system",
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "production"),
        "host.name": socket.gethostname(),
        "process.pid": str(os.getpid()),
    })
    
    # OTLP endpoint configuration
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    # Only set providers if not already set
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
        
        # Console exporter for debugging (can be removed in production)
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
            export_interval_millis=10000,  # 10 seconds
        )
        
        metric_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(metric_provider)
        
        print("✅ OpenTelemetry initialized with EDOT configuration")
        
    except Exception as e:
        print(f"⚠️  OpenTelemetry provider already set or error: {e}")
    
    # Set up propagators for distributed tracing
    set_global_textmap(B3MultiFormat())
    
    # Manual instrumentation only if not using auto-instrumentation
    if not os.getenv("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"):
        try:
            LoggingInstrumentor().instrument(set_logging_format=True)
            RequestsInstrumentor().instrument()
        except Exception as e:
            print(f"⚠️  Instrumentation already applied or error: {e}")
    
    return trace.get_tracer(__name__), metrics.get_meter(__name__)

# Initialize telemetry
tracer, meter = init_telemetry()

# Custom metrics
service_startup_counter = meter.create_counter(
    "service_startup_total",
    description="Total number of service startups",
    unit="1"
)

service_health_gauge = meter.create_up_down_counter(
    "service_health_status",
    description="Service health status (1=healthy, 0=unhealthy)",
    unit="1"
)

process_monitoring_gauge = meter.create_up_down_counter(
    "process_monitoring_active",
    description="Number of active process monitors",
    unit="1"
)

# Configure structured logging with trace correlation
class TraceFormatter(logging.Formatter):
    """Custom formatter to include trace and span IDs"""
    
    def format(self, record):
        # Get current span context
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

# Set up logging with trace correlation
formatter = TraceFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler('orchestrator.log')
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set"""
    with tracer.start_as_current_span("check_environment") as span:
        logger.info("🔧 Checking environment configuration...")
        
        span.set_attribute("check.type", "environment")
        
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = []

        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        span.set_attribute("required_vars.count", len(required_vars))
        span.set_attribute("missing_vars.count", len(missing_vars))
        
        if missing_vars:
            span.set_attribute("environment.status", "invalid")
            span.add_event("Missing environment variables", {
                "missing_vars": str(missing_vars)
            })
            logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            logger.info("   Please set them in your .env file")
            return False

        # Check optional but recommended variables
        google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if not google_drive_folder_id:
            span.add_event("Google Drive monitoring disabled")
            logger.warning("⚠️  Google Drive Folder ID not set - Google Drive monitoring will be disabled")
        else:
            span.set_attribute("google_drive.folder_id", google_drive_folder_id)
            span.add_event("Google Drive monitoring enabled")
            logger.info(f"✅ Google Drive monitoring enabled for folder: {google_drive_folder_id}")

        span.set_attribute("environment.status", "valid")
        logger.info("✅ Environment configuration OK")
        return True

def check_qdrant():
    """Check if Qdrant is accessible"""
    with tracer.start_as_current_span("check_qdrant") as span:
        logger.info("🔍 Checking Qdrant connection...")
        
        span.set_attribute("check.type", "qdrant")
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        span.set_attribute("qdrant.url", qdrant_url)

        try:
            import requests
            
            with tracer.start_as_current_span("qdrant_health_check") as health_span:
                response = requests.get(qdrant_url, timeout=5)
                health_span.set_attribute("http.status_code", response.status_code)
                health_span.set_attribute("http.url", qdrant_url)
                
                if response.status_code == 200:
                    span.set_attribute("qdrant.status", "accessible")
                    span.add_event("Qdrant connection successful")
                    logger.info(f"✅ Qdrant accessible at {qdrant_url}")
                    return True
                else:
                    span.set_attribute("qdrant.status", "error")
                    span.add_event("Qdrant connection failed", {
                        "status_code": response.status_code
                    })
                    logger.warning(f"⚠️  Qdrant responded with status {response.status_code}")
                    return False
                    
        except Exception as e:
            span.set_attribute("qdrant.status", "unreachable")
            span.record_exception(e)
            span.add_event("Qdrant connection exception", {
                "error": str(e)
            })
            logger.error(f"❌ Could not connect to Qdrant: {e}")
            logger.info("   Make sure Qdrant is running or update QDRANT_URL in .env")
            return False

def start_backend_service():
    """Start the backend document processing service"""
    with tracer.start_as_current_span("start_backend_service") as span:
        logger.info("🚀 Starting backend document processing service...")
        
        span.set_attribute("service.name", "backend_service")
        span.set_attribute("service.port", 8001)
        span.set_attribute("service.host", "0.0.0.0")
        
        service_startup_counter.add(1, {"service": "backend", "status": "attempt"})

        try:
            # Set environment variables for child process to inherit trace context
            env = os.environ.copy()
            env.update({
                "OTEL_SERVICE_NAME": "document-rag-backend",
                "OTEL_RESOURCE_ATTRIBUTES": f"service.name=document-rag-backend,service.version=1.0.0",
                "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            })
            
            # Inject trace context into environment for manual propagation
            from opentelemetry.propagate import inject
            headers = {}
            inject(headers)
            for key, value in headers.items():
                env[f"OTEL_PROPAGATED_{key.upper().replace('-', '_')}"] = value
            
            with tracer.start_as_current_span("subprocess_start") as subprocess_span:
                process = subprocess.Popen([
                    sys.executable, "backend_service.py",
                    "--host", "0.0.0.0",
                    "--port", "8001"
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
                
                subprocess_span.set_attribute("subprocess.pid", process.pid)
                subprocess_span.set_attribute("subprocess.command", "backend_service.py")

            # Wait a bit to see if it starts successfully
            time.sleep(3)

            if process.poll() is None:
                span.set_attribute("service.status", "started")
                span.set_attribute("process.pid", process.pid)
                span.add_event("Backend service started successfully")
                service_startup_counter.add(1, {"service": "backend", "status": "success"})
                service_health_gauge.add(1, {"service": "backend"})
                logger.info("✅ Backend service started successfully on port 8001")
                return process
            else:
                span.set_attribute("service.status", "failed")
                span.add_event("Backend service failed to start")
                service_startup_counter.add(1, {"service": "backend", "status": "failure"})
                logger.error("❌ Backend service failed to start")
                return None

        except Exception as e:
            span.set_attribute("service.status", "error")
            span.record_exception(e)
            service_startup_counter.add(1, {"service": "backend", "status": "error"})
            logger.error(f"❌ Error starting backend service: {e}")
            return None

def start_api_ui_server():
    """Start the API+UI server"""
    with tracer.start_as_current_span("start_api_ui_server") as span:
        logger.info("🌐 Starting API+UI server...")
        
        span.set_attribute("service.name", "api_ui_server")
        span.set_attribute("service.port", 8000)
        
        service_startup_counter.add(1, {"service": "api_ui", "status": "attempt"})

        try:
            # Set environment variables for child process to inherit trace context
            env = os.environ.copy()
            env.update({
                "OTEL_SERVICE_NAME": "document-rag-api-ui",
                "OTEL_RESOURCE_ATTRIBUTES": f"service.name=document-rag-api-ui,service.version=1.0.0",
                "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            })
            
            # Inject trace context into environment for manual propagation
            from opentelemetry.propagate import inject
            headers = {}
            inject(headers)
            for key, value in headers.items():
                env[f"OTEL_PROPAGATED_{key.upper().replace('-', '_')}"] = value
            
            with tracer.start_as_current_span("subprocess_start") as subprocess_span:
                process = subprocess.Popen([
                    sys.executable, "start_server.py"
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
                
                subprocess_span.set_attribute("subprocess.pid", process.pid)
                subprocess_span.set_attribute("subprocess.command", "start_server.py")

            # Wait a bit to see if it starts successfully
            time.sleep(3)

            if process.poll() is None:
                span.set_attribute("service.status", "started")
                span.set_attribute("process.pid", process.pid)
                span.add_event("API+UI server started successfully")
                service_startup_counter.add(1, {"service": "api_ui", "status": "success"})
                service_health_gauge.add(1, {"service": "api_ui"})
                logger.info("✅ API+UI server started successfully on port 8000")
                return process
            else:
                span.set_attribute("service.status", "failed")
                span.add_event("API+UI server failed to start")
                service_startup_counter.add(1, {"service": "api_ui", "status": "failure"})
                logger.error("❌ API+UI server failed to start")
                return None

        except Exception as e:
            span.set_attribute("service.status", "error")
            span.record_exception(e)
            service_startup_counter.add(1, {"service": "api_ui", "status": "error"})
            logger.error(f"❌ Error starting API+UI server: {e}")
            return None

def monitor_process(process, name):
    """Monitor a process and log its output with tracing"""
    with tracer.start_as_current_span(f"monitor_process_{name.lower()}") as span:
        span.set_attribute("process.name", name)
        span.set_attribute("process.pid", process.pid)
        
        process_monitoring_gauge.add(1, {"process": name.lower()})
        logger.info(f"🔍 Starting process monitor for {name}")
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Create a child span for each log line to maintain context
                    with tracer.start_as_current_span(f"process_log_{name.lower()}") as log_span:
                        log_span.set_attribute("log.source", name)
                        log_span.set_attribute("log.content", line.rstrip()[:200])  # Truncate long logs
                        logger.info(f"[{name}] {line.rstrip()}")
                        
        except Exception as e:
            span.record_exception(e)
            logger.error(f"❌ Error monitoring {name}: {e}")
        finally:
            process_monitoring_gauge.add(-1, {"process": name.lower()})
            span.add_event("Process monitoring ended")

@tracer.start_as_current_span("main_startup")
def main():
    """Main startup function with distributed tracing"""
    current_span = trace.get_current_span()
    current_span.set_attribute("system.name", "document-rag-system")
    current_span.set_attribute("startup.version", "1.0.0")
    
    # Add baggage for cross-service correlation
    baggage.set_baggage("system.component", "orchestrator")
    baggage.set_baggage("startup.session", str(uuid.uuid4()))
    
    logger.info("🎯 Starting Complete Document RAG System")
    logger.info("=" * 60)

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    current_span.set_attribute("working.directory", str(script_dir))
    logger.info(f"📁 Working directory: {script_dir}")

    # Check environment
    with tracer.start_as_current_span("environment_validation") as env_span:
        if not check_environment():
            env_span.set_attribute("validation.result", "failed")
            logger.error("Environment validation failed, exiting")
            sys.exit(1)
        env_span.set_attribute("validation.result", "passed")

    # Check Qdrant
    with tracer.start_as_current_span("qdrant_validation") as qdrant_span:
        qdrant_healthy = check_qdrant()
        qdrant_span.set_attribute("validation.result", "passed" if qdrant_healthy else "warning")
        if not qdrant_healthy:
            logger.warning("⚠️  Continuing without Qdrant check - service will handle connection errors")

    logger.info("")
    logger.info("🚀 Starting services...")
    logger.info("")

    # Start backend service
    with tracer.start_as_current_span("backend_service_startup") as backend_span:
        backend_process = start_backend_service()
        if not backend_process:
            backend_span.set_attribute("startup.result", "failed")
            logger.error("❌ Failed to start backend service")
            sys.exit(1)
        backend_span.set_attribute("startup.result", "success")
        backend_span.set_attribute("process.pid", backend_process.pid)

    # Start API+UI server
    with tracer.start_as_current_span("api_ui_service_startup") as api_span:
        api_process = start_api_ui_server()
        if not api_process:
            api_span.set_attribute("startup.result", "failed")
            logger.error("❌ Failed to start API+UI server")
            backend_process.terminate()
            sys.exit(1)
        api_span.set_attribute("startup.result", "success")
        api_span.set_attribute("process.pid", api_process.pid)

    current_span.set_attribute("startup.status", "completed")
    current_span.add_event("All services started successfully")
    
    logger.info("")
    logger.info("✅ All services started successfully!")
    logger.info("")
    logger.info("🌐 System URLs:")
    logger.info("   📊 Main UI:           http://localhost:8000")
    logger.info("   🔧 API Documentation: http://localhost:8000/docs")
    logger.info("   ⚙️  Backend Status:    http://localhost:8001/status")
    logger.info("   🔍 Manual Scan:       http://localhost:8001/scan")
    logger.info("")
    logger.info("📋 System Components:")
    logger.info("   🔄 Backend Service:    Monitors Google Drive & processes documents")
    logger.info("   🌐 API Server:         Handles queries & retrieval")
    logger.info("   💻 Web UI:            User interface for document search")
    logger.info("")
    logger.info("🔍 OpenTelemetry Configuration:")
    logger.info(f"   📡 OTLP Endpoint:     {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4317')}")
    logger.info(f"   🏷️  Service Name:      document-rag-orchestrator")
    logger.info(f"   📊 Trace ID:          {format(current_span.get_span_context().trace_id, '032x')}")
    logger.info("")
    logger.info("Press Ctrl+C to stop all services")
    logger.info("=" * 60)

    # Set up signal handler for clean shutdown
    def signal_handler(signum, frame):
        with tracer.start_as_current_span("system_shutdown") as shutdown_span:
            shutdown_span.set_attribute("shutdown.signal", signum)
            logger.info("\n🛑 Shutting down all services...")
            
            # Update service health metrics
            service_health_gauge.add(-1, {"service": "backend"})
            service_health_gauge.add(-1, {"service": "api_ui"})
            
            backend_process.terminate()
            api_process.terminate()

            # Wait for processes to terminate
            try:
                with tracer.start_as_current_span("graceful_shutdown") as graceful_span:
                    backend_process.wait(timeout=10)
                    api_process.wait(timeout=10)
                    graceful_span.set_attribute("shutdown.type", "graceful")
            except subprocess.TimeoutExpired:
                with tracer.start_as_current_span("force_shutdown") as force_span:
                    logger.warning("⚠️  Force killing processes...")
                    backend_process.kill()
                    api_process.kill()
                    force_span.set_attribute("shutdown.type", "forced")

            shutdown_span.add_event("All services stopped")
            logger.info("✅ All services stopped")
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start monitoring threads
    with tracer.start_as_current_span("start_monitoring_threads") as monitor_span:
        backend_thread = threading.Thread(
            target=monitor_process,
            args=(backend_process, "BACKEND"),
            daemon=True
        )
        api_thread = threading.Thread(
            target=monitor_process,
            args=(api_process, "API"),
            daemon=True
        )

        backend_thread.start()
        api_thread.start()
        
        monitor_span.set_attribute("monitoring.threads", 2)
        monitor_span.add_event("Process monitoring threads started")

    # Keep main thread alive with health monitoring
    with tracer.start_as_current_span("system_monitoring") as monitoring_span:
        monitoring_span.set_attribute("monitoring.type", "main_loop")
        
        try:
            while True:
                # Check if processes are still running
                if backend_process.poll() is not None:
                    monitoring_span.add_event("Backend service stopped unexpectedly")
                    service_health_gauge.add(-1, {"service": "backend"})
                    logger.error("❌ Backend service stopped unexpectedly")
                    break
                if api_process.poll() is not None:
                    monitoring_span.add_event("API service stopped unexpectedly")
                    service_health_gauge.add(-1, {"service": "api_ui"})
                    logger.error("❌ API service stopped unexpectedly")
                    break

                time.sleep(1)
        except KeyboardInterrupt:
            monitoring_span.add_event("Keyboard interrupt received")
            pass

    # Clean shutdown
    signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()

