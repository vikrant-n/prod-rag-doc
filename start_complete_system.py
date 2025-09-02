#!/usr/bin/env python3
"""
FIXED: Enhanced Orchestrator with Proper Trace Context Propagation
"""

import os
import sys
import subprocess
import asyncio
import signal
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
from aiohttp import web

from dotenv import load_dotenv

# CRITICAL: Initialize OpenTelemetry FIRST before any other imports
os.environ.update({
    "OTEL_SERVICE_NAME": "document-rag-orchestrator", 
    "OTEL_SERVICE_VERSION": "2.0.0",
    "OTEL_ENVIRONMENT": "production",
    "OTEL_RESOURCE_ATTRIBUTES": "service.namespace=document-rag-system,deployment.environment=production",
    "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://172.31.41.170:4317"),
    "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
    "OTEL_EXPORTER_OTLP_INSECURE": "true",
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_METRICS_EXPORTER": "otlp", 
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_TRACES_SAMPLER": "always_on",  # Always sample for debugging
    "OTEL_TRACES_SAMPLER_ARG": "1.0",
    "OTEL_PYTHON_LOG_CORRELATION": "true"
})

load_dotenv()

# Initialize OpenTelemetry ONCE at module level
from otel_config import initialize_opentelemetry, get_service_tracer, get_correlated_logger
from opentelemetry import trace, context as otel_context
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.context import attach, detach

# Initialize orchestrator tracing IMMEDIATELY
orchestrator_tracer, orchestrator_meter = initialize_opentelemetry(
    service_name="document-rag-orchestrator",
    service_version="2.0.0", 
    environment="production"
)

# Create root trace context for orchestrator
ROOT_SPAN = orchestrator_tracer.start_span(
    "orchestrator_root_session",
    kind=SpanKind.SERVER
)
ROOT_CONTEXT_TOKEN = attach(trace.set_span_in_context(ROOT_SPAN))

print(f"🚀 ORCHESTRATOR ROOT TRACE ESTABLISHED")
print(f"   Root Trace ID: {format(ROOT_SPAN.get_span_context().trace_id, '032x')}")
print(f"   Root Span ID: {format(ROOT_SPAN.get_span_context().span_id, '016x')}")

# Initialize logger with trace context
logger = get_correlated_logger(__name__)

@dataclass
class ServiceConfig:
    name: str
    command: List[str]
    port: int
    environment: Dict[str, str]

class ServiceStatus(Enum):
    STARTING = "starting"
    RUNNING = "running" 
    STOPPED = "stopped"
    FAILED = "failed"

@dataclass
class ServiceProcess:
    name: str
    process: subprocess.Popen
    config: ServiceConfig
    status: ServiceStatus
    started_at: datetime
    pid: int

class ProcessManager:
    def __init__(self, tracer):
        self.tracer = tracer
        self.processes: Dict[str, ServiceProcess] = {}
        self.logger = get_correlated_logger(f"{__name__}.ProcessManager")

    def start_service(self, config: ServiceConfig) -> ServiceProcess:
        """FIXED: Start service with proper trace context propagation"""
        with self.tracer.start_as_current_span(
            f"start_service_{config.name}",
            kind=SpanKind.INTERNAL
        ) as span:
            span.set_attributes({
                "service.child": config.name,
                "service.port": config.port,
                "service.name": "document-rag-orchestrator",
                "operation.name": "start_service"
            })
            
            # Get current trace context for propagation
            current_context = otel_context.get_current()
            
            # Create headers dict for context injection
            context_headers = {}
            from opentelemetry import propagate
            propagate.inject(context_headers, context=current_context)
            
            # Extract trace information for logging
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x')
            
            self.logger.info_with_context(
                f"Starting service: {config.name}",
                extra_attributes={
                    "service.name": config.name,
                    "service.port": config.port,
                    "trace.parent_id": trace_id,
                    "operation": "service_startup"
                }
            )
            
            print(f"🔄 Starting {config.name}")
            print(f"   Parent Trace ID: {trace_id}")
            print(f"   Parent Span ID: {span_id}")
            print(f"   W3C Headers: {context_headers}")
            
            # Start with current environment
            env = os.environ.copy()
            
            # CRITICAL: Proper environment variable setup for child service
            service_env = {
                # OpenTelemetry configuration
                "OTEL_SERVICE_NAME": config.name,
                "OTEL_SERVICE_VERSION": "2.0.0",
                "OTEL_SERVICE_NAMESPACE": "document-rag-system",
                "OTEL_ENVIRONMENT": "production",
                "OTEL_RESOURCE_ATTRIBUTES": f"service.name={config.name},service.version=2.0.0,service.namespace=document-rag-system,deployment.environment=production,service.parent=document-rag-orchestrator",
                "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
                "OTEL_EXPORTER_OTLP_INSECURE": "true",
                "OTEL_TRACES_EXPORTER": "otlp",
                "OTEL_METRICS_EXPORTER": "otlp", 
                "OTEL_LOGS_EXPORTER": "otlp",
                "OTEL_TRACES_SAMPLER": "always_on",
                "OTEL_TRACES_SAMPLER_ARG": "1.0",
                "OTEL_PYTHON_LOG_CORRELATION": "true",
                
                # CRITICAL FIX: Ensure W3C context is properly formatted
                "OTEL_TRACE_PARENT": context_headers.get("traceparent", ""),
                "OTEL_TRACE_STATE": context_headers.get("tracestate", ""),
                "OTEL_PARENT_TRACE_ID": trace_id,
                "OTEL_PARENT_SPAN_ID": span_id,
                "OTEL_SERVICE_PARENT": "document-rag-orchestrator",
                
                # CRITICAL: Force proper trace propagation
                "OTEL_PROPAGATORS": "tracecontext,baggage",
                "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS": "",
                
                # Service communication
                "ORCHESTRATOR_URL": "http://localhost:8002",
                
                # Preserve critical environment variables
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "GOOGLE_DRIVE_FOLDER_ID": os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
                "GOOGLE_CREDENTIALS_PATH": os.getenv("GOOGLE_CREDENTIALS_PATH"),
                "GOOGLE_TOKEN_PATH": os.getenv("GOOGLE_TOKEN_PATH"),
                "ELASTICSEARCH_URL": os.getenv("ELASTICSEARCH_URL", "https://172.31.23.77:9200"),
                "ELASTICSEARCH_USERNAME": os.getenv("ELASTICSEARCH_USERNAME", "elastic"),
                "ELASTICSEARCH_PASSWORD": os.getenv("ELASTICSEARCH_PASSWORD", "elastic"),
                "ELASTICSEARCH_INDEX": os.getenv("ELASTICSEARCH_INDEX", "documents"),
                "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
                "CHUNK_SIZE": os.getenv("CHUNK_SIZE", "3000"),
                "CHUNK_OVERLAP": os.getenv("CHUNK_OVERLAP", "300"),
                "BATCH_SIZE": os.getenv("BATCH_SIZE", "5"),
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            }
            
            # Add service-specific environment variables
            if config.environment:
                service_env.update(config.environment)
            
            # Remove None values
            service_env = {k: v for k, v in service_env.items() if v is not None}
            env.update(service_env)

            try:
                # Start the process
                process = subprocess.Popen(
                    config.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    cwd=str(Path(__file__).parent),
                    bufsize=1,
                    universal_newlines=True
                )

                # Start thread to read and display output
                output_thread = threading.Thread(
                    target=self._log_stream_reader,
                    args=(process.stdout, config.name),
                    daemon=True
                )
                output_thread.start()

                # Wait for process to stabilize
                time.sleep(3)

                if process.poll() is None:
                    service_proc = ServiceProcess(
                        name=config.name,
                        process=process,
                        config=config,
                        status=ServiceStatus.RUNNING,
                        started_at=datetime.now(),
                        pid=process.pid
                    )
                    self.processes[config.name] = service_proc
                    
                    self.logger.info_with_context(
                        f"Service started successfully: {config.name}",
                        extra_attributes={
                            "service.name": config.name,
                            "service.pid": process.pid,
                            "trace.child_context_sent": True,
                            "operation": "service_startup",
                            "status": "success"
                        }
                    )
                    
                    print(f"✅ {config.name} started (PID: {process.pid})")
                    return service_proc
                else:
                    raise RuntimeError(f"Process failed to start: {process.poll()}")
                    
            except Exception as e:
                self.logger.error_with_context(
                    f"Failed to start service: {config.name}",
                    extra_attributes={
                        "service.name": config.name,
                        "error.type": type(e).__name__,
                        "error.message": str(e),
                        "operation": "service_startup"
                    },
                    exc_info=True
                )
                raise

    def _log_stream_reader(self, stream, service_name):
        """Read service output and log with correlation - reduced noise"""
        try:
            for line in iter(stream.readline, ''):
                if line and line.strip():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] [{service_name}] {line.strip()}")
                    
                    # Skip logging heartbeat and routine operations to reduce noise
                    line_lower = line.strip().lower()
                    if any(skip_term in line_lower for skip_term in [
                        "heartbeat", "service keepalive", "periodic heartbeat",
                        "no new files found", "elasticsearch instrumentation",
                        "warnings.warn"
                    ]):
                        continue
                    
                    # Only log important service events
                    if any(important_term in line_lower for important_term in [
                        "error", "failed", "exception", "starting", "initialized", 
                        "processing", "completed", "scan cycle"
                    ]):
                        logger.info_with_context(
                            f"Service log: {line.strip()}",
                            extra_attributes={
                                "service.name": service_name,
                                "log.source": "subprocess"
                            }
                        )
        except Exception as e:
            logger.error_with_context(
                f"Error reading stream for {service_name}",
                extra_attributes={
                    "service.name": service_name,
                    "error.message": str(e)
                }
            )
        finally:
            stream.close()

    def terminate_all(self):
        """Gracefully terminate all services"""
        with self.tracer.start_as_current_span("terminate_all_services") as span:
            self.logger.info_with_context(
                "Initiating graceful shutdown of all services",
                extra_attributes={
                    "services.count": len(self.processes),
                    "operation": "system_shutdown"
                }
            )
            
            for name, svc_proc in self.processes.items():
                if svc_proc.process.poll() is None:
                    print(f"🛑 Stopping {name}...")
                    svc_proc.process.terminate()
                    try:
                        svc_proc.process.wait(timeout=10)
                        print(f"✅ {name} stopped")
                    except subprocess.TimeoutExpired:
                        svc_proc.process.kill()
                        print(f"🔥 {name} force terminated")

class EnhancedOrchestrator:
    def __init__(self):
        self.tracer = orchestrator_tracer
        self.service_name = "document-rag-orchestrator"
        self.orchestrator_url = "http://localhost:8002"
        self.process_manager = ProcessManager(self.tracer)
        self.is_running = False
        self.startup_time = datetime.now()
        self.logger = get_correlated_logger(f"{__name__}.EnhancedOrchestrator")

        self.service_configs = {
            "backend": ServiceConfig(
                name="document-rag-backend",
                command=[sys.executable, "backend_service.py", "--host", "0.0.0.0", "--port", "8001"],
                port=8001,
                environment={}
            ),
            "api": ServiceConfig(
                name="document-rag-api", 
                command=[sys.executable, "start_server.py"],
                port=8000,
                environment={}
            )
        }

    async def start_system(self) -> bool:
        """Start system with proper trace correlation"""
        with self.tracer.start_as_current_span("system_startup") as span:
            span.set_attributes({
                "service.name": "document-rag-orchestrator",
                "operation.name": "system_startup",
                "services.count": len(self.service_configs)
            })
            
            self.logger.info_with_context(
                "Starting Document RAG System with trace correlation",
                extra_attributes={
                    "system.component": "orchestrator",
                    "services.count": len(self.service_configs),
                    "operation": "system_startup"
                }
            )
            
            print("🔥 Starting Document RAG System")
            print("=" * 70)
            
            # Start HTTP server first
            try:
                await self._start_http_server()
            except Exception as e:
                self.logger.error_with_context(
                    "Failed to start HTTP server",
                    extra_attributes={
                        "error.type": type(e).__name__,
                        "error.message": str(e)
                    }
                )
                return False
            
            # Start services with proper trace context
            services_started = 0
            for service_name in ["backend", "api"]:
                try:
                    with self.tracer.start_as_current_span(f"start_{service_name}") as service_span:
                        service_span.set_attributes({
                            "service.child": service_name,
                            "service.order": services_started + 1
                        })
                        
                        print(f"\n🔧 Starting service: {service_name}")
                        
                        self.process_manager.start_service(self.service_configs[service_name])
                        await asyncio.sleep(8)  # Give more time for proper startup
                        services_started += 1
                        
                        print(f"✅ {service_name} is ready")
                        
                except Exception as e:
                    self.logger.error_with_context(
                        f"Failed to start service: {service_name}",
                        extra_attributes={
                            "service.name": service_name,
                            "error.type": type(e).__name__,
                            "error.message": str(e)
                        },
                        exc_info=True
                    )
                    
                    self.process_manager.terminate_all()
                    return False
            
            self.is_running = True
            self._display_system_info()
            self._setup_signal_handlers()
            
            return True

    async def _start_http_server(self):
        """Start HTTP server for orchestrator communication"""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_get('/health', self._health_handler)
        app.router.add_post('/heartbeat', self._heartbeat_handler)
        
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', 8002)
        await site.start()
        
        print("📡 Orchestrator HTTP server started on http://localhost:8002")

    async def _health_handler(self, request):
        """Health check with trace context"""
        with self.tracer.start_as_current_span("orchestrator_health_check") as span:
            return web.json_response({
                "status": "healthy",
                "service": "document-rag-orchestrator",
                "timestamp": datetime.now().isoformat()
            })

    async def _heartbeat_handler(self, request):
        """Handle heartbeats from child services - minimal logging"""
        # Don't create spans for routine heartbeats to reduce noise
        data = await request.json()
        service_name = data.get("service", "unknown")
        
        # Only log heartbeats occasionally to confirm connectivity
        if not hasattr(self, '_heartbeat_counts'):
            self._heartbeat_counts = {}
        
        self._heartbeat_counts[service_name] = self._heartbeat_counts.get(service_name, 0) + 1
        
        # Log every 20th heartbeat (10 minutes) per service
        if self._heartbeat_counts[service_name] % 20 == 0:
            with self.tracer.start_as_current_span("heartbeat_status") as span:
                span.set_attributes({
                    "heartbeat.from": service_name,
                    "heartbeat.count": self._heartbeat_counts[service_name],
                    "heartbeat.status": data.get("status")
                })
                
                self.logger.info_with_context(
                    f"Heartbeat status from {service_name} - {self._heartbeat_counts[service_name]} received",
                    extra_attributes={
                        "heartbeat.from_service": service_name,
                        "heartbeat.count": self._heartbeat_counts[service_name],
                        "heartbeat.service_status": data.get("status"),
                        "operation": "heartbeat_status"
                    }
                )
        
        return web.json_response({
            "status": "acknowledged",
            "service": "document-rag-orchestrator",
            "timestamp": datetime.now().isoformat()
        })

    def _display_system_info(self):
        """Display system information with trace context"""
        current_span = trace.get_current_span()
        trace_id = "unknown"
        if current_span and current_span.is_recording():
            trace_id = format(current_span.get_span_context().trace_id, '032x')
        
        print("\n" + "="*70)
        print("✅ Document RAG System Ready!")
        print("=" * 70)
        print(f"🆔 System Trace ID: {trace_id}")
        print(f"📡 Orchestrator: http://localhost:8002")
        print(f"🌐 API Service: http://localhost:8000")
        print(f"⚙️ Backend Service: http://localhost:8001")
        print("=" * 70)

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            print(f"\n🛑 Shutdown signal received ({signal_name})...")
            self.is_running = False
            self.process_manager.terminate_all()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main orchestrator entry point with proper trace context"""
    
    print("🚀 DOCUMENT RAG ORCHESTRATOR v2.0")
    print(f"🆔 Root Trace ID: {format(ROOT_SPAN.get_span_context().trace_id, '032x')}")
    print(f"📡 OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
    print()
    
    orchestrator = EnhancedOrchestrator()
    
    try:
        success = await orchestrator.start_system()
        
        if not success:
            print("❌ Startup failed")
            return 1
        
        print("🔄 System running... (Press Ctrl+C to stop)")
        
        # Main orchestrator loop
        heartbeat_count = 0
        while orchestrator.is_running:
            with orchestrator.tracer.start_as_current_span("orchestrator_heartbeat") as heartbeat_span:
                heartbeat_count += 1
                
                heartbeat_span.set_attributes({
                    "service.name": "document-rag-orchestrator",
                    "heartbeat.count": heartbeat_count,
                    "operation.name": "orchestrator_heartbeat"
                })
                
                if heartbeat_count % 60 == 0:  # Every 10 minutes
                    current_trace = format(heartbeat_span.get_span_context().trace_id, '032x')
                    print(f"💓 Orchestrator heartbeat #{heartbeat_count} (Trace: {current_trace[:8]}...)")
                
                await asyncio.sleep(10)
                
    except KeyboardInterrupt:
        print("\n🔄 Shutdown requested")
    except Exception as e:
        logger.error_with_context(
            "Unexpected error in main loop",
            extra_attributes={
                "error.type": type(e).__name__,
                "error.message": str(e)
            },
            exc_info=True
        )
    finally:
        orchestrator.process_manager.terminate_all()
        
        # Clean up root context
        detach(ROOT_CONTEXT_TOKEN)
        ROOT_SPAN.end()

if __name__ == "__main__":
    asyncio.run(main())