#!/usr/bin/env python3
"""
Enhanced Orchestrator with HTTP Server for Service Map Connectivity and Correlated Logging
Fixed to properly start backend services with console output and proper environment handling
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

# CRITICAL: Set OTEL environment variables BEFORE any imports
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
    "OTEL_TRACES_SAMPLER": "traceidratio",
    "OTEL_TRACES_SAMPLER_ARG": "1.0",
    "OTEL_PYTHON_LOG_CORRELATION": "false"
})

load_dotenv()

# Add correlated logging import
from otel_config import (
    initialize_opentelemetry, get_service_tracer,
    get_current_trace_id, extract_and_activate_context, propagate,
    get_correlated_logger, enhanced_error_logging
)
from opentelemetry.trace import SpanKind
from opentelemetry.context import attach, detach
from opentelemetry import trace

# Initialize correlated logger
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

def log_stream_reader(stream, service_name, logger, log_level="INFO"):
    """Read from subprocess stream and log to console with proper formatting"""
    try:
        for line in iter(stream.readline, ''):
            if line:
                line = line.strip()
                if line:
                    # Format the output with service name and timestamp
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] [{service_name}] {line}")
                    
                    # Also log through structured logging
                    logger.info_with_context(
                        f"Service log: {line}",
                        extra_attributes={
                            "service.name": service_name,
                            "log.source": "subprocess",
                            "operation": "service_logging"
                        }
                    )
    except Exception as e:
        logger.error_with_context(
            f"Error reading stream for {service_name}",
            extra_attributes={
                "service.name": service_name,
                "error.type": type(e).__name__,
                "error.message": str(e),
                "operation": "stream_reading"
            }
        )
    finally:
        stream.close()

class ProcessManager:
    def __init__(self, tracer):
        self.tracer = tracer
        self.processes: Dict[str, ServiceProcess] = {}
        self.logger = get_correlated_logger(f"{__name__}.ProcessManager")

    def start_service(self, config: ServiceConfig) -> ServiceProcess:
        with self.tracer.start_as_current_span(
            f"start_{config.name}",
            kind=SpanKind.INTERNAL
        ) as span:
            span.set_attributes({
                "service.child": config.name,
                "service.port": config.port,
                "service.name": "document-rag-orchestrator",
                "operation.name": "start_service"
            })
            
            self.logger.info_with_context(
                f"Starting service: {config.name}",
                extra_attributes={
                    "service.name": config.name,
                    "service.port": config.port,
                    "service.command": " ".join(config.command),
                    "operation": "service_startup"
                }
            )
            
            # Start with current environment
            env = os.environ.copy()
            
            # Get span context for propagation
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x')
            
            # Create proper trace headers for child service
            trace_headers = {
                "traceparent": f"00-{trace_id}-{span_id}-01",
                "tracestate": "",
            }
            
            self.logger.info_with_context(
                f"Creating trace context for {config.name}",
                extra_attributes={
                    "service.name": config.name,
                    "trace.parent_id": trace_id,
                    "span.parent_id": span_id,
                    "trace.headers": trace_headers,
                    "operation": "trace_propagation"
                }
            )
            
            # Preserve all existing environment variables and add new ones
            service_env = {
                # OpenTelemetry configuration - CRITICAL for service map
                "OTEL_SERVICE_NAME": config.name,
                "OTEL_SERVICE_VERSION": "2.0.0",
                "OTEL_SERVICE_NAMESPACE": "document-rag-system",
                "OTEL_ENVIRONMENT": os.getenv("OTEL_ENVIRONMENT", "production"),
                "OTEL_RESOURCE_ATTRIBUTES": f"service.name={config.name},service.version=2.0.0,service.namespace=document-rag-system,deployment.environment=production",
                "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
                "OTEL_EXPORTER_OTLP_INSECURE": "true",
                "OTEL_TRACES_EXPORTER": "otlp",
                "OTEL_METRICS_EXPORTER": "otlp", 
                "OTEL_LOGS_EXPORTER": "otlp",
                "OTEL_TRACES_SAMPLER": "always_on",  # Changed from traceidratio for debugging
                "OTEL_TRACES_SAMPLER_ARG": "1.0",
                "OTEL_PYTHON_LOG_CORRELATION": "true",  # Enable for better correlation
                "OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED": "true",
                
                # Trace propagation
                "OTEL_PARENT_TRACE_ID": trace_id,
                "OTEL_PARENT_SPAN_ID": span_id,
                "OTEL_SERVICE_PARENT": "document-rag-orchestrator",
                "TRACEPARENT": trace_headers["traceparent"],
                
                # Service communication
                "ORCHESTRATOR_URL": "http://localhost:8002",
                
                # Preserve critical environment variables
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "GOOGLE_DRIVE_FOLDER_ID": os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
                "GOOGLE_CREDENTIALS_PATH": os.getenv("GOOGLE_CREDENTIALS_PATH"),
                "GOOGLE_TOKEN_PATH": os.getenv("GOOGLE_TOKEN_PATH"),
                "LOCAL_WATCH_DIRS": os.getenv("LOCAL_WATCH_DIRS"),
                "SCAN_INTERVAL": os.getenv("SCAN_INTERVAL"),
                
                # Embedding and processing configuration
                "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
                "EMBEDDING_VECTOR_SIZE": os.getenv("EMBEDDING_VECTOR_SIZE", "3072"),
                "CHUNK_SIZE": os.getenv("CHUNK_SIZE", "3000"),
                "CHUNK_OVERLAP": os.getenv("CHUNK_OVERLAP", "300"),
                "BATCH_SIZE": os.getenv("BATCH_SIZE", "5"),
                "RETRIEVAL_INITIAL_K": os.getenv("RETRIEVAL_INITIAL_K", "20"),
                "RETRIEVAL_FETCH_K": os.getenv("RETRIEVAL_FETCH_K", "10"),
                "RETRIEVAL_FINAL_K": os.getenv("RETRIEVAL_FINAL_K", "5"),
                "RETRIEVAL_TOP_N": os.getenv("RETRIEVAL_TOP_N", "5"),
                "CONTEXT_OUTPUT_FILE": os.getenv("CONTEXT_OUTPUT_FILE"),
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
                
                # Python path to ensure imports work
                "PYTHONPATH": os.pathsep.join([
                    str(Path(__file__).parent),
                    os.getenv("PYTHONPATH", "")
                ]).rstrip(os.pathsep),
            }
            
            # Add service-specific environment variables
            if config.environment:
                service_env.update(config.environment)
            
            # Remove None values
            service_env = {k: v for k, v in service_env.items() if v is not None}
            
            # Update the environment
            env.update(service_env)

            self.logger.info_with_context(
                f"Environment prepared for service: {config.name}",
                extra_attributes={
                    "service.name": config.name,
                    "trace.parent_id": trace_id,
                    "span.parent_id": span_id,
                    "orchestrator.url": "http://localhost:8002",
                    "otel.endpoint": service_env.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
                    "otel.service_name": service_env.get("OTEL_SERVICE_NAME"),
                    "python_path": service_env.get("PYTHONPATH"),
                    "working_dir": str(Path(__file__).parent),
                    "operation": "service_startup"
                }
            )

            print(f"🚀 Starting {config.name}")
            print(f"   Command: {' '.join(config.command)}")
            print(f"   Working Dir: {Path(__file__).parent}")
            print(f"   OTEL Endpoint: {service_env.get('OTEL_EXPORTER_OTLP_ENDPOINT')}")
            print(f"   Trace Parent: {trace_id}")
            
            try:
                # Start the process with real-time output
                process = subprocess.Popen(
                    config.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    cwd=str(Path(__file__).parent),
                    bufsize=1,  # Line buffering
                    universal_newlines=True
                )

                # Start thread to read and display output in real-time
                output_thread = threading.Thread(
                    target=log_stream_reader,
                    args=(process.stdout, config.name, self.logger),
                    daemon=True
                )
                output_thread.start()

                # Wait a bit to see if the process starts successfully
                time.sleep(5)

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
                            "service.port": config.port,
                            "service.status": ServiceStatus.RUNNING.value,
                            "startup.duration_seconds": 5,
                            "trace.child_id": trace_id,
                            "operation": "service_startup",
                            "status": "success"
                        }
                    )
                    
                    print(f"✅ {config.name} started successfully (PID: {process.pid})")
                    print(f"   Trace ID: {trace_id}")
                    return service_proc
                else:
                    exit_code = process.poll()
                    error_msg = f"Process exited with code {exit_code}"
                    
                    # Try to read any remaining output
                    try:
                        remaining_output = process.stdout.read()
                        if remaining_output:
                            print(f"   Last output: {remaining_output}")
                    except:
                        pass
                    
                    self.logger.error_with_context(
                        f"Service failed to start: {config.name}",
                        extra_attributes={
                            "service.name": config.name,
                            "service.exit_code": exit_code,
                            "error.message": error_msg,
                            "operation": "service_startup",
                            "status": "failed"
                        }
                    )
                    raise RuntimeError(f"Failed to start {config.name}: {error_msg}")
                    
            except Exception as e:
                enhanced_error_logging(
                    self.logger,
                    f"Exception during service startup: {config.name}",
                    extra_attributes={
                        "service.name": config.name,
                        "service.command": " ".join(config.command),
                        "working_dir": str(Path(__file__).parent),
                        "operation": "service_startup",
                        "status": "exception"
                    }
                )
                print(f"❌ Exception starting {config.name}: {e}")
                raise

    def terminate_all(self):
        self.logger.info_with_context(
            "Initiating graceful shutdown of all services",
            extra_attributes={
                "services.count": len(self.processes),
                "operation": "system_shutdown"
            }
        )
        
        print("🛑 Initiating graceful shutdown...")
        terminated_count = 0
        failed_count = 0
        
        for name, svc_proc in self.processes.items():
            if svc_proc.process.poll() is None:
                self.logger.info_with_context(
                    f"Stopping service: {name}",
                    extra_attributes={
                        "service.name": name,
                        "service.pid": svc_proc.pid,
                        "operation": "service_shutdown"
                    }
                )
                
                print(f"⏹️  Stopping {name}...")
                svc_proc.process.terminate()
                
                try:
                    svc_proc.process.wait(timeout=10)
                    terminated_count += 1
                    
                    self.logger.info_with_context(
                        f"Service stopped gracefully: {name}",
                        extra_attributes={
                            "service.name": name,
                            "service.pid": svc_proc.pid,
                            "shutdown.method": "graceful",
                            "operation": "service_shutdown",
                            "status": "success"
                        }
                    )
                    
                    print(f"✅ {name} stopped")
                except subprocess.TimeoutExpired:
                    svc_proc.process.kill()
                    failed_count += 1
                    
                    self.logger.warning_with_context(
                        f"Service force terminated: {name}",
                        extra_attributes={
                            "service.name": name,
                            "service.pid": svc_proc.pid,
                            "shutdown.method": "force_kill",
                            "shutdown.reason": "timeout_expired",
                            "operation": "service_shutdown",
                            "status": "forced"
                        }
                    )
                    
                    print(f"🔥 {name} force terminated")
            else:
                self.logger.debug_with_context(
                    f"Service already stopped: {name}",
                    extra_attributes={
                        "service.name": name,
                        "operation": "service_shutdown"
                    }
                )
        
        self.logger.info_with_context(
            "Service shutdown completed",
            extra_attributes={
                "services.total": len(self.processes),
                "services.terminated_gracefully": terminated_count,
                "services.force_killed": failed_count,
                "operation": "system_shutdown",
                "status": "completed"
            }
        )

class OrchestratorHTTPServer:
    """HTTP server for orchestrator to receive calls from children"""
    
    def __init__(self, tracer):
        self.tracer = tracer
        self.logger = get_correlated_logger(f"{__name__}.OrchestratorHTTPServer")
        self.app = web.Application()
        self.setup_routes()
        self.runner = None
        
    def setup_routes(self):
        self.app.router.add_get('/health', self.health_handler)
        self.app.router.add_post('/heartbeat', self.heartbeat_handler)
        self.app.router.add_get('/status', self.status_handler)
        
        self.logger.debug_with_context(
            "HTTP server routes configured",
            extra_attributes={
                "routes.count": 3,
                "routes": ["/health", "/heartbeat", "/status"],
                "operation": "http_server_setup"
            }
        )
        
    async def health_handler(self, request):
        """Health check endpoint with trace context extraction"""
        # Extract trace context from headers
        context = propagate.extract(dict(request.headers))
        token = attach(context)
        
        try:
            with self.tracer.start_as_current_span(
                "orchestrator.health_check",
                kind=SpanKind.SERVER
            ) as span:
                span.set_attributes({
                    "http.method": "GET",
                    "http.route": "/health",
                    "http.status_code": 200
                })
                
                self.logger.debug_with_context(
                    "Health check requested",
                    extra_attributes={
                        "http.method": "GET",
                        "http.route": "/health",
                        "http.status_code": 200,
                        "operation": "health_check"
                    }
                )
                
                return web.json_response({
                    "status": "healthy",
                    "service": "document-rag-orchestrator",
                    "timestamp": datetime.now().isoformat()
                })
        finally:
            detach(token)
    
    async def heartbeat_handler(self, request):
        """Receive heartbeats from child services"""
        # Extract trace context
        context = propagate.extract(dict(request.headers))
        token = attach(context)
        
        try:
            with self.tracer.start_as_current_span(
                "orchestrator.receive_heartbeat",
                kind=SpanKind.SERVER
            ) as span:
                data = await request.json()
                service_name = data.get("service", "unknown")
                service_status = data.get("status", "unknown")
                stats = data.get("stats", {})
                
                span.set_attributes({
                    "http.method": "POST",
                    "http.route": "/heartbeat",
                    "heartbeat.from": service_name,
                    "heartbeat.status": service_status,
                    "http.status_code": 200
                })
                
                self.logger.info_with_context(
                    f"Heartbeat received from service: {service_name}",
                    extra_attributes={
                        "http.method": "POST",
                        "http.route": "/heartbeat",
                        "heartbeat.from_service": service_name,
                        "heartbeat.service_status": service_status,
                        "heartbeat.stats": stats,
                        "operation": "heartbeat_receive"
                    }
                )
                
                return web.json_response({
                    "status": "acknowledged",
                    "service": "document-rag-orchestrator",
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            self.logger.error_with_context(
                "Error processing heartbeat",
                extra_attributes={
                    "http.method": "POST",
                    "http.route": "/heartbeat",
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "heartbeat_receive",
                    "status": "failed"
                },
                exc_info=True
            )
            raise
        finally:
            detach(token)
    
    async def status_handler(self, request):
        """Status endpoint"""
        context = propagate.extract(dict(request.headers))
        token = attach(context)
        
        try:
            with self.tracer.start_as_current_span(
                "orchestrator.status",
                kind=SpanKind.SERVER
            ) as span:
                span.set_attributes({
                    "http.method": "GET",
                    "http.route": "/status"
                })
                
                self.logger.debug_with_context(
                    "Status request received",
                    extra_attributes={
                        "http.method": "GET",
                        "http.route": "/status",
                        "operation": "status_check"
                    }
                )
                
                return web.json_response({
                    "status": "running",
                    "service": "document-rag-orchestrator",
                    "timestamp": datetime.now().isoformat()
                })
        finally:
            detach(token)
    
    async def start(self):
        """Start HTTP server"""
        self.logger.info_with_context(
            "Starting orchestrator HTTP server",
            extra_attributes={
                "server.host": "0.0.0.0",
                "server.port": 8002,
                "operation": "http_server_startup"
            }
        )
        
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, '0.0.0.0', 8002)
            await site.start()
            
            self.logger.info_with_context(
                "Orchestrator HTTP server started successfully",
                extra_attributes={
                    "server.host": "0.0.0.0",
                    "server.port": 8002,
                    "server.url": "http://localhost:8002",
                    "operation": "http_server_startup",
                    "status": "success"
                }
            )
            
            print("📡 Orchestrator HTTP server started on http://localhost:8002")
        except Exception as e:
            self.logger.error_with_context(
                "Failed to start HTTP server",
                extra_attributes={
                    "server.host": "0.0.0.0",
                    "server.port": 8002,
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "http_server_startup",
                    "status": "failed"
                },
                exc_info=True
            )
            raise
    
    async def stop(self):
        """Stop HTTP server"""
        self.logger.info_with_context(
            "Stopping orchestrator HTTP server",
            extra_attributes={
                "operation": "http_server_shutdown"
            }
        )
        
        if self.runner:
            try:
                await self.runner.cleanup()
                self.logger.info_with_context(
                    "HTTP server stopped successfully",
                    extra_attributes={
                        "operation": "http_server_shutdown",
                        "status": "success"
                    }
                )
            except Exception as e:
                self.logger.error_with_context(
                    "Error stopping HTTP server",
                    extra_attributes={
                        "error.type": type(e).__name__,
                        "error.message": str(e),
                        "operation": "http_server_shutdown",
                        "status": "failed"
                    },
                    exc_info=True
                )

class EnhancedOrchestrator:
    def __init__(self):
        self.tracer, self.meter = initialize_opentelemetry(
            "document-rag-orchestrator", "2.0.0", "production"
        )
        # Add correlated logger for the orchestrator
        self.logger = get_correlated_logger(f"{__name__}.EnhancedOrchestrator")
        
        self.process_manager = ProcessManager(self.tracer)
        self.http_server = OrchestratorHTTPServer(self.tracer)
        self.is_running = False
        self.startup_time = datetime.now()

        self.service_configs = {
            "backend": ServiceConfig(
                name="document-rag-backend",
                command=[sys.executable, "backend_service.py", "--host", "0.0.0.0", "--port", "8001"],
                port=8001,
                environment={
                    "HOST": "0.0.0.0", 
                    "PORT": "8001"
                }
            ),
            "api": ServiceConfig(
                name="document-rag-api", 
                command=[sys.executable, "start_server.py"],
                port=8000,
                environment={
                    "HOST": "0.0.0.0",
                    "PORT": "8000",
                    "BACKEND_SERVICE_URL": "http://localhost:8001",
                    "SERVER_HOST": "0.0.0.0",
                    "SERVER_PORT": "8000"
                }
            )
        }

        self.logger.info_with_context(
            "Enhanced Orchestrator initialized",
            extra_attributes={
                "orchestrator.version": "2.0.0",
                "services.configured": len(self.service_configs),
                "services.list": list(self.service_configs.keys()),
                "working_dir": str(Path(__file__).parent),
                "operation": "orchestrator_init"
            }
        )

    def check_environment(self) -> bool:
        with self.tracer.start_as_current_span("environment_check") as span:
            span.set_attributes({
                "service.name": "document-rag-orchestrator",
                "operation.name": "environment_check"
            })
            
            self.logger.info_with_context(
                "Starting environment validation",
                extra_attributes={
                    "operation": "environment_check"
                }
            )
            
            required_vars = ["OPENAI_API_KEY", "OTEL_EXPORTER_OTLP_ENDPOINT"]
            missing = [var for var in required_vars if not os.getenv(var)]
            
            # Check critical environment variables
            env_check = {
                "OPENAI_API_KEY": "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Missing",
                "OTEL_EXPORTER_OTLP_ENDPOINT": f"✅ {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}" if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") else "❌ Missing",
                "GOOGLE_DRIVE_FOLDER_ID": f"✅ {os.getenv('GOOGLE_DRIVE_FOLDER_ID')}" if os.getenv("GOOGLE_DRIVE_FOLDER_ID") else "⚠️ Optional",
                "EMBEDDING_MODEL": f"✅ {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')}",
                "CHUNK_SIZE": f"✅ {os.getenv('CHUNK_SIZE', '3000')}",
                "CHUNK_OVERLAP": f"✅ {os.getenv('CHUNK_OVERLAP', '300')}",
            }
            
            print("🔍 Environment Check:")
            for var, status in env_check.items():
                print(f"   {var}: {status}")
            
            # Test OTEL connectivity
            otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if otel_endpoint:
                try:
                    import requests
                    # Try to connect to OTEL endpoint (without authentication)
                    # Just check if the host is reachable
                    host = otel_endpoint.replace("http://", "").replace("https://", "").split(":")[0]
                    port = otel_endpoint.split(":")[-1] if ":" in otel_endpoint else "4317"
                    
                    # Test OTEL connectivity
                    print(f"🔗 Testing OTEL connectivity to {host}:{port}...")
                    
                    # Create a test span to verify OTEL is working
                    with self.tracer.start_as_current_span("otel_connectivity_test") as test_span:
                        test_span.set_attributes({
                            "test.type": "connectivity",
                            "otel.endpoint": otel_endpoint,
                            "service.name": "document-rag-orchestrator",
                            "test.timestamp": datetime.now().isoformat()
                        })
                        
                        test_trace_id = format(test_span.get_span_context().trace_id, '032x')
                        print(f"✅ OTEL test span created")
                        print(f"   Test Trace ID: {test_trace_id}")
                        print(f"   Should appear in Kibana within 30 seconds")
                        
                        # Force flush to send immediately
                        try:
                            from opentelemetry.sdk.trace import TracerProvider
                            from opentelemetry.sdk.trace.export import BatchSpanProcessor
                            
                            # Get the tracer provider and force flush
                            tracer_provider = trace.get_tracer_provider()
                            if hasattr(tracer_provider, '_active_span_processor'):
                                tracer_provider._active_span_processor.force_flush(timeout_millis=5000)
                                print(f"✅ OTEL spans flushed to endpoint")
                        except Exception as flush_error:
                            print(f"⚠️ Could not force flush OTEL spans: {flush_error}")
                        
                except Exception as e:
                    print(f"⚠️ OTEL connectivity test failed: {e}")
                    print(f"   This may affect service map visibility in Kibana")
                    self.logger.warning_with_context(
                        "OTEL connectivity test failed",
                        extra_attributes={
                            "otel.endpoint": otel_endpoint,
                            "error.type": type(e).__name__,
                            "error.message": str(e),
                            "operation": "environment_check"
                        }
                    )
            
            if missing:
                self.logger.error_with_context(
                    "Environment validation failed - missing required variables",
                    extra_attributes={
                        "missing_variables": missing,
                        "required_variables": required_vars,
                        "operation": "environment_check",
                        "status": "failed"
                    }
                )
                print(f"❌ Missing required variables: {', '.join(missing)}")
                return False
            
            self.logger.info_with_context(
                "Environment validation passed",
                extra_attributes={
                    "required_variables": required_vars,
                    "environment_status": env_check,
                    "operation": "environment_check",
                    "status": "success"
                }
            )
            
            print("✅ Environment validation passed")
            return True

    async def start_system(self) -> bool:
        with self.tracer.start_as_current_span("system_startup") as span:
            self.logger.info_with_context(
                "Starting Document RAG System",
                extra_attributes={
                    "system.component": "orchestrator",
                    "startup.phase": "initialization",
                    "services.count": len(self.service_configs),
                    "operation": "system_startup"
                }
            )
            
            print("🔥 Starting Document RAG System")
            print("=" * 70)
            
            if not self.check_environment():
                self.logger.error_with_context(
                    "Environment validation failed",
                    extra_attributes={
                        "startup.phase": "environment_validation",
                        "status": "failed",
                        "operation": "system_startup"
                    }
                )
                return False
            
            # Start HTTP server first
            try:
                await self.http_server.start()
                self.logger.info_with_context(
                    "Orchestrator HTTP server started",
                    extra_attributes={
                        "server.port": 8002,
                        "server.url": "http://localhost:8002",
                        "startup.phase": "http_server",
                        "operation": "system_startup"
                    }
                )
            except Exception as e:
                self.logger.error_with_context(
                    "Failed to start HTTP server",
                    extra_attributes={
                        "startup.phase": "http_server",
                        "error.type": type(e).__name__,
                        "error.message": str(e),
                        "operation": "system_startup",
                        "status": "failed"
                    },
                    exc_info=True
                )
                return False
            
            # Ensure we're in the correct working directory
            original_cwd = os.getcwd()
            target_dir = Path(__file__).parent
            if original_cwd != str(target_dir):
                print(f"📁 Changing working directory: {original_cwd} -> {target_dir}")
                os.chdir(target_dir)
            
            # Start each service with logging
            services_started = 0
            for service_name in ["backend", "api"]:
                try:
                    print(f"\n🔄 Starting service: {service_name}")
                    
                    self.logger.info_with_context(
                        f"Starting service: {service_name}",
                        extra_attributes={
                            "service.name": service_name,
                            "service.port": self.service_configs[service_name].port,
                            "startup.phase": "service_startup",
                            "services.started": services_started,
                            "services.remaining": len(self.service_configs) - services_started,
                            "operation": "system_startup"
                        }
                    )
                    
                    self.process_manager.start_service(self.service_configs[service_name])
                    await asyncio.sleep(6)  # Give more time for service to fully start
                    services_started += 1
                    
                    self.logger.info_with_context(
                        f"Service started successfully: {service_name}",
                        extra_attributes={
                            "service.name": service_name,
                            "service.port": self.service_configs[service_name].port,
                            "startup.phase": "service_startup",
                            "services.started": services_started,
                            "services.total": len(self.service_configs),
                            "status": "success",
                            "operation": "system_startup"
                        }
                    )
                    
                    print(f"✅ {service_name} is ready\n")
                    
                except Exception as e:
                    self.logger.error_with_context(
                        f"Failed to start service: {service_name}",
                        extra_attributes={
                            "service.name": service_name,
                            "service.port": self.service_configs[service_name].port,
                            "startup.phase": "service_startup",
                            "services.started": services_started,
                            "status": "failed",
                            "error.type": type(e).__name__,
                            "error.message": str(e),
                            "operation": "system_startup"
                        },
                        exc_info=True
                    )
                    
                    print(f"❌ Failed to start {service_name}: {e}")
                    
                    # Cleanup on failure
                    self.logger.info_with_context(
                        "Initiating cleanup due to service startup failure",
                        extra_attributes={
                            "failed_service": service_name,
                            "services.started": services_started,
                            "operation": "startup_cleanup"
                        }
                    )
                    
                    self.process_manager.terminate_all()
                    return False
            
            self.is_running = True
            self.display_system_info()
            self.setup_signal_handlers()
            
            self.logger.info_with_context(
                "Document RAG System startup completed",
                extra_attributes={
                    "system.component": "orchestrator",
                    "startup.phase": "completed",
                    "status": "success",
                    "services.running": len(self.service_configs),
                    "startup.duration_seconds": (datetime.now() - self.startup_time).total_seconds(),
                    "operation": "system_startup"
                }
            )
            
            print("🎉 System startup completed!")
            return True

    def display_system_info(self):
        # Get current trace context
        current_span = trace.get_current_span()
        trace_id = "unknown"
        if current_span != trace.INVALID_SPAN:
            trace_id = format(current_span.get_span_context().trace_id, '032x')
        
        self.logger.info_with_context(
            "System information display",
            extra_attributes={
                "orchestrator.url": "http://localhost:8002",
                "api.url": "http://localhost:8000", 
                "backend.url": "http://localhost:8001",
                "trace.id": trace_id,
                "system.status": "ready",
                "operation": "system_info"
            }
        )
        
        print("\n" + "="*70)
        print("✅ Document RAG System Ready!")
        print("=" * 70)
        print("📡 Orchestrator HTTP: http://localhost:8002")
        print("📊 Main UI: http://localhost:8000")
        print("⚙️  Backend: http://localhost:8001")
        print(f"🆔 System Trace ID: {trace_id}")
        print(f"📤 OTEL Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
        print("=" * 70)
        print("📝 Service logs will appear below:")
        print("-" * 70)
        
        # Show process information
        for name, proc in self.process_manager.processes.items():
            print(f"🔧 {name}: PID {proc.pid}, Port {proc.config.port}, Status {proc.status.value}")
        print("-" * 70)

    def setup_signal_handlers(self):
        def signal_handler(signum, frame):
            signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            
            self.logger.info_with_context(
                f"Shutdown signal received: {signal_name}",
                extra_attributes={
                    "signal.name": signal_name,
                    "signal.number": signum,
                    "operation": "signal_handling"
                }
            )
            
            print(f"\n🛑 Shutdown signal received ({signal_name})...")
            self.is_running = False
            self.process_manager.terminate_all()
            asyncio.create_task(self.http_server.stop())
            
            self.logger.info_with_context(
                "System shutdown initiated via signal",
                extra_attributes={
                    "signal.name": signal_name,
                    "operation": "system_shutdown"
                }
            )
            
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        self.logger.debug_with_context(
            "Signal handlers configured",
            extra_attributes={
                "signals": ["SIGINT", "SIGTERM"],
                "operation": "signal_setup"
            }
        )

async def main():
    startup_logger = get_correlated_logger("startup")
    
    # Initialize the tracer early to get proper trace context
    tracer, meter = initialize_opentelemetry(
        "document-rag-orchestrator", "2.0.0", "production"
    )
    
    with tracer.start_as_current_span("orchestrator_main_startup") as main_span:
        main_span.set_attributes({
            "service.name": "document-rag-orchestrator",
            "service.version": "2.0.0",
            "operation.name": "main_startup"
        })
        
        trace_id = format(main_span.get_span_context().trace_id, '032x')
        span_id = format(main_span.get_span_context().span_id, '016x')
        
        startup_logger.info_with_context(
            "Orchestrator starting up",
            extra_attributes={
                "orchestrator.version": "2.0",
                "trace.id": trace_id,
                "span.id": span_id,
                "otel.endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                "operation": "main_startup"
            }
        )
        
        print("🚀 ORCHESTRATOR v2.0 with HTTP Server")
        print(f"🆔 Main Trace ID: {trace_id}")
        print(f"📡 OTEL Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
        print()
        
        orchestrator = EnhancedOrchestrator()
        
        try:
            success = await orchestrator.start_system()
            
            if not success:
                startup_logger.error_with_context(
                    "Orchestrator startup failed",
                    extra_attributes={
                        "operation": "main_startup",
                        "status": "failed"
                    }
                )
                print("❌ Startup failed")
                sys.exit(1)
            
            startup_logger.info_with_context(
                "Orchestrator startup completed successfully",
                extra_attributes={
                    "operation": "main_startup",
                    "status": "success"
                }
            )
            
            print("🔄 System running... (Press Ctrl+C to stop)")
            
            # Main orchestrator loop with periodic heartbeats
            heartbeat_count = 0
            while orchestrator.is_running:
                with orchestrator.tracer.start_as_current_span("orchestrator.heartbeat") as heartbeat_span:
                    heartbeat_count += 1
                    
                    heartbeat_span.set_attributes({
                        "service.name": "document-rag-orchestrator",
                        "heartbeat.count": heartbeat_count,
                        "operation.name": "orchestrator_heartbeat"
                    })
                    
                    if heartbeat_count % 6 == 0:  # Log every minute (6 * 10 seconds)
                        current_trace = format(heartbeat_span.get_span_context().trace_id, '032x')
                        
                        orchestrator.logger.debug_with_context(
                            "Orchestrator heartbeat",
                            extra_attributes={
                                "heartbeat.count": heartbeat_count,
                                "system.uptime_seconds": (datetime.now() - orchestrator.startup_time).total_seconds(),
                                "services.running": len(orchestrator.process_manager.processes),
                                "trace.current": current_trace,
                                "operation": "orchestrator_heartbeat"
                            }
                        )
                        
                        print(f"💓 Orchestrator heartbeat #{heartbeat_count} (Trace: {current_trace[:8]}...)")
                    
                    await asyncio.sleep(10)
                    
        except KeyboardInterrupt:
            orchestrator.logger.info_with_context(
                "Shutdown requested via keyboard interrupt",
                extra_attributes={
                    "operation": "main_shutdown",
                    "shutdown.reason": "keyboard_interrupt"
                }
            )
            print("\n🔄 Shutdown requested")
        except Exception as e:
            orchestrator.logger.error_with_context(
                "Unexpected error in main loop",
                extra_attributes={
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "operation": "main_loop",
                    "status": "unexpected_error"
                },
                exc_info=True
            )
            print(f"\n❌ Unexpected error: {e}")
        finally:
            orchestrator.logger.info_with_context(
                "Initiating final cleanup",
                extra_attributes={
                    "operation": "main_cleanup"
                }
            )
            
            orchestrator.process_manager.terminate_all()
            await orchestrator.http_server.stop()
            
            orchestrator.logger.info_with_context(
                "Orchestrator shutdown completed",
                extra_attributes={
                    "operation": "main_cleanup",
                    "status": "completed"
                }
            )

if __name__ == "__main__":
    asyncio.run(main())