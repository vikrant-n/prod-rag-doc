#!/usr/bin/env python3
"""
Enhanced Orchestrator with HTTP Server for Service Map Connectivity and Correlated Logging
"""

import os
import sys
import subprocess
import asyncio
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
from aiohttp import web
import threading

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
    get_correlated_logger, enhanced_error_logging # NEW: Import correlated logger
)
from opentelemetry.trace import SpanKind
from opentelemetry.context import attach, detach

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
                "service.port": config.port
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
            
            env = os.environ.copy()
            if config.environment:
                env.update(config.environment)

            # Get span context for propagation
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x') 
            
            # Pass orchestrator URL to children
            env.update({
                "OTEL_PARENT_TRACE_ID": trace_id,
                "OTEL_PARENT_SPAN_ID": span_id,
                "OTEL_SERVICE_NAME": config.name,
                "OTEL_SERVICE_PARENT": "document-rag-orchestrator",
                "ORCHESTRATOR_URL": "http://localhost:8002",  # Orchestrator HTTP endpoint
                "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                "OTEL_PYTHON_LOG_CORRELATION": "false"
            })

            self.logger.debug_with_context(
                f"Environment prepared for service: {config.name}",
                extra_attributes={
                    "service.name": config.name,
                    "trace.parent_id": trace_id,
                    "span.parent_id": span_id,
                    "orchestrator.url": "http://localhost:8002",
                    "operation": "service_startup"
                }
            )

            print(f"🚀 Starting {config.name}")
            
            try:
                process = subprocess.Popen(
                    config.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    cwd=Path(__file__).parent
                )

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
                            "service.port": config.port,
                            "service.status": ServiceStatus.RUNNING.value,
                            "startup.duration_seconds": 3,
                            "operation": "service_startup",
                            "status": "success"
                        }
                    )
                    
                    print(f"✅ {config.name} started (PID: {process.pid})")
                    return service_proc
                else:
                    error_msg = f"Process exited with code {process.poll()}"
                    self.logger.error_with_context(
                        f"Service failed to start: {config.name}",
                        extra_attributes={
                            "service.name": config.name,
                            "service.exit_code": process.poll(),
                            "error.message": error_msg,
                            "operation": "service_startup",
                            "status": "failed"
                        }
                    )
                    raise RuntimeError(f"Failed to start {config.name}: {error_msg}")
                    
            except Exception as e:
                # Replace existing error logging
                enhanced_error_logging(
                    self.logger,
                    f"Exception during service startup: {config.name}",
                    extra_attributes={
                        "service.name": config.name,
                        "operation": "service_startup",
                        "status": "exception"
                    }
                )
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
                environment={"HOST": "0.0.0.0", "PORT": "8001"}
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
                "operation": "orchestrator_init"
            }
        )

    def check_environment(self) -> bool:
        with self.tracer.start_as_current_span("environment_check") as span:
            self.logger.info_with_context(
                "Starting environment validation",
                extra_attributes={
                    "operation": "environment_check"
                }
            )
            
            required_vars = ["OPENAI_API_KEY", "OTEL_EXPORTER_OTLP_ENDPOINT"]
            missing = [var for var in required_vars if not os.getenv(var)]
            
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
                print(f"❌ Missing: {', '.join(missing)}")
                return False
            
            # Log available environment variables (without values for security)
            env_status = {}
            for var in required_vars:
                env_status[var] = "set" if os.getenv(var) else "missing"
            
            self.logger.info_with_context(
                "Environment validation passed",
                extra_attributes={
                    "required_variables": required_vars,
                    "environment_status": env_status,
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
            
            os.chdir(Path(__file__).parent)
            
            # Start each service with logging
            services_started = 0
            for service_name in ["backend", "api"]:
                try:
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
                    await asyncio.sleep(4)
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
        trace_id = get_current_trace_id()
        
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
        
        print("\n✅ Document RAG System Ready!")
        print("📡 Orchestrator HTTP: http://localhost:8002")
        print("📊 Main UI: http://localhost:8000")
        print("⚙️  Backend: http://localhost:8001")
        print(f"🆔 Trace ID: {trace_id}")
        print("=" * 70)

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
    
    startup_logger.info_with_context(
        "Orchestrator starting up",
        extra_attributes={
            "orchestrator.version": "2.0",
            "operation": "main_startup"
        }
    )
    
    print("🚀 ORCHESTRATOR v2.0 with HTTP Server")
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
        
        print("🔄 System running...")
        
        # Main orchestrator loop with periodic heartbeats
        heartbeat_count = 0
        while orchestrator.is_running:
            with orchestrator.tracer.start_as_current_span("orchestrator.heartbeat"):
                heartbeat_count += 1
                
                if heartbeat_count % 6 == 0:  # Log every minute (6 * 10 seconds)
                    orchestrator.logger.debug_with_context(
                        "Orchestrator heartbeat",
                        extra_attributes={
                            "heartbeat.count": heartbeat_count,
                            "system.uptime_seconds": (datetime.now() - orchestrator.startup_time).total_seconds(),
                            "services.running": len(orchestrator.process_manager.processes),
                            "operation": "orchestrator_heartbeat"
                        }
                    )
                
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
