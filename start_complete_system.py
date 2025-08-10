#!/usr/bin/env python3
"""
Enhanced Orchestrator with Complete W3C Trace Context Propagation
Fixed for proper parent-child service relationships and logging conflicts
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
    # CRITICAL: Disable automatic logging correlation to prevent KeyError
    "OTEL_PYTHON_LOG_CORRELATION": "false"
})

load_dotenv()

from otel_config import (
    initialize_opentelemetry, get_service_tracer,
    get_current_trace_id, SERVICE_HIERARCHY
)

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

    def start_service(self, config: ServiceConfig) -> ServiceProcess:
        with self.tracer.start_as_current_span("start_service") as span:
            span.set_attributes({
                "service.name": config.name,
                "service.port": config.port,
                "service.type": "child_process",
                "service.parent": "document-rag-orchestrator"
            })
            
            # Prepare environment with complete OTEL configuration
            env = os.environ.copy()
            if config.environment:
                env.update(config.environment)

            # CRITICAL: Get complete span context for W3C propagation
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x') 
            trace_flags = format(span_context.trace_flags, '02x')
            
            # Construct W3C traceparent header
            traceparent = f"00-{trace_id}-{span_id}-{trace_flags}"

            # CRITICAL: Pass complete OTEL configuration to child processes
            env.update({
                # W3C Trace Context
                "OTEL_TRACEPARENT": traceparent,
                "OTEL_PARENT_TRACE_ID": trace_id,
                "OTEL_PARENT_SPAN_ID": span_id,
                "OTEL_TRACE_FLAGS": trace_flags,
                
                # Service Configuration  
                "OTEL_SERVICE_NAME": config.name,
                "OTEL_SERVICE_VERSION": "1.0.0",
                "OTEL_ENVIRONMENT": "production",
                
                # OTEL Exporter Configuration
                "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
                "OTEL_EXPORTER_OTLP_INSECURE": "true",
                "OTEL_RESOURCE_ATTRIBUTES": "service.namespace=document-rag-system,deployment.environment=production",
                "OTEL_TRACES_EXPORTER": "otlp",
                "OTEL_METRICS_EXPORTER": "otlp",
                "OTEL_LOGS_EXPORTER": "otlp", 
                "OTEL_TRACES_SAMPLER": "traceidratio",
                "OTEL_TRACES_SAMPLER_ARG": "1.0",
                
                # CRITICAL: Disable logging correlation in child processes to prevent conflicts
                "OTEL_PYTHON_LOG_CORRELATION": "false"
            })

            # Add service hierarchy for proper parent-child relationships
            hierarchy_info = SERVICE_HIERARCHY.get(config.name, {})
            if hierarchy_info.get("parent"):
                env["OTEL_SERVICE_PARENT"] = hierarchy_info["parent"]
                env["OTEL_SERVICE_TYPE"] = hierarchy_info.get("type", "service")

            print(f"🚀 Starting {config.name}")
            print(f"   📊 Trace ID: {trace_id[:16]}...")
            print(f"   🔗 Parent Span: {span_id[:8]}...")

            # Start process with complete trace context
            process = subprocess.Popen(
                config.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=Path(__file__).parent
            )

            time.sleep(3)  # Allow process initialization

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
                span.set_attributes({
                    "startup.success": True,
                    "process.pid": process.pid,
                    "trace.propagated": True
                })
                print(f"✅ {config.name} started (PID: {process.pid})")
                return service_proc
            else:
                span.set_attributes({
                    "startup.success": False,
                    "error.message": "Process failed to start"
                })
                raise RuntimeError(f"Failed to start {config.name}")

    def terminate_all(self):
        with self.tracer.start_as_current_span("terminate_all_services") as span:
            print("🛑 Initiating graceful shutdown...")
            terminated = 0
            
            for name, svc_proc in self.processes.items():
                if svc_proc.process.poll() is None:
                    print(f"⏹️  Stopping {name}...")
                    svc_proc.process.terminate()
                    
                    try:
                        svc_proc.process.wait(timeout=10)
                        terminated += 1
                        print(f"✅ {name} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        svc_proc.process.kill()
                        print(f"🔥 {name} force terminated")
                        terminated += 1
                    
                    svc_proc.status = ServiceStatus.STOPPED
            
            span.set_attributes({
                "termination.count": terminated,
                "termination.success": True
            })
            print(f"✅ {terminated} services terminated")

class EnhancedOrchestrator:
    def __init__(self):
        # Initialize OpenTelemetry with proper orchestrator configuration
        self.tracer, self.meter = initialize_opentelemetry(
            "document-rag-orchestrator", "2.0.0", "production"
        )
        self.process_manager = ProcessManager(self.tracer)
        self.is_running = False
        self.startup_time = datetime.now()

        # Service configurations with complete OTEL setup
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

    def check_environment(self) -> bool:
        with self.tracer.start_as_current_span("environment_validation") as span:
            required_vars = ["OPENAI_API_KEY", "OTEL_EXPORTER_OTLP_ENDPOINT"]
            missing = [var for var in required_vars if not os.getenv(var)]
            
            if missing:
                span.set_attributes({
                    "validation.failed": True,
                    "missing_vars": ",".join(missing)
                })
                print(f"❌ Missing required environment variables: {', '.join(missing)}")
                return False
            
            span.set_attribute("validation.passed", True)
            print("✅ Environment validation passed")
            print(f"📡 OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
            return True

    async def start_system(self) -> bool:
        with self.tracer.start_as_current_span("system_startup") as span:
            span.set_attributes({
                "system.name": "document-rag",
                "system.version": "2.0.0",
                "startup.mode": "orchestrator"
            })
            
            print("🔥 Starting Document RAG System with W3C Trace Propagation")
            print("=" * 70)
            
            # Environment validation
            if not self.check_environment():
                span.set_attribute("startup.failed_reason", "environment_validation")
                return False
            
            # Set working directory
            os.chdir(Path(__file__).parent)
            
            # Start services in dependency order (backend first, then API)
            service_order = ["backend", "api"]
            started_services = []
            
            for service_name in service_order:
                try:
                    with self.tracer.start_as_current_span(f"start_{service_name}") as service_span:
                        service_span.set_attributes({
                            "service.target": service_name,
                            "service.startup_order": len(started_services) + 1
                        })
                        
                        service_process = self.process_manager.start_service(
                            self.service_configs[service_name]
                        )
                        started_services.append(service_name)
                        service_span.set_attribute("service.started", True)
                        
                        # Allow service initialization time
                        await asyncio.sleep(4)
                        
                except Exception as e:
                    span.set_attributes({
                        "startup.failed_reason": f"{service_name}_startup_error",
                        "error.message": str(e)
                    })
                    print(f"❌ Failed to start {service_name}: {e}")
                    
                    # Clean shutdown of already started services
                    self.process_manager.terminate_all()
                    return False
            
            self.is_running = True
            span.set_attributes({
                "startup.success": True,
                "services.started": ",".join(started_services),
                "startup.duration_seconds": (datetime.now() - self.startup_time).total_seconds(),
                "trace.root_id": get_current_trace_id()
            })
            
            # Display system information
            self.display_system_info()
            
            # Setup signal handlers for graceful shutdown
            self.setup_signal_handlers()
            
            print("🎉 System startup completed successfully!")
            return True

    def display_system_info(self):
        current_trace_id = get_current_trace_id()
        
        print()
        print("✅ Document RAG System Ready!")
        print("🔗 W3C Trace Propagation: ACTIVE")
        print("📊 Service Map: Connected Hierarchy")
        print()
        print("🌐 Service Access Points:")
        print("   📊 Main UI:           http://localhost:8000")
        print("   📚 API Documentation: http://localhost:8000/docs") 
        print("   ⚙️  Backend Health:    http://localhost:8001/health")
        print("   🔍 Backend Status:     http://localhost:8001/status")
        print()
        print("📊 Expected Service Hierarchy (Kibana Service Map):")
        print("   document-rag-orchestrator (root)")
        print("   ├── document-rag-backend (processing)")
        print("   └── document-rag-api (frontend)")
        print("       ├── query-processor")
        print("       ├── response-generator")
        print("       ├── session-manager")
        print("       └── backend-proxy")
        print()
        print(f"🆔 Root Trace ID: {current_trace_id}")
        print(f"⏰ System Started: {self.startup_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏷️ Environment: production")
        print()
        print("🔍 Trace Verification Commands:")
        print("   curl http://localhost:8000/api/health")
        print("   curl http://localhost:8001/health")
        print("=" * 70)

    def setup_signal_handlers(self):
        def signal_handler(signum, frame):
            with self.tracer.start_as_current_span("graceful_shutdown") as span:
                print("\n🛑 Graceful shutdown signal received...")
                self.is_running = False
                
                # Terminate all child services
                self.process_manager.terminate_all()
                
                span.set_attributes({
                    "shutdown.graceful": True,
                    "shutdown.signal": signum
                })
                
                print("✅ System shutdown completed")
                sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Docker/systemd stop

async def main():
    print("🚀 ENHANCED ORCHESTRATOR v2.0")
    print("🔗 W3C Trace Context Propagation: ENABLED")
    print("📊 Service Map Generation: ACTIVE")
    print("🛠️  Logging Conflicts: RESOLVED")
    print()
    
    orchestrator = EnhancedOrchestrator()
    
    try:
        with orchestrator.tracer.start_as_current_span("main_execution") as span:
            span.set_attributes({
                "execution.mode": "orchestrator_main",
                "system.name": "document-rag",
                "trace.root": True
            })
            
            # Start the complete system
            success = await orchestrator.start_system()
            
            if not success:
                span.set_attribute("execution.failed", True)
                print("❌ System startup failed - exiting")
                sys.exit(1)
            
            span.set_attribute("execution.success", True)
            
            # Keep system running until shutdown signal
            print("🔄 System running - monitoring services...")
            print("🔍 Check Kibana service map for connected hierarchy")
            
            while orchestrator.is_running:
                await asyncio.sleep(5)
                
    except KeyboardInterrupt:
        print("\n🔄 Keyboard interrupt received")
    except Exception as e:
        print(f"❌ Unexpected system error: {e}")
        sys.exit(1)
    finally:
        # Ensure cleanup on any exit
        if hasattr(orchestrator, 'process_manager'):
            orchestrator.process_manager.terminate_all()
        
        # Shutdown OpenTelemetry
        try:
            from otel_config import shutdown_opentelemetry
            shutdown_opentelemetry()
        except ImportError:
            pass

if __name__ == "__main__":
    asyncio.run(main())
