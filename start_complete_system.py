#!/usr/bin/env python3
"""
Enhanced Orchestrator with HTTP Server for Service Map Connectivity
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

from otel_config import (
    initialize_opentelemetry, get_service_tracer,
    get_current_trace_id, extract_and_activate_context, propagate
)
from opentelemetry.trace import SpanKind
from opentelemetry.context import attach, detach

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
        with self.tracer.start_as_current_span(
            f"start_{config.name}",
            kind=SpanKind.INTERNAL
        ) as span:
            span.set_attributes({
                "service.child": config.name,
                "service.port": config.port
            })
            
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

            print(f"🚀 Starting {config.name}")
            
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
                print(f"✅ {config.name} started (PID: {process.pid})")
                return service_proc
            else:
                raise RuntimeError(f"Failed to start {config.name}")

    def terminate_all(self):
        print("🛑 Initiating graceful shutdown...")
        for name, svc_proc in self.processes.items():
            if svc_proc.process.poll() is None:
                print(f"⏹️  Stopping {name}...")
                svc_proc.process.terminate()
                try:
                    svc_proc.process.wait(timeout=10)
                    print(f"✅ {name} stopped")
                except subprocess.TimeoutExpired:
                    svc_proc.process.kill()
                    print(f"🔥 {name} force terminated")

class OrchestratorHTTPServer:
    """HTTP server for orchestrator to receive calls from children"""
    
    def __init__(self, tracer):
        self.tracer = tracer
        self.app = web.Application()
        self.setup_routes()
        self.runner = None
        
    def setup_routes(self):
        self.app.router.add_get('/health', self.health_handler)
        self.app.router.add_post('/heartbeat', self.heartbeat_handler)
        self.app.router.add_get('/status', self.status_handler)
        
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
                
                span.set_attributes({
                    "http.method": "POST",
                    "http.route": "/heartbeat",
                    "heartbeat.from": service_name,
                    "http.status_code": 200
                })
                
                return web.json_response({
                    "status": "acknowledged",
                    "service": "document-rag-orchestrator"
                })
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
                
                return web.json_response({
                    "status": "running",
                    "service": "document-rag-orchestrator",
                    "timestamp": datetime.now().isoformat()
                })
        finally:
            detach(token)
    
    async def start(self):
        """Start HTTP server"""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', 8002)
        await site.start()
        print("📡 Orchestrator HTTP server started on http://localhost:8002")
    
    async def stop(self):
        """Stop HTTP server"""
        if self.runner:
            await self.runner.cleanup()

class EnhancedOrchestrator:
    def __init__(self):
        self.tracer, self.meter = initialize_opentelemetry(
            "document-rag-orchestrator", "2.0.0", "production"
        )
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

    def check_environment(self) -> bool:
        with self.tracer.start_as_current_span("environment_check") as span:
            required_vars = ["OPENAI_API_KEY", "OTEL_EXPORTER_OTLP_ENDPOINT"]
            missing = [var for var in required_vars if not os.getenv(var)]
            
            if missing:
                print(f"❌ Missing: {', '.join(missing)}")
                return False
            
            print("✅ Environment validation passed")
            return True

    async def start_system(self) -> bool:
        with self.tracer.start_as_current_span("system_startup") as span:
            print("🔥 Starting Document RAG System")
            print("=" * 70)
            
            if not self.check_environment():
                return False
            
            # Start HTTP server first
            await self.http_server.start()
            
            os.chdir(Path(__file__).parent)
            
            # Start services
            for service_name in ["backend", "api"]:
                try:
                    self.process_manager.start_service(self.service_configs[service_name])
                    await asyncio.sleep(4)
                except Exception as e:
                    print(f"❌ Failed to start {service_name}: {e}")
                    self.process_manager.terminate_all()
                    return False
            
            self.is_running = True
            self.display_system_info()
            self.setup_signal_handlers()
            
            print("🎉 System startup completed!")
            return True

    def display_system_info(self):
        print("\n✅ Document RAG System Ready!")
        print("📡 Orchestrator HTTP: http://localhost:8002")
        print("📊 Main UI: http://localhost:8000")
        print("⚙️  Backend: http://localhost:8001")
        print(f"🆔 Trace ID: {get_current_trace_id()}")
        print("=" * 70)

    def setup_signal_handlers(self):
        def signal_handler(signum, frame):
            print("\n🛑 Shutdown signal received...")
            self.is_running = False
            self.process_manager.terminate_all()
            asyncio.create_task(self.http_server.stop())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    print("🚀 ORCHESTRATOR v2.0 with HTTP Server")
    print()
    
    orchestrator = EnhancedOrchestrator()
    
    try:
        success = await orchestrator.start_system()
        
        if not success:
            print("❌ Startup failed")
            sys.exit(1)
        
        print("🔄 System running...")
        
        while orchestrator.is_running:
            with orchestrator.tracer.start_as_current_span("orchestrator.heartbeat"):
                await asyncio.sleep(10)
                
    except KeyboardInterrupt:
        print("\n🔄 Shutdown requested")
    finally:
        orchestrator.process_manager.terminate_all()
        await orchestrator.http_server.stop()

if __name__ == "__main__":
    asyncio.run(main())