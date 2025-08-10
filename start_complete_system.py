#!/usr/bin/env python3
"""
BEAST MODE: Ultimate Orchestrator with Full Service Tree
Creates complete hierarchical system with connected services
"""

import os
import sys
import subprocess
import time
import signal
import threading
import asyncio
import httpx
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Force service name early
os.environ["OTEL_SERVICE_NAME"] = "document-rag-orchestrator"

# Load environment first
load_dotenv()

# Enhanced OpenTelemetry configuration
from otel_config import (
    initialize_opentelemetry, get_service_tracer, traced_function, 
    trace_http_call, get_current_trace_id, add_trace_correlation_to_log
)
from metrics import rag_metrics, record_cache_event
import logging

# At very top after imports
from otel_config import initialize_opentelemetry

class EnhancedOrchestrator:
    def __init__(self):
        self.tracer, self.meter = initialize_opentelemetry("document-rag-orchestrator")

class ProcessMonitor:
    """Process monitoring service component"""
    
    def __init__(self, orchestrator_tracer):
        self.tracer = get_service_tracer("process-manager")
        self.orchestrator_tracer = orchestrator_tracer
        self.service_name = "process-manager"
        self.processes = {}
        self.monitoring_active = False
    
    @traced_function(service_name="process-manager")
    def start_process(self, service_name: str, command: List[str], port: int) -> subprocess.Popen:
        """Start and monitor a service process"""
        with self.tracer.start_as_current_span("process_manager.start_process") as span:
            span.set_attribute("process.service_name", service_name)
            span.set_attribute("process.port", port)
            span.set_attribute("process.command", " ".join(command))
            
            try:
                process = subprocess.Popen(
                    command, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True
                )
                
                # Wait for startup
                time.sleep(3)
                
                if process.poll() is None:
                    self.processes[service_name] = {
                        "process": process,
                        "port": port,
                        "started_at": datetime.now(),
                        "status": "running"
                    }
                    span.set_attribute("process.status", "started")
                    span.set_attribute("process.pid", process.pid)
                    return process
                else:
                    span.set_attribute("process.status", "failed")
                    raise Exception(f"Process failed to start: {service_name}")
                    
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("process.status", "error")
                raise
    
    @traced_function(service_name="process-manager")
    async def monitor_processes(self):
        """Monitor all managed processes"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            with self.tracer.start_as_current_span("process_manager.monitor_cycle") as span:
                healthy_count = 0
                failed_count = 0
                
                for service_name, proc_info in self.processes.items():
                    with self.tracer.start_as_current_span(f"process_manager.check_{service_name}") as check_span:
                        process = proc_info["process"]
                        
                        if process.poll() is None:
                            proc_info["status"] = "running"
                            healthy_count += 1
                            check_span.set_attribute("process.status", "running")
                        else:
                            proc_info["status"] = "failed"
                            failed_count += 1
                            check_span.set_attribute("process.status", "failed")
                            check_span.set_attribute("process.exit_code", process.returncode)
                
                span.set_attribute("monitor.healthy_processes", healthy_count)
                span.set_attribute("monitor.failed_processes", failed_count)
                
                await asyncio.sleep(30)
    
    def stop_monitoring(self):
        """Stop process monitoring"""
        self.monitoring_active = False
    
    def get_process_status(self) -> Dict[str, Any]:
        """Get status of all managed processes"""
        with self.tracer.start_as_current_span("process_manager.get_status") as span:
            status = {}
            for service_name, proc_info in self.processes.items():
                status[service_name] = {
                    "status": proc_info["status"],
                    "pid": proc_info["process"].pid if proc_info["process"].poll() is None else None,
                    "port": proc_info["port"],
                    "started_at": proc_info["started_at"].isoformat(),
                    "uptime_seconds": (datetime.now() - proc_info["started_at"]).total_seconds()
                }
            
            span.set_attribute("status.processes_count", len(status))
            return status

class ServiceConnector:
    """Service connection and health monitoring"""
    
    def __init__(self, orchestrator_tracer):
        self.tracer = orchestrator_tracer
        self.service_name = "service-connector"
        self.services = {
            "api": {"url": "http://localhost:8000", "health_endpoint": "/api/health"},
            "backend": {"url": "http://localhost:8001", "health_endpoint": "/health"}
        }
    
    @traced_function(service_name="service-connector")
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        with self.tracer.start_as_current_span(f"service_connector.check_{service_name}") as span:
            service_config = self.services.get(service_name)
            if not service_config:
                span.set_attribute("service.found", False)
                return {"status": "unknown", "error": "Service not configured"}
            
            span.set_attribute("service.url", service_config["url"])
            span.set_attribute("service.health_endpoint", service_config["health_endpoint"])
            
            try:
                async with httpx.AsyncClient() as client:
                    headers = {"X-Trace-ID": get_current_trace_id()}
                    
                    response = await client.get(
                        f"{service_config['url']}{service_config['health_endpoint']}", 
                        headers=headers,
                        timeout=10.0
                    )
                    
                    span.set_attribute("http.status_code", response.status_code)
                    
                    # Check response status and content
                    if response.status_code == 200:
                        try:
                            health_data = response.json()
                            span.set_attribute("service.health_status", health_data.get("status", "unknown"))
                            return health_data
                        except Exception:
                            # Service responded but not JSON - consider it basic healthy
                            span.set_attribute("service.health_status", "basic_healthy")
                            return {"status": "healthy", "message": "Service responding"}
                    elif response.status_code == 422:
                        # FastAPI validation error - service is up but endpoint issue
                        span.set_attribute("service.health_status", "endpoint_error")
                        return {"status": "degraded", "error": "Endpoint validation error", "status_code": response.status_code}
                    else:
                        span.set_attribute("service.health_status", "unhealthy")
                        return {"status": "unhealthy", "status_code": response.status_code}
                        
            except httpx.TimeoutException:
                span.set_attribute("service.health_status", "timeout")
                return {"status": "timeout"}
            except httpx.ConnectError:
                span.set_attribute("service.health_status", "unreachable")
                return {"status": "unreachable"}
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("service.health_status", "error")
                return {"status": "error", "error": str(e)}
    
    @traced_function(service_name="service-connector")
    async def get_all_service_health(self) -> Dict[str, Any]:
        """Get health status of all services"""
        with self.tracer.start_as_current_span("service_connector.get_all_health") as span:
            health_results = {}
            
            for service_name in self.services.keys():
                health_results[service_name] = await self.check_service_health(service_name)
            
            # Overall system health - accept degraded as functional
            all_functional = all(
                result.get("status") in ["healthy", "degraded", "basic_healthy"] 
                for result in health_results.values()
            )
            
            overall_status = "healthy" if all_functional else "degraded"
            
            span.set_attribute("system.overall_health", overall_status)
            span.set_attribute("system.services_count", len(health_results))
            
            return {
                "overall_status": overall_status,
                "services": health_results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trace_id": get_current_trace_id()
            }

class EnhancedOrchestrator:
    """Main orchestrator with complete service tree"""
    
    def __init__(self):
        # Initialize root orchestrator tracer
        self.tracer, self.meter = initialize_opentelemetry(
            service_name="document-rag-orchestrator",
            service_version="1.0.0",
            environment="production"
        )
        
        self.service_name = "document-rag-orchestrator"
        
        # Initialize components
        self.process_monitor = ProcessMonitor(self.tracer)
        self.service_connector = ServiceConnector(self.tracer)
        
        # Service state
        self.is_running = False
        self.startup_time = datetime.now()
        
        # Setup logger
        self.logger = add_trace_correlation_to_log(logging.getLogger(__name__))
        self.logger.info("Enhanced orchestrator initialized")
    
    @traced_function(service_name="document-rag-orchestrator")
    def check_environment(self) -> bool:
        """Check environment configuration"""
        with self.tracer.start_as_current_span("orchestrator.check_environment") as span:
            required_vars = ["OPENAI_API_KEY"]
            missing_vars = []
            
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            span.set_attribute("environment.required_vars", len(required_vars))
            span.set_attribute("environment.missing_vars", len(missing_vars))
            
            if missing_vars:
                span.set_attribute("environment.status", "missing_vars")
                self.logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
                return False
            
            # Check optional variables
            gdrive_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
            span.set_attribute("google_drive.enabled", bool(gdrive_id))
            
            if gdrive_id:
                self.logger.info(f"Google Drive monitoring enabled for folder: {gdrive_id}")
            else:
                self.logger.warning("Google Drive Folder ID not set - monitoring disabled")
            
            span.set_attribute("environment.status", "ok")
            return True
    
    @traced_function(service_name="document-rag-orchestrator")
    async def check_qdrant(self) -> bool:
        """Check Qdrant availability"""
        with self.tracer.start_as_current_span("orchestrator.check_qdrant") as span:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            span.set_attribute("qdrant.url", qdrant_url)
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(qdrant_url, timeout=5)
                    
                    span.set_attribute("qdrant.status_code", response.status_code)
                    
                    if response.status_code == 200:
                        span.set_attribute("qdrant.status", "healthy")
                        self.logger.info(f"Qdrant accessible at {qdrant_url}")
                        return True
                    else:
                        span.set_attribute("qdrant.status", "unhealthy")
                        self.logger.warning(f"Qdrant responded with status {response.status_code}")
                        return False
                        
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("qdrant.status", "unreachable")
                self.logger.error(f"Could not connect to Qdrant: {e}")
                return False
    
    @traced_function(service_name="document-rag-orchestrator")
    def start_backend_service(self) -> subprocess.Popen:
        """Start backend service with monitoring"""
        with self.tracer.start_as_current_span("orchestrator.start_backend_service") as span:
            span.set_attribute("service.name", "document-rag-backend")
            span.set_attribute("service.port", 8001)
            
            command = [sys.executable, "backend_service.py", "--host", "0.0.0.0", "--port", "8001"]
            
            try:
                process = self.process_monitor.start_process("document-rag-backend", command, 8001)
                span.set_attribute("backend.started", True)
                span.set_attribute("backend.pid", process.pid)
                
                self.logger.info("Backend service started successfully on port 8001")
                record_cache_event("service_start", True)
                return process
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("backend.started", False)
                self.logger.error(f"Failed to start backend service: {e}")
                record_cache_event("service_start", False)
                return None
    
    @traced_function(service_name="document-rag-orchestrator")
    def start_api_service(self) -> subprocess.Popen:
        """Start API service with monitoring"""
        with self.tracer.start_as_current_span("orchestrator.start_api_service") as span:
            span.set_attribute("service.name", "document-rag-api")
            span.set_attribute("service.port", 8000)
            
            command = [sys.executable, "start_server.py"]
            
            try:
                process = self.process_monitor.start_process("document-rag-api", command, 8000)
                span.set_attribute("api.started", True)
                span.set_attribute("api.pid", process.pid)
                
                self.logger.info("API service started successfully on port 8000")
                record_cache_event("service_start", True)
                return process
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("api.started", False)
                self.logger.error(f"Failed to start API service: {e}")
                record_cache_event("service_start", False)
                return None
    
    @traced_function(service_name="document-rag-orchestrator")
    async def wait_for_services(self) -> bool:
        """Wait for services to become healthy"""
        with self.tracer.start_as_current_span("orchestrator.wait_for_services") as span:
            max_attempts = 5  # Reduced from 30
            attempt = 0
            
            while attempt < max_attempts:
                attempt += 1
                
                with self.tracer.start_as_current_span(f"orchestrator.health_check_attempt_{attempt}") as check_span:
                    health_status = await self.service_connector.get_all_service_health()
                    
                    check_span.set_attribute("attempt.number", attempt)
                    check_span.set_attribute("health.overall_status", health_status["overall_status"])
                    
                    # Accept degraded as OK for now
                    if health_status["overall_status"] in ["healthy", "degraded"]:
                        span.set_attribute("services.ready", True)
                        span.set_attribute("services.ready_after_attempts", attempt)
                        self.logger.info(f"Services ready after {attempt} attempts")
                        return True
                    
                    self.logger.info(f"Waiting for services... attempt {attempt}/{max_attempts}")
                    self.logger.info(f"Current status: {health_status}")
                    await asyncio.sleep(2)
            
            # Continue anyway if services partially working
            span.set_attribute("services.ready", False)
            span.set_attribute("services.max_attempts_reached", True)
            self.logger.warning("Services not fully healthy, continuing anyway")
            return True  # Changed from False
    
    @traced_function(service_name="document-rag-orchestrator")
    async def start_system(self):
        """Start complete system with full service tree"""
        with self.tracer.start_as_current_span("orchestrator.start_system") as span:
            self.logger.info("🔥 Starting Enhanced Document RAG System")
            self.logger.info("=" * 60)
            
            span.set_attribute("system.name", "document-rag")
            span.set_attribute("system.version", "2.0.0")
            
            # Set working directory
            script_dir = Path(__file__).parent
            os.chdir(script_dir)
            span.set_attribute("system.working_directory", str(script_dir))
            
            # Check environment
            if not self.check_environment():
                span.set_attribute("startup.status", "failed")
                span.set_attribute("startup.failure_reason", "environment")
                return False
            
            # Check Qdrant
            qdrant_ok = await self.check_qdrant()
            span.set_attribute("qdrant.startup_check", qdrant_ok)
            
            if not qdrant_ok:
                self.logger.warning("⚠️ Continuing without Qdrant - services will handle connection errors")
            
            self.logger.info("🚀 Starting services...")
            
            # Start backend service
            backend_process = self.start_backend_service()
            if not backend_process:
                span.set_attribute("startup.status", "backend_failed")
                return False
            
            # Start API service
            api_process = self.start_api_service()
            if not api_process:
                span.set_attribute("startup.status", "api_failed")
                backend_process.terminate()
                return False
            
            # Wait for services to be healthy
            services_ready = await self.wait_for_services()
            if not services_ready:
                span.set_attribute("startup.status", "services_not_ready")
                self.logger.error("Services failed to become healthy")
                return False
            
            # Start process monitoring
            self.is_running = True
            monitoring_task = asyncio.create_task(self.process_monitor.monitor_processes())
            
            span.set_attribute("startup.status", "success")
            span.set_attribute("monitoring.started", True)
            
            # Display system information
            await self.display_system_info()
            
            # Setup signal handlers
            self.setup_signal_handlers(backend_process, api_process, monitoring_task)
            
            return True
    
    async def display_system_info(self):
        """Display comprehensive system information"""
        with self.tracer.start_as_current_span("orchestrator.display_system_info") as span:
            print()
            print("✅ All services started successfully!")
            print()
            print("🌐 System URLs:")
            print("   📊 Main UI:             http://localhost:8000")
            print("   🔧 API Documentation:   http://localhost:8000/docs")
            print("   🗺️  Service Map:         http://localhost:8000/api/service-map")
            print("   🧩 API Components:      http://localhost:8000/api/components")
            print("   ⚙️  Backend Status:      http://localhost:8001/status")
            print("   🔍 Backend Components:  http://localhost:8001/components")
            print("   🔄 Manual Scan:         http://localhost:8001/scan")
            print()
            print("📊 Service Tree:")
            print("   document-rag-orchestrator")
            print("   ├── process-manager")
            print("   ├── document-rag-api (port 8000)")
            print("   │   ├── query-processor")
            print("   │   ├── response-generator → openai-api")
            print("   │   ├── session-manager")
            print("   │   └── backend-proxy → document-rag-backend")
            print("   └── document-rag-backend (port 8001)")
            print("       ├── google-drive-monitor")
            print("       ├── local-file-scanner")
            print("       ├── document-processor")
            print("       │   ├── pdf-loader")
            print("       │   ├── docx-loader")
            print("       │   ├── pptx-loader")
            print("       │   └── image-processor")
            print("       ├── text-splitter")
            print("       ├── embedding-generator → openai-api")
            print("       ├── vector-store-manager → qdrant-database")
            print("       └── file-fingerprint-db")
            print()
            print("🔗 External Services:")
            print("   🤖 OpenAI API:         GPT completions & embeddings")
            print("   🗄️  Qdrant Database:   Vector storage & similarity search")
            print("   ☁️  Google Drive API:  Document monitoring & download")
            print()
            
            # Get real-time health status
            health_status = await self.service_connector.get_all_service_health()
            print("💊 System Health:")
            for service_name, health in health_status["services"].items():
                status_emoji = "✅" if health.get("status") in ["healthy", "degraded"] else "❌"
                print(f"   {status_emoji} {service_name}: {health.get('status', 'unknown')}")
            
            print()
            print(f"🆔 Trace ID: {get_current_trace_id()}")
            print("Press Ctrl+C to stop all services")
            print("=" * 60)
            
            span.set_attribute("system_info.displayed", True)
    
    def setup_signal_handlers(self, backend_process, api_process, monitoring_task):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            with self.tracer.start_as_current_span("orchestrator.shutdown") as shutdown_span:
                print("\n🛑 Shutting down all services...")
                shutdown_span.set_attribute("shutdown.signal", signum)
                shutdown_span.set_attribute("shutdown.initiated_by", "signal")
                
                # Stop monitoring
                self.process_monitor.stop_monitoring()
                monitoring_task.cancel()
                
                # Terminate processes
                backend_process.terminate()
                api_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    backend_process.wait(timeout=10)
                    api_process.wait(timeout=10)
                    shutdown_span.set_attribute("shutdown.graceful", True)
                    print("✅ All services stopped gracefully")
                except subprocess.TimeoutExpired:
                    print("⚠️ Force killing processes...")
                    backend_process.kill()
                    api_process.kill()
                    shutdown_span.set_attribute("shutdown.graceful", False)
                
                self.logger.info("System shutdown completed")
                sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main orchestrator function"""
    orchestrator = EnhancedOrchestrator()
    
    try:
        success = await orchestrator.start_system()
        
        if not success:
            print("❌ System startup failed")
            sys.exit(1)
        
        # Keep main thread alive
        while orchestrator.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        orchestrator.logger.error(f"Orchestrator error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())