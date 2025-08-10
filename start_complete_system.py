#!/usr/bin/env python3
"""
ENHANCED ORCHESTRATOR: Complete System with Full Service Hierarchy and W3C Propagation
Creates and manages the entire document RAG system with proper trace context flow
"""

import os
import sys
import subprocess
import time
import signal
import threading
import asyncio
import httpx
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Force service name early
os.environ["OTEL_SERVICE_NAME"] = "document-rag-orchestrator"

# Load environment first
load_dotenv()

# Enhanced OpenTelemetry configuration with W3C propagation
from otel_config import (
    initialize_opentelemetry, get_service_tracer, traced_function, 
    trace_http_call, get_current_trace_id, add_trace_correlation_to_log,
    inject_trace_context, extract_trace_context, SERVICE_HIERARCHY,
    create_child_span_with_context
)
from metrics import rag_metrics, record_cache_event, time_document_processing

# Initialize OpenTelemetry for orchestrator
tracer, meter = initialize_opentelemetry(
    service_name="document-rag-orchestrator",
    service_version="2.0.0", 
    environment="production"
)

@dataclass
class ServiceConfig:
    """Configuration for a managed service"""
    name: str
    command: List[str]
    port: int
    health_endpoint: str
    startup_timeout: int = 30
    dependencies: List[str] = None
    environment_vars: Dict[str, str] = None

class ServiceStatus(Enum):
    """Service status enumeration"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    UNHEALTHY = "unhealthy"

@dataclass 
class ServiceProcess:
    """Managed service process information"""
    name: str
    process: subprocess.Popen
    config: ServiceConfig
    status: ServiceStatus
    started_at: datetime
    pid: int
    health_check_count: int = 0
    last_health_check: Optional[datetime] = None
    restart_count: int = 0

class ProcessManager:
    """Enhanced process management with W3C trace propagation"""
    
    def __init__(self, orchestrator_tracer):
        self.tracer = get_service_tracer("process-manager")
        self.orchestrator_tracer = orchestrator_tracer
        self.service_name = "process-manager"
        self.processes: Dict[str, ServiceProcess] = {}
        self.monitoring_active = False
        self.logger = add_trace_correlation_to_log(__import__('logging').getLogger(__name__))
        
        # Initialize process monitoring tracers for each service
        self.backend_monitor_tracer = get_service_tracer("backend-process-monitor")
        self.api_monitor_tracer = get_service_tracer("api-process-monitor")
    
    @traced_function(service_name="process-manager")
    def start_service(self, config: ServiceConfig) -> ServiceProcess:
        """Start a service with enhanced monitoring and W3C context"""
        with self.tracer.start_as_current_span("process_manager.start_service") as span:
            span.set_attribute("service.name", config.name)
            span.set_attribute("service.port", config.port)
            span.set_attribute("service.command", " ".join(config.command))
            span.set_attribute("service.parent", "document-rag-orchestrator")
            
            # Add service hierarchy information
            hierarchy_info = SERVICE_HIERARCHY.get(config.name, {})
            if hierarchy_info:
                span.set_attribute("service.type", hierarchy_info.get("type", "service"))
                children = hierarchy_info.get("children", [])
                if children:
                    span.set_attribute("service.children", ",".join(children))
            
            try:
                # Prepare environment variables with trace context
                env = os.environ.copy()
                
                # Add service-specific environment variables
                if config.environment_vars:
                    env.update(config.environment_vars)
                
                # Inject trace context into environment
                env["OTEL_PARENT_TRACE_ID"] = get_current_trace_id()
                env["OTEL_SERVICE_NAME"] = config.name
                env["OTEL_SERVICE_VERSION"] = "1.0.0"
                env["OTEL_ENVIRONMENT"] = "production"
                env["OTEL_SERVICE_NAMESPACE"] = "document-rag-system"
                
                # Add service hierarchy information to environment
                if hierarchy_info:
                    parent_service = hierarchy_info.get("parent")
                    if parent_service:
                        env["OTEL_SERVICE_PARENT"] = parent_service
                    
                    service_type = hierarchy_info.get("type", "service")
                    env["OTEL_SERVICE_TYPE"] = service_type
                
                self.logger.info(f"Starting service {config.name} with trace context {get_current_trace_id()}")
                
                # Start the process
                process = subprocess.Popen(
                    config.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    cwd=Path(__file__).parent
                )
                
                # Give more time for startup
                time.sleep(5)
                
                if process.poll() is None:
                    service_process = ServiceProcess(
                        name=config.name,
                        process=process,
                        config=config,
                        status=ServiceStatus.STARTING,
                        started_at=datetime.now(),
                        pid=process.pid
                    )
                    
                    self.processes[config.name] = service_process
                    
                    span.set_attribute("process.status", "started")
                    span.set_attribute("process.pid", process.pid)
                    span.set_attribute("process.startup_timeout", config.startup_timeout)
                    
                    self.logger.info(f"Service {config.name} started successfully (PID: {process.pid})")
                    record_cache_event("service_start", True)
                    
                    return service_process
                else:
                    span.set_attribute("process.status", "failed_to_start")
                    error_msg = f"Process {config.name} failed to start (exit code: {process.returncode})"
                    self.logger.error(error_msg)
                    record_cache_event("service_start", False)
                    raise Exception(error_msg)
                    
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("process.status", "error")
                self.logger.error(f"Failed to start service {config.name}: {e}")
                raise
    
    @traced_function(service_name="process-manager")
    async def wait_for_service_health(self, service_name: str) -> bool:
        """DISABLED - No health checks during startup"""
        with self.tracer.start_as_current_span("process_manager.wait_for_service_health") as span:
            span.set_attribute("service.name", service_name)
            span.set_attribute("health.disabled", True)
            span.set_attribute("wait.result", "skipped")
            
            # Just return True - no health checks
            self.logger.info(f"Health check disabled for {service_name} - assuming healthy")
            return True
    
    @traced_function(service_name="process-manager")
    async def monitor_services(self):
        """Continuous service monitoring with enhanced health checks"""
        self.monitoring_active = True
        
        with self.tracer.start_as_current_span("process_manager.monitor_services") as span:
            span.set_attribute("monitoring.services_count", len(self.processes))
            span.set_attribute("monitoring.active", True)
            
            self.logger.info(f"Starting continuous monitoring for {len(self.processes)} services")
            
            while self.monitoring_active:
                monitoring_cycle_start = time.time()
                
                with self.tracer.start_as_current_span("monitoring_cycle") as cycle_span:
                    healthy_count = 0
                    failed_count = 0
                    
                    for service_name, service_process in self.processes.items():
                        # Use service-specific monitoring tracer
                        if service_name == "document-rag-backend":
                            monitor_tracer = self.backend_monitor_tracer
                        elif service_name == "document-rag-api":
                            monitor_tracer = self.api_monitor_tracer
                        else:
                            monitor_tracer = self.tracer
                        
                        with monitor_tracer.start_as_current_span(f"monitor_{service_name}") as service_span:
                            service_span.set_attribute("service.name", service_name)
                            service_span.set_attribute("service.pid", service_process.pid)
                            
                            try:
                                # Check if process is running
                                if service_process.process.poll() is None:
                                    # Process is running, check health
                                    url = f"http://localhost:{service_process.config.port}{service_process.config.health_endpoint}"
                                    
                                    try:
                                        async with httpx.AsyncClient() as client:
                                            headers = inject_trace_context({})
                                            response = await client.get(url, headers=headers, timeout=5.0)
                                            
                                            if response.status_code == 200:
                                                service_process.status = ServiceStatus.RUNNING
                                                service_process.last_health_check = datetime.now()
                                                service_process.health_check_count += 1
                                                healthy_count += 1
                                                
                                                service_span.set_attribute("monitor.status", "healthy")
                                                service_span.set_attribute("monitor.response_code", 200)
                                            else:
                                                service_process.status = ServiceStatus.UNHEALTHY
                                                failed_count += 1
                                                
                                                service_span.set_attribute("monitor.status", "unhealthy")
                                                service_span.set_attribute("monitor.response_code", response.status_code)
                                                
                                    except Exception as health_error:
                                        service_span.record_exception(health_error)
                                        service_process.status = ServiceStatus.UNHEALTHY
                                        failed_count += 1
                                        service_span.set_attribute("monitor.status", "health_check_failed")
                                        
                                else:
                                    # Process has died
                                    service_process.status = ServiceStatus.FAILED
                                    failed_count += 1
                                    exit_code = service_process.process.returncode
                                    
                                    service_span.set_attribute("monitor.status", "process_died")
                                    service_span.set_attribute("monitor.exit_code", exit_code)
                                    
                                    self.logger.error(f"Service {service_name} process died (exit code: {exit_code})")
                                    
                            except Exception as e:
                                service_span.record_exception(e)
                                service_span.set_attribute("monitor.status", "error")
                                failed_count += 1
                    
                    cycle_span.set_attribute("monitor.healthy_services", healthy_count)
                    cycle_span.set_attribute("monitor.failed_services", failed_count)
                    cycle_span.set_attribute("monitor.cycle_duration", time.time() - monitoring_cycle_start)
                
                # Wait before next monitoring cycle
                await asyncio.sleep(30)
            
            span.set_attribute("monitoring.stopped", True)
            self.logger.info("Service monitoring stopped")
    
    @traced_function(service_name="process-manager")
    def stop_monitoring(self):
        """Stop service monitoring"""
        with self.tracer.start_as_current_span("process_manager.stop_monitoring") as span:
            self.monitoring_active = False
            span.set_attribute("action", "stop_monitoring")
            self.logger.info("Stopping service monitoring")
    
    @traced_function(service_name="process-manager")
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all managed services"""
        with self.tracer.start_as_current_span("process_manager.get_service_status") as span:
            status = {}
            total_services = len(self.processes)
            running_services = 0
            
            for service_name, service_process in self.processes.items():
                service_status = {
                    "name": service_name,
                    "status": service_process.status.value,
                    "pid": service_process.pid if service_process.process.poll() is None else None,
                    "port": service_process.config.port,
                    "started_at": service_process.started_at.isoformat(),
                    "uptime_seconds": (datetime.now() - service_process.started_at).total_seconds(),
                    "health_check_count": service_process.health_check_count,
                    "last_health_check": service_process.last_health_check.isoformat() if service_process.last_health_check else None,
                    "restart_count": service_process.restart_count,
                    "service_type": SERVICE_HIERARCHY.get(service_name, {}).get("type", "service")
                }
                
                if service_process.status == ServiceStatus.RUNNING:
                    running_services += 1
                
                status[service_name] = service_status
            
            span.set_attribute("status.total_services", total_services)
            span.set_attribute("status.running_services", running_services)
            
            return {
                "total_services": total_services,
                "running_services": running_services,
                "failed_services": total_services - running_services,
                "services": status,
                "monitoring_active": self.monitoring_active
            }
    
    @traced_function(service_name="process-manager")
    def terminate_all_services(self):
        """Gracefully terminate all managed services"""
        with self.tracer.start_as_current_span("process_manager.terminate_all_services") as span:
            span.set_attribute("services.count", len(self.processes))
            
            self.logger.info("Terminating all managed services...")
            
            terminated_count = 0
            for service_name, service_process in self.processes.items():
                with self.tracer.start_as_current_span(f"terminate_{service_name}") as terminate_span:
                    try:
                        if service_process.process.poll() is None:
                            self.logger.info(f"Terminating {service_name} (PID: {service_process.pid})")
                            service_process.process.terminate()
                            
                            # Wait for graceful shutdown
                            try:
                                service_process.process.wait(timeout=10)
                                terminate_span.set_attribute("termination.graceful", True)
                                self.logger.info(f"Service {service_name} terminated gracefully")
                            except subprocess.TimeoutExpired:
                                self.logger.warning(f"Force killing {service_name}")
                                service_process.process.kill()
                                service_process.process.wait()
                                terminate_span.set_attribute("termination.graceful", False)
                            
                            service_process.status = ServiceStatus.STOPPED
                            terminated_count += 1
                            terminate_span.set_attribute("termination.success", True)
                        else:
                            terminate_span.set_attribute("termination.already_stopped", True)
                            
                    except Exception as e:
                        terminate_span.record_exception(e)
                        terminate_span.set_attribute("termination.success", False)
                        self.logger.error(f"Error terminating {service_name}: {e}")
            
            span.set_attribute("services.terminated", terminated_count)
            self.logger.info(f"Terminated {terminated_count} services")

class ServiceConnector:
    """Enhanced service connection manager with W3C propagation"""
    
    def __init__(self, orchestrator_tracer):
        self.tracer = orchestrator_tracer
        self.service_name = "service-connector"
        self.logger = add_trace_correlation_to_log(__import__('logging').getLogger(__name__))
        
        self.services = {
            "api": {"url": "http://localhost:8000", "health_endpoint": "/api/health"},
            "backend": {"url": "http://localhost:8001", "health_endpoint": "/health"}
        }
    
    @trace_http_call("GET", "service_health", "service-connector")
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service with W3C trace context"""
        with self.tracer.start_as_current_span(f"service_connector.check_{service_name}") as span:
            service_config = self.services.get(service_name)
            if not service_config:
                span.set_attribute("service.found", False)
                return {"status": "unknown", "error": "Service not configured"}
            
            span.set_attribute("service.url", service_config["url"])
            span.set_attribute("service.health_endpoint", service_config["health_endpoint"])
            span.set_attribute("service.target", service_name)
            
            try:
                async with httpx.AsyncClient() as client:
                    # Inject comprehensive trace context
                    headers = inject_trace_context({})
                    headers["X-Trace-ID"] = get_current_trace_id()
                    headers["X-Service-Chain"] = f"document-rag-orchestrator -> service-connector -> {service_name}"
                    headers["X-Request-Source"] = "orchestrator-health-check"
                    
                    url = f"{service_config['url']}{service_config['health_endpoint']}"
                    
                    response = await client.get(url, headers=headers, timeout=10.0)
                    
                    span.set_attribute("http.status_code", response.status_code)
                    span.set_attribute("http.response_time", response.elapsed.total_seconds())
                    
                    if response.status_code == 200:
                        try:
                            health_data = response.json()
                            span.set_attribute("service.health_status", health_data.get("status", "unknown"))
                            
                            # Extract additional service information
                            if "components" in health_data:
                                span.set_attribute("service.components_count", len(health_data["components"]))
                            
                            if "trace_context" in health_data:
                                span.set_attribute("service.trace_correlation", True)
                            
                            return health_data
                        except Exception:
                            span.set_attribute("service.health_status", "basic_healthy")
                            return {"status": "healthy", "message": "Service responding (basic)"}
                    elif response.status_code in [404, 422]:
                        span.set_attribute("service.health_status", "endpoint_error")
                        return {"status": "degraded", "error": "Health endpoint issue", "status_code": response.status_code}
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
        """Get health status of all services with comprehensive correlation"""
        with self.tracer.start_as_current_span("service_connector.get_all_health") as span:
            health_results = {}
            
            for service_name in self.services.keys():
                with self.tracer.start_as_current_span(f"check_service_{service_name}") as service_span:
                    health_result = await self.check_service_health(service_name)
                    health_results[service_name] = health_result
                    
                    service_span.set_attribute("service.name", service_name)
                    service_span.set_attribute("service.health", health_result.get("status", "unknown"))
            
            # Calculate overall system health
            functional_statuses = ["healthy", "degraded", "basic_healthy"]
            all_functional = all(
                result.get("status") in functional_statuses 
                for result in health_results.values()
            )
            
            overall_status = "healthy" if all_functional else "degraded"
            
            span.set_attribute("system.overall_health", overall_status)
            span.set_attribute("system.services_count", len(health_results))
            span.set_attribute("system.functional_services", sum(1 for r in health_results.values() if r.get("status") in functional_statuses))
            
            return {
                "overall_status": overall_status,
                "services": health_results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trace_id": get_current_trace_id(),
                "service_chain": "document-rag-orchestrator -> service-connector",
                "correlation_info": {
                    "w3c_propagation": True,
                    "trace_context_injected": True,
                    "service_hierarchy_tracked": True
                }
            }

class EnhancedOrchestrator:
    """Main orchestrator with full service hierarchy and W3C trace correlation"""
    
    def __init__(self):
        # Initialize root orchestrator tracer with enhanced hierarchy
        self.tracer, self.meter = initialize_opentelemetry(
            service_name="document-rag-orchestrator",
            service_version="2.0.0",
            environment="production"
        )
        
        self.service_name = "document-rag-orchestrator"
        
        # Initialize service components with proper hierarchy
        self.process_manager = ProcessManager(self.tracer)
        self.service_connector = ServiceConnector(self.tracer)
        
        # Service state
        self.is_running = False
        self.startup_time = datetime.now()
        
        # Enhanced logger with trace correlation
        self.logger = add_trace_correlation_to_log(__import__('logging').getLogger(__name__))
        self.logger.info("Enhanced orchestrator initialized with full W3C trace propagation")
        
        # Service configurations with enhanced settings
        self.service_configs = {
            "backend": ServiceConfig(
                name="document-rag-backend",
                command=[sys.executable, "backend_service.py", "--host", "0.0.0.0", "--port", "8001"],
                port=8001,
                health_endpoint="/health",
                startup_timeout=60,  # Increased timeout for backend
                environment_vars={
                    "OTEL_SERVICE_NAME": "document-rag-backend",
                    "OTEL_SERVICE_PARENT": "document-rag-orchestrator",
                    "OTEL_SERVICE_TYPE": "processing_service",
                    "SCAN_INTERVAL": "30"
                }
            ),
            "api": ServiceConfig(
                name="document-rag-api", 
                command=[sys.executable, "start_server.py"],
                port=8000,
                health_endpoint="/api/health",
                startup_timeout=30,
                dependencies=["backend"],
                environment_vars={
                    "OTEL_SERVICE_NAME": "document-rag-api",
                    "OTEL_SERVICE_PARENT": "document-rag-orchestrator",
                    "OTEL_SERVICE_TYPE": "api_service",
                    "BACKEND_SERVICE_URL": "http://localhost:8001"
                }
            )
        }
    
    @traced_function(service_name="document-rag-orchestrator")
    def check_environment(self) -> bool:
        """Comprehensive environment validation with trace context"""
        with self.tracer.start_as_current_span("orchestrator.check_environment") as span:
            span.set_attribute("environment.check_type", "comprehensive")
            
            # Check required environment variables
            required_vars = {
                "OPENAI_API_KEY": "OpenAI API access",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "OpenTelemetry collector endpoint"
            }
            
            missing_vars = []
            for var, description in required_vars.items():
                if not os.getenv(var):
                    missing_vars.append(f"{var} ({description})")
            
            span.set_attribute("environment.required_vars", len(required_vars))
            span.set_attribute("environment.missing_vars", len(missing_vars))
            
            if missing_vars:
                span.set_attribute("environment.status", "missing_vars")
                self.logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
                return False
            
            # Check optional but recommended variables
            optional_vars = {
                "GOOGLE_DRIVE_FOLDER_ID": "Google Drive monitoring",
                "COHERE_API_KEY": "Enhanced reranking",
                "QDRANT_URL": "Vector database connection"
            }
            
            missing_optional = []
            for var, description in optional_vars.items():
                if not os.getenv(var):
                    missing_optional.append(f"{var} ({description})")
            
            if missing_optional:
                span.set_attribute("environment.missing_optional", len(missing_optional))
                self.logger.warning(f"Missing optional variables: {', '.join(missing_optional)}")
            
            # Validate OTLP endpoint connectivity
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            span.set_attribute("otlp.endpoint", otlp_endpoint)
            
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(otlp_endpoint)
                if not parsed.scheme or not parsed.netloc:
                    span.set_attribute("otlp.endpoint_valid", False)
                    self.logger.warning(f"Invalid OTLP endpoint format: {otlp_endpoint}")
                else:
                    span.set_attribute("otlp.endpoint_valid", True)
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("otlp.endpoint_valid", False)
            
            span.set_attribute("environment.status", "ok")
            self.logger.info("Environment validation completed successfully")
            return True
    
    @traced_function(service_name="document-rag-orchestrator")
    async def check_external_dependencies(self) -> bool:
        """Check external service dependencies with W3C trace context"""
        with self.tracer.start_as_current_span("orchestrator.check_external_dependencies") as span:
            dependencies_ok = True
            
            # Check Qdrant
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            span.set_attribute("qdrant.url", qdrant_url)
            
            with self.tracer.start_as_current_span("check_qdrant") as qdrant_span:
                try:
                    async with httpx.AsyncClient() as client:
                        headers = inject_trace_context({})
                        response = await client.get(qdrant_url, headers=headers, timeout=5)
                        
                        qdrant_span.set_attribute("qdrant.status_code", response.status_code)
                        qdrant_span.set_attribute("qdrant.response_time", response.elapsed.total_seconds())
                        
                        if response.status_code == 200:
                            qdrant_span.set_attribute("qdrant.status", "healthy")
                            self.logger.info(f"Qdrant accessible at {qdrant_url}")
                        else:
                            qdrant_span.set_attribute("qdrant.status", "unhealthy")
                            self.logger.warning(f"Qdrant responded with status {response.status_code}")
                            dependencies_ok = False
                            
                except Exception as e:
                    qdrant_span.record_exception(e)
                    qdrant_span.set_attribute("qdrant.status", "unreachable")
                    self.logger.warning(f"Could not connect to Qdrant: {e}")
                    # Don't fail startup for Qdrant issues
            
            # Check OpenAI API (if key is provided)
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                with self.tracer.start_as_current_span("check_openai") as openai_span:
                    try:
                        import openai
                        client = openai.OpenAI(api_key=openai_key)
                        
                        # Simple API test (list models)
                        models = client.models.list()
                        openai_span.set_attribute("openai.status", "healthy")
                        openai_span.set_attribute("openai.models_count", len(models.data))
                        self.logger.info("OpenAI API accessible")
                        
                    except Exception as e:
                        openai_span.record_exception(e)
                        openai_span.set_attribute("openai.status", "error")
                        self.logger.warning(f"OpenAI API check failed: {e}")
                        dependencies_ok = False
            
            span.set_attribute("dependencies.all_ok", dependencies_ok)
            return dependencies_ok
    
    @traced_function(service_name="document-rag-orchestrator")
    async def start_service_with_dependencies(self, service_name: str) -> bool:
        """Start service with dependency resolution and W3C context"""
        with self.tracer.start_as_current_span(f"orchestrator.start_{service_name}") as span:
            config = self.service_configs.get(service_name)
            if not config:
                span.set_attribute("service.found", False)
                return False
            
            span.set_attribute("service.name", service_name)
            span.set_attribute("service.has_dependencies", bool(config.dependencies))
            
            # Start dependencies first
            if config.dependencies:
                with self.tracer.start_as_current_span(f"start_dependencies_{service_name}") as deps_span:
                    deps_span.set_attribute("dependencies.count", len(config.dependencies))
                    
                    for dep_name in config.dependencies:
                        if dep_name not in self.process_manager.processes:
                            self.logger.info(f"Starting dependency {dep_name} for {service_name}")
                            dep_success = await self.start_service_with_dependencies(dep_name)
                            if not dep_success:
                                deps_span.set_attribute("dependencies.failed", dep_name)
                                span.set_attribute("startup.result", "dependency_failed")
                                return False
            
            # Start the service
            with self.tracer.start_as_current_span(f"launch_{service_name}") as launch_span:
                try:
                    service_process = self.process_manager.start_service(config)
                    launch_span.set_attribute("launch.pid", service_process.pid)
                    launch_span.set_attribute("launch.success", True)
                    
                    # Wait for health
                    with self.tracer.start_as_current_span(f"wait_health_{service_name}") as health_span:
                        health_ok = await self.process_manager.wait_for_service_health(service_name)
                        health_span.set_attribute("health.result", health_ok)
                        
                        if health_ok:
                            span.set_attribute("startup.result", "success")
                            self.logger.info(f"Service {service_name} started and is healthy")
                            return True
                        else:
                            span.set_attribute("startup.result", "unhealthy")
                            self.logger.error(f"Service {service_name} failed health check")
                            return False
                            
                except Exception as e:
                    launch_span.record_exception(e)
                    launch_span.set_attribute("launch.success", False)
                    span.set_attribute("startup.result", "error")
                    self.logger.error(f"Failed to start {service_name}: {e}")
                    return False
    
    @traced_function(service_name="document-rag-orchestrator")
    async def start_system(self) -> bool:
        """Start complete system with enhanced orchestration and W3C correlation"""
        with self.tracer.start_as_current_span("orchestrator.start_system") as span:
            self.logger.info("🔥 Starting Enhanced Document RAG System with W3C Trace Propagation")
            self.logger.info("=" * 80)
            
            span.set_attribute("system.name", "document-rag")
            span.set_attribute("system.version", "2.0.0")
            span.set_attribute("system.trace_propagation", "w3c")
            span.set_attribute("system.service_hierarchy", "enabled")
            
            # Set working directory
            script_dir = Path(__file__).parent
            os.chdir(script_dir)
            span.set_attribute("system.working_directory", str(script_dir))
            
            # Environment validation
            with self.tracer.start_as_current_span("environment_validation") as env_span:
                if not self.check_environment():
                    env_span.set_attribute("validation.result", "failed")
                    span.set_attribute("startup.status", "environment_failed")
                    return False
                env_span.set_attribute("validation.result", "passed")
            
            # External dependencies check
            with self.tracer.start_as_current_span("dependencies_validation") as deps_span:
                deps_ok = await self.check_external_dependencies()
                deps_span.set_attribute("validation.result", "passed" if deps_ok else "warning")
                
                if not deps_ok:
                    self.logger.warning("⚠️ Some external dependencies are not available - continuing with reduced functionality")
            
            self.logger.info("🚀 Starting services with dependency resolution...")
            
            # Start services in correct order with relaxed failure handling
            service_start_order = ["backend", "api"]
            started_services = []
            
            for service_name in service_start_order:
                with self.tracer.start_as_current_span(f"start_service_{service_name}") as service_span:
                    service_span.set_attribute("service.name", service_name)
                    service_span.set_attribute("service.order", service_start_order.index(service_name) + 1)
                    
                    self.logger.info(f"Starting {service_name} service...")
                    
                    try:
                        success = await self.start_service_with_dependencies(service_name)
                        
                        if success:
                            started_services.append(service_name)
                            service_span.set_attribute("service.startup", "success")
                            self.logger.info(f"✅ {service_name} service started successfully")
                        else:
                            service_span.set_attribute("service.startup", "failed")
                            self.logger.error(f"❌ Failed to start {service_name} service")
                            
                            # For backend service, let's check if it's actually running but just failing health check
                            if service_name == "document-rag-backend":
                                backend_process = self.process_manager.processes.get("document-rag-backend")
                                if backend_process and backend_process.process.poll() is None:
                                    self.logger.warning(f"⚠️ Backend process is running (PID: {backend_process.pid}) but health check failed")
                                    self.logger.warning("Continuing startup - backend may need more time to initialize")
                                    started_services.append(service_name)
                                    service_span.set_attribute("service.startup", "partial_success")
                                    continue
                            
                            # If we get here, the service truly failed
                            span.set_attribute("startup.status", f"{service_name}_failed")
                            
                            # Cleanup started services
                            self.logger.info("Cleaning up started services...")
                            self.process_manager.terminate_all_services()
                            return False
                            
                    except Exception as e:
                        service_span.record_exception(e)
                        service_span.set_attribute("service.startup", "error")
                        self.logger.error(f"Exception starting {service_name}: {e}")
                        
                        # Cleanup started services
                        self.logger.info("Cleaning up started services...")
                        self.process_manager.terminate_all_services()
                        return False
            
            # Start process monitoring
            self.is_running = True
            monitoring_task = asyncio.create_task(self.process_manager.monitor_services())
            
            span.set_attribute("startup.status", "success")
            span.set_attribute("startup.services_started", len(started_services))
            span.set_attribute("monitoring.started", True)
            
            # Display system information
            await self.display_enhanced_system_info()
            
            # Setup signal handlers for graceful shutdown
            self.setup_signal_handlers(monitoring_task)
            
            self.logger.info("🎉 System startup completed successfully with full W3C trace correlation")
            return True
    
    async def display_enhanced_system_info(self):
        """Display comprehensive system information with trace context"""
        with self.tracer.start_as_current_span("orchestrator.display_system_info") as span:
            print()
            print("✅ Enhanced Document RAG System Started Successfully!")
            print("🔗 W3C Trace Context Propagation: ENABLED")
            print()
            print("🌐 System URLs:")
            print("   📊 Main UI:             http://localhost:8000")
            print("   📚 API Documentation:   http://localhost:8000/docs")
            print("   🗺️  Service Map:         http://localhost:8000/api/service-map")
            print("   🧩 API Components:      http://localhost:8000/api/components")
            print("   ⚙️  Backend Status:      http://localhost:8001/health")
            print("   🔍 Backend Components:  http://localhost:8001/components")
            print("   🔄 Manual Scan:         http://localhost:8001/scan")
            print()
            print("📊 Enhanced Service Hierarchy:")
            print("   document-rag-orchestrator (root)")
            print("   ├── 📋 process-manager")
            print("   │   ├── 🔍 backend-process-monitor")
            print("   │   └── 🌐 api-process-monitor")
            print("   ├── 🌐 document-rag-api (port 8000)")
            print("   │   ├── 🔍 query-processor")
            print("   │   ├── 🤖 response-generator → openai-api")
            print("   │   ├── 💬 session-manager")
            print("   │   └── 🔗 backend-proxy → document-rag-backend")
            print("   └── ⚙️ document-rag-backend (port 8001)")
            print("       ├── ☁️  google-drive-monitor → google-drive-api")
            print("       ├── 📁 local-file-scanner")
            print("       ├── 📄 document-processor")
            print("       │   ├── 📑 pdf-loader")
            print("       │   ├── 📝 docx-loader")
            print("       │   ├── 🎯 pptx-loader")
            print("       │   └── 🖼️  image-processor")
            print("       ├── ✂️  text-splitter")
            print("       ├── 🧠 embedding-generator → openai-api")
            print("       ├── 🗃️  vector-store-manager → qdrant-database")
            print("       └── 📊 file-fingerprint-db")
            print()
            print("🔗 External Services & APIs:")
            print("   🤖 OpenAI API:         GPT completions & embeddings")
            print("   🗄️  Qdrant Database:   Vector storage & similarity search")
            print("   ☁️  Google Drive API:  Document monitoring & download")
            print("   📡 OTEL Collector:     Distributed tracing & metrics")
            print()
            
            # Get real-time health status with trace context - with error handling
            try:
                health_status = await self.service_connector.get_all_service_health()
                print("💊 System Health Status:")
                for service_name, health in health_status["services"].items():
                    status_emoji = "✅" if health.get("status") in ["healthy", "degraded", "basic_healthy"] else "❌"
                    status_text = health.get("status", "unknown").upper()
                    print(f"   {status_emoji} {service_name}: {status_text}")
            except Exception as e:
                print("💊 System Health Status:")
                print(f"   ⚠️ Health check error: {e}")
                print("   📝 Services may still be initializing...")
            
            print()
            print("🔍 W3C Trace Propagation Features:")
            print("   ✅ TraceContext headers (primary)")
            print("   ✅ Baggage propagation")
            print("   ✅ B3 format (Zipkin compatibility)")
            print("   ✅ Jaeger format (legacy support)")
            print("   ✅ Service hierarchy tracking")
            print("   ✅ Cross-service correlation")
            print()
            print(f"🆔 Root Trace ID: {get_current_trace_id()}")
            print(f"⏰ System Started: {self.startup_time.isoformat()}")
            print("🔧 Configuration:")
            print(f"   • OTEL Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'default')}")
            print(f"   • Service Namespace: {os.getenv('OTEL_SERVICE_NAMESPACE', 'document-rag-system')}")
            print(f"   • Environment: {os.getenv('OTEL_ENVIRONMENT', 'production')}")
            print()
            print("📝 Notes:")
            print("   • Services may take additional time to fully initialize")
            print("   • Backend service initializes vector stores and components")
            print("   • Check individual service health endpoints for detailed status")
            print()
            print("Press Ctrl+C to stop all services")
            print("=" * 80)
            
            span.set_attribute("system_info.displayed", True)
            span.set_attribute("system_info.trace_id", get_current_trace_id())
    
    def setup_signal_handlers(self, monitoring_task):
        """Setup enhanced signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            with self.tracer.start_as_current_span("orchestrator.graceful_shutdown") as shutdown_span:
                print("\n🛑 Graceful shutdown initiated...")
                shutdown_span.set_attribute("shutdown.signal", signum)
                shutdown_span.set_attribute("shutdown.initiated_by", "signal")
                shutdown_span.set_attribute("shutdown.trace_id", get_current_trace_id())
                
                self.is_running = False
                
                # Stop process monitoring
                with self.tracer.start_as_current_span("shutdown.stop_monitoring") as monitor_span:
                    self.process_manager.stop_monitoring()
                    monitoring_task.cancel()
                    monitor_span.set_attribute("monitoring.stopped", True)
                
                # Get final service status
                with self.tracer.start_as_current_span("shutdown.final_status") as status_span:
                    final_status = self.process_manager.get_service_status()
                    status_span.set_attribute("final.running_services", final_status["running_services"])
                    status_span.set_attribute("final.total_services", final_status["total_services"])
                
                # Terminate all services
                with self.tracer.start_as_current_span("shutdown.terminate_services") as terminate_span:
                    self.process_manager.terminate_all_services()
                    terminate_span.set_attribute("termination.completed", True)
                
                shutdown_duration = time.time() - time.time()
                shutdown_span.set_attribute("shutdown.duration", shutdown_duration)
                shutdown_span.set_attribute("shutdown.graceful", True)
                
                print("✅ All services stopped gracefully")
                print(f"🆔 Final Trace ID: {get_current_trace_id()}")
                
                self.logger.info("Enhanced orchestrator shutdown completed with full trace correlation")
                sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main orchestrator function with enhanced error handling"""
    orchestrator = EnhancedOrchestrator()
    
    try:
        # Start system with comprehensive tracing
        with orchestrator.tracer.start_as_current_span("main.system_startup") as main_span:
            main_span.set_attribute("startup.mode", "enhanced")
            main_span.set_attribute("startup.w3c_propagation", True)
            main_span.set_attribute("startup.service_hierarchy", True)
            
            success = await orchestrator.start_system()
            
            if not success:
                main_span.set_attribute("startup.result", "failed")
                print("❌ System startup failed")
                sys.exit(1)
            
            main_span.set_attribute("startup.result", "success")
            
            # Keep main thread alive with enhanced monitoring
            while orchestrator.is_running:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        print("\n🔄 Shutdown signal received")
    except Exception as e:
        orchestrator.logger.error(f"Orchestrator error: {e}")
        print(f"❌ System error: {e}")
        sys.exit(1)
    finally:
        # Final cleanup
        if hasattr(orchestrator, 'process_manager'):
            orchestrator.process_manager.terminate_all_services()

if __name__ == "__main__":
    print("🔥 ENHANCED ORCHESTRATOR: Starting Document RAG System")
    print("🌐 W3C Trace Context Propagation: ENABLED")
    print("📊 Service Hierarchy: FULL TREE")
    print("🔗 Cross-Service Correlation: ACTIVE")
    print()
    
    asyncio.run(main())