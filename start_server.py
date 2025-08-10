#!/usr/bin/env python3
"""
Enhanced API Service Launcher with W3C Trace Propagation
Starts the Document RAG API service with proper OpenTelemetry integration
"""

import os
import sys
import time
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Force service name early for proper initialization
os.environ["OTEL_SERVICE_NAME"] = "document-rag-api"

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Enhanced OpenTelemetry configuration with W3C propagation
from otel_config import (
    initialize_opentelemetry, get_service_tracer, traced_function,
    add_trace_correlation_to_log, get_current_trace_id,
    inject_trace_context, SERVICE_HIERARCHY
)

# Initialize OpenTelemetry for API service launcher
tracer, meter = initialize_opentelemetry(
    service_name="document-rag-api",
    service_version="2.0.0",
    environment="production"
)

# Setup enhanced logging with trace correlation
logger = add_trace_correlation_to_log(logging.getLogger(__name__))

class APIServiceLauncher:
    """Enhanced API service launcher with proper service hierarchy"""
    
    def __init__(self):
        self.tracer = tracer
        self.service_name = "document-rag-api"
        self.logger = logger
        
        # Get orchestrator tracer for parent context
        self.orchestrator_tracer = get_service_tracer("document-rag-orchestrator")
        
        # Service configuration
        self.host = os.getenv("SERVER_HOST", "0.0.0.0")
        self.port = int(os.getenv("SERVER_PORT", "8000"))
        
        self.logger.info("API Service Launcher initialized with W3C trace propagation")
    
    @traced_function(service_name="document-rag-api")
    def check_environment(self) -> bool:
        """Comprehensive environment validation with tracing"""
        with self.tracer.start_as_current_span("api_launcher.check_environment") as span:
            span.set_attribute("environment.check_type", "comprehensive")
            span.set_attribute("service.component", self.service_name)
            span.set_attribute("service.parent", "document-rag-orchestrator")
            
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
                "QDRANT_HOST": "Vector database host",
                "QDRANT_PORT": "Vector database port", 
                "COLLECTION_NAME": "Vector collection name",
                "EMBEDDING_MODEL": "OpenAI embedding model"
            }
            
            missing_optional = []
            for var, description in optional_vars.items():
                if not os.getenv(var):
                    missing_optional.append(f"{var} ({description})")
            
            if missing_optional:
                span.set_attribute("environment.missing_optional", len(missing_optional))
                self.logger.warning(f"Using defaults for optional variables: {', '.join(missing_optional)}")
            
            # Set defaults for optional variables
            os.environ.setdefault("QDRANT_HOST", "localhost")
            os.environ.setdefault("QDRANT_PORT", "6333")
            os.environ.setdefault("COLLECTION_NAME", "documents")
            os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-large")
            
            span.set_attribute("environment.status", "ok")
            self.logger.info("Environment validation completed successfully")
            return True
    
    @traced_function(service_name="document-rag-api")
    def check_dependencies(self) -> bool:
        """Check service dependencies with enhanced tracing"""
        with self.tracer.start_as_current_span("api_launcher.check_dependencies") as span:
            dependencies_ok = True
            
            # Check Qdrant connection
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            
            with self.tracer.start_as_current_span("check_qdrant") as qdrant_span:
                qdrant_span.set_attribute("qdrant.host", qdrant_host)
                qdrant_span.set_attribute("qdrant.port", qdrant_port)
                qdrant_span.set_attribute("service.external", "qdrant-database")
                
                try:
                    from qdrant_client import QdrantClient
                    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=5.0)
                    collections = client.get_collections()
                    
                    qdrant_span.set_attribute("qdrant.status", "healthy")
                    qdrant_span.set_attribute("qdrant.collections_count", len(collections.collections))
                    self.logger.info(f"Qdrant accessible at {qdrant_host}:{qdrant_port} ({len(collections.collections)} collections)")
                    
                    # Check if documents collection exists
                    collection_name = os.getenv("COLLECTION_NAME", "documents")
                    try:
                        collection_info = client.get_collection(collection_name)
                        point_count = collection_info.points_count
                        qdrant_span.set_attribute("qdrant.documents_count", point_count)
                        self.logger.info(f"Collection '{collection_name}' has {point_count} documents")
                        
                        if point_count == 0:
                            self.logger.warning(f"Collection '{collection_name}' is empty - no documents available for search")
                    except Exception as e:
                        qdrant_span.set_attribute("qdrant.collection_status", "not_found")
                        self.logger.warning(f"Collection '{collection_name}' not found: {e}")
                        
                except Exception as e:
                    qdrant_span.record_exception(e)
                    qdrant_span.set_attribute("qdrant.status", "unreachable")
                    self.logger.error(f"Could not connect to Qdrant: {e}")
                    dependencies_ok = False
            
            # Check OpenAI API
            with self.tracer.start_as_current_span("check_openai") as openai_span:
                openai_span.set_attribute("service.external", "openai-api")
                
                try:
                    import openai
                    openai_key = os.getenv("OPENAI_API_KEY")
                    
                    if openai_key:
                        client = openai.OpenAI(api_key=openai_key)
                        # Test with a simple model list call
                        models = client.models.list()
                        openai_span.set_attribute("openai.status", "healthy")
                        openai_span.set_attribute("openai.models_accessible", len(models.data))
                        self.logger.info("OpenAI API accessible")
                    else:
                        openai_span.set_attribute("openai.status", "no_key")
                        self.logger.error("OpenAI API key not provided")
                        dependencies_ok = False
                        
                except Exception as e:
                    openai_span.record_exception(e)
                    openai_span.set_attribute("openai.status", "error")
                    self.logger.warning(f"OpenAI API check failed: {e}")
                    dependencies_ok = False
            
            # Check backend service availability
            backend_url = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8001")
            with self.tracer.start_as_current_span("check_backend") as backend_span:
                backend_span.set_attribute("backend.url", backend_url)
                backend_span.set_attribute("service.internal", "document-rag-backend")
                
                try:
                    import httpx
                    import asyncio
                    
                    async def check_backend():
                        async with httpx.AsyncClient() as client:
                            headers = inject_trace_context({})
                            response = await client.get(f"{backend_url}/health", headers=headers, timeout=5.0)
                            return response.status_code == 200
                    
                    backend_healthy = asyncio.run(check_backend())
                    backend_span.set_attribute("backend.status", "healthy" if backend_healthy else "unhealthy")
                    
                    if backend_healthy:
                        self.logger.info(f"Backend service accessible at {backend_url}")
                    else:
                        self.logger.warning(f"Backend service at {backend_url} is not responding properly")
                        
                except Exception as e:
                    backend_span.record_exception(e)
                    backend_span.set_attribute("backend.status", "unreachable")
                    self.logger.warning(f"Could not connect to backend service: {e}")
                    # Don't fail startup for backend issues - API can work independently
            
            span.set_attribute("dependencies.all_ok", dependencies_ok)
            return dependencies_ok
    
    @traced_function(service_name="document-rag-api")
    def setup_signal_handlers(self):
        """Setup enhanced signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            with self.tracer.start_as_current_span("api_launcher.graceful_shutdown") as shutdown_span:
                shutdown_span.set_attribute("shutdown.signal", signum)
                shutdown_span.set_attribute("shutdown.initiated_by", "signal")
                shutdown_span.set_attribute("shutdown.trace_id", get_current_trace_id())
                
                self.logger.info("🛑 Graceful shutdown initiated for API service")
                
                shutdown_span.set_attribute("shutdown.graceful", True)
                self.logger.info("✅ API service stopped gracefully")
                self.logger.info(f"🆔 Final Trace ID: {get_current_trace_id()}")
                
                sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    @traced_function(service_name="document-rag-api")
    def start_service(self) -> bool:
        """Start the API service with enhanced tracing and error handling"""
        with self.orchestrator_tracer.start_as_current_span("api_service.start") as span:
            span.set_attribute("service.name", self.service_name)
            span.set_attribute("service.host", self.host)
            span.set_attribute("service.port", self.port)
            span.set_attribute("service.parent", "document-rag-orchestrator")
            span.set_attribute("w3c.propagation", "enabled")
            
            self.logger.info("🚀 Starting Enhanced Document RAG API Service")
            self.logger.info("=" * 60)
            
            # Environment validation
            with self.tracer.start_as_current_span("api_service.environment_check") as env_span:
                if not self.check_environment():
                    env_span.set_attribute("validation.result", "failed")
                    span.set_attribute("startup.status", "environment_failed")
                    return False
                env_span.set_attribute("validation.result", "passed")
            
            # Dependencies check
            with self.tracer.start_as_current_span("api_service.dependencies_check") as deps_span:
                deps_ok = self.check_dependencies()
                deps_span.set_attribute("validation.result", "passed" if deps_ok else "warning")
                
                if not deps_ok:
                    self.logger.warning("⚠️ Some dependencies are not available - continuing with reduced functionality")
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Set working directory
            script_dir = Path(__file__).parent
            os.chdir(script_dir)
            span.set_attribute("api.working_directory", str(script_dir))
            
            self.logger.info("🌐 API Service Configuration:")
            self.logger.info(f"   📍 Host: {self.host}")
            self.logger.info(f"   🔌 Port: {self.port}")
            self.logger.info(f"   🆔 Service: {self.service_name}")
            self.logger.info(f"   📊 Parent: document-rag-orchestrator")
            self.logger.info(f"   🔗 W3C Propagation: ENABLED")
            self.logger.info(f"   📡 OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
            
            # Service hierarchy info
            hierarchy_info = SERVICE_HIERARCHY.get(self.service_name, {})
            children = hierarchy_info.get("children", [])
            if children:
                self.logger.info(f"   🧩 Components: {', '.join(children)}")
            
            self.logger.info("")
            self.logger.info("🌐 API Endpoints will be available at:")
            self.logger.info(f"   📊 Main UI:             http://{self.host}:{self.port}")
            self.logger.info(f"   🔧 API Documentation:   http://{self.host}:{self.port}/docs")
            self.logger.info(f"   🔍 Query API:           http://{self.host}:{self.port}/api/query")
            self.logger.info(f"   💊 Health Check:        http://{self.host}:{self.port}/api/health")
            self.logger.info(f"   🗺️  Service Map:         http://{self.host}:{self.port}/api/service-map")
            self.logger.info(f"   🧩 Components:          http://{self.host}:{self.port}/api/components")
            self.logger.info("")
            
            # Start the server
            try:
                import uvicorn
                
                with self.tracer.start_as_current_span("api_service.uvicorn_start") as uvicorn_span:
                    uvicorn_span.set_attribute("uvicorn.host", self.host)
                    uvicorn_span.set_attribute("uvicorn.port", self.port)
                    uvicorn_span.set_attribute("uvicorn.app", "ui.api_integrated_clean:app")
                    
                    self.logger.info("🔥 Starting FastAPI server with enhanced service tree...")
                    self.logger.info(f"🆔 Startup Trace ID: {get_current_trace_id()}")
                    
                    # Run the FastAPI application
                    uvicorn.run(
                        "ui.api_integrated_clean:app",
                        host=self.host,
                        port=self.port,
                        reload=False,  # Disable reload in production
                        log_level="info",
                        access_log=True
                    )
                    
                    uvicorn_span.set_attribute("uvicorn.startup", "success")
                    span.set_attribute("startup.status", "success")
                    return True
                    
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("startup.status", "failed")
                self.logger.error(f"❌ Failed to start API service: {e}")
                return False

def main():
    """Main function to start the API service"""
    # Initialize with orchestrator context if running standalone
    orchestrator_tracer = get_service_tracer("document-rag-orchestrator")
    
    with orchestrator_tracer.start_as_current_span("api_service.main_startup") as main_span:
        main_span.set_attribute("startup.mode", "launcher")
        main_span.set_attribute("startup.w3c_propagation", True)
        main_span.set_attribute("startup.service_hierarchy", True)
        
        launcher = APIServiceLauncher()
        
        try:
            success = launcher.start_service()
            
            if not success:
                main_span.set_attribute("startup.result", "failed")
                print("❌ API service startup failed")
                sys.exit(1)
            
            main_span.set_attribute("startup.result", "success")
            
        except KeyboardInterrupt:
            main_span.add_event("shutdown_requested")
            print("\n🔄 Shutdown signal received")
        except Exception as e:
            main_span.record_exception(e)
            main_span.set_attribute("startup.result", "error")
            launcher.logger.error(f"API service error: {e}")
            print(f"❌ API service error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    print("🔥 ENHANCED API SERVICE: Starting Document RAG API")
    print("🌐 W3C Trace Context Propagation: ENABLED")
    print("📊 Service Hierarchy: document-rag-orchestrator -> document-rag-api")
    print("🧩 Component Tree: query-processor, response-generator, session-manager, backend-proxy")
    print("🔗 Cross-Service Correlation: ACTIVE")
    print()
    
    main()
