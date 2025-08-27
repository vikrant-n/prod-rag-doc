#!/usr/bin/env python3
"""
Enhanced API Service Launcher with W3C Trace Propagation
Middleware-based approach without decorators
"""

import os
import sys
import time
import signal
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

# Force service name early
os.environ["OTEL_SERVICE_NAME"] = "document-rag-api"

from dotenv import load_dotenv
load_dotenv()

# OpenTelemetry imports - NO DECORATORS
from otel_config import (
    initialize_opentelemetry, get_service_tracer,
    get_current_trace_id, inject_trace_context, 
    extract_and_activate_context, SERVICE_HIERARCHY
)

# Initialize OpenTelemetry
tracer, meter = initialize_opentelemetry("document-rag-api", "2.0.0", "production")

class APIServiceLauncher:
    """API service launcher with middleware-based tracing"""
    
    def __init__(self):
        self.tracer = tracer
        self.service_name = "document-rag-api"
        self.host = os.getenv("SERVER_HOST", "0.0.0.0")
        self.port = int(os.getenv("SERVER_PORT", "8000"))
        
        # Extract parent context from environment if available
        parent_trace_id = os.getenv("OTEL_PARENT_TRACE_ID")
        if parent_trace_id:
            print(f"🔗 Inheriting parent trace: {parent_trace_id[:8]}...")
    
    def check_environment(self) -> bool:
        """Environment validation with manual spans"""
        with self.tracer.start_as_current_span("environment_validation") as span:
            span.set_attributes({
                "service.component": self.service_name,
                "service.parent": "document-rag-orchestrator",
                "check.type": "environment"
            })
            
            required_vars = ["OPENAI_API_KEY", "OTEL_EXPORTER_OTLP_ENDPOINT"]
            missing = [var for var in required_vars if not os.getenv(var)]
            
            if missing:
                span.set_attribute("validation.failed", True)
                print(f"❌ Missing required variables: {', '.join(missing)}")
                return False
            
            # Set defaults for optional variables
            os.environ.setdefault("QDRANT_HOST", "localhost")
            os.environ.setdefault("QDRANT_PORT", "6333")
            os.environ.setdefault("COLLECTION_NAME", "documents")
            os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-large")
            
            span.set_attribute("validation.passed", True)
            print("✅ Environment validation passed")
            return True
    
    def check_dependencies(self) -> bool:
        """Check service dependencies with manual tracing"""
        with self.tracer.start_as_current_span("dependency_check") as span:
            dependencies_ok = True
            
            # Check Qdrant
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            
            with self.tracer.start_as_current_span("check_qdrant") as qdrant_span:
                qdrant_span.set_attributes({
                    "qdrant.host": qdrant_host,
                    "qdrant.port": qdrant_port,
                    "service.external": "qdrant-database"
                })
                
                try:
                    from qdrant_client import QdrantClient
                    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=5.0)
                    collections = client.get_collections()
                    
                    qdrant_span.set_attributes({
                        "qdrant.status": "healthy",
                        "qdrant.collections_count": len(collections.collections)
                    })
                    print(f"✅ Qdrant accessible ({len(collections.collections)} collections)")
                    
                    # Check documents collection
                    collection_name = os.getenv("COLLECTION_NAME", "documents")
                    try:
                        collection_info = client.get_collection(collection_name)
                        point_count = collection_info.points_count
                        qdrant_span.set_attribute("qdrant.documents_count", point_count)
                        print(f"📄 Collection '{collection_name}' has {point_count} documents")
                    except Exception:
                        print(f"⚠️ Collection '{collection_name}' not found")
                        
                except Exception as e:
                    qdrant_span.record_exception(e)
                    qdrant_span.set_attribute("qdrant.status", "unreachable")
                    print(f"⚠️ Qdrant connection failed: {e}")
                    dependencies_ok = False
            
            # Check OpenAI API
            with self.tracer.start_as_current_span("check_openai") as openai_span:
                openai_span.set_attribute("service.external", "openai-api")
                
                try:
                    import openai
                    openai_key = os.getenv("OPENAI_API_KEY")
                    
                    if openai_key:
                        client = openai.OpenAI(api_key=openai_key)
                        models = client.models.list()
                        openai_span.set_attributes({
                            "openai.status": "healthy",
                            "openai.models_accessible": len(models.data)
                        })
                        print("✅ OpenAI API accessible")
                    else:
                        openai_span.set_attribute("openai.status", "no_key")
                        print("❌ OpenAI API key missing")
                        dependencies_ok = False
                        
                except Exception as e:
                    openai_span.record_exception(e)
                    print(f"⚠️ OpenAI API check failed: {e}")
                    dependencies_ok = False
            
            # Check backend service with proper context injection
            backend_url = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8001")
            with self.tracer.start_as_current_span("check_backend") as backend_span:
                backend_span.set_attributes({
                    "backend.url": backend_url,
                    "service.internal": "document-rag-backend"
                })
                
                try:
                    import httpx
                    
                    async def check_backend():
                        # CRITICAL: Inject trace context for backend call
                        headers = inject_trace_context({})
                        async with httpx.AsyncClient() as client:
                            response = await client.get(f"{backend_url}/health", headers=headers, timeout=5.0)
                            return response.status_code == 200
                    
                    backend_healthy = asyncio.run(check_backend())
                    backend_span.set_attribute("backend.status", "healthy" if backend_healthy else "unhealthy")
                    
                    if backend_healthy:
                        print(f"✅ Backend service accessible at {backend_url}")
                    else:
                        print(f"⚠️ Backend service not responding properly")
                        
                except Exception as e:
                    backend_span.record_exception(e)
                    print(f"⚠️ Backend service check failed: {e}")
            
            span.set_attribute("dependencies.all_ok", dependencies_ok)
            return dependencies_ok
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            with self.tracer.start_as_current_span("graceful_shutdown") as span:
                span.set_attributes({
                    "shutdown.signal": signum,
                    "shutdown.trace_id": get_current_trace_id()
                })
                
                print("\n🛑 Graceful shutdown initiated for API service")
                print(f"🆔 Final Trace ID: {get_current_trace_id()}")
                
                # Shutdown OpenTelemetry
                from otel_config import shutdown_opentelemetry
                shutdown_opentelemetry()
                
                sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_service(self) -> bool:
        """Start API service with W3C trace propagation"""
        with self.tracer.start_as_current_span("api_service_startup") as span:
            span.set_attributes({
                "service.name": self.service_name,
                "service.host": self.host,
                "service.port": self.port,
                "service.parent": "document-rag-orchestrator",
                "w3c.propagation": "enabled"
            })
            
            print("🚀 Starting Enhanced Document RAG API Service")
            print("=" * 65)
            
            # Environment validation
            if not self.check_environment():
                span.set_attribute("startup.failed_reason", "environment")
                return False
            
            # Dependencies check
            deps_ok = self.check_dependencies()
            if not deps_ok:
                print("⚠️ Some dependencies unavailable - continuing with reduced functionality")
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Set working directory
            script_dir = Path(__file__).parent
            os.chdir(script_dir)
            
            # Display service information
            print()
            print("✅ API Service Ready!")
            print("🔗 W3C Trace Propagation: ENABLED")
            print("🗺️ Service Map: ACTIVE")
            print()
            print("🌐 Service Configuration:")
            print(f"   📍 Host: {self.host}")
            print(f"   🔌 Port: {self.port}")
            print(f"   🆔 Service: {self.service_name}")
            print(f"   📊 Parent: document-rag-orchestrator")
            print(f"   🔗 Middleware: TraceContextMiddleware")
            
            # Service hierarchy
            hierarchy_info = SERVICE_HIERARCHY.get(self.service_name, {})
            children = hierarchy_info.get("children", [])
            if children:
                print(f"   🧩 Components: {', '.join(children)}")
            
            print()
            print("🌐 API Endpoints:")
            print(f"   📊 Main UI:           http://{self.host}:{self.port}")
            print(f"   📚 API Docs:          http://{self.host}:{self.port}/docs")
            print(f"   🔍 Query API:         http://{self.host}:{self.port}/api/query")
            print(f"   💊 Health Check:      http://{self.host}:{self.port}/api/health")
            print(f"   🗺️  Service Map:       http://{self.host}:{self.port}/api/service-map")
            
            print()
            print(f"🆔 Startup Trace ID: {get_current_trace_id()}")
            print(f"📡 OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
            print("=" * 65)
            
            # Start FastAPI server
            try:
                import uvicorn
                
                with self.tracer.start_as_current_span("uvicorn_server_start") as uvicorn_span:
                    uvicorn_span.set_attributes({
                        "uvicorn.host": self.host,
                        "uvicorn.port": self.port,
                        "uvicorn.app": "ui.api_integrated_clean:app"
                    })
                    
                    print("🔥 Starting FastAPI server with middleware...")
                    
                    # CRITICAL: This will use the middleware from api_integrated_clean.py
                    uvicorn.run(
                        "ui.api_integrated_clean:app",
                        host=self.host,
                        port=self.port,
                        reload=False,
                        log_level="info",
                        access_log=True
                    )
                    
                    span.set_attribute("startup.success", True)
                    return True
                    
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("startup.failed", True)
                print(f"❌ Failed to start API service: {e}")
                return False

def main():
    """Main entry point with parent context extraction"""
    
    # Extract parent context from environment if running as child process
    parent_trace_id = os.getenv("OTEL_PARENT_TRACE_ID")
    parent_service = os.getenv("OTEL_SERVICE_PARENT", "document-rag-orchestrator")
    
    # Create span in orchestrator context if available
    orchestrator_tracer = get_service_tracer(parent_service)
    
    with orchestrator_tracer.start_as_current_span("api_service_main") as main_span:
        main_span.set_attributes({
            "startup.mode": "launcher",
            "startup.parent_trace_id": parent_trace_id or "none",
            "startup.w3c_propagation": True,
            "service.hierarchy": f"{parent_service} -> document-rag-api"
        })
        
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
            print(f"❌ API service error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    print("🔥 ENHANCED API SERVICE LAUNCHER")
    print("🌐 W3C Trace Context Propagation: ENABLED")
    print("📊 Service Hierarchy: orchestrator → api → components")
    print("🗺️ Middleware-Based Trace Continuity: ACTIVE")
    print()
    
    main()
