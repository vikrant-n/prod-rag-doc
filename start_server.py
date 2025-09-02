#!/usr/bin/env python3
"""
Enhanced API Service Launcher with W3C Trace Propagation
Updated for Elasticsearch support
"""

import os
import sys
import time
import signal
import logging
import asyncio
import urllib3
from pathlib import Path
from datetime import datetime
from typing import Optional

from opentelemetry import trace, context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


# Suppress SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

def extract_parent_context_from_environment():
    """Extract parent trace context from environment variables set by orchestrator"""
    
    # CRITICAL FIX: Import propagate inside function to avoid import issues
    from opentelemetry import propagate
    from opentelemetry.context import attach
    
    # Check for W3C trace context in environment
    traceparent = os.getenv("OTEL_TRACE_PARENT")
    tracestate = os.getenv("OTEL_TRACE_STATE", "")
    
    print(f"🔍 Looking for parent context in environment:")
    print(f"   OTEL_TRACE_PARENT: {traceparent}")
    print(f"   OTEL_TRACE_STATE: {tracestate}")
    
    if traceparent:
        # Create headers dict for W3C propagator
        headers = {
            "traceparent": traceparent,
            "tracestate": tracestate
        }
        
        print(f"📥 Extracting W3C context from headers: {headers}")
        
        try:
            # Extract context using OpenTelemetry propagator
            parent_context = propagate.extract(headers)
            
            # Activate the extracted context
            context_token = attach(parent_context)
            
            print(f"✅ Parent context extracted and activated")
            print(f"   Context: {parent_context}")
            
            return context_token, parent_context
        except Exception as e:
            print(f"❌ Error extracting parent context: {e}")
            return None, None
    else:
        print(f"ℹ️ No parent context found in environment")
        return None, None

def initialize_service_with_parent_context(service_name: str):
    """Initialize service with proper parent context extraction"""
    
    print(f"🚀 Initializing {service_name} with parent context extraction")
    
    # Step 1: Extract parent context BEFORE initializing OpenTelemetry
    context_token, parent_context = extract_parent_context_from_environment()
    
    # Step 2: Initialize OpenTelemetry in the context of parent
    from otel_config import initialize_opentelemetry
    tracer, meter = initialize_opentelemetry(
        service_name=service_name,
        service_version="2.0.0",
        environment=os.getenv("OTEL_ENVIRONMENT", "production")
    )
    
    # Step 3: Create startup span within parent context
    with tracer.start_as_current_span(f"{service_name}_startup_with_parent") as span:
        span.set_attributes({
            "service.name": service_name,
            "service.version": "2.0.0",
            "service.parent": "document-rag-orchestrator",
            "startup.with_parent_context": parent_context is not None,
            "operation.name": "service_startup_with_context"
        })
        
        # Log the trace information
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, '032x')
        span_id = format(span_context.span_id, '016x')
        
        print(f"🆔 Service {service_name} trace context:")
        print(f"   Trace ID: {trace_id}")
        print(f"   Span ID: {span_id}")
        print(f"   Parent Context: {'✅ Yes' if parent_context else '❌ No'}")
        print(f"   Context Token: {context_token}")
        
        # Store context token for cleanup later if needed
        return tracer, meter, span, context_token


class APIServiceLauncher:
    """API service launcher with middleware-based tracing and Elasticsearch support"""
    
    def __init__(self):
        self.tracer = tracer
        self.service_name = "document-rag-api"
        self.host = os.getenv("SERVER_HOST", "0.0.0.0")
        self.port = int(os.getenv("SERVER_PORT", "8000"))
        
        # Extract parent context from environment if available
        parent_trace_id = os.getenv("OTEL_PARENT_TRACE_ID")
        if parent_trace_id:
            print(f"Inheriting parent trace: {parent_trace_id[:8]}...")
    
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
                print(f"Missing required variables: {', '.join(missing)}")
                return False
            
            # Set defaults for Elasticsearch variables
            os.environ.setdefault("ELASTICSEARCH_URL", "https://172.31.23.77:9200")
            os.environ.setdefault("ELASTICSEARCH_USERNAME", "elastic")
            os.environ.setdefault("ELASTICSEARCH_PASSWORD", "elastic")
            os.environ.setdefault("ELASTICSEARCH_INDEX", "documents")
            os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-large")
            
            span.set_attribute("validation.passed", True)
            print("Environment validation passed")
            return True
    
    def check_dependencies(self) -> bool:
        """Check service dependencies with Elasticsearch support"""
        with self.tracer.start_as_current_span("dependency_check") as span:
            dependencies_ok = True
            
            # Check Elasticsearch
            elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "https://172.31.23.77:9200")
            elasticsearch_username = os.getenv("ELASTICSEARCH_USERNAME", "elastic")
            elasticsearch_password = os.getenv("ELASTICSEARCH_PASSWORD", "elastic")
            elasticsearch_index = os.getenv("ELASTICSEARCH_INDEX", "documents")
            
            with self.tracer.start_as_current_span("check_elasticsearch") as es_span:
                es_span.set_attributes({
                    "elasticsearch.url": elasticsearch_url,
                    "elasticsearch.username": elasticsearch_username,
                    "elasticsearch.index": elasticsearch_index,
                    "service.external": "elasticsearch-database"
                })
                
                try:
                    from elasticsearch import Elasticsearch
                    
                    client = Elasticsearch(
                        [elasticsearch_url],
                        basic_auth=(elasticsearch_username, elasticsearch_password),
                        verify_certs=False,
                        ssl_show_warn=False,
                        request_timeout=10,
                        retry_on_timeout=True,
                        max_retries=2
                    )
                    
                    # Test cluster health
                    health = client.cluster.health()
                    
                    es_span.set_attributes({
                        "elasticsearch.status": health.get("status", "unknown"),
                        "elasticsearch.cluster_name": health.get("cluster_name", "unknown"),
                        "elasticsearch.nodes": health.get("number_of_nodes", 0),
                        "elasticsearch.indices": health.get("number_of_indices", 0)
                    })
                    
                    print(f"Elasticsearch accessible - Status: {health.get('status')}")
                    print(f"Cluster: {health.get('cluster_name')} ({health.get('number_of_nodes')} nodes)")
                    
                    # Check documents index
                    try:
                        if client.indices.exists(index=elasticsearch_index):
                            stats = client.indices.stats(index=elasticsearch_index)
                            doc_count = stats['indices'][elasticsearch_index]['total']['docs']['count']
                            size_bytes = stats['indices'][elasticsearch_index]['total']['store']['size_in_bytes']
                            
                            es_span.set_attributes({
                                "elasticsearch.documents_count": doc_count,
                                "elasticsearch.index_size_bytes": size_bytes
                            })
                            
                            print(f"Index '{elasticsearch_index}' has {doc_count:,} documents ({size_bytes:,} bytes)")
                        else:
                            print(f"Index '{elasticsearch_index}' not found (will be created automatically)")
                            
                    except Exception as index_error:
                        print(f"Could not check index status: {index_error}")
                        
                except Exception as e:
                    es_span.record_exception(e)
                    es_span.set_attribute("elasticsearch.status", "unreachable")
                    print(f"Elasticsearch connection failed: {e}")
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
                        print("OpenAI API accessible")
                        
                        # Check if embedding model is available
                        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
                        available_models = [model.id for model in models.data]
                        if embedding_model in available_models:
                            print(f"Embedding model '{embedding_model}' available")
                        else:
                            print(f"Warning: Embedding model '{embedding_model}' not found in available models")
                    else:
                        openai_span.set_attribute("openai.status", "no_key")
                        print("OpenAI API key missing")
                        dependencies_ok = False
                        
                except Exception as e:
                    openai_span.record_exception(e)
                    print(f"OpenAI API check failed: {e}")
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
                        print(f"Backend service accessible at {backend_url}")
                    else:
                        print(f"Backend service not responding properly")
                        
                except Exception as e:
                    backend_span.record_exception(e)
                    print(f"Backend service check failed: {e}")
            
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
                
                print("\nGraceful shutdown initiated for API service")
                print(f"Final Trace ID: {get_current_trace_id()}")
                
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
                "w3c.propagation": "enabled",
                "vector_database": "elasticsearch"
            })
            
            print("Starting Enhanced Document RAG API Service")
            print("=" * 65)
            
            # Environment validation
            if not self.check_environment():
                span.set_attribute("startup.failed_reason", "environment")
                return False
            
            # Dependencies check
            deps_ok = self.check_dependencies()
            if not deps_ok:
                print("Some dependencies unavailable - continuing with reduced functionality")
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Set working directory
            script_dir = Path(__file__).parent
            os.chdir(script_dir)
            
            # Display service information
            print()
            print("API Service Ready!")
            print("W3C Trace Propagation: ENABLED")
            print("Service Map: ACTIVE")
            print("Vector Database: Elasticsearch")
            print()
            print("Service Configuration:")
            print(f"   Host: {self.host}")
            print(f"   Port: {self.port}")
            print(f"   Service: {self.service_name}")
            print(f"   Parent: document-rag-orchestrator")
            print(f"   Middleware: TraceContextMiddleware")
            print(f"   Elasticsearch: {os.getenv('ELASTICSEARCH_URL')}")
            print(f"   Index: {os.getenv('ELASTICSEARCH_INDEX')}")
            
            # Service hierarchy
            hierarchy_info = SERVICE_HIERARCHY.get(self.service_name, {})
            children = hierarchy_info.get("children", [])
            if children:
                print(f"   Components: {', '.join(children)}")
            
            print()
            print("API Endpoints:")
            print(f"   Main UI:           http://{self.host}:{self.port}")
            print(f"   API Docs:          http://{self.host}:{self.port}/docs")
            print(f"   Query API:         http://{self.host}:{self.port}/api/query")
            print(f"   Health Check:      http://{self.host}:{self.port}/api/health")
            print(f"   Service Map:       http://{self.host}:{self.port}/api/service-map")
            
            print()
            print(f"Startup Trace ID: {get_current_trace_id()}")
            print(f"OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
            print("=" * 65)
            
            # Start FastAPI server
            try:
                import uvicorn
                
                with self.tracer.start_as_current_span("uvicorn_server_start") as uvicorn_span:
                    uvicorn_span.set_attributes({
                        "uvicorn.host": self.host,
                        "uvicorn.port": self.port,
                        "uvicorn.app": "ui.api_integrated_clean:app",
                        "vector_database": "elasticsearch"
                    })
                    
                    print("Starting FastAPI server with middleware...")
                    
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
                print(f"Failed to start API service: {e}")
                return False

def main():
    """Main entry point with parent context extraction"""
    
    # Extract parent context from environment if running as child process
    parent_trace_id = os.getenv("OTEL_PARENT_TRACE_ID")
    parent_service = os.getenv("OTEL_SERVICE_PARENT", "document-rag-orchestrator")
    
    # Create span in orchestrator context if available
    orchestrator_tracer = get_service_tracer(parent_service)

    tracer, meter, startup_span, context_token = initialize_service_with_parent_context("document-rag-api")
    
    with tracer.start_as_current_span("api_service_main") as main_span:
        main_span.set_attributes({
            "startup.mode": "launcher",
            "startup.parent_trace_id": parent_trace_id or "none",
            "startup.w3c_propagation": True,
            "service.hierarchy": f"{parent_service} -> document-rag-api",
            "vector_database": "elasticsearch"
        })
        
        launcher = APIServiceLauncher()
        
        try:
            success = launcher.start_service()
            
            if not success:
                main_span.set_attribute("startup.result", "failed")
                print("API service startup failed")
                sys.exit(1)
            
            main_span.set_attribute("startup.result", "success")
            
        except KeyboardInterrupt:
            main_span.add_event("shutdown_requested")
            print("\nShutdown signal received")
        except Exception as e:
            main_span.record_exception(e)
            main_span.set_attribute("startup.result", "error")
            print(f"API service error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    # Initialize with parent context
    tracer, meter, startup_span, context_token = initialize_service_with_parent_context("document-rag-api")

    print("ENHANCED API SERVICE LAUNCHER")
    print("W3C Trace Context Propagation: ENABLED")
    print("Service Hierarchy: orchestrator → api → components")
    print("Middleware-Based Trace Continuity: ACTIVE")
    print("Vector Database: Elasticsearch with Authentication")
    print()
    
    main()