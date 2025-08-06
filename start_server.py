#!/usr/bin/env python3
"""
Startup script for the Document RAG FastAPI server
with comprehensive OpenTelemetry instrumentation
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Initialize OpenTelemetry early
from otel_config import initialize_opentelemetry, shutdown_opentelemetry, trace_function, traced_operation
from opentelemetry import trace, baggage

# Initialize OpenTelemetry for the API server
tracer, meter = initialize_opentelemetry(
    service_name="document-rag-api",
    service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
    environment=os.getenv("OTEL_ENVIRONMENT", "development")
)

@trace_function("api_server.main", {"component": "api_server", "operation": "startup"})
def main():
    """Start the FastAPI server"""
    with traced_operation("api_server_startup") as span:
        try:
            # Check if required environment variables are set
            required_vars = ["OPENAI_API_KEY"]
            optional_vars = ["COHERE_API_KEY", "QDRANT_HOST", "QDRANT_PORT"]
            
            span.set_attribute("config.required_vars_count", len(required_vars))
            span.set_attribute("config.optional_vars_count", len(optional_vars))
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                span.set_attribute("config.missing_vars", missing_vars)
                span.add_event("missing_environment_variables", {"missing_vars": missing_vars})
                print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
                print("Please set them in your .env file or environment")
                sys.exit(1)
            
            # Set default values for optional variables
            os.environ.setdefault("QDRANT_HOST", "localhost")
            os.environ.setdefault("QDRANT_PORT", "6333")
            
            qdrant_host = os.getenv("QDRANT_HOST")
            qdrant_port = os.getenv("QDRANT_PORT")
            
            span.set_attribute("config.qdrant_host", qdrant_host)
            span.set_attribute("config.qdrant_port", qdrant_port)
            span.set_attribute("server.host", "0.0.0.0")
            span.set_attribute("server.port", 8000)
            span.add_event("api_server_configuration_complete", {
                "qdrant_host": qdrant_host,
                "qdrant_port": qdrant_port,
                "server_port": 8000
            })
            
            print("🚀 Starting Document RAG FastAPI server...")
            print(f"📍 Server will be available at: http://localhost:8000")
            print(f"📊 API documentation at: http://localhost:8000/docs")
            print(f"🔍 Qdrant connection: {qdrant_host}:{qdrant_port}")
            
            # Start the server
            span.add_event("uvicorn_server_starting")
            uvicorn.run(
                "ui.api_integrated:app",
                host="0.0.0.0",
                port=8000,
                reload=False,  # Set to True for development
                log_level="info"
            )
        except KeyboardInterrupt:
            span.add_event("server_stopped_by_user")
            print("\n🛑 Server stopped by user")
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("startup.result", "error")
            span.add_event("api_server_startup_failed", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            print(f"❌ Error starting server: {e}")
            sys.exit(1)
        finally:
            shutdown_opentelemetry()

if __name__ == "__main__":
    main()