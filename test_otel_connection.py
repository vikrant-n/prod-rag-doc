#!/usr/bin/env python3
"""
Test OpenTelemetry Connection and Trace Export
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set required environment variables
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
os.environ.setdefault("OTEL_SERVICE_NAME", "document-rag-test")
os.environ.setdefault("OTEL_SERVICE_VERSION", "1.0.0")
os.environ.setdefault("OTEL_ENVIRONMENT", "development")

from otel_config import initialize_opentelemetry, get_tracer, shutdown_opentelemetry

def test_trace_export():
    """Test trace export to OpenTelemetry collector"""
    print("🔧 Initializing OpenTelemetry...")
    
    # Initialize OpenTelemetry
    tracer, meter = initialize_opentelemetry()
    
    print("🔭 Creating test trace...")
    
    # Create a test span
    with tracer.start_as_current_span("test_connection") as span:
        span.set_attribute("test.type", "connection_check")
        span.set_attribute("test.timestamp", time.time())
        
        print("   ✓ Test span created")
        
        # Create a child span
        with tracer.start_as_current_span("test_operation") as child_span:
            child_span.set_attribute("operation.type", "test")
            child_span.set_attribute("operation.result", "success")
            
            print("   ✓ Child span created")
            
            # Simulate some work
            time.sleep(1)
            
            print("   ✓ Operation completed")
    
    print("🔭 Test trace completed")
    
    # Wait a moment for export
    print("⏳ Waiting for trace export...")
    time.sleep(3)
    
    # Shutdown
    shutdown_opentelemetry()
    print("✅ Test completed")

if __name__ == "__main__":
    test_trace_export() 