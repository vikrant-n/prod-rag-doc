#!/usr/bin/env python3
"""
Test script to verify trace export to OTLP collector
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables explicitly
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "localhost:4317"
os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"
os.environ["OTEL_CONSOLE_EXPORT"] = "true"

# Import OpenTelemetry after setting environment
from otel_config import initialize_opentelemetry, get_tracer, shutdown_opentelemetry

def test_trace_export():
    """Test trace export to OTLP collector"""
    print("🔭 Initializing OpenTelemetry...")
    
    # Initialize OpenTelemetry
    tracer, meter = initialize_opentelemetry(
        service_name="test-trace-export",
        service_version="1.0.0",
        environment="development"
    )
    
    print(f"✅ OpenTelemetry initialized")
    print(f"   Service: test-trace-export")
    print(f"   Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
    print(f"   Protocol: {os.getenv('OTEL_EXPORTER_OTLP_PROTOCOL')}")
    
    # Create some test traces
    with tracer.start_as_current_span("test_operation") as span:
        span.set_attribute("test.attribute", "test_value")
        span.add_event("test_event", {"message": "This is a test event"})
        
        print("📊 Created test span with attributes and events")
        
        # Create a child span
        with tracer.start_as_current_span("child_operation") as child_span:
            child_span.set_attribute("child.attribute", "child_value")
            print("📊 Created child span")
            time.sleep(1)  # Simulate some work
    
    print("✅ Test traces created")
    print("🔄 Waiting for traces to be exported...")
    time.sleep(10)  # Wait longer for traces to be exported
    
    # Force flush of traces
    from opentelemetry import trace
    trace.get_tracer_provider().force_flush()
    print("🔄 Forced flush of traces")
    
    # Shutdown OpenTelemetry
    shutdown_opentelemetry()
    print("🛑 OpenTelemetry shutdown complete")

if __name__ == "__main__":
    test_trace_export() 