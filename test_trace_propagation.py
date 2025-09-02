#!/usr/bin/env python3
"""
Test script to verify trace propagation in the Document RAG system
"""

import os
import time
import asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenTelemetry
os.environ["OTEL_SERVICE_NAME"] = "trace-test"
from otel_config import initialize_opentelemetry, get_current_trace_id, TracedHTTPXClient

def test_trace_propagation():
    """Test trace propagation between services"""
    
    print("🔍 Testing Trace Propagation in Document RAG System")
    print("=" * 60)
    
    # Initialize OpenTelemetry
    tracer, meter = initialize_opentelemetry(
        service_name="trace-test",
        service_version="1.0.0",
        environment="test"
    )
    
    with tracer.start_as_current_span("test_root_span") as root_span:
        root_trace_id = get_current_trace_id()
        root_span.set_attributes({
            "test.type": "trace_propagation",
            "test.timestamp": datetime.now().isoformat()
        })
        
        print(f"🆔 Root Trace ID: {root_trace_id}")
        print(f"🆔 Root Span ID: {format(root_span.get_span_context().span_id, '016x')}")
        
        # Test child span creation
        with tracer.start_as_current_span("test_child_span") as child_span:
            child_trace_id = get_current_trace_id()
            child_span.set_attributes({
                "test.level": "child",
                "test.parent_trace": root_trace_id
            })
            
            print(f"👶 Child Trace ID: {child_trace_id}")
            print(f"👶 Child Span ID: {format(child_span.get_span_context().span_id, '016x')}")
            
            # Verify trace IDs match
            if root_trace_id == child_trace_id and root_trace_id != "00000000000000000000000000000000":
                print("✅ Trace propagation working correctly")
            else:
                print("❌ Trace propagation failed")
                print(f"   Root: {root_trace_id}")
                print(f"   Child: {child_trace_id}")
                
        print()
        print("🌐 Testing HTTP Client Trace Propagation")
        
        async def test_http_propagation():
            try:
                async with TracedHTTPXClient(service_name="test-client") as client:
                    # Test backend health check with trace propagation
                    response = await client.get("http://localhost:8001/health", timeout=5.0)
                    print(f"✅ Backend health check: {response.status_code}")
                    
                    # Test API health check with trace propagation
                    response = await client.get("http://localhost:8000/api/health", timeout=5.0)
                    print(f"✅ API health check: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ HTTP test failed: {e}")
        
        # Run async HTTP test
        asyncio.run(test_http_propagation())
        
        print()
        print("📊 Test Results Summary:")
        print(f"   Root Trace ID: {root_trace_id}")
        print(f"   Trace Valid: {'Yes' if root_trace_id != '00000000000000000000000000000000' else 'No'}")
        print(f"   OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
        print(f"   Service Name: trace-test")
        
        # Add a small delay to ensure span export
        time.sleep(2)

if __name__ == "__main__":
    test_trace_propagation()
