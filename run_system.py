#!/usr/bin/env python3
"""
Run Document RAG System without OpenTelemetry auto-instrumentation
This avoids conflicts with our manual instrumentation
"""

import os
import subprocess
import sys

def main():
    """Run the system without opentelemetry-instrument wrapper"""
    print("🚀 Starting Document RAG System (Manual Instrumentation)")
    print("=" * 60)
    
    # Set environment variables for OpenTelemetry
    env = os.environ.copy()
    env.update({
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://172.31.41.170:4317",
        "OTEL_SERVICE_NAME": "document-rag-orchestrator",  # Will be overridden per service
        "OTEL_SERVICE_VERSION": "1.0.0",
        "OTEL_ENVIRONMENT": "development",  # Explicitly set to development
        "SCAN_INTERVAL": "30",  # 30 seconds scan interval
        "OTEL_RESOURCE_ATTRIBUTES": "service.namespace=document-rag-system,deployment.environment=development",
    })
    
    # Run start_complete_system.py directly
    try:
        subprocess.run([sys.executable, "start_complete_system.py"], env=env)
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Error starting system: {e}")

if __name__ == "__main__":
    main()