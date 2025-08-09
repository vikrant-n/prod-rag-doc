#!/usr/bin/env python3
"""
Complete System Startup Script - Instrumented with OpenTelemetry

Starts both the backend document processing service and the API+UI server
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path
from dotenv import load_dotenv

# Load environment first
load_dotenv()

# Initialize OpenTelemetry before other imports
from otel_config import initialize_opentelemetry, get_tracer
from metrics import rag_metrics
import logging

# Initialize OpenTelemetry
tracer, meter = initialize_opentelemetry(
    service_name="document-rag-orchestrator",
    service_version="1.0.0",
    environment="development"
)

logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set"""
    with tracer.start_as_current_span("system.check_environment") as span:
        print("🔧 Checking environment configuration...")
        
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        span.set_attribute("environment.required_vars", len(required_vars))
        span.set_attribute("environment.missing_vars", len(missing_vars))
        
        if missing_vars:
            span.set_attribute("environment.status", "missing_vars")
            span.record_exception(Exception(f"Missing variables: {missing_vars}"))
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            print("   Please set them in your .env file")
            return False
        
        # Check optional but recommended variables
        gdrive_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if not gdrive_id:
            span.set_attribute("google_drive.enabled", False)
            print("⚠️  Google Drive Folder ID not set - Google Drive monitoring will be disabled")
        else:
            span.set_attribute("google_drive.enabled", True)
            span.set_attribute("google_drive.folder_id", gdrive_id[:10] + "...")  # Truncated for security
            print(f"✅ Google Drive monitoring enabled for folder: {gdrive_id}")
        
        span.set_attribute("environment.status", "ok")
        print("✅ Environment configuration OK")
        logger.info("Environment configuration validated successfully")
        return True

def check_qdrant():
    """Check if Qdrant is accessible"""
    with tracer.start_as_current_span("system.check_qdrant") as span:
        print("🔍 Checking Qdrant connection...")
        
        try:
            import requests
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            span.set_attribute("qdrant.url", qdrant_url)
            
            with tracer.start_as_current_span("qdrant.health_check") as health_span:
                response = requests.get(qdrant_url, timeout=5)
                health_span.set_attribute("http.status_code", response.status_code)
                health_span.set_attribute("http.url", qdrant_url)
                
                if response.status_code == 200:
                    span.set_attribute("qdrant.status", "healthy")
                    print(f"✅ Qdrant accessible at {qdrant_url}")
                    logger.info(f"Qdrant health check passed: {qdrant_url}")
                    return True
                else:
                    span.set_attribute("qdrant.status", "unhealthy")
                    span.set_attribute("qdrant.status_code", response.status_code)
                    print(f"⚠️  Qdrant responded with status {response.status_code}")
                    logger.warning(f"Qdrant unhealthy - status code: {response.status_code}")
                    return False
                    
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("qdrant.status", "connection_failed")
            print(f"❌ Could not connect to Qdrant: {e}")
            print("   Make sure Qdrant is running or update QDRANT_URL in .env")
            logger.error(f"Qdrant connection failed: {e}")
            return False

def start_backend_service():
    """Start the backend document processing service"""
    with tracer.start_as_current_span("system.start_backend_service") as span:
        print("🚀 Starting backend document processing service...")
        
        try:
            span.set_attribute("service.name", "backend_service")
            span.set_attribute("service.port", 8001)
            
            process = subprocess.Popen([
                sys.executable, "backend_service.py", 
                "--host", "0.0.0.0", 
                "--port", "8001"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            span.set_attribute("process.pid", process.pid)
            
            # Wait a bit to see if it starts successfully
            time.sleep(3)
            
            if process.poll() is None:
                span.set_attribute("service.status", "started")
                print("✅ Backend service started successfully on port 8001")
                logger.info(f"Backend service started with PID: {process.pid}")
                rag_metrics.record_cache_event("service_start", True)
                return process
            else:
                span.set_attribute("service.status", "failed")
                span.record_exception(Exception("Backend service failed to start"))
                print("❌ Backend service failed to start")
                logger.error("Backend service failed to start")
                rag_metrics.record_cache_event("service_start", False)
                return None
                
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("service.status", "error")
            print(f"❌ Error starting backend service: {e}")
            logger.error(f"Error starting backend service: {e}")
            return None

def start_api_ui_server():
    """Start the API+UI server"""
    with tracer.start_as_current_span("system.start_api_server") as span:
        print("🌐 Starting API+UI server...")
        
        try:
            span.set_attribute("service.name", "api_ui_server")
            span.set_attribute("service.port", 8000)
            
            process = subprocess.Popen([
                sys.executable, "start_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            span.set_attribute("process.pid", process.pid)
            
            # Wait a bit to see if it starts successfully
            time.sleep(3)
            
            if process.poll() is None:
                span.set_attribute("service.status", "started")
                print("✅ API+UI server started successfully on port 8000")
                logger.info(f"API+UI server started with PID: {process.pid}")
                rag_metrics.record_cache_event("service_start", True)
                return process
            else:
                span.set_attribute("service.status", "failed")
                span.record_exception(Exception("API+UI server failed to start"))
                print("❌ API+UI server failed to start")
                logger.error("API+UI server failed to start")
                rag_metrics.record_cache_event("service_start", False)
                return None
                
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("service.status", "error")
            print(f"❌ Error starting API+UI server: {e}")
            logger.error(f"Error starting API+UI server: {e}")
            return None

def monitor_process(process, name):
    """Monitor a process and log its output"""
    with tracer.start_as_current_span(f"system.monitor_process.{name.lower()}"):
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[{name}] {line.rstrip()}")
                    # Log critical errors with trace correlation
                    if "error" in line.lower() or "failed" in line.lower():
                        logger.error(f"[{name}] {line.rstrip()}")
        except Exception as e:
            logger.error(f"Error monitoring {name}: {e}")

def main():
    """Main startup function"""
    with tracer.start_as_current_span("system.startup") as main_span:
        print("🎯 Starting Complete Document RAG System")
        print("=" * 60)
        
        main_span.set_attribute("system.name", "document-rag")
        main_span.set_attribute("system.version", "1.0.0")
        
        # Change to script directory
        script_dir = Path(__file__).parent
        os.chdir(script_dir)
        main_span.set_attribute("system.working_directory", str(script_dir))
        print(f"📁 Working directory: {script_dir}")
        
        # Check environment
        if not check_environment():
            main_span.set_attribute("startup.status", "failed")
            main_span.record_exception(Exception("Environment check failed"))
            sys.exit(1)
        
        # Check Qdrant
        qdrant_ok = check_qdrant()
        if not qdrant_ok:
            main_span.set_attribute("qdrant.startup_check", "failed")
            print("⚠️  Continuing without Qdrant check - service will handle connection errors")
        else:
            main_span.set_attribute("qdrant.startup_check", "passed")
        
        print()
        print("🚀 Starting services...")
        print()
        
        # Start backend service
        backend_process = start_backend_service()
        if not backend_process:
            main_span.set_attribute("startup.status", "backend_failed")
            print("❌ Failed to start backend service")
            sys.exit(1)
        
        # Start API+UI server
        api_process = start_api_ui_server()
        if not api_process:
            main_span.set_attribute("startup.status", "api_failed")
            print("❌ Failed to start API+UI server")
            backend_process.terminate()
            sys.exit(1)
        
        main_span.set_attribute("startup.status", "success")
        main_span.set_attribute("backend_process.pid", backend_process.pid)
        main_span.set_attribute("api_process.pid", api_process.pid)
        
        print()
        print("✅ All services started successfully!")
        print()
        print("🌐 System URLs:")
        print("   📊 Main UI:           http://localhost:8000")
        print("   🔧 API Documentation: http://localhost:8000/docs")
        print("   ⚙️  Backend Status:    http://localhost:8001/status")
        print("   🔍 Manual Scan:       http://localhost:8001/scan")
        print()
        print("📋 System Components:")
        print("   🔄 Backend Service:    Monitors Google Drive & processes documents")
        print("   🌐 API Server:         Handles queries & retrieval")
        print("   💻 Web UI:            User interface for document search")
        print()
        print("Press Ctrl+C to stop all services")
        print("=" * 60)
        
        logger.info("Document RAG system startup completed successfully")
        
        # Set up signal handler for clean shutdown
        def signal_handler(signum, frame):
            with tracer.start_as_current_span("system.shutdown") as shutdown_span:
                print("\n🛑 Shutting down all services...")
                shutdown_span.set_attribute("shutdown.signal", signum)
                
                backend_process.terminate()
                api_process.terminate()
                
                # Wait for processes to terminate
                try:
                    backend_process.wait(timeout=10)
                    api_process.wait(timeout=10)
                    shutdown_span.set_attribute("shutdown.graceful", True)
                except subprocess.TimeoutExpired:
                    print("⚠️  Force killing processes...")
                    backend_process.kill()
                    api_process.kill()
                    shutdown_span.set_attribute("shutdown.graceful", False)
                
                print("✅ All services stopped")
                logger.info("System shutdown completed")
                sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start monitoring threads
        backend_thread = threading.Thread(
            target=monitor_process, 
            args=(backend_process, "BACKEND"),
            daemon=True
        )
        api_thread = threading.Thread(
            target=monitor_process, 
            args=(api_process, "API"),
            daemon=True
        )
        
        backend_thread.start()
        api_thread.start()
        
        # Keep main thread alive with monitoring
        try:
            while True:
                # Check if processes are still running
                if backend_process.poll() is not None:
                    main_span.add_event("backend_service_stopped")
                    logger.error("Backend service stopped unexpectedly")
                    print("❌ Backend service stopped unexpectedly")
                    break
                if api_process.poll() is not None:
                    main_span.add_event("api_service_stopped")
                    logger.error("API service stopped unexpectedly")
                    print("❌ API service stopped unexpectedly")
                    break
                
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        # Clean shutdown
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()