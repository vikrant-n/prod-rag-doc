#!/usr/bin/env python3
"""
Complete System Startup Script

Starts both the backend document processing service and the API+UI server
with comprehensive OpenTelemetry instrumentation for traces, metrics, and logs.
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Initialize OpenTelemetry early
from otel_config import initialize_opentelemetry, shutdown_opentelemetry, trace_function, traced_operation
from opentelemetry import trace, metrics, baggage
from opentelemetry.trace import Status, StatusCode

# Initialize OpenTelemetry for the main system
tracer, meter = initialize_opentelemetry(
    service_name="document-rag-orchestrator",
    service_version="1.0.0"
)

# Create metrics for system monitoring
process_counter = meter.create_counter(
    "system_processes_started",
    description="Number of system processes started",
    unit="1"
)

process_gauge = meter.create_up_down_counter(
    "system_processes_active",
    description="Number of active system processes",
    unit="1"
)

startup_duration = meter.create_histogram(
    "system_startup_duration",
    description="Time taken to start the complete system",
    unit="s"
)

health_check_counter = meter.create_counter(
    "system_health_checks",
    description="Number of health checks performed",
    unit="1"
)

@trace_function("system.check_environment", {"component": "orchestrator", "operation": "environment_check"})
def check_environment():
    """Check if all required environment variables are set"""
    with traced_operation("environment_validation") as span:
        print("🔧 Checking environment configuration...")
        
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = []
        
        span.set_attribute("env.required_vars.count", len(required_vars))
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        span.set_attribute("env.missing_vars.count", len(missing_vars))
        
        if missing_vars:
            span.set_attribute("env.validation.result", "failed")
            span.set_attribute("env.missing_vars", ",".join(missing_vars))
            span.add_event("environment_validation_failed", {
                "missing_variables": missing_vars,
                "required_count": len(required_vars)
            })
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            print("   Please set them in your .env file")
            return False
        
        # Check optional but recommended variables
        google_drive_folder = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        span.set_attribute("env.google_drive.configured", bool(google_drive_folder))
        
        if not google_drive_folder:
            print("⚠️  Google Drive Folder ID not set - Google Drive monitoring will be disabled")
            span.add_event("google_drive_monitoring_disabled")
        else:
            print(f"✅ Google Drive monitoring enabled for folder: {google_drive_folder}")
            span.set_attribute("env.google_drive.folder_id", google_drive_folder[:8] + "...")  # Truncate for security
            span.add_event("google_drive_monitoring_enabled")
        
        span.set_attribute("env.validation.result", "success")
        span.add_event("environment_validation_success")
        print("✅ Environment configuration OK")
        return True

@trace_function("system.check_qdrant", {"component": "orchestrator", "operation": "qdrant_health_check"})
def check_qdrant():
    """Check if Qdrant is accessible"""
    with traced_operation("qdrant_health_check") as span:
        print("🔍 Checking Qdrant connection...")
        
        try:
            import requests
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            
            span.set_attribute("qdrant.url", qdrant_url)
            span.set_attribute("qdrant.timeout", 5)
            span.add_event("qdrant_connection_attempt", {"url": qdrant_url})
            
            health_check_counter.add(1, {"service": "qdrant", "type": "connection_check"})
            
            response = requests.get(qdrant_url, timeout=5)
            
            span.set_attribute("qdrant.response.status_code", response.status_code)
            span.set_attribute("qdrant.response.time", response.elapsed.total_seconds())
            
            if response.status_code == 200:
                span.set_attribute("qdrant.connection.result", "success")
                span.add_event("qdrant_connection_success", {
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                })
                print(f"✅ Qdrant accessible at {qdrant_url}")
                return True
            else:
                span.set_attribute("qdrant.connection.result", "failed")
                span.set_attribute("qdrant.error.type", "http_error")
                span.add_event("qdrant_connection_failed", {
                    "status_code": response.status_code,
                    "reason": "unexpected_status_code"
                })
                print(f"⚠️  Qdrant responded with status {response.status_code}")
                return False
                
        except Exception as e:
            span.set_attribute("qdrant.connection.result", "failed")
            span.set_attribute("qdrant.error.type", "connection_error")
            span.set_attribute("qdrant.error.message", str(e))
            span.record_exception(e)
            span.add_event("qdrant_connection_error", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            print(f"❌ Could not connect to Qdrant: {e}")
            print("   Make sure Qdrant is running or update QDRANT_URL in .env")
            return False

@trace_function("system.start_backend_service", {"component": "orchestrator", "operation": "service_startup"})
def start_backend_service():
    """Start the backend document processing service"""
    with traced_operation("backend_service_startup") as span:
        print("🚀 Starting backend document processing service...")
        
        try:
            span.set_attribute("service.name", "backend_service")
            span.set_attribute("service.port", 8001)
            span.set_attribute("service.host", "0.0.0.0")
            span.add_event("service_startup_attempt", {"service": "backend"})
            
            # Set trace context in environment for child process
            env = os.environ.copy()
            # Inject trace context into environment for the child process
            from opentelemetry.propagate import inject
            headers = {}
            inject(headers)
            for key, value in headers.items():
                env[f"OTEL_PROPAGATED_{key.upper().replace('-', '_')}"] = value
            
            process = subprocess.Popen([
                sys.executable, "backend_service.py", 
                "--host", "0.0.0.0", 
                "--port", "8001"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            
            span.set_attribute("process.pid", process.pid)
            span.add_event("process_created", {"pid": process.pid})
            
            # Wait a bit to see if it starts successfully
            startup_wait_time = 3
            span.set_attribute("startup.wait_time", startup_wait_time)
            time.sleep(startup_wait_time)
            
            if process.poll() is None:
                span.set_attribute("service.startup.result", "success")
                span.set_attribute("service.status", "running")
                span.add_event("service_startup_success", {
                    "service": "backend",
                    "pid": process.pid,
                    "port": 8001
                })
                process_counter.add(1, {"service": "backend", "status": "started"})
                process_gauge.add(1, {"service": "backend"})
                print("✅ Backend service started successfully on port 8001")
                return process
            else:
                span.set_attribute("service.startup.result", "failed")
                span.set_attribute("service.status", "failed")
                span.set_attribute("process.exit_code", process.poll())
                span.add_event("service_startup_failed", {
                    "service": "backend",
                    "exit_code": process.poll(),
                    "reason": "process_exited_early"
                })
                print("❌ Backend service failed to start")
                return None
                
        except Exception as e:
            span.set_attribute("service.startup.result", "failed")
            span.set_attribute("service.error.type", type(e).__name__)
            span.set_attribute("service.error.message", str(e))
            span.record_exception(e)
            span.add_event("service_startup_error", {
                "service": "backend",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            print(f"❌ Error starting backend service: {e}")
            return None

@trace_function("system.start_api_ui_server", {"component": "orchestrator", "operation": "service_startup"})
def start_api_ui_server():
    """Start the API+UI server"""
    with traced_operation("api_ui_server_startup") as span:
        print("🌐 Starting API+UI server...")
        
        try:
            span.set_attribute("service.name", "api_ui_server")
            span.set_attribute("service.port", 8000)
            span.add_event("service_startup_attempt", {"service": "api_ui"})
            
            # Set trace context in environment for child process
            env = os.environ.copy()
            # Inject trace context into environment for the child process
            from opentelemetry.propagate import inject
            headers = {}
            inject(headers)
            for key, value in headers.items():
                env[f"OTEL_PROPAGATED_{key.upper().replace('-', '_')}"] = value
            
            process = subprocess.Popen([
                sys.executable, "start_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            
            span.set_attribute("process.pid", process.pid)
            span.add_event("process_created", {"pid": process.pid})
            
            # Wait a bit to see if it starts successfully
            startup_wait_time = 3
            span.set_attribute("startup.wait_time", startup_wait_time)
            time.sleep(startup_wait_time)
            
            if process.poll() is None:
                span.set_attribute("service.startup.result", "success")
                span.set_attribute("service.status", "running")
                span.add_event("service_startup_success", {
                    "service": "api_ui",
                    "pid": process.pid,
                    "port": 8000
                })
                process_counter.add(1, {"service": "api_ui", "status": "started"})
                process_gauge.add(1, {"service": "api_ui"})
                print("✅ API+UI server started successfully on port 8000")
                return process
            else:
                span.set_attribute("service.startup.result", "failed")
                span.set_attribute("service.status", "failed")
                span.set_attribute("process.exit_code", process.poll())
                span.add_event("service_startup_failed", {
                    "service": "api_ui",
                    "exit_code": process.poll(),
                    "reason": "process_exited_early"
                })
                print("❌ API+UI server failed to start")
                return None
                
        except Exception as e:
            span.set_attribute("service.startup.result", "failed")
            span.set_attribute("service.error.type", type(e).__name__)
            span.set_attribute("service.error.message", str(e))
            span.record_exception(e)
            span.add_event("service_startup_error", {
                "service": "api_ui",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            print(f"❌ Error starting API+UI server: {e}")
            return None

def monitor_process(process, name):
    """Monitor a process and log its output"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[{name}] {line.rstrip()}")
    except Exception as e:
        print(f"❌ Error monitoring {name}: {e}")

@trace_function("system.main", {"component": "orchestrator", "operation": "system_startup"})
def main():
    """Main startup function with comprehensive OpenTelemetry instrumentation"""
    startup_start_time = time.time()
    
    with traced_operation("complete_system_startup") as main_span:
        print("🎯 Starting Complete Document RAG System")
        print("=" * 60)
        
        main_span.set_attribute("system.name", "document-rag-system")
        main_span.set_attribute("system.version", "1.0.0")
        main_span.set_attribute("startup.timestamp", int(startup_start_time))
        main_span.add_event("system_startup_initiated")
        
        # Change to script directory
        script_dir = Path(__file__).parent
        os.chdir(script_dir)
        main_span.set_attribute("system.working_directory", str(script_dir))
        print(f"📁 Working directory: {script_dir}")
        
        try:
            # Check environment
            main_span.add_event("environment_check_start")
            if not check_environment():
                main_span.set_attribute("startup.result", "failed")
                main_span.set_attribute("startup.failure_reason", "environment_check_failed")
                main_span.add_event("system_startup_failed", {"reason": "environment_check"})
                sys.exit(1)
            main_span.add_event("environment_check_complete")
            
            # Check Qdrant
            main_span.add_event("qdrant_check_start")
            qdrant_available = check_qdrant()
            main_span.set_attribute("dependencies.qdrant.available", qdrant_available)
            
            if not qdrant_available:
                main_span.add_event("qdrant_check_warning", {"message": "Qdrant not available, continuing anyway"})
                print("⚠️  Continuing without Qdrant check - service will handle connection errors")
            main_span.add_event("qdrant_check_complete")
            
            print()
            print("🚀 Starting services...")
            print()
            
            # Start backend service
            main_span.add_event("backend_service_start_attempt")
            backend_process = start_backend_service()
            if not backend_process:
                main_span.set_attribute("startup.result", "failed")
                main_span.set_attribute("startup.failure_reason", "backend_service_failed")
                main_span.add_event("system_startup_failed", {"reason": "backend_service"})
                print("❌ Failed to start backend service")
                sys.exit(1)
            
            main_span.set_attribute("services.backend.pid", backend_process.pid)
            main_span.add_event("backend_service_started", {"pid": backend_process.pid})
            
            # Start API+UI server
            main_span.add_event("api_ui_service_start_attempt")
            api_process = start_api_ui_server()
            if not api_process:
                main_span.set_attribute("startup.result", "failed")
                main_span.set_attribute("startup.failure_reason", "api_ui_service_failed")
                main_span.add_event("system_startup_failed", {"reason": "api_ui_service"})
                print("❌ Failed to start API+UI server")
                backend_process.terminate()
                process_gauge.add(-1, {"service": "backend"})
                sys.exit(1)
            
            main_span.set_attribute("services.api_ui.pid", api_process.pid)
            main_span.add_event("api_ui_service_started", {"pid": api_process.pid})
            
            startup_duration_seconds = time.time() - startup_start_time
            startup_duration.record(startup_duration_seconds)
            main_span.set_attribute("startup.duration_seconds", startup_duration_seconds)
            main_span.set_attribute("startup.result", "success")
            main_span.add_event("system_startup_complete", {
                "duration_seconds": startup_duration_seconds,
                "services_started": 2
            })
            
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
            print("🔭 OpenTelemetry:")
            print(f"   📈 Traces: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4317')}")
            print(f"   🏷️  Service: {os.getenv('OTEL_SERVICE_NAME', 'document-rag-orchestrator')}")
            print()
            print("Press Ctrl+C to stop all services")
            print("=" * 60)
            
            # Set up signal handler for clean shutdown
            def signal_handler(signum, frame):
                with traced_operation("system_shutdown") as shutdown_span:
                    print("\n🛑 Shutting down all services...")
                    shutdown_span.set_attribute("shutdown.signal", signum)
                    shutdown_span.add_event("shutdown_initiated", {"signal": signum})
                    
                    # Update metrics
                    process_gauge.add(-1, {"service": "backend"})
                    process_gauge.add(-1, {"service": "api_ui"})
                    
                    backend_process.terminate()
                    api_process.terminate()
                    shutdown_span.add_event("processes_terminated")
                    
                    # Wait for processes to terminate
                    try:
                        backend_process.wait(timeout=10)
                        api_process.wait(timeout=10)
                        shutdown_span.add_event("processes_gracefully_stopped")
                    except subprocess.TimeoutExpired:
                        print("⚠️  Force killing processes...")
                        shutdown_span.add_event("force_killing_processes")
                        backend_process.kill()
                        api_process.kill()
                        shutdown_span.add_event("processes_force_killed")
                    
                    shutdown_span.set_attribute("shutdown.result", "success")
                    shutdown_span.add_event("shutdown_complete")
                    print("✅ All services stopped")
                    
                    # Shutdown OpenTelemetry
                    shutdown_opentelemetry()
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
            main_span.add_event("monitoring_threads_started")
            
            # Keep main thread alive and monitor processes
            with traced_operation("system_monitoring") as monitor_span:
                monitor_span.set_attribute("monitoring.active", True)
                try:
                    while True:
                        # Check if processes are still running
                        if backend_process.poll() is not None:
                            monitor_span.add_event("backend_service_unexpected_stop", {
                                "exit_code": backend_process.poll()
                            })
                            print("❌ Backend service stopped unexpectedly")
                            process_gauge.add(-1, {"service": "backend"})
                            break
                        if api_process.poll() is not None:
                            monitor_span.add_event("api_service_unexpected_stop", {
                                "exit_code": api_process.poll()
                            })
                            print("❌ API service stopped unexpectedly")
                            process_gauge.add(-1, {"service": "api_ui"})
                            break
                        
                        time.sleep(1)
                except KeyboardInterrupt:
                    monitor_span.add_event("keyboard_interrupt_received")
                    pass
                finally:
                    monitor_span.set_attribute("monitoring.active", False)
            
            # Clean shutdown
            signal_handler(signal.SIGINT, None)
        
        except Exception as e:
            main_span.record_exception(e)
            main_span.set_attribute("startup.result", "failed")
            main_span.set_attribute("startup.error.type", type(e).__name__)
            main_span.set_attribute("startup.error.message", str(e))
            main_span.add_event("system_startup_exception", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            print(f"❌ System startup failed with exception: {e}")
            shutdown_opentelemetry()
            raise

if __name__ == "__main__":
    main()