#!/usr/bin/env python3
"""
Complete System Startup Script

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

load_dotenv()

def check_environment():
    """Check if all required environment variables are set"""
    print("🔧 Checking environment configuration...")
    
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("   Please set them in your .env file")
        return False
    
    # Check optional but recommended variables
    if not os.getenv("GOOGLE_DRIVE_FOLDER_ID"):
        print("⚠️  Google Drive Folder ID not set - Google Drive monitoring will be disabled")
    else:
        print(f"✅ Google Drive monitoring enabled for folder: {os.getenv('GOOGLE_DRIVE_FOLDER_ID')}")
    
    print("✅ Environment configuration OK")
    return True

def check_qdrant():
    """Check if Qdrant is accessible"""
    print("🔍 Checking Qdrant connection...")
    
    try:
        import requests
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        response = requests.get(qdrant_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ Qdrant accessible at {qdrant_url}")
            return True
        else:
            print(f"⚠️  Qdrant responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to Qdrant: {e}")
        print("   Make sure Qdrant is running or update QDRANT_URL in .env")
        return False

def start_backend_service():
    """Start the backend document processing service"""
    print("🚀 Starting backend document processing service...")
    
    try:
        process = subprocess.Popen([
            sys.executable, "backend_service.py", 
            "--host", "0.0.0.0", 
            "--port", "8001"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Wait a bit to see if it starts successfully
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Backend service started successfully on port 8001")
            return process
        else:
            print("❌ Backend service failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend service: {e}")
        return None

def start_api_ui_server():
    """Start the API+UI server"""
    print("🌐 Starting API+UI server...")
    
    try:
        process = subprocess.Popen([
            sys.executable, "start_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Wait a bit to see if it starts successfully
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ API+UI server started successfully on port 8000")
            return process
        else:
            print("❌ API+UI server failed to start")
            return None
            
    except Exception as e:
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

def main():
    """Main startup function"""
    print("🎯 Starting Complete Document RAG System")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check Qdrant
    if not check_qdrant():
        print("⚠️  Continuing without Qdrant check - service will handle connection errors")
    
    print()
    print("🚀 Starting services...")
    print()
    
    # Start backend service
    backend_process = start_backend_service()
    if not backend_process:
        print("❌ Failed to start backend service")
        sys.exit(1)
    
    # Start API+UI server
    api_process = start_api_ui_server()
    if not api_process:
        print("❌ Failed to start API+UI server")
        backend_process.terminate()
        sys.exit(1)
    
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
    
    # Set up signal handler for clean shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutting down all services...")
        backend_process.terminate()
        api_process.terminate()
        
        # Wait for processes to terminate
        try:
            backend_process.wait(timeout=10)
            api_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing processes...")
            backend_process.kill()
            api_process.kill()
        
        print("✅ All services stopped")
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
    
    # Keep main thread alive
    try:
        while True:
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend service stopped unexpectedly")
                break
            if api_process.poll() is not None:
                print("❌ API service stopped unexpectedly")
                break
            
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    # Clean shutdown
    signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()