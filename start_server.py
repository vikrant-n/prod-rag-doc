#!/usr/bin/env python3
"""
Startup script for the Document RAG FastAPI server
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the FastAPI server"""
    
    # Check if required environment variables are set
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["COHERE_API_KEY", "QDRANT_HOST", "QDRANT_PORT"]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment")
        sys.exit(1)
    
    # Set default values for optional variables
    os.environ.setdefault("QDRANT_HOST", "localhost")
    os.environ.setdefault("QDRANT_PORT", "6333")
    
    print("🚀 Starting Document RAG FastAPI server...")
    print(f"📍 Server will be available at: http://localhost:8000")
    print(f"📊 API documentation at: http://localhost:8000/docs")
    print(f"🔍 Qdrant connection: {os.getenv('QDRANT_HOST')}:{os.getenv('QDRANT_PORT')}")
    
    # Start the server
    try:
        uvicorn.run(
            "ui.api_integrated:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Set to True for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()