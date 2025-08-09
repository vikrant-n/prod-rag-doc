#!/usr/bin/env python3
"""
Debug API startup issues
"""

import sys
import traceback

def test_imports():
    """Test all imports required by the API"""
    print("🔍 Testing imports...")
    
    try:
        print("  ✓ Basic imports...")
        import asyncio
        import logging
        import os
        import json
        import uuid
        from datetime import datetime, timezone
        from typing import Dict, List, Optional, Any
        from dataclasses import dataclass
        from enum import Enum
        print("  ✅ Basic imports OK")
        
        print("  ✓ FastAPI imports...")
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import HTMLResponse, JSONResponse
        from pydantic import BaseModel, Field
        print("  ✅ FastAPI imports OK")
        
        print("  ✓ Environment...")
        from dotenv import load_dotenv
        load_dotenv()
        print("  ✅ Environment OK")
        
        print("  ✓ OpenTelemetry...")
        from otel_config import initialize_opentelemetry, get_tracer, add_trace_correlation_to_log
        print("  ✅ OpenTelemetry OK")
        
        print("  ✓ Metrics...")
        from metrics import (
            rag_metrics, time_query_processing, record_query_processed, 
            record_cache_event
        )
        print("  ✅ Metrics OK")
        
        print("  ✓ LangChain...")
        from qdrant_client import QdrantClient
        from langchain_openai import OpenAIEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from langchain.docstore.document import Document
        print("  ✅ LangChain OK")
        
        print("  ✓ HTTP client...")
        import httpx
        print("  ✅ HTTP client OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_api_creation():
    """Test creating the FastAPI app"""
    print("🔍 Testing API creation...")
    
    try:
        from fastapi import FastAPI
        
        app = FastAPI(
            title="Test API",
            description="Test API creation",
            version="1.0.0"
        )
        
        print("  ✅ FastAPI app created successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ API creation failed: {e}")
        traceback.print_exc()
        return False

def test_otel_init():
    """Test OpenTelemetry initialization"""
    print("🔍 Testing OpenTelemetry initialization...")
    
    try:
        from otel_config import initialize_opentelemetry
        
        tracer, meter = initialize_opentelemetry(
            service_name="test-api",
            service_version="1.0.0",
            environment="development"
        )
        
        print("  ✅ OpenTelemetry initialized successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ OpenTelemetry initialization failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 API Startup Debug Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("API Creation Test", test_api_creation),
        ("OpenTelemetry Test", test_otel_init)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! API should start successfully.")
    else:
        print("\n⚠️  Some tests failed. Fix issues before starting API.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)