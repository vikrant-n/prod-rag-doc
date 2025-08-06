#!/bin/bash

# Start Jaeger for OpenTelemetry tracing
# This script starts Jaeger all-in-one container for local development

set -e

echo "🔍 Starting Jaeger for OpenTelemetry tracing..."
echo "============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop existing Jaeger container if running
if docker ps -q -f name=jaeger > /dev/null; then
    echo "🛑 Stopping existing Jaeger container..."
    docker stop jaeger > /dev/null
fi

# Remove existing Jaeger container if exists
if docker ps -a -q -f name=jaeger > /dev/null; then
    echo "🗑️  Removing existing Jaeger container..."
    docker rm jaeger > /dev/null
fi

# Start Jaeger all-in-one
echo "🚀 Starting Jaeger all-in-one container..."
docker run -d \
  --name jaeger \
  -p 16686:16686 \
  -p 14250:14250 \
  -p 14268:14268 \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -e COLLECTOR_OTLP_ENABLED=true \
  jaegertracing/all-in-one:latest

# Wait for Jaeger to be ready
echo "⏳ Waiting for Jaeger to be ready..."
sleep 5

# Check if Jaeger is accessible
if curl -s http://localhost:16686/api/services > /dev/null; then
    echo "✅ Jaeger is running and accessible!"
    echo ""
    echo "🌐 Access Points:"
    echo "   • Jaeger UI:      http://localhost:16686"
    echo "   • OTLP gRPC:      localhost:14250"
    echo "   • OTLP HTTP:      localhost:14268"
    echo "   • Jaeger Thrift:  localhost:6831 (UDP)"
    echo ""
    echo "🔧 To use with your application, set:"
    echo "   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:14268"
    echo "   OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf"
    echo ""
    echo "🛑 To stop Jaeger:"
    echo "   docker stop jaeger && docker rm jaeger"
else
    echo "⚠️  Jaeger may not be fully ready yet. Please wait a moment and try accessing:"
    echo "   http://localhost:16686"
fi

echo ""
echo "🎉 Jaeger setup complete!"