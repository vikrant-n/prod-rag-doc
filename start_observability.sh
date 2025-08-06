#!/bin/bash

# Start Observability Stack for Document RAG System
# This script starts Jaeger, Prometheus, Grafana, and the OpenTelemetry Collector
# with proper network configuration and health checks

set -e

echo "🔭 Starting OpenTelemetry Observability Stack..."
echo "================================================"

# Check if Docker is running
if ! sudo docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
sudo docker-compose -f docker-compose.otel.yml down --remove-orphans 2>/dev/null || true
sudo docker stop jaeger grafana prometheus otel-collector 2>/dev/null || true
sudo docker rm jaeger grafana prometheus otel-collector 2>/dev/null || true

# Create necessary directories with proper permissions
sudo mkdir -p grafana/provisioning/dashboards
sudo mkdir -p grafana/provisioning/datasources
sudo chmod -R 755 grafana/

# Create Grafana datasource configuration
sudo tee grafana/provisioning/datasources/datasources.yml > /dev/null << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
EOF

# Create Grafana dashboard provisioning
sudo tee grafana/provisioning/dashboards/dashboards.yml > /dev/null << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

echo "📁 Created Grafana configuration files"

# Fix OpenTelemetry Collector configuration for proper networking
echo "🔧 Ensuring OpenTelemetry Collector configuration is correct..."
if grep -q "endpoint: 172\." otel-collector-config.yaml; then
    echo "   Fixing collector endpoint to use container hostname..."
    sed -i 's/endpoint: 172\.[0-9]*\.[0-9]*\.[0-9]*:14250/endpoint: jaeger:14250/g' otel-collector-config.yaml
fi

# Start the observability stack
echo "🚀 Starting services with docker-compose..."
sudo docker-compose -f docker-compose.otel.yml up -d

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        echo "   Attempt $attempt/$max_attempts - $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "⚠️  $service_name may not be fully ready after $max_attempts attempts"
    return 1
}

# Wait for services to be ready with proper health checks
echo "🔍 Performing comprehensive health checks..."

# Check Jaeger (wait for it first since collector depends on it)
wait_for_service "Jaeger" "http://localhost:16686/api/services"

# Check Prometheus
wait_for_service "Prometheus" "http://localhost:9090/-/healthy"

# Check Grafana
wait_for_service "Grafana" "http://localhost:3000/api/health"

# Check OpenTelemetry Collector (should be last since it depends on others)
wait_for_service "OpenTelemetry Collector" "http://localhost:8889/metrics"

# Additional verification - check if collector can connect to Jaeger
echo "🔗 Verifying OpenTelemetry Collector connectivity..."
sleep 5
if sudo docker logs otel-collector 2>&1 | grep -q "connection refused\|connection error"; then
    echo "⚠️  OpenTelemetry Collector may have connectivity issues. Restarting..."
    sudo docker restart otel-collector
    sleep 10
fi

# Final verification
if sudo docker logs otel-collector --tail 5 2>&1 | grep -q "Everything is ready"; then
    echo "✅ OpenTelemetry Collector is properly connected and ready"
else
    echo "⚠️  OpenTelemetry Collector may need manual verification"
fi

echo ""
echo "🎉 Observability stack started successfully!"
echo "================================================"
echo "📊 Access your observability tools:"
echo "   • Jaeger (Tracing):     http://localhost:16686"
echo "   • Prometheus (Metrics): http://localhost:9090"
echo "   • Grafana (Dashboards): http://localhost:3000 (admin/admin)"
echo ""
echo "🔧 OpenTelemetry Collector endpoints:"
echo "   • gRPC:  localhost:4317"
echo "   • HTTP:  localhost:4318"
echo "   • Metrics: http://localhost:8889/metrics"
echo ""
echo "🔍 Container Status:"
sudo docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(jaeger|prometheus|grafana|otel-collector)"
echo ""
echo "📋 Configuration Summary:"
echo "   • All services are on 'otel-network' for proper connectivity"
echo "   • OpenTelemetry Collector forwards traces to Jaeger"
echo "   • Prometheus scrapes metrics from the collector"
echo "   • Grafana has pre-configured datasources for both Prometheus and Jaeger"
echo ""
echo "▶️  Now start your Document RAG system:"
echo "   python start_complete_system.py"
echo ""
echo "🔍 To view traces in Jaeger:"
echo "   1. Start your application: python start_complete_system.py"
echo "   2. Open Jaeger UI: http://localhost:16686"
echo "   3. Select service: 'document-rag-orchestrator' or 'document-rag-api'"
echo "   4. Click 'Find Traces' to see your application traces"
echo ""
echo "🛑 To stop the observability stack:"
echo "   sudo docker-compose -f docker-compose.otel.yml down"
echo "   # Or use: sudo docker stop jaeger prometheus grafana otel-collector"
echo ""