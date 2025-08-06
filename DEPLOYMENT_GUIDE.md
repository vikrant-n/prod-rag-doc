# Production Deployment Guide

This guide provides step-by-step instructions for deploying the Document RAG System in production environments.

## 🏗️ Production Architecture

```
Internet → Load Balancer → API Servers (Port 8000)
                       → Backend Services (Port 8001)
                       → Qdrant Cluster (Port 6333)
                       → Shared Storage (Documents/Logs)
```

## 🐳 Docker Deployment (Recommended)

### 1. Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for generated files
RUN mkdir -p docx_images pptx_images pdf_images

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "start_complete_system.py"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
    restart: unless-stopped

  document-rag:
    build: .
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - QDRANT_URL=http://qdrant:6333
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    depends_on:
      - qdrant
    restart: unless-stopped

volumes:
  qdrant_data:
```

### 3. Deploy with Docker Compose

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f document-rag

# Scale API servers
docker-compose up -d --scale document-rag=3
```

## ☁️ Cloud Deployment Options

### AWS Deployment

#### Using ECS Fargate

1. **Create ECR Repository**
```bash
aws ecr create-repository --repository-name document-rag
```

2. **Build and Push Image**
```bash
# Get login token
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

# Build and tag image
docker build -t document-rag .
docker tag document-rag:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/document-rag:latest

# Push image
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/document-rag:latest
```

3. **Create ECS Task Definition**
```json
{
  "family": "document-rag",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "document-rag",
      "image": "<account-id>.dkr.ecr.us-west-2.amazonaws.com/document-rag:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        },
        {
          "containerPort": 8001,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "QDRANT_URL",
          "value": "http://qdrant-service:6333"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:<account-id>:secret:openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/document-rag",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Using Lambda (Serverless)

```python
# lambda_handler.py
import json
from mangum import Mangum
from ui.api_integrated import app

handler = Mangum(app)

def lambda_handler(event, context):
    return handler(event, context)
```

### Google Cloud Deployment

#### Using Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/document-rag
gcloud run deploy document-rag \
  --image gcr.io/PROJECT_ID/document-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 1 \
  --set-env-vars QDRANT_URL=http://qdrant-service:6333
```

### Azure Deployment

#### Using Container Instances

```bash
# Create resource group
az group create --name document-rag-rg --location eastus

# Deploy container
az container create \
  --resource-group document-rag-rg \
  --name document-rag \
  --image your-registry/document-rag:latest \
  --ports 8000 8001 \
  --environment-variables QDRANT_URL=http://qdrant:6333 \
  --secure-environment-variables OPENAI_API_KEY=your-key \
  --memory 2 \
  --cpu 1
```

## 🔧 Production Configuration

### 1. Environment Variables

```bash
# Production .env
OPENAI_API_KEY=sk-prod-...
GOOGLE_DRIVE_FOLDER_ID=1ABcD2EfG3HiJ4KlM5NpQ6RsT7UvW8XyZ9
QDRANT_URL=http://qdrant-cluster:6333
LOG_LEVEL=INFO
ENVIRONMENT=production

# Performance Settings
CHUNK_SIZE=2000
CHUNK_OVERLAP=200
MAX_CONCURRENT_REQUESTS=100

# Security Settings
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com
CORS_ORIGINS=https://yourdomain.com
```

### 2. Reverse Proxy Configuration

#### Nginx Configuration

```nginx
upstream document_rag_api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8002;  # Additional instances
    server 127.0.0.1:8003;
}

upstream document_rag_backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8004;  # Additional instances
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL Configuration
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # API Routes
    location /api/ {
        proxy_pass http://document_rag_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Backend Routes
    location /backend/ {
        proxy_pass http://document_rag_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Static Files
    location / {
        proxy_pass http://document_rag_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Health Check
    location /health {
        access_log off;
        proxy_pass http://document_rag_api/health;
    }
}
```

## 📊 Monitoring & Observability

### 1. Application Monitoring

```python
# monitoring.py
import logging
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')

# Start metrics server
start_http_server(9090)
```

### 2. Log Aggregation

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  document-rag:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  elasticsearch:
    image: elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  kibana:
    image: kibana:7.14.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

  logstash:
    image: logstash:7.14.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch
```

### 3. Health Checks

```python
# health.py
from fastapi import APIRouter
from qdrant_client import QdrantClient
import openai

router = APIRouter()

@router.get("/health")
async def health_check():
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check Qdrant
    try:
        qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))
        qdrant.get_collections()
        checks["services"]["qdrant"] = "healthy"
    except Exception as e:
        checks["services"]["qdrant"] = f"unhealthy: {str(e)}"
        checks["status"] = "unhealthy"
    
    # Check OpenAI
    try:
        openai.models.list()
        checks["services"]["openai"] = "healthy"
    except Exception as e:
        checks["services"]["openai"] = f"unhealthy: {str(e)}"
        checks["status"] = "unhealthy"
    
    return checks
```

## 🔒 Security Hardening

### 1. API Security

```python
# security.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Apply to protected routes
@app.post("/api/query")
async def query_endpoint(request: QueryRequest, user=Depends(verify_token)):
    # Protected endpoint logic
    pass
```

### 2. Rate Limiting

```python
# rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/query")
@limiter.limit("10/minute")
async def query_endpoint(request: Request, query: QueryRequest):
    # Rate-limited endpoint
    pass
```

## 🚀 Performance Optimization

### 1. Caching Strategy

```python
# caching.py
import redis
import json
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_key(query: str) -> str:
    return f"query:{hashlib.md5(query.encode()).hexdigest()}"

def get_cached_result(query: str):
    key = cache_key(query)
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None

def cache_result(query: str, result: dict, ttl: int = 3600):
    key = cache_key(query)
    redis_client.setex(key, ttl, json.dumps(result))
```

### 2. Database Optimization

```python
# qdrant_optimization.py
from qdrant_client.http.models import OptimizersConfigDiff, VectorParams

# Optimize collection settings
qdrant.update_collection(
    collection_name="documents",
    optimizer_config=OptimizersConfigDiff(
        indexing_threshold=20000,
        max_segment_size=100000,
        max_optimization_threads=4
    )
)
```

## 📋 Production Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database backups configured
- [ ] Monitoring setup
- [ ] Log aggregation configured
- [ ] Security review completed
- [ ] Performance testing completed
- [ ] Disaster recovery plan documented

### Post-Deployment
- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] Log rotation configured
- [ ] Backup verification
- [ ] Performance metrics baseline established
- [ ] Security scanning completed
- [ ] Documentation updated

### Ongoing Maintenance
- [ ] Regular security updates
- [ ] Performance monitoring
- [ ] Log analysis
- [ ] Backup verification
- [ ] Capacity planning
- [ ] API key rotation

## 🆘 Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Backup Qdrant data
docker exec qdrant-container qdrant-backup create backup-$(date +%Y%m%d)

# Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz .env credentials.json

# Upload to cloud storage
aws s3 cp backup-$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
```

### Recovery Procedures

```bash
#!/bin/bash
# restore.sh

# Restore Qdrant data
docker exec qdrant-container qdrant-backup restore backup-20241201

# Restore configuration
tar -xzf config-backup-20241201.tar.gz

# Restart services
docker-compose restart
```

This deployment guide provides comprehensive instructions for production deployment across various platforms and environments.