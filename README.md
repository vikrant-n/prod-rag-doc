# Document RAG System - Production Ready

A production-ready Document Retrieval-Augmented Generation (RAG) system that processes documents from Google Drive and provides intelligent query responses using OpenAI embeddings and Qdrant vector database.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** installed
2. **Qdrant vector database** running on port 6333
3. **OpenAI API key**
4. **Google Drive API credentials** (optional, for document monitoring)

### 1. Installation

```bash
# Clone or download this folder
cd document-rag-production

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env.template .env

# Edit .env file with your credentials
nano .env  # or use your preferred editor
```

**Required Environment Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key
- `GOOGLE_DRIVE_FOLDER_ID`: Google Drive folder ID to monitor
- `QDRANT_URL`: Qdrant database URL (default: http://localhost:6333)

### 3. Setup Qdrant Database

**Option A: Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Local Installation**
```bash
# Follow Qdrant installation guide
# https://qdrant.tech/documentation/guides/installation/
```

### 4. Google Drive Setup (Optional)

If you want automatic document monitoring:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Drive API
4. Create credentials (OAuth 2.0 or Service Account)
5. Download `credentials.json` to the project root
6. Set `GOOGLE_DRIVE_FOLDER_ID` in `.env`

### 5. Start Observability Stack (Optional but Recommended)

For production monitoring and debugging, start the OpenTelemetry observability stack:

```bash
# Start Jaeger, Prometheus, Grafana, and OpenTelemetry Collector
./start_observability.sh
```

This provides:
- **🔍 Distributed Tracing**: End-to-end request tracing with Jaeger
- **📊 Metrics Collection**: Performance metrics with Prometheus  
- **📈 Visualization**: Dashboards with Grafana
- **🔗 Correlation**: Unified trace and span IDs across all services

**Observability Access Points:**
- **Jaeger (Tracing)**: http://localhost:16686
- **Prometheus (Metrics)**: http://localhost:9090
- **Grafana (Dashboards)**: http://localhost:3000 (admin/admin)

### 6. Run the System

```bash
# Start the complete system (backend + API + UI)
python start_complete_system.py
```

The system now includes comprehensive OpenTelemetry instrumentation for:
- ✅ Distributed tracing across all services
- ✅ Performance metrics and monitoring
- ✅ Structured logging with trace correlation
- ✅ Real-time observability dashboards

## 🌐 Access Points

Once running, access these URLs:

- **Main UI**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Backend Status**: http://localhost:8001/status
- **Manual Document Scan**: http://localhost:8001/scan

## 📁 System Architecture

### Core Components

```
start_complete_system.py  # Main entry point
├── backend_service.py    # Document processing service (port 8001)
└── start_server.py       # API + UI server (port 8000)
    └── ui/
        ├── api_integrated.py    # FastAPI application
        └── static/
            ├── index.html       # Web interface
            ├── styles.css       # UI styling
            └── app.js          # Frontend logic
```

### Document Processing Pipeline

```
Google Drive → Document Loaders → Text Splitting → OpenAI Embeddings → Qdrant Storage
```

### Supported Document Types

- **PDF** files (.pdf)
- **Word** documents (.docx)
- **PowerPoint** presentations (.pptx)
- **Excel** spreadsheets (.xlsx)
- **CSV** files (.csv)
- **Images** with OCR support (.png, .jpg, .jpeg)

## 🔧 Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | - | OpenAI API key for embeddings and chat |
| `GOOGLE_DRIVE_FOLDER_ID` | ✅ | - | Google Drive folder to monitor |
| `QDRANT_URL` | ❌ | http://localhost:6333 | Qdrant database URL |
| `QDRANT_COLLECTION` | ❌ | documents | Collection name in Qdrant |
| `EMBEDDING_MODEL` | ❌ | text-embedding-3-large | OpenAI embedding model |
| `CHUNK_SIZE` | ❌ | 3000 | Text chunk size for processing |
| `CHUNK_OVERLAP` | ❌ | 300 | Overlap between text chunks |
| `LOG_LEVEL` | ❌ | INFO | Logging level |

### Advanced Features

- **Duplicate Source Deduplication**: UI automatically consolidates duplicate sources
- **Real-time Document Monitoring**: Automatically processes new Google Drive files
- **Intelligent Chunking**: Optimized text splitting for better retrieval
- **Multi-format Support**: Handles various document formats seamlessly
- **🔭 OpenTelemetry Observability**: Comprehensive distributed tracing, metrics, and logging
  - End-to-end request tracing across all services
  - Performance metrics and real-time monitoring
  - Structured logging with trace correlation
  - Integration with Jaeger, Prometheus, and Grafana

## 🛠️ Development & Customization

### Adding New Document Types

1. Create a new loader in `loaders/` directory
2. Add the loader to `loaders/master_loaders.py`
3. Update file type detection logic

### Customizing UI

- Modify `ui/static/index.html` for layout changes
- Update `ui/static/styles.css` for styling
- Edit `ui/static/app.js` for functionality changes

### API Endpoints

The system provides RESTful API endpoints:

- `POST /api/query` - Submit queries
- `GET /api/status` - System status
- `GET /docs` - Interactive API documentation

## 📊 Monitoring & Observability

### OpenTelemetry Integration

The system includes comprehensive observability with OpenTelemetry:

- **Distributed Tracing**: View end-to-end request flows in Jaeger
- **Metrics Collection**: Monitor performance with Prometheus
- **Structured Logging**: Correlated logs with trace context
- **Real-time Dashboards**: Visualize system health in Grafana

**Quick Start:**
```bash
./start_observability.sh  # Start observability stack
python start_complete_system.py  # Start instrumented application
```

**Access Points:**
- **Jaeger**: http://localhost:16686 (Distributed tracing)
- **Prometheus**: http://localhost:9090 (Metrics)
- **Grafana**: http://localhost:3000 (Dashboards, admin/admin)

### Log Files

- `backend_service.log` - Document processing logs with trace context
- Console output - Real-time system status with OpenTelemetry spans

### Health Checks

- Backend Status: http://localhost:8001/status
- Manual Scan: http://localhost:8001/scan
- OpenTelemetry Collector: http://localhost:8889/metrics
- System URLs displayed on startup

### Key Metrics to Monitor

- Query processing latency (p95, p99)
- Document processing throughput
- Error rates and success rates
- Active WebSocket connections
- Database operation performance

For detailed setup instructions, see [OPENTELEMETRY_SETUP.md](OPENTELEMETRY_SETUP.md)

## 🔒 Security Considerations

1. **API Keys**: Never commit `.env` files to version control
2. **Network Access**: Configure firewall rules for production
3. **Authentication**: Consider adding authentication for production use
4. **HTTPS**: Use reverse proxy (nginx) with SSL for production

## 🚨 Troubleshooting

### Common Issues

**1. "Could not connect to Qdrant"**
```bash
# Check if Qdrant is running
curl http://localhost:6333

# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant
```

**2. "Missing required environment variables"**
```bash
# Verify .env file exists and has required variables
cat .env | grep OPENAI_API_KEY
```

**3. "Google Drive authentication failed"**
```bash
# Check credentials.json exists
ls -la credentials.json

# Verify Google Drive API is enabled in Google Cloud Console
```

**4. "Backend service failed to start"**
```bash
# Check logs for specific error
tail -f backend_service.log

# Verify all dependencies are installed
pip install -r requirements.txt
```

### Performance Optimization

1. **Embedding Model**: Use `text-embedding-3-small` for faster processing
2. **Chunk Size**: Reduce `CHUNK_SIZE` for faster indexing
3. **Qdrant**: Use persistent storage for production
4. **Batch Processing**: Process multiple documents simultaneously

## 📈 Scaling for Production

### Horizontal Scaling

1. **Load Balancer**: Use nginx or similar
2. **Multiple Instances**: Run multiple API servers
3. **Database**: Use Qdrant cluster for high availability
4. **Caching**: Implement Redis for query caching

### Deployment Options

**Docker Deployment:**
```bash
# Create Dockerfile (not included, but recommended)
docker build -t document-rag .
docker run -p 8000:8000 -p 8001:8001 document-rag
```

**Cloud Deployment:**
- AWS: Use ECS or Lambda
- Google Cloud: Use Cloud Run or GKE
- Azure: Use Container Instances or AKS

## 📝 License

This project is provided as-is for production use. Ensure compliance with:
- OpenAI API Terms of Service
- Google Drive API Terms of Service
- Qdrant License Terms

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review system logs for error messages
3. Verify all environment variables are correctly set
4. Ensure all dependencies are properly installed

---

**System Requirements:**
- Python 3.9+
- 4GB+ RAM (8GB+ recommended)
- 10GB+ disk space
- Internet connection for API calls

**Production Checklist:**
- [ ] Environment variables configured
- [ ] Qdrant database running
- [ ] Google Drive credentials setup
- [ ] SSL/HTTPS configured
- [ ] Monitoring and logging setup
- [ ] Backup strategy implemented
- [ ] Security review completed