# Document RAG System - Production Ready

A production-ready Document Retrieval-Augmented Generation (RAG) system that processes documents from Google Drive and provides intelligent query responses using OpenAI embeddings and Qdrant vector database.

## 🎯 What is This System?

The Document RAG System is an intelligent document processing and query system that:

- **Automatically monitors** a Google Drive folder for new documents
- **Processes multiple file formats** (PDF, DOCX, PPTX, Excel, CSV, Images)
- **Extracts text and images** using advanced OCR and Vision AI
- **Creates vector embeddings** for semantic search
- **Provides intelligent Q&A** about your documents with source citations
- **Maintains conversation context** for better user experience

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Google Drive  │───▶│  Backend Service │───▶│  Qdrant Vector  │
│   (Documents)   │    │   (Port 8001)    │    │    Database     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   API + UI       │
                       │  (Port 8000)     │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Web Interface  │
                       │  (Chat UI)       │
                       └──────────────────┘
```

### Core Services

1. **Backend Service** (`backend_service.py`): Document processing, Google Drive monitoring
2. **API Server** (`start_server.py`): FastAPI server with query processing
3. **Web Interface** (`ui/static/`): Modern chat interface for document queries
4. **Orchestrator** (`start_complete_system.py`): Manages all services

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** installed
2. **Qdrant vector database** running on port 6333
3. **OpenAI API key**
4. **Google Drive API credentials** (optional, for document monitoring)

### System Requirements

- **RAM**: 4GB+ (8GB+ recommended)
- **Storage**: 10GB+ free space
- **Network**: Internet connection for API calls
- **OS**: Linux, macOS, or Windows

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

### 5. Run the System

```bash
# Start the complete system (backend + API + UI)
python start_complete_system.py
```

### Alternative Startup Methods

**Option A: Run Individual Services**
```bash
# Terminal 1: Start backend service
python backend_service.py

# Terminal 2: Start API server
python start_server.py
```

**Option B: Use the launcher script**
```bash
python run_system.py
```

**Option C: Debug mode (test imports first)**
```bash
python debug_api.py
```

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
├── start_server.py       # API + UI server (port 8000)
├── run_system.py         # Alternative launcher
├── debug_api.py          # Debug and testing tool
└── ui/
    ├── api_integrated.py    # FastAPI application
    ├── api_enhanced.py      # Enhanced features
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

- **PDF** files (.pdf) - Text extraction + OCR + Vision AI
- **Word** documents (.docx) - Text + tables + images + OCR
- **PowerPoint** presentations (.pptx) - Slides + text + images
- **Excel** spreadsheets (.xlsx) - Data + formulas + charts
- **CSV** files (.csv) - Tabular data processing
- **Images** (.png, .jpg, .jpeg) - OCR + Vision AI analysis

### Document Processing Pipeline

```
Google Drive → Document Loaders → Text Splitting → OpenAI Embeddings → Qdrant Storage
```

### Supported Document Types

- **PDF** files (.pdf) - Text extraction + OCR + Vision AI
- **Word** documents (.docx) - Text + tables + images + OCR
- **PowerPoint** presentations (.pptx) - Slides + text + images
- **Excel** spreadsheets (.xlsx) - Data + formulas + charts
- **CSV** files (.csv) - Tabular data processing
- **Images** (.png, .jpg, .jpeg) - OCR + Vision AI analysis

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
- **OpenTelemetry Integration**: Comprehensive tracing and monitoring
- **Vision AI Analysis**: Advanced image understanding with OpenAI Vision
- **OCR Support**: Text extraction from images using EasyOCR
- **Hybrid Search**: Combines vector search with BM25 for better results

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
- `GET /api/health` - Health check
- `GET /api/backend/status` - Backend service status
- `POST /api/backend/scan` - Trigger manual document scan
- `GET /docs` - Interactive API documentation

### Development Tools

- `debug_api.py` - Test imports and API creation
- `run_system.py` - Alternative system launcher
- OpenTelemetry integration for debugging and monitoring

## 📊 Monitoring & Logs

### Log Files

- `backend_service.log` - Document processing logs
- Console output - Real-time system status
- OpenTelemetry traces and metrics

### Health Checks

- Backend Status: http://localhost:8001/status
- Manual Scan: http://localhost:8001/scan
- System URLs displayed on startup
- OpenTelemetry metrics endpoint (if configured)

### Monitoring Features

- **Real-time Logging**: Correlated logs with trace IDs
- **Performance Metrics**: Document processing times, API response times
- **Service Health**: Backend and API service status monitoring
- **Error Tracking**: Comprehensive error logging with context

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