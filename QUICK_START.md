# Quick Start Guide - Document RAG System

Get your Document RAG System running in under 10 minutes!

## 🚀 Prerequisites

- Python 3.9+ installed
- Docker (for Qdrant database)
- OpenAI API key
- Google Drive folder (optional)

## ⚡ 5-Minute Setup

### Step 1: Download & Setup
```bash
# Navigate to the project folder
cd document-rag-production

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start Qdrant Database
```bash
# Start Qdrant with Docker (one command!)
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### Step 3: Configure Environment
```bash
# Copy environment template
cp env.template .env

# Edit .env file (add your OpenAI API key)
nano .env  # or use any text editor
```

**Minimum required in .env:**
```bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_DRIVE_FOLDER_ID=your_google_drive_folder_id
```

### Step 4: Launch System
```bash
# Start the complete system
python start_complete_system.py
```

## 🌐 Access Your System

Once running, open these URLs:

- **Main Interface**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **System Status**: http://localhost:8001/status

## 📝 Test Your System

1. **Upload a document** to your Google Drive folder
2. **Wait 30 seconds** for automatic processing
3. **Go to** http://localhost:8000
4. **Ask a question** about your document
5. **Get AI-powered answers** with source citations!

## 🔧 Troubleshooting

**Issue: "Could not connect to Qdrant"**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not running, start it
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

**Issue: "Missing API key"**
```bash
# Verify your .env file
cat .env | grep OPENAI_API_KEY
```

**Issue: "Google Drive not working"**
- Google Drive integration is optional
- System works without it for manual document upload
- Check `GOOGLE_DRIVE_FOLDER_ID` in .env if needed

## 🎯 What's Next?

- **Add more documents** to your Google Drive folder
- **Customize the UI** by editing `ui/static/` files
- **Scale for production** using the DEPLOYMENT_GUIDE.md
- **Monitor performance** via the status endpoints

## 💡 Pro Tips

1. **Use text-embedding-3-large** for best quality (default)
2. **Use text-embedding-3-small** for faster processing
3. **Check logs** if something isn't working: `tail -f backend_service.log`
4. **Manual document scan**: Visit http://localhost:8001/scan

---

**Need help?** Check the full README.md for detailed documentation!