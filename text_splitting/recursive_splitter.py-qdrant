import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders.google_drive_loader import GoogleDriveMasterLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os

def split_documents(documents, chunk_size=1000, chunk_overlap=120):
    """
    Universal chunker with hierarchical and semantic splitting:
    - PPTX: group all content from a slide (title + bullets) into one chunk, split further if >chunk_size
    - DOCX/PDF: group by section/heading, split further if >chunk_size
    - Excel: group by row, include column headers, split further if needed
    - TXT: uses RecursiveCharacterTextSplitter
    - Images: returns as-is (with metadata)
    - Enriches metadata for all chunks
    - Filters out empty/boilerplate chunks
    """
    from collections import defaultdict
    from langchain_core.documents import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Validate input parameters
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        chunk_size = 1000
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        chunk_overlap = 120
        
    # Ensure overlap is not greater than chunk_size
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4  # Use 25% overlap as fallback
    chunks = []
    pptx_slides = defaultdict(list)
    docx_sections = defaultdict(list)
    pdf_sections = defaultdict(list)
    excel_sheets = defaultdict(list)
    other_docs = []
    
    # Validate documents input
    if not documents:
        return []
    
    # Ensure documents is a list
    if not isinstance(documents, list):
        documents = [documents]
    
    for doc in documents:
        # Skip invalid documents
        if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
            continue
        file_type = doc.metadata.get("file_type")
        if file_type == "pptx" and doc.metadata.get("content_type") == "slide_content":
            slide_number = doc.metadata.get("slide_number")
            pptx_slides[slide_number].append(doc)
        elif file_type == "docx" and doc.metadata.get("content_type") == "section":
            section = doc.metadata.get("section_title", "Unknown Section")
            docx_sections[section].append(doc)
        elif file_type == "pdf" and doc.metadata.get("content_type") == "section":
            section = doc.metadata.get("section_title", "Unknown Section")
            pdf_sections[section].append(doc)
        elif file_type == "excel" and doc.metadata.get("content_type") == "row":
            sheet = doc.metadata.get("sheet_name", "Sheet1")
            excel_sheets[sheet].append(doc)
        else:
            other_docs.append(doc)
    # PPTX: Chunk by slide, split further if needed
    for slide_number, slide_docs in pptx_slides.items():
        # Since each slide now has one comprehensive document, just use it directly
        for slide_doc in slide_docs:
            slide_text = slide_doc.page_content.strip()
            if not slide_text or len(slide_text.strip()) < 10:
                continue
            meta = dict(slide_doc.metadata)
            meta["document_type"] = meta.get("file_type", "pptx")
            meta["file_name"] = meta.get("drive_file_name", "unknown")
            meta["slide_number"] = slide_number
            meta["heading"] = meta.get("slide_title", "Unknown")
            meta["content_type"] = "slide_content"
            # Split if too large
            try:
                if len(slide_text) > chunk_size:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    text_chunks = splitter.split_text(slide_text)
                    for chunk in text_chunks:
                        if chunk and chunk.strip():
                            chunks.append(Document(page_content=chunk, metadata=meta))
                else:
                    chunks.append(Document(page_content=slide_text, metadata=meta))
            except Exception as e:
                # If splitting fails, add the original document
                chunks.append(Document(page_content=slide_text, metadata=meta))
    # DOCX: Chunk by section, split further if needed
    for section, docs_in_section in docx_sections.items():
        section_text = "\n".join(d.page_content.strip() for d in docs_in_section if d.page_content and d.page_content.strip())
        if not section_text or len(section_text.strip()) < 10:
            continue
        meta = dict(docs_in_section[0].metadata)
        meta["document_type"] = meta.get("file_type", "docx")
        meta["file_name"] = meta.get("file_name", "unknown")
        meta["section_title"] = section
        meta["heading"] = meta.get("section_title", "Unknown")
        meta["content_type"] = "section"
        if len(section_text) > chunk_size:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for chunk in splitter.split_text(section_text):
                chunks.append(Document(page_content=chunk, metadata=meta))
        else:
            chunks.append(Document(page_content=section_text, metadata=meta))
    # PDF: Chunk by section, split further if needed
    for section, docs_in_section in pdf_sections.items():
        section_text = "\n".join(d.page_content.strip() for d in docs_in_section if d.page_content and d.page_content.strip())
        if not section_text or len(section_text.strip()) < 10:
            continue
        meta = dict(docs_in_section[0].metadata)
        meta["document_type"] = meta.get("file_type", "pdf")
        meta["file_name"] = meta.get("file_name", "unknown")
        meta["section_title"] = section
        meta["heading"] = meta.get("section_title", "Unknown")
        meta["content_type"] = "section"
        if len(section_text) > chunk_size:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for chunk in splitter.split_text(section_text):
                chunks.append(Document(page_content=chunk, metadata=meta))
        else:
            chunks.append(Document(page_content=section_text, metadata=meta))
    # Excel: Chunk by row, include column headers
    for sheet, rows in excel_sheets.items():
        for row_doc in rows:
            if row_doc.page_content and len(row_doc.page_content.strip()) >= 10:
                meta = dict(row_doc.metadata)
                meta["document_type"] = meta.get("file_type", "excel")
                meta["file_name"] = meta.get("file_name", "unknown")
                meta["sheet_name"] = sheet
                meta["heading"] = meta.get("sheet_name", "Unknown")
                meta["content_type"] = "row"
                # Optionally, add column headers if available
                if "column_headers" in row_doc.metadata:
                    meta["column_headers"] = row_doc.metadata["column_headers"]
                # Split if row is very long
                if len(row_doc.page_content) > chunk_size:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    for chunk in splitter.split_text(row_doc.page_content):
                        chunks.append(Document(page_content=chunk, metadata=meta))
                else:
                    chunks.append(Document(page_content=row_doc.page_content, metadata=meta))
    # Other docs (TXT, images, etc.)
    for doc in other_docs:
        if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
            if not doc.page_content or len(doc.page_content.strip()) < 10:
                continue
            meta = dict(doc.metadata)
            meta["document_type"] = meta.get("file_type", "unknown")
            meta["file_name"] = meta.get("file_name", "unknown")
            meta["heading"] = meta.get("heading", "Unknown")
            meta["content_type"] = meta.get("content_type", "unknown")
            # For images, just attach metadata
            if doc.metadata.get("file_type") == "image":
                chunks.append(Document(page_content=doc.page_content, metadata=meta))
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                # Split the text content directly, then create Document objects
                text_chunks = splitter.split_text(doc.page_content)
                for chunk_text in text_chunks:
                    if chunk_text and len(chunk_text.strip()) >= 10:
                        chunk_meta = dict(meta)
                        chunk_meta["document_type"] = chunk_meta.get("file_type", meta["document_type"])
                        chunk_meta["file_name"] = chunk_meta.get("file_name", meta["file_name"])
                        chunk_meta["heading"] = chunk_meta.get("heading", meta["heading"])
                        chunk_meta["content_type"] = chunk_meta.get("content_type", meta["content_type"])
                        chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))
        else:
            chunks.append(doc)
    return chunks

# --- CONFIGURATION FROM ENVIRONMENT ---
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
VECTOR_SIZE = 3072  # For text-embedding-3-large

# Default chunking constants for export
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

# Note: FOLDER_ID validation moved to main() function to avoid import-time errors

def main():
    """Main function to process documents from Google Drive"""
    # Validate required environment variables
    if not FOLDER_ID:
        raise ValueError("GOOGLE_DRIVE_FOLDER_ID environment variable is required")
    
    # --- 1. Load all documents from Google Drive ---
    loader = GoogleDriveMasterLoader(
        folder_id=FOLDER_ID,
        credentials_path=CREDENTIALS_PATH,
        token_path=TOKEN_PATH,
        split=False  # We'll chunk manually
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} raw documents from Google Drive.")

    # --- 2. Chunk all documents using universal chunker ---
    chunks = split_documents(docs, chunk_size=1024, chunk_overlap=100)
    print(f"Created {len(chunks)} chunks from all Google Drive docs.")

    # --- 3. Connect to Qdrant and (re)create collection ---
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    if client.collection_exists(COLLECTION_NAME):
        print("Deleting existing collection...")
        client.delete_collection(collection_name=COLLECTION_NAME)
    print("Creating new collection...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"}
    )

    # --- 4. Embed and upload chunks to Qdrant ---
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_function
    )
    print("Uploading chunks to Qdrant...")
    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        vectorstore.add_documents(batch)
    print("Done! All chunks embedded and uploaded to Qdrant.")

    # --- (Optional) Print a few chunks for verification ---
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"--- Chunk {i} ---")
        print("Metadata:", chunk.metadata)
        print("Content:", chunk.page_content[:200], "...")

if __name__ == "__main__":
    main()

    