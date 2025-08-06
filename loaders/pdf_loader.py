# To run: python3 -m loaders.pdf_loader from the project root
import os
from typing import List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
from loaders.image_extractor import extract_images_from_pdf
from tools.vision_tool import analyze_images_with_vision_model, VISION_PROMPT

# OCR imports
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: easyocr not installed. OCR text extraction will be skipped.")

load_dotenv()

PDF_IMAGE_DIR = "pdf_images"
os.makedirs(PDF_IMAGE_DIR, exist_ok=True)

def extract_text_with_ocr(image_path: str) -> str:
    """Extract text from a PDF image using OCR."""
    if not OCR_AVAILABLE:
        return ""
    
    try:
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image_path)
        
        # Combine all detected text
        ocr_text = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Only include high-confidence text
                ocr_text.append(text)
        
        return "\n".join(ocr_text)
    except Exception as e:
        print(f"OCR extraction failed for {image_path}: {e}")
        return ""

def load_pdf_as_langchain_docs(file_path: str) -> List[Document]:
    """
    Load a PDF file and return a list of LangChain Document objects with comprehensive text extraction.
    
    Uses multiple extraction methods:
    1. Traditional text extraction via PDFPlumber
    2. OCR on extracted images
    3. Vision AI analysis for context
    
    Args:
        file_path (str): Path to the PDF file to process.
        
    Returns:
        List[Document]: List of LangChain Document objects containing all extractable content.
    """
    docs = []
    
    # 1. Traditional text and tables extraction
    loader = PDFPlumberLoader(file_path)
    text_docs = loader.load()
    
    # 2. Extract images from PDF
    images = extract_images_from_pdf(file_path, PDF_IMAGE_DIR)
    
    # 3. Enhanced Vision AI analysis with comprehensive prompt
    vision_prompt = """
    Analyze this PDF page/image and extract:
    1. ALL visible text (exact transcription)
    2. Key concepts and information
    3. Context and meaning
    4. Any structured data (lists, tables, diagrams, charts)
    5. Technical diagrams or flowcharts
    6. Any text that might be in images or graphics
    
    If this appears to be a document page, transcribe all text exactly as it appears.
    Focus on extracting actionable information that would be useful for answering questions.
    Include any text that might be embedded in images, charts, or diagrams.
    """
    
    image_docs = analyze_images_with_vision_model(images, prompt=vision_prompt)
    
    # 4. Add OCR text extraction to image documents
    for img_doc in image_docs:
        img_path = img_doc.metadata.get("image_path")
        if img_path and os.path.exists(img_path):
            ocr_text = extract_text_with_ocr(img_path)
            if ocr_text:
                # Combine vision AI analysis with OCR text
                combined_content = f"{img_doc.page_content}\n\n[OCR Extracted Text]\n{ocr_text}"
                img_doc.page_content = combined_content
                img_doc.metadata["extraction_methods"] = "vision_ai,ocr"
            else:
                img_doc.metadata["extraction_methods"] = "vision_ai"
    
    # 5. Enhance text documents with comprehensive metadata
    for doc in text_docs:
        doc.metadata["file_type"] = "pdf"
        doc.metadata["extraction_methods"] = "pdfplumber,vision_ai,ocr"
        doc.metadata["content_type"] = "text"
    
    # 6. Link images to text chunks by page number
    page_to_images = {}
    for img_doc in image_docs:
        page = img_doc.metadata.get("page", img_doc.metadata.get("page_number"))
        page_to_images.setdefault(page, []).append(img_doc.metadata.get("image_path"))
    
    for doc in text_docs:
        page = doc.metadata.get("page", doc.metadata.get("page_number"))
        if page in page_to_images:
            doc.metadata["related_images"] = page_to_images[page]
    
    # 7. Return all documents
    return text_docs + image_docs

def test_pdf_loader():
    sample_path = "sample.pdf"  # Place a sample PDF in the project root for testing
    print(f"Testing comprehensive PDF loader with multi-modal extraction: {sample_path}")
    if not os.path.isfile(sample_path):
        print("[Test Skipped] sample.pdf not found.")
        return
    docs = load_pdf_as_langchain_docs(sample_path)
    print(f"Loaded {len(docs)} Document objects.")
    for doc in docs[:3]:
        print("---\nMetadata:", doc.metadata)
        print("Content (first 300 chars):", doc.page_content[:300])
    print(docs)

if __name__ == "__main__":
    test_pdf_loader() 