import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from docx import Document as DocxDocument
import mammoth
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loaders.image_extractor import extract_images_from_docx
from tools.vision_tool import analyze_images_with_vision_model, VISION_PROMPT

# OCR imports
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: easyocr not installed. OCR text extraction will be skipped.")

load_dotenv()

DOCX_IMAGE_DIR = "docx_images"
os.makedirs(DOCX_IMAGE_DIR, exist_ok=True)

def extract_text_with_ocr(image_path: str) -> str:
    """Extract text from a DOCX image using OCR."""
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

def extract_tables(doc: DocxDocument) -> List[Dict[str, Any]]:
    """Extract tables from the document with their structure preserved."""
    tables = []
    for table in doc.tables:
        # Get header row
        header = []
        for cell in table.rows[0].cells:
            header.append(cell.text.strip())
        
        # Get data rows
        data = []
        for row in table.rows[1:]:
            row_data = {}
            for i, cell in enumerate(row.cells):
                if i < len(header):  # Ensure we have a header for this column
                    row_data[header[i]] = cell.text.strip()
            data.append(row_data)
        
        tables.append({
            "header": header,
            "data": data
        })
    return tables

def extract_structured_content(doc: DocxDocument) -> List[Dict[str, Any]]:
    """Extract content with structure (paragraphs, headings, lists) preserved."""
    content = []
    
    for paragraph in doc.paragraphs:
        # Skip empty paragraphs
        if not paragraph.text.strip():
            continue
        
        # Determine content type based on style
        style = paragraph.style.name.lower()
        content_type = "paragraph"
        if "heading" in style:
            content_type = f"heading{style[-1]}" if style[-1].isdigit() else "heading"
        elif "list" in style:
            content_type = "list_item"
            
        content.append({
            "type": content_type,
            "text": paragraph.text.strip(),
            "style": style
        })
    
    return content

def extract_semantic_html(file_path: str) -> str:
    """Convert DOCX to semantic HTML using Mammoth."""
    with open(file_path, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file)
        return result.value

def parse_html_content(html: str) -> List[Dict[str, Any]]:
    """Parse HTML content into structured blocks."""
    soup = BeautifulSoup(html, 'html.parser')
    content = []
    
    for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'table']):
        if element.name.startswith('h'):
            content.append({
                "type": f"heading{element.name[1]}",
                "text": element.get_text().strip()
            })
        elif element.name == 'p':
            content.append({
                "type": "paragraph",
                "text": element.get_text().strip()
            })
        elif element.name in ['ul', 'ol']:
            list_type = "unordered" if element.name == 'ul' else "ordered"
            items = [item.get_text().strip() for item in element.find_all('li')]
            content.append({
                "type": "list",
                "list_type": list_type,
                "items": items
            })
        elif element.name == 'table':
            rows = []
            for row in element.find_all('tr'):
                cells = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                rows.append(cells)
            content.append({
                "type": "table",
                "rows": rows
            })
    
    return content

def load_docx_as_langchain_docs(file_path: str) -> List[Document]:
    """
    Load a DOCX file and return a list of LangChain Document objects with comprehensive text extraction.
    
    Uses multiple extraction methods:
    1. Traditional text extraction via python-docx
    2. Semantic HTML extraction via Mammoth
    3. OCR on extracted images
    4. Vision AI analysis for context
    
    Args:
        file_path (str): Path to the DOCX file to process.
        
    Returns:
        List[Document]: List of LangChain Document objects containing all extractable content.
    """
    docs = []
    
    # 1. Extract structured content using python-docx
    doc = DocxDocument(file_path)
    structured_content = extract_structured_content(doc)
    tables = extract_tables(doc)
    
    # 2. Extract semantic HTML using Mammoth
    html_content = extract_semantic_html(file_path)
    semantic_content = parse_html_content(html_content)
    
    # 3. Create documents from structured content
    for content in structured_content:
        docs.append(Document(
            page_content=content["text"],
            metadata={
                "source": file_path,
                "file_type": "docx",
                "content_type": content["type"],
                "style": content.get("style", "default"),
                "extraction_methods": "python_docx,mammoth"
            }
        ))
    
    # 4. Create documents from tables
    for table in tables:
        # Convert table to string representation
        table_str = "\n".join([
            "\t".join(table["header"]),  # Header row
            *["\t".join(str(val) for val in row.values()) for row in table["data"]]  # Data rows
        ])
        docs.append(Document(
            page_content=table_str,
            metadata={
                "source": file_path,
                "file_type": "docx",
                "content_type": "table",
                "columns": table["header"],
                "extraction_methods": "python_docx"
            }
        ))
    
    # 5. Extract and analyze images with comprehensive approach
    images = extract_images_from_docx(file_path, DOCX_IMAGE_DIR)
    
    # Enhanced Vision AI analysis with comprehensive prompt
    vision_prompt = """
    Analyze this DOCX image and extract:
    1. ALL visible text (exact transcription)
    2. Key concepts and information
    3. Context and meaning
    4. Any structured data (lists, tables, diagrams, charts)
    5. Technical diagrams or flowcharts
    6. Any text that might be in images or graphics
    
    If this appears to be a document page or embedded content, transcribe all text exactly as it appears.
    Focus on extracting actionable information that would be useful for answering questions.
    Include any text that might be embedded in images, charts, or diagrams.
    """
    
    image_docs = analyze_images_with_vision_model(images, prompt=vision_prompt)
    
    # 6. Add OCR text extraction to image documents
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
    
    docs.extend(image_docs)
    
    return docs

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        docs = load_docx_as_langchain_docs(file_path)
        print(f"\nProcessed {len(docs)} documents from {file_path}")
        print(docs)
        
        # Group documents by type for better overview
        content_types = {}
        for doc in docs:
            content_type = doc.metadata.get("content_type", "unknown")
            if content_type not in content_types:
                content_types[content_type] = 0
            content_types[content_type] += 1
        
        print("\nContent breakdown:")
        for content_type, count in content_types.items():
            print(f"- {content_type}: {count} items")
        
        print("\nFirst few documents preview:")
        for i, doc in enumerate(docs[:3], 1):
            print(f"\nDocument {i}:")
            print(f"Type: {doc.metadata.get('content_type', 'unknown')}")
            print(f"Content preview: {doc.page_content[:200]}...") 
    