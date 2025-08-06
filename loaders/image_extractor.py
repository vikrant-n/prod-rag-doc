import os
import hashlib
from typing import List, Dict, Optional
from datetime import datetime
import fitz  # PyMuPDF for PDF
import docx2txt  # for DOCX
import shutil
from pathlib import Path


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-_.") else "_" for c in name)


def generate_unique_image_path(output_dir: str, doc_name: str, img_index: int, extension: str) -> str:
    """Generate a unique path for an image, avoiding conflicts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{sanitize_filename(doc_name)}_{timestamp}_{img_index}{extension}"
    return os.path.join(output_dir, base_name)


def extract_images_from_pdf(doc_path: str, output_dir: str) -> List[Dict]:
    """Extract images from a PDF file."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(doc_path)
    doc_name = os.path.splitext(os.path.basename(doc_path))[0]
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        img_list = page.get_images(full=True)
        
        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = f".{base_image['ext']}"
            
            # Generate unique path for this image
            image_path = generate_unique_image_path(output_dir, doc_name, len(images), image_ext)
            
            # Save the image
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            # Record image metadata
            images.append({
                "path": image_path,
                "page": page_num + 1,
                "index": img_index,
                "extension": image_ext,
                "source_doc": doc_path,
                "source_type": "pdf"
            })
    
    return images


def extract_images_from_docx(doc_path: str, output_dir: str) -> List[Dict]:
    """Extract images from a DOCX file."""
    os.makedirs(output_dir, exist_ok=True)
    doc_name = os.path.splitext(os.path.basename(doc_path))[0]
    
    # Create a temporary directory for docx2txt to extract images
    temp_dir = os.path.join(output_dir, f"temp_{doc_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Extract text and images (docx2txt saves images to temp_dir)
        docx2txt.process(doc_path, temp_dir)
        
        # Process extracted images
        images = []
        for img_index, img_file in enumerate(os.listdir(temp_dir)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                temp_path = os.path.join(temp_dir, img_file)
                image_ext = os.path.splitext(img_file)[1]
                
                # Generate unique path for this image
                final_path = generate_unique_image_path(output_dir, doc_name, img_index, image_ext)
                
                # Move the image from temp to final location
                shutil.move(temp_path, final_path)
                
                # Record image metadata
                images.append({
                    "path": final_path,
                    "index": img_index,
                    "extension": image_ext,
                    "source_doc": doc_path,
                    "source_type": "docx",
                    "original_name": img_file
                })
        
        return images
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def extract_images(doc_path: str, output_dir: str) -> List[Dict]:
    """Extract images from a document based on its file type."""
    ext = os.path.splitext(doc_path)[1].lower()
    
    if ext == '.pdf':
        return extract_images_from_pdf(doc_path, output_dir)
    elif ext == '.docx':
        return extract_images_from_docx(doc_path, output_dir)
    else:
        raise ValueError(f"Unsupported document type: {ext}")

# Extend with extract_images_from_html, etc. as needed. 