import os
from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime
import imghdr
from PIL import Image, ExifTags
from langchain_core.documents import Document
from tools.vision_tool import analyze_images_with_vision_model, VISION_PROMPT

class RawImageLoader:
    """Loader for raw image files (PNG, JPEG, JPG, GIF, TIFF, etc.)."""
    
    SUPPORTED_FORMATS = {
        'png', 'jpeg', 'jpg', 'gif', 'tiff', 'tif', 'bmp', 'webp'
    }
    
    def __init__(self, file_paths: Union[str, List[str], Path, List[Path]]):
        """Initialize the raw image loader.
        
        Args:
            file_paths: Single path or list of paths to image files
        """
        # Convert single path to list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
            
        # Convert all paths to Path objects
        self.file_paths = [Path(p) for p in file_paths]
        
        # Validate files exist and are supported formats
        for path in self.file_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            
            # Check file extension
            ext = path.suffix.lower().lstrip('.')
            if ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported image format: {ext}. Must be one of {self.SUPPORTED_FORMATS}")
    
    def _extract_image_metadata(self, image_path: Path, index: int) -> Dict:
        """Extract metadata from an image file.
        
        Args:
            image_path: Path to the image file
            index: Index of the image in the batch
            
        Returns:
            Dictionary containing image metadata
        """
        try:
            with Image.open(image_path) as img:
                # Basic metadata
                metadata = {
                    "path": str(image_path),
                    "format": img.format.lower() if img.format else imghdr.what(image_path),
                    "mode": img.mode,
                    "width": img.width,
                    "height": img.height,
                    "aspect_ratio": round(img.width / img.height, 2),
                    "file_size": image_path.stat().st_size,
                    "created": datetime.fromtimestamp(image_path.stat().st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(image_path.stat().st_mtime).isoformat(),
                    "index": index,  # Add index to metadata
                    "extension": image_path.suffix.lower().lstrip('.'),
                    "source_type": "raw_image",
                    "source_doc": str(image_path)
                }
                
                # Try to extract EXIF data
                try:
                    exif = img.getexif()
                    if exif:
                        exif_data = {}
                        for tag_id in exif:
                            # Get the tag name, default to the tag ID if not found
                            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                            value = exif.get(tag_id)
                            # Only include string-serializable values
                            if isinstance(value, (str, int, float)):
                                exif_data[tag] = value
                        metadata["exif"] = exif_data
                except Exception as e:
                    print(f"Warning: Could not extract EXIF data from {image_path}: {e}")
                
                return metadata
                
        except Exception as e:
            print(f"Error extracting metadata from {image_path}: {e}")
            # Return basic metadata if image processing fails
            return {
                "path": str(image_path),
                "format": imghdr.what(image_path),
                "file_size": image_path.stat().st_size,
                "created": datetime.fromtimestamp(image_path.stat().st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(image_path.stat().st_mtime).isoformat(),
                "index": index,  # Add index to metadata
                "extension": image_path.suffix.lower().lstrip('.'),
                "source_type": "raw_image",
                "source_doc": str(image_path)
            }
    
    def load(self, batch_size: Optional[int] = None) -> List[Document]:
        """Load and analyze the image files, converting them to LangChain documents.
        
        Args:
            batch_size: Optional batch size for processing images. If None, process all at once.
            
        Returns:
            List of LangChain Document objects containing image analysis and metadata.
        """
        all_docs = []
        image_metadata = []
        
        # Process images in batches if specified
        paths_to_process = self.file_paths
        if batch_size:
            paths_to_process = self.file_paths[:batch_size]
        
        # Extract metadata for each image
        for idx, path in enumerate(paths_to_process):
            try:
                metadata = self._extract_image_metadata(path, idx)
                image_metadata.append(metadata)
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                continue
        
        if not image_metadata:
            raise ValueError("No valid images were processed")
        
        try:
            # Analyze images with vision model
            vision_docs = analyze_images_with_vision_model(
                images=image_metadata,  # Pass the complete metadata dictionaries
                prompt=VISION_PROMPT
            )
            
            # Combine vision analysis with metadata
            for doc, meta in zip(vision_docs, image_metadata):
                # Update metadata with image details
                doc.metadata.update({
                    "image_format": meta["format"],
                    "image_mode": meta.get("mode"),
                    "image_width": meta.get("width"),
                    "image_height": meta.get("height"),
                    "aspect_ratio": meta.get("aspect_ratio"),
                    "file_size": meta["file_size"],
                    "created_at": meta["created"],
                    "modified_at": meta["modified"]
                })
                
                # Add EXIF data if available
                if "exif" in meta:
                    doc.metadata["exif"] = meta["exif"]
                
                all_docs.append(doc)
            
            return all_docs
            
        except Exception as e:
            print(f"Error during vision analysis: {str(e)}")
            raise ValueError("Failed to analyze images with vision model") from e

if __name__ == "__main__":
    import argparse
    import json
    from pprint import pprint
    
    parser = argparse.ArgumentParser(description="Load and analyze image files using LangChain and Vision model")
    parser.add_argument("images", nargs="+", help="One or more image files to process")
    parser.add_argument("--batch-size", type=int, help="Optional batch size for processing multiple images")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()
    
    try:
        # Initialize loader with provided images
        loader = RawImageLoader(args.images)
        
        # Process images
        docs = loader.load(batch_size=args.batch_size)
        
        # Output results
        if args.json:
            # Convert to JSON-serializable format
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            print(json.dumps(results, indent=2))
        else:
            print(f"\nProcessed {len(docs)} images successfully:")
            for i, doc in enumerate(docs, 1):
                print(f"\n--- Image {i} ---")
                print(f"File: {doc.metadata['image_path']}")
                print(f"Format: {doc.metadata['image_format']}")
                print(f"Dimensions: {doc.metadata['image_width']}x{doc.metadata['image_height']}")
                print("\nAnalysis:")
                print(doc.page_content)
                print("\nMetadata:")
                pprint(doc.metadata)
                print("-" * 80)
                
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1) 