import os
import chardet
import mimetypes
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader

class TextLoader(BaseLoader):
    """Loader for text files and other text-based formats.
    
    This loader can handle:
    - .txt files directly
    - .conf, .log, .ini and other text-based formats
    - Attempts to detect and handle various file encodings
    - Extracts metadata like file type, encoding, size
    """
    
    def __init__(
        self,
        file_path: str,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = True
    ):
        """Initialize the TextLoader.
        
        Args:
            file_path: Path to the text file
            encoding: Specific encoding to use (overrides autodetection)
            autodetect_encoding: Whether to attempt encoding autodetection
        """
        self.file_path = file_path
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Get MIME type
        self.mime_type, _ = mimetypes.guess_type(file_path)
        
        # Register common text file extensions
        mimetypes.add_type('text/plain', '.conf')
        mimetypes.add_type('text/plain', '.log')
        mimetypes.add_type('text/plain', '.ini')
        
    def _detect_encoding(self, raw_bytes: bytes) -> str:
        """Detect the file encoding."""
        if self.encoding:
            return self.encoding
            
        if self.autodetect_encoding:
            result = chardet.detect(raw_bytes)
            if result['encoding']:
                return result['encoding']
                
        return 'utf-8'  # Default fallback
        
    def _is_text_file(self) -> bool:
        """Check if the file appears to be text-based."""
        # First check MIME type
        if self.mime_type and self.mime_type.startswith('text/'):
            return True
            
        # Then try reading as text
        try:
            with open(self.file_path, 'rb') as f:
                raw_bytes = f.read(1024)  # Read first 1KB
                encoding = self._detect_encoding(raw_bytes)
                raw_bytes.decode(encoding)
            return True
        except UnicodeDecodeError:
            return False
            
    def _get_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the file."""
        stats = os.stat(self.file_path)
        return {
            'source': self.file_path,
            'filename': os.path.basename(self.file_path),
            'file_type': self.mime_type or 'text/plain',
            'file_size': stats.st_size,
            'created_at': stats.st_ctime,
            'modified_at': stats.st_mtime,
            'accessed_at': stats.st_atime
        }
        
    def load(self) -> List[Document]:
        """Load the text file and return it as a Document."""
        if not self._is_text_file():
            raise ValueError(f"File does not appear to be text-based: {self.file_path}")
            
        with open(self.file_path, 'rb') as f:
            raw_bytes = f.read()
            
        encoding = self._detect_encoding(raw_bytes)
        text_content = raw_bytes.decode(encoding)
        
        metadata = self._get_metadata()
        metadata['detected_encoding'] = encoding
        
        return [Document(
            page_content=text_content,
            metadata=metadata
        )] 