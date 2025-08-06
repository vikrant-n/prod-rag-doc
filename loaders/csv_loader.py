import os
from typing import List
import pandas as pd
from langchain_core.documents import Document
from .tabular_loader import TabularLoader, TabularMetadata

class CSVLoader(TabularLoader):
    """Loader for CSV and TSV files."""
    
    def __init__(self, file_path: str, delimiter: str = None, encoding: str = 'utf-8'):
        """Initialize the CSV/TSV loader.
        
        Args:
            file_path: Path to the CSV/TSV file
            delimiter: Optional delimiter to use. If None, will try to detect from file extension
                      ('.csv' uses ',' and '.tsv' uses '\t')
            encoding: File encoding to use (default: utf-8)
        """
        self.file_path = file_path
        self.encoding = encoding
        
        # Auto-detect delimiter if not provided
        if delimiter is None:
            ext = os.path.splitext(file_path)[1].lower()
            self.delimiter = '\t' if ext == '.tsv' else ','
        else:
            self.delimiter = delimiter
            
        # Cache for metadata
        self._metadata = None
    
    def load(self) -> List[Document]:
        """Load the CSV/TSV file and convert to LangChain documents.
        
        Returns:
            List of LangChain Document objects.
        """
        # Read the CSV/TSV file
        df = pd.read_csv(self.file_path, delimiter=self.delimiter, encoding=self.encoding)
        metadata = self.get_metadata()
        
        # Create a single document for the entire file
        return [Document(
            page_content=df.to_string(index=False),
            metadata={
                "source": self.file_path,
                "file_type": metadata.source_type,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns)
            }
        )]
    
    def get_metadata(self) -> TabularMetadata:
        """Extract metadata from the CSV/TSV file.
        
        Returns:
            TabularMetadata object containing source information.
        """
        # Return cached metadata if available
        if self._metadata is not None:
            return self._metadata
        
        # Read the file to extract metadata
        df = pd.read_csv(self.file_path, delimiter=self.delimiter, encoding=self.encoding)
        
        # Create metadata object
        self._metadata = TabularMetadata(
            source_type="csv" if self.delimiter == ',' else "tsv",
            sheet_names=["default"],  # CSV/TSV files don't have sheets
            total_rows=len(df),
            total_columns=len(df.columns),
            column_types=self._get_column_types(df),
            file_path=self.file_path
        )
        
        return self._metadata

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        loader = CSVLoader(file_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} documents")
        if docs:
            print("\nFirst document preview:")
            doc = docs[0]
            print(f"Metadata: {doc.metadata}")
            print(f"Content preview: {doc.page_content[:200]}...") 