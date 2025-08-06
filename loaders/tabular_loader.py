from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
from langchain_core.documents import Document

@dataclass
class TabularMetadata:
    """Metadata for tabular data sources."""
    source_type: str  # xlsx, csv, gsheet, etc.
    sheet_names: List[str]  # List of sheet names (for multi-sheet sources)
    total_rows: int  # Total number of rows across all sheets
    total_columns: int  # Total number of unique columns
    column_types: Dict[str, str]  # Mapping of column names to their data types
    file_path: Optional[str] = None  # Path for local files
    sheet_url: Optional[str] = None  # URL for Google Sheets
    additional_metadata: Optional[Dict[str, Any]] = None  # Any additional format-specific metadata

class TabularLoader(ABC):
    """Abstract base class for tabular data loaders."""
    
    @abstractmethod
    def load(self) -> List[Document]:
        """Load the tabular data and convert to LangChain documents.
        
        Returns:
            List of LangChain Document objects.
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> TabularMetadata:
        """Extract metadata from the tabular data source.
        
        Returns:
            TabularMetadata object containing source information.
        """
        pass
    
    def _get_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Helper method to get column types from a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to their data types
        """
        return {col: str(dtype) for col, dtype in df.dtypes.items()} 