"""Utilities for splitting text into manageable chunks."""
from .recursive_splitter import (
    split_documents,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)

__all__ = [
    "split_documents",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
]
