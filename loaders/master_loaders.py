# Master loader that delegates file loading to specific loaders based on file extension
import os
from typing import List

from langchain_core.documents import Document

from .pdf_loader import load_pdf_as_langchain_docs
from .docx_loader import load_docx_as_langchain_docs
from .pptx_loader import load_pptx_as_langchain_docs
from .csv_loader import CSVLoader
from .excel_loader import ExcelLoader
from .raw_image_loader import RawImageLoader
from .text_loader import TextLoader


def load_file(path: str) -> List[Document]:
    """Load a single file using the appropriate loader based on extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return load_pdf_as_langchain_docs(path)
    if ext == ".docx":
        return load_docx_as_langchain_docs(path)
    if ext == ".pptx":
        return load_pptx_as_langchain_docs(path)
    if ext in {".csv", ".tsv"}:
        return CSVLoader(path).load()
    if ext in {".xlsx", ".xls"}:
        return ExcelLoader(path).load()
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".bmp", ".webp"}:
        return RawImageLoader(path).load()

    # Default to generic text loader for any other extension
    return TextLoader(path).load()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load a document using the appropriate loader")
    parser.add_argument("file", help="Path to the file to load")
    args = parser.parse_args()

    docs = load_file(args.file)
    print(f"Loaded {len(docs)} documents")
    for doc in docs[:3]:
        print("---")
        print("Metadata:", doc.metadata)
        print("Content preview:", doc.page_content[:200])


if __name__ == "__main__":
    main()
    