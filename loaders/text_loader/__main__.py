"""Command-line interface for the text loader."""
import argparse
import json
from datetime import datetime
import sys
from ..text_loader import TextLoader

def main():
    """Command line interface for the text loader."""
    parser = argparse.ArgumentParser(description='Load and analyze text files')
    parser.add_argument('file_path', help='Path to the text file to load')
    parser.add_argument('--encoding', help='Specify file encoding (optional)')
    parser.add_argument('--no-autodetect', action='store_true', 
                      help='Disable encoding autodetection')
    args = parser.parse_args()
    
    try:
        loader = TextLoader(
            args.file_path,
            encoding=args.encoding,
            autodetect_encoding=not args.no_autodetect
        )
        
        docs = loader.load()
        
        # Convert datetime objects to strings for JSON serialization
        metadata = docs[0].metadata.copy()
        metadata['created_at'] = datetime.fromtimestamp(
            metadata['created_at']).isoformat()
        metadata['modified_at'] = datetime.fromtimestamp(
            metadata['modified_at']).isoformat()
        metadata['accessed_at'] = datetime.fromtimestamp(
            metadata['accessed_at']).isoformat()
        
        print("\nFile Analysis Results:")
        print("-" * 50)
        print(f"Content Preview (first 200 chars):")
        print(docs[0].page_content[:200] + "...")
        print("\nMetadata:")
        print(json.dumps(metadata, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == '__main__':
    sys.exit(main()) 