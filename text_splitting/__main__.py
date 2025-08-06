"""CLI for recursively splitting documents."""
import argparse

from loaders.master_loaders import load_file
from .recursive_splitter import split_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def main() -> int:
    parser = argparse.ArgumentParser(description="Split documents for embedding")
    parser.add_argument("file", help="Path to the file to split")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in characters")
    parser.add_argument("--show-all", action="store_true", help="Print all chunks instead of a short preview")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=3,
        help="Number of chunks to preview if --show-all is not set",
    )
    args = parser.parse_args()

    docs = load_file(args.file)
    chunks = split_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Created {len(chunks)} chunks")
    selection = chunks if args.show_all else chunks[: args.max_chunks]
    for i, chunk in enumerate(selection, start=1):
        print("---")
        print("metadata:", chunk.metadata)
        if args.show_all:
            content = chunk.page_content
        else:
            content = chunk.page_content.replace("\n", " ")[:200]
            if len(chunk.page_content) > 200:
                content += "..."
        print("content:")
        print(content)
    if not args.show_all and len(chunks) > args.max_chunks:
        print(f"... (use --show-all to print all {len(chunks)} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
