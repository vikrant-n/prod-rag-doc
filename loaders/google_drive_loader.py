"""Loader for fetching files from Google Drive and processing them with master_loaders."""

import io
import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from langchain_core.documents import Document
from .master_loaders import load_file

# Remove circular import: import these only inside methods
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 100

SCOPES = ["https://www.googleapis.com/auth/drive"]

class GoogleDriveMasterLoader:
    """Recursively fetch files from a Google Drive folder and load them.

    Parameters
    ----------
    folder_id : str
        Google Drive folder ID to traverse.
    credentials_path : str, optional
        Path to OAuth 2.0 credentials JSON, by default ``"credentials.json"``.
    token_path : str, optional
        Path to the token JSON file, by default ``"token.json"``.
    chunk_size : int, optional
        Desired chunk size when ``split`` is ``True``.
    chunk_overlap : int, optional
        Desired chunk overlap when ``split`` is ``True``.
    split : bool, optional
        Whether to split loaded documents using :func:`split_documents`,
        by default ``True``.
    """

    def __init__(
        self,
        folder_id: str,
        credentials_path: str = "credentials.json",
        token_path: str = "token.json",
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        split: bool = True,
    ):
        self.folder_id = folder_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split = split
        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    creds = None
            if not creds or not creds.valid:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token_path, "w") as token:
                token.write(creds.to_json())
        self.service = build("drive", "v3", credentials=creds)
        self.image_folder_id = self._get_or_create_image_folder()

    def _get_or_create_image_folder(self) -> str:
        """Ensure a subfolder for extracted images exists and return its ID."""
        query = (
            f"'{self.folder_id}' in parents and "
            "mimeType='application/vnd.google-apps.folder' and "
            "name='extracted_images' and trashed=false"
        )
        resp = (
            self.service.files()
            .list(q=query, fields="files(id, name)")
            .execute()
        )
        files = resp.get("files", [])
        if files:
            return files[0]["id"]

        metadata = {
            "name": "extracted_images",
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [self.folder_id],
        }
        folder = (
            self.service.files()
            .create(body=metadata, fields="id")
            .execute()
        )
        return folder["id"]

    def _upload_file_to_drive(self, local_path: str, filename: str) -> str:
        """Upload a file to the images folder and return the file ID."""
        media = MediaFileUpload(local_path, resumable=True)
        body = {"name": filename, "parents": [self.image_folder_id]}
        uploaded = (
            self.service.files()
            .create(body=body, media_body=media, fields="id")
            .execute()
        )
        return uploaded["id"]

    def _process_docs(self, docs: List[Document], file_meta: Dict) -> None:
        """Update metadata and upload extracted images."""
        drive_link = f"https://drive.google.com/file/d/{file_meta['id']}/view?usp=drive_link"
        path_map: Dict[str, str] = {}

        for doc in docs:
            doc.metadata["source"] = drive_link
            doc.metadata["drive_file_id"] = file_meta["id"]
            doc.metadata["drive_file_name"] = file_meta["name"]

        for doc in docs:
            img_path = doc.metadata.get("image_path")
            if img_path:
                img_id = self._upload_file_to_drive(img_path, os.path.basename(img_path))
                drive_img_link = f"https://drive.google.com/file/d/{img_id}/view?usp=drive_link"
                path_map[img_path] = drive_img_link
                doc.metadata["image_path"] = drive_img_link
                doc.metadata["image_file_name"] = os.path.basename(img_path)
                try:
                    os.remove(img_path)
                except OSError:
                    pass

        for doc in docs:
            if "related_images" in doc.metadata:
                doc.metadata["related_images"] = [path_map.get(p, p) for p in doc.metadata["related_images"]]

    def _list_files(self, folder_id: str) -> List[dict]:
        """Yield file metadata for all files in the folder (recursively)."""
        if folder_id == self.image_folder_id:
            return []

        query = f"'{folder_id}' in parents and trashed=false"
        page_token = None
        files = []
        while True:
            response = (
                self.service.files()
                .list(q=query, fields="nextPageToken, files(id, name, mimeType)", pageToken=page_token)
                .execute()
            )
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        result = []
        for f in files:
            if f["id"] == self.image_folder_id:
                continue
            if f["mimeType"] == "application/vnd.google-apps.folder":
                result.extend(self._list_files(f["id"]))
            else:
                result.append(f)
        return result

    def _download_file(self, file_info: dict) -> str:
        """Download a Drive file and return the local path."""
        mime = file_info["mimeType"]
        file_id = file_info["id"]
        name = file_info["name"]

        # Handle Google Docs types via export
        if mime == "application/vnd.google-apps.document":
            request = self.service.files().export_media(
                fileId=file_id,
                mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            suffix = ".docx"
        elif mime == "application/vnd.google-apps.spreadsheet":
            request = self.service.files().export_media(
                fileId=file_id,
                mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            suffix = ".xlsx"
        elif mime == "application/vnd.google-apps.presentation":
            request = self.service.files().export_media(
                fileId=file_id,
                mimeType="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
            suffix = ".pptx"
        else:
            request = self.service.files().get_media(fileId=file_id)
            _, suffix = os.path.splitext(name)

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(fh.read())
        tmp.close()
        return tmp.name

    def load(self) -> List[Document]:
        docs: List[Document] = []
        for file_meta in self._list_files(self.folder_id):
            local_path = self._download_file(file_meta)
            try:
                loaded_docs = load_file(local_path)
                self._process_docs(loaded_docs, file_meta)
                if self.split:
                    from text_splitting import split_documents
                    loaded_docs = split_documents(
                        loaded_docs,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                docs.extend(loaded_docs)
            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)
        return docs

def is_markdown(content: str) -> bool:
    md_indicators = ["#", "-", "*", "`", ">", "[", "!", "```"]
    lines = content.strip().splitlines()
    return any(line.strip().startswith(tuple(md_indicators)) for line in lines[:5])  # Check first few lines


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load documents from a Google Drive folder")
    parser.add_argument("folder_id", help="ID of the Google Drive folder")
    parser.add_argument(
        "--credentials",
        default="credentials.json",
        help="Path to OAuth 2.0 credentials JSON",
    )
    parser.add_argument("--token", default="token.json", help="Path to token JSON")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap in characters",
    )
    parser.add_argument(
        "--output-file",
        default="output_documents.md",
        help="Write all chunks to this file",
    )
    parser.add_argument(
        "--print-chunks",
        action="store_true",
        help="Print each chunk to stdout",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Return raw documents without splitting",
    )
    args = parser.parse_args()

    loader = GoogleDriveMasterLoader(
        args.folder_id,
        credentials_path=args.credentials,
        token_path=args.token,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        split=not args.no_split,
    )
    documents = loader.load()

    print(f"\n✅ Loaded {len(documents)} documents from Google Drive\n")

    markdown_lines = []

    for i, doc in enumerate(documents, start=1):
        markdown_lines.append(f"## 📄 Document {i}\n")

        markdown_lines.append(
            f"**Metadata:**\n```json\n{json.dumps(doc.metadata, indent=2)}\n```\n"
        )

        markdown_lines.append("**Content:**")
        content = doc.page_content.strip()

        if is_markdown(content):
            markdown_lines.append(content + "\n")
        else:
            markdown_lines.append(f"```\n{content}\n```\n")

        if args.print_chunks:
            print("---")
            print(json.dumps(doc.metadata, indent=2))
            print(content)

    output_path = Path(args.output_file)
    output_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"📝 Output saved to {output_path}")


if __name__ == "__main__":
    main()
