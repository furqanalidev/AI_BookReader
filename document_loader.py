"""
document_loader.py
Handles text extraction and cleaning from PDF, TXT, and DOCX files.
"""
import os
import re
from typing import List

import pdfplumber
import docx


class DocumentLoader:
    """Utility class to load and clean text from common document types.

    Supports: PDF, TXT, DOCX.
    """

    @staticmethod
    def load_txt(file_path: str) -> str:
        """Read a plain text file as UTF-8."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Extract text from PDF using pdfplumber.

        Returns concatenated text from all pages. If a page has no extractable
        text, an empty string is used for that page.
        """
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    @staticmethod
    def load_docx(file_path: str) -> str:
        """Extract text from a Word (.docx) document."""
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace and remove control/special characters.

        Keeps common punctuation so contexts remain readable.
        """
        if not text:
            return ''
        # Collapse whitespace/newlines into single spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable/control characters (keep punctuation)
        text = re.sub(r'[\x00-\x1f\x7f]', '', text)
        return text.strip()

    @staticmethod
    def load_and_clean(file_path: str) -> str:
        """Detect file type, extract text, and clean it for downstream processing."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            text = DocumentLoader.load_pdf(file_path)
        elif ext == '.txt':
            text = DocumentLoader.load_txt(file_path)
        elif ext == '.docx':
            text = DocumentLoader.load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return DocumentLoader.clean_text(text)
