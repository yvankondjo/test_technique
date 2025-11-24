from pathlib import Path
from typing import List, Dict, Optional
import re
import pandas as pd
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup

class DocumentExtractor:
    def __init__(self):
        self.supported_extensions = {
            ".txt": self.extract_text,
            ".pdf": self.extract_pdf,
            ".docx": self.extract_docx,
            ".html": self.extract_html,
            ".csv": self.extract_csv,
        }
        self.data: Dict[str, List[Dict]] = {}
        self.errors: List[str] = []

    def extract_file(self, file_path: str) -> Dict:
        try:
            extension = Path(file_path).suffix.lower()
            if extension not in self.supported_extensions:
                raise ValueError(f"Extension {extension} non supportée")
            return self.supported_extensions[extension](file_path)
        except Exception as e:
            self.errors.append(f"Erreur lors du traitement de {file_path}: {str(e)}")
            return {"error": str(e)}

    def extract_text(self, file_path: str) -> Dict:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return {"content": text,"file_path": file_path}

    def extract_pdf(self, file_path: str) -> Dict:
        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text and page_text.strip():
                texts.append(page_text)

        text = "\n\n".join(texts)
        return {"content": text, "file_path": file_path}

    def extract_docx(self, file_path: str) -> Dict:
        doc = Document(file_path)
        text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
        return {"content": text,"file_path": file_path}

    def extract_html(self, file_path: str) -> Dict:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n\n', strip=True)
        return {"content": text,"file_path": file_path}

    def extract_csv(self, file_path: str) -> Dict:
        df = pd.read_csv(file_path)
        file_name = Path(file_path).stem
        content_parts = [
            f"[Table: {file_name}]",
            f"Columns: {', '.join(df.columns.tolist())}",
        ]
        for index, row in df.iterrows():
            row_str = " |".join([f"{col}:{row[col]}" for col in df.columns])
            content_parts.append(f"Line {index + 1}: {row_str}")
        return {"content": "\n\n".join(content_parts),"file_path": file_path}
