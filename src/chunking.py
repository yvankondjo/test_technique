from typing import List
import os
import re
import asyncio
from openai import AsyncOpenAI

class DocumentChunker:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128, context: bool = True, model: str = "gpt-4o-mini"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context = context
        self.model = model
        self.async_client = None
        if self.context:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set when context=True")
            self.async_client = AsyncOpenAI(api_key=api_key)

    def chunk_text(self, content_text: str) -> List[str]:
        length = len(content_text)
        overlap = min(self.chunk_overlap, self.chunk_size - 1)
        step = self.chunk_size - overlap
        start = 0
        chunks = []
        while start < length:
            end = min(start + self.chunk_size, length)
            chunk = content_text[start:end]
            chunk = self.clean_chunk(chunk)
            if chunk:
                chunks.append(chunk)
            if end == length:
                break      
            start += step
        if self.context and chunks:
            chunks = asyncio.run(self.add_context_to_chunk_multiple(chunks, content_text))
        return chunks
    
    def chunk_csv(self, content_csv: str) -> List[str]:
        lines = content_csv.split('\n')
        headers = []
        data_lines = []

        for line in lines:
            if line.startswith('[Table:') or line.startswith('Columns:'):
                headers.append(line)
            elif line.startswith('Line ') and line.strip():
                data_lines.append(line)
        
        header = '\n'.join(headers)
        chunks = []
        for line in data_lines:
            chunk = f'{header}\n\n{line}'
            chunk = self.clean_chunk(chunk)
            if chunk:
                chunks.append(chunk)
        chunks = [chunk for chunk in chunks if chunk and len(chunk.strip()) >= 15]
        if self.context and chunks:
            return asyncio.run(self.add_context_to_chunk_multiple(chunks, content_csv))
        return chunks
        
    async def add_context_to_chunk_single(self, chunk_text: str, document_text: str) -> str:
        if not chunk_text or not chunk_text.strip():
            return ""
        
        if len(document_text) > 100000:
            document_text = document_text[:100000]
        messages = [
            {'role': 'system', 'content': 'You are a retrieval assistant. Given a chunk from the user, return a concise 1-4 sentence context label that situates the chunk within the cached document. Be specific, no fluff. YOU REPLY IN THE SAME LANGUAGE AS THE DOCUMENT.'},
            {'role': 'user', 'content': f'\n<document>\n{document_text}\n</document>\n<chunk>\n{chunk_text}\n</chunk>\nGive only the succinct context (same language as the document). You must reply in the same language as the document.'}
        ]
        try:
            r = await self.async_client.chat.completions.create(model=self.model, messages=messages, temperature=0.75, max_tokens=256)
            ctx = (r.choices[0].message.content or '').strip()
            result = f'{ctx}\n\n{chunk_text}'.strip()
            if not result or len(result) < 15:
                return chunk_text
            return result
        except Exception:
            return chunk_text

    async def add_context_to_chunk_multiple(self, chunks: List[str], document_text: str) -> List[str]:
        tasks = [self.add_context_to_chunk_single(chunk, document_text) for chunk in chunks if chunk and chunk.strip()]
        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_chunks = []
        for result in results:
            if isinstance(result, Exception):
                continue
            if result and isinstance(result, str) and result.strip() and len(result.strip()) >= 15:
                valid_chunks.append(result)
        return valid_chunks


    def clean_chunk(self, chunk: str) -> str:
        chunk = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', chunk)
        chunk = re.sub(r'\s+', ' ', chunk)
        chunk = chunk.strip()
        
        if len(chunk) < 15:
            return ""
        
        return chunk