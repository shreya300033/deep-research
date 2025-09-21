"""
Document processing and indexing pipeline for various file formats.
Handles text extraction, chunking, and metadata generation.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import hashlib
import json
from datetime import datetime

# Document processing libraries
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    
    def __init__(self, content: str, metadata: Dict[str, Any], chunk_id: str = None):
        self.content = content.strip()
        self.metadata = metadata
        self.chunk_id = chunk_id or self._generate_chunk_id()
        self.timestamp = datetime.now().isoformat()
    
    def _generate_chunk_id(self) -> str:
        """Generate a unique ID for this chunk."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return f"chunk_{content_hash[:12]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create chunk from dictionary."""
        return cls(
            content=data['content'],
            metadata=data['metadata'],
            chunk_id=data['chunk_id']
        )


class DocumentProcessor:
    """Handles document processing, text extraction, and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = {'.txt', '.pdf', '.docx', '.html', '.md'}
    
    def process_document(self, file_path: Union[str, Path], 
                        source_type: str = "file") -> List[DocumentChunk]:
        """
        Process a document and return chunks.
        
        Args:
            file_path: Path to the document or URL
            source_type: Type of source ("file", "url", "text")
            
        Returns:
            List of document chunks
        """
        try:
            if source_type == "file":
                return self._process_file(file_path)
            elif source_type == "url":
                return self._process_url(file_path)
            elif source_type == "text":
                return self._process_text(file_path)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return []
    
    def _process_file(self, file_path: Union[str, Path]) -> List[DocumentChunk]:
        """Process a local file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text based on file extension
        if file_path.suffix.lower() == '.pdf':
            text = self._extract_pdf_text(file_path)
        elif file_path.suffix.lower() == '.docx':
            text = self._extract_docx_text(file_path)
        elif file_path.suffix.lower() in ['.html', '.htm']:
            text = self._extract_html_text(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            text = self._extract_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Create metadata
        metadata = {
            'source_type': 'file',
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix,
            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        return self._chunk_text(text, metadata)
    
    def _process_url(self, url: str) -> List[DocumentChunk]:
        """Process a URL and extract text content."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse URL
            parsed_url = urlparse(url)
            
            # Extract text based on content type
            content_type = response.headers.get('content-type', '').lower()
            
            if 'html' in content_type:
                text = self._extract_html_from_content(response.text)
            else:
                text = response.text
            
            # Create metadata
            metadata = {
                'source_type': 'url',
                'url': url,
                'domain': parsed_url.netloc,
                'content_type': content_type,
                'content_length': len(response.text),
                'status_code': response.status_code
            }
            
            return self._chunk_text(text, metadata)
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return []
    
    def _process_text(self, text: str) -> List[DocumentChunk]:
        """Process raw text."""
        metadata = {
            'source_type': 'text',
            'text_length': len(text)
        }
        
        return self._chunk_text(text, metadata)
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
        return text
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return self._extract_html_from_content(file.read())
        except Exception as e:
            logger.error(f"Error extracting HTML text: {e}")
            return ""
    
    def _extract_html_from_content(self, html_content: str) -> str:
        """Extract text from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return html_content
    
    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading text file: {e}")
                return ""
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata for the chunks
            
        Returns:
            List of document chunks
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(start, end - 200)
                sentence_endings = ['.', '!', '?', '\n\n']
                
                for i in range(end - 1, search_start - 1, -1):
                    if text[i] in sentence_endings:
                        end = i + 1
                        break
            
            # Extract chunk content
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                # Create chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                    'start_position': start,
                    'end_position': end,
                    'chunk_length': len(chunk_content)
                })
                
                chunk = DocumentChunk(chunk_content, chunk_metadata)
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def process_directory(self, directory_path: Union[str, Path], 
                         recursive: bool = True) -> List[DocumentChunk]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
            
        Returns:
            List of all document chunks
        """
        directory_path = Path(directory_path)
        all_chunks = []
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return all_chunks
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        files = []
        for ext in self.supported_extensions:
            files.extend(directory_path.glob(f"{pattern}{ext}"))
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process each file
        for file_path in files:
            try:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
                logger.info(f"Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

