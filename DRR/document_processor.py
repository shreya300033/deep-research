"""
Document processing and indexing system for the Deep Researcher Agent
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
import hashlib
import json
from datetime import datetime

# Document processing libraries
import PyPDF2
from docx import Document
import markdown
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse

# Text processing
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Our modules
import config
from embedding_engine import embedding_manager
import numpy as np

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content.strip()
        self.metadata = metadata or {}
        self.id = self._generate_id()
        self.embedding = None
        
    def _generate_id(self) -> str:
        """Generate unique ID for the chunk"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return f"chunk_{content_hash[:12]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create chunk from dictionary"""
        chunk = cls(data["content"], data["metadata"])
        chunk.id = data["id"]
        if data.get("embedding"):
            chunk.embedding = np.array(data["embedding"])
        return chunk


class DocumentProcessor:
    """Handles document processing and chunking"""
    
    def __init__(self):
        self.embedding_engine = embedding_manager.get_default_engine()
        self.stop_words = set(stopwords.words('english'))
        
    def process_file(self, file_path: Union[str, Path]) -> List[DocumentChunk]:
        """
        Process a single file and return chunks
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of DocumentChunk objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.stat().st_size > config.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_path}")
        
        # Get file metadata
        metadata = self._get_file_metadata(file_path)
        
        # Extract text based on file type
        text = self._extract_text(file_path)
        
        if not text.strip():
            logger.warning(f"No text extracted from {file_path}")
            return []
        
        # Clean and preprocess text
        cleaned_text = self._clean_text(text)
        
        # Split into chunks
        chunks = self._split_into_chunks(cleaned_text, metadata)
        
        # Generate embeddings for chunks
        self._generate_chunk_embeddings(chunks)
        
        logger.info(f"Processed {file_path}: {len(chunks)} chunks created")
        return chunks
    
    def process_url(self, url: str) -> List[DocumentChunk]:
        """
        Process a web URL and return chunks
        
        Args:
            url: URL to process
            
        Returns:
            List of DocumentChunk objects
        """
        try:
            # Fetch content
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Create metadata
            metadata = {
                "source_type": "web",
                "url": url,
                "domain": urlparse(url).netloc,
                "title": soup.title.string if soup.title else url,
                "processed_at": datetime.now().isoformat()
            }
            
            # Clean and chunk
            cleaned_text = self._clean_text(text)
            chunks = self._split_into_chunks(cleaned_text, metadata)
            
            # Generate embeddings
            self._generate_chunk_embeddings(chunks)
            
            logger.info(f"Processed URL {url}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {e}")
            return []
    
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file"""
        stat = file_path.stat()
        return {
            "source_type": "file",
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": stat.st_size,
            "file_extension": file_path.suffix,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "processed_at": datetime.now().isoformat()
        }
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text from file based on its type"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self._extract_pdf_text(file_path)
        elif suffix == '.docx':
            return self._extract_docx_text(file_path)
        elif suffix == '.txt':
            return self._extract_txt_text(file_path)
        elif suffix == '.md':
            return self._extract_markdown_text(file_path)
        elif suffix == '.html':
            return self._extract_html_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
        return text
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to extract DOCX text: {e}")
            return ""
    
    def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Failed to extract TXT text: {e}")
                return ""
    
    def _extract_markdown_text(self, file_path: Path) -> str:
        """Extract text from Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                # Convert markdown to HTML then extract text
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"Failed to extract Markdown text: {e}")
            return ""
    
    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"Failed to extract HTML text: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Remove very short lines
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(lines)
    
    def _split_into_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text into chunks"""
        # Split by sentences first
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, create a new chunk
            if current_length + sentence_length > config.CHUNK_SIZE and current_chunk:
                chunks.append(DocumentChunk(current_chunk, metadata.copy()))
                current_chunk = sentence
                current_length = sentence_length
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(DocumentChunk(current_chunk, metadata.copy()))
        
        # Add overlap between chunks
        if config.CHUNK_OVERLAP > 0 and len(chunks) > 1:
            chunks = self._add_chunk_overlap(chunks)
        
        return chunks
    
    def _add_chunk_overlap(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add overlap between chunks for better context"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.content
            
            # Add overlap from previous chunk
            if i > 0:
                prev_content = chunks[i-1].content
                overlap_text = prev_content[-config.CHUNK_OVERLAP:]
                content = overlap_text + " " + content
            
            # Add overlap to next chunk
            if i < len(chunks) - 1:
                next_content = chunks[i+1].content
                overlap_text = next_content[:config.CHUNK_OVERLAP]
                content = content + " " + overlap_text
            
            # Create new chunk with overlap
            new_chunk = DocumentChunk(content, chunk.metadata.copy())
            new_chunk.id = chunk.id  # Keep original ID
            overlapped_chunks.append(new_chunk)
        
        return overlapped_chunks
    
    def _generate_chunk_embeddings(self, chunks: List[DocumentChunk]):
        """Generate embeddings for all chunks"""
        if not chunks:
            return
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_engine.embed_texts(texts, show_progress=True)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding


class DocumentIndexer:
    """Handles indexing of processed documents"""
    
    def __init__(self, vector_store_type: str = None):
        self.vector_store_type = vector_store_type or config.VECTOR_STORE_TYPE
        self.embedding_engine = embedding_manager.get_default_engine()
        self.index = None
        self.metadata_store = {}
        
    def index_documents(self, documents: List[DocumentChunk]) -> bool:
        """
        Index a list of document chunks
        
        Args:
            documents: List of DocumentChunk objects
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            return False
        
        try:
            # Extract embeddings and metadata
            embeddings = []
            metadata = []
            
            for doc in documents:
                if doc.embedding is not None:
                    embeddings.append(doc.embedding)
                    metadata.append(doc.to_dict())
                    self.metadata_store[doc.id] = doc.to_dict()
            
            if not embeddings:
                logger.warning("No embeddings found in documents")
                return False
            
            embeddings = np.array(embeddings)
            
            # Create or update index
            if self.index is None:
                self._create_index(embeddings)
            else:
                self._update_index(embeddings)
            
            logger.info(f"Indexed {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False
    
    def _create_index(self, embeddings: np.ndarray):
        """Create a new vector index"""
        if self.vector_store_type == "faiss":
            self._create_faiss_index(embeddings)
        elif self.vector_store_type == "chroma":
            self._create_chroma_index(embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
    
    def _create_faiss_index(self, embeddings: np.ndarray):
        """Create FAISS index"""
        import faiss
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def _create_chroma_index(self, embeddings: np.ndarray):
        """Create ChromaDB index"""
        import chromadb
        
        self.chroma_client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME
        )
        
        # Add documents to ChromaDB
        ids = [doc_id for doc_id in self.metadata_store.keys()]
        documents = [self.metadata_store[doc_id]["content"] for doc_id in ids]
        metadatas = [self.metadata_store[doc_id]["metadata"] for doc_id in ids]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
    
    def _update_index(self, embeddings: np.ndarray):
        """Update existing index with new embeddings"""
        if self.vector_store_type == "faiss":
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        elif self.vector_store_type == "chroma":
            # ChromaDB handles updates automatically
            pass
    
    def save_index(self, path: str = None):
        """Save the index to disk"""
        path = path or str(config.VECTOR_STORE_DIR / "index")
        
        if self.vector_store_type == "faiss" and self.index is not None:
            faiss.write_index(self.index, f"{path}.faiss")
            
            # Save metadata
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(self.metadata_store, f, indent=2)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str = None):
        """Load the index from disk"""
        path = path or str(config.VECTOR_STORE_DIR / "index")
        
        try:
            if self.vector_store_type == "faiss":
                self.index = faiss.read_index(f"{path}.faiss")
                
                # Load metadata
                with open(f"{path}_metadata.json", 'r') as f:
                    self.metadata_store = json.load(f)
            
            logger.info(f"Index loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False


# Global instances
document_processor = DocumentProcessor()
document_indexer = DocumentIndexer()
