"""
Local embedding generation for document indexing and retrieval
"""
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import tiktoken
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_DIR


class EmbeddingGenerator:
    """Handles local embedding generation and storage"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, 
                   overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode([text], convert_to_numpy=True)[0]
    
    def save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], 
                       filename: str) -> None:
        """Save embeddings and metadata to disk"""
        save_path = EMBEDDINGS_DIR / f"{filename}.pkl"
        
        data = {
            'embeddings': embeddings,
            'metadata': metadata,
            'model_name': self.model_name
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_embeddings(self, filename: str) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load embeddings and metadata from disk"""
        load_path = EMBEDDINGS_DIR / f"{filename}.pkl"
        
        if not load_path.exists():
            raise FileNotFoundError(f"Embeddings file {filename} not found")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        return data['embeddings'], data['metadata']
    
    def process_document(self, document: Dict[str, Any]) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        """Process a document and generate embeddings for its chunks"""
        text = document.get('content', '')
        chunks = self.chunk_text(text)
        
        # Generate embeddings for all chunks
        embeddings = self.generate_embeddings(chunks)
        
        # Create metadata for each chunk
        metadata = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                'chunk_id': f"{document.get('id', 'unknown')}_{i}",
                'document_id': document.get('id', 'unknown'),
                'title': document.get('title', 'Unknown'),
                'source': document.get('source', 'Unknown'),
                'chunk_index': i,
                'chunk_text': chunk,
                'total_chunks': len(chunks)
            }
            metadata.append(chunk_metadata)
        
        return embeddings, metadata
