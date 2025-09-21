"""
Efficient storage and retrieval pipeline using FAISS
"""
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from config import INDEX_DIR, TOP_K_RESULTS, SIMILARITY_THRESHOLD


class VectorStore:
    """Handles vector storage and similarity search using FAISS"""
    
    def __init__(self, dimension: int = 384):  # all-MiniLM-L6-v2 dimension
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.metadata = []
        self.id_to_metadata = {}
        
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings and metadata to the index"""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        start_id = len(self.metadata)
        for i, meta in enumerate(metadata):
            meta['vector_id'] = start_id + i
            self.metadata.append(meta)
            self.id_to_metadata[start_id + i] = meta
    
    def search(self, query_embedding: np.ndarray, k: int = TOP_K_RESULTS, 
               threshold: float = SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
                
            if score >= threshold:
                result = self.id_to_metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def search_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific document"""
        results = []
        for meta in self.metadata:
            if meta.get('document_id') == document_id:
                results.append(meta)
        return results
    
    def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Get summary information for a document"""
        chunks = self.search_by_document(document_id)
        if not chunks:
            return {}
        
        # Get document info from first chunk
        first_chunk = chunks[0]
        return {
            'document_id': document_id,
            'title': first_chunk.get('title', 'Unknown'),
            'source': first_chunk.get('source', 'Unknown'),
            'total_chunks': len(chunks),
            'chunks': chunks
        }
    
    def save_index(self, filename: str) -> None:
        """Save the FAISS index and metadata"""
        index_path = INDEX_DIR / f"{filename}.index"
        metadata_path = INDEX_DIR / f"{filename}_metadata.pkl"
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'id_to_metadata': self.id_to_metadata,
                'dimension': self.dimension
            }, f)
    
    def load_index(self, filename: str) -> None:
        """Load the FAISS index and metadata"""
        index_path = INDEX_DIR / f"{filename}.index"
        metadata_path = INDEX_DIR / f"{filename}_metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index files {filename} not found")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.id_to_metadata = data['id_to_metadata']
            self.dimension = data['dimension']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'unique_documents': len(set(meta.get('document_id') for meta in self.metadata)),
            'total_chunks': len(self.metadata)
        }
