"""
Local embedding generation for document indexing and retrieval.
Uses sentence-transformers for high-quality embeddings without external APIs.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict, Any
import logging
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles local embedding generation for documents and queries."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "./cache"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Sentence transformer model to use
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load the sentence transformer model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Cache for embeddings
        self.embedding_cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def generate_embeddings(self, texts: Union[str, List[str]], 
                          batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache first
        cache_key = hash(tuple(texts))
        if cache_key in self.embedding_cache:
            logger.debug("Using cached embeddings")
            return self.embedding_cache[cache_key]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Cache the result
        self.embedding_cache[cache_key] = embeddings
        self._save_cache()
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as numpy array
        """
        return self.generate_embeddings(query).flatten()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # Generate a test embedding to get dimension
        test_embedding = self.generate_embeddings("test")
        return test_embedding.shape[1]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def batch_similarity(self, query_embedding: np.ndarray, 
                        document_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between query and multiple document embeddings.
        
        Args:
            query_embedding: Query embedding
            document_embeddings: Document embeddings matrix
            
        Returns:
            Array of similarity scores
        """
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(document_embeddings.shape[0])
        
        # Normalize document embeddings
        doc_norms = np.linalg.norm(document_embeddings, axis=1)
        doc_norms[doc_norms == 0] = 1  # Avoid division by zero
        
        # Calculate cosine similarities
        similarities = np.dot(document_embeddings, query_embedding) / (doc_norms * query_norm)
        return similarities

