"""
Local embedding generation system for the Deep Researcher Agent
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional, Dict, Any
import logging
from pathlib import Path
import pickle
import hashlib
from tqdm import tqdm
import config

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Handles local embedding generation using sentence-transformers
    """
    
    def __init__(self, model_name: str = None, cache_dir: str = None):
        """
        Initialize the embedding engine
        
        Args:
            model_name: Name of the sentence-transformer model to use
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize the model
        self.model = None
        self.embedding_dim = config.EMBEDDING_DIMENSION
        self.batch_size = config.EMBEDDING_BATCH_SIZE
        
        self._load_model()
        
    def _load_model(self):
        """Load the sentence-transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a smaller model
            self.model_name = "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Using fallback model: {self.model_name}")
    
    def _get_cache_path(self, text: str) -> Path:
        """Generate cache file path for a text"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{self.model_name}_{text_hash}.pkl"
    
    def _load_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Load embedding from cache if available"""
        if not config.ENABLE_CACHING:
            return None
            
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for text: {e}")
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache"""
        if not config.ENABLE_CACHING:
            return
            
        cache_path = self._get_cache_path(text)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for text: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array containing the embedding
        """
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)
        
        # Check cache first
        cached_embedding = self._load_from_cache(text)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Save to cache
            self._save_to_cache(text, embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return np.zeros((len(texts), self.embedding_dim))
        
        # Check cache for all texts
        embeddings = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(valid_texts):
            cached_embedding = self._load_from_cache(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                iterator = tqdm(
                    range(0, len(uncached_texts), self.batch_size),
                    desc="Generating embeddings",
                    disable=not show_progress
                )
                
                for start_idx in iterator:
                    end_idx = min(start_idx + self.batch_size, len(uncached_texts))
                    batch_texts = uncached_texts[start_idx:end_idx]
                    
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    
                    # Update embeddings list and cache
                    for i, embedding in enumerate(batch_embeddings):
                        global_idx = uncached_indices[start_idx + i]
                        embeddings[global_idx] = embedding
                        self._save_to_cache(uncached_texts[start_idx + i], embedding)
                        
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                # Fill with zeros for failed embeddings
                for i in uncached_indices:
                    embeddings[i] = np.zeros(self.embedding_dim)
        
        # Handle empty texts
        final_embeddings = []
        for i, text in enumerate(texts):
            if text and text.strip():
                # Find the embedding for this text
                valid_idx = sum(1 for j in range(i) if texts[j] and texts[j].strip())
                final_embeddings.append(embeddings[valid_idx])
            else:
                final_embeddings.append(np.zeros(self.embedding_dim))
        
        return np.array(final_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query with query-specific preprocessing
        
        Args:
            query: Search query
            
        Returns:
            numpy array containing the query embedding
        """
        # Preprocess query for better search
        processed_query = self._preprocess_query(query)
        return self.embed_text(processed_query)
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query for better embedding generation
        
        Args:
            query: Raw query
            
        Returns:
            Preprocessed query
        """
        # Basic preprocessing
        query = query.strip()
        
        # Add context if query is too short
        if len(query.split()) < 3:
            query = f"search query: {query}"
        
        return query
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
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
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "batch_size": self.batch_size,
            "cache_enabled": config.ENABLE_CACHING,
            "cache_dir": str(self.cache_dir)
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob(f"{self.model_name}_*.pkl"):
                cache_file.unlink()
            logger.info("Embedding cache cleared")


class EmbeddingManager:
    """
    Manager class for handling multiple embedding engines
    """
    
    def __init__(self):
        self.engines = {}
        self.default_engine = None
    
    def get_engine(self, model_name: str = None) -> EmbeddingEngine:
        """
        Get an embedding engine, creating it if necessary
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            EmbeddingEngine instance
        """
        model_name = model_name or config.EMBEDDING_MODEL
        
        if model_name not in self.engines:
            self.engines[model_name] = EmbeddingEngine(model_name)
        
        if self.default_engine is None:
            self.default_engine = self.engines[model_name]
        
        return self.engines[model_name]
    
    def get_default_engine(self) -> EmbeddingEngine:
        """Get the default embedding engine"""
        if self.default_engine is None:
            self.default_engine = self.get_engine()
        return self.default_engine


# Global embedding manager instance
embedding_manager = EmbeddingManager()
