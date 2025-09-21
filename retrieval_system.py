"""
Retrieval and search system for the Deep Researcher Agent
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json

# Vector search libraries
import faiss
import chromadb

# Our modules
import config
from embedding_engine import embedding_manager
from document_processor import DocumentChunk, document_indexer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with relevance score and metadata"""
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "source": self.source
        }


@dataclass
class SearchQuery:
    """Represents a search query with options"""
    query: str
    top_k: int = None
    similarity_threshold: float = None
    filters: Dict[str, Any] = None
    include_metadata: bool = True
    
    def __post_init__(self):
        self.top_k = self.top_k or config.TOP_K_RESULTS
        self.similarity_threshold = self.similarity_threshold or config.SIMILARITY_THRESHOLD
        self.filters = self.filters or {}


class RetrievalSystem:
    """Handles document retrieval and search operations"""
    
    def __init__(self, vector_store_type: str = None):
        self.vector_store_type = vector_store_type or config.VECTOR_STORE_TYPE
        self.embedding_engine = embedding_manager.get_default_engine()
        self.index = None
        self.metadata_store = {}
        self.chroma_client = None
        self.collection = None
        
        # Load existing index if available
        self._load_existing_index()
    
    def _load_existing_index(self):
        """Load existing index from disk"""
        try:
            if self.vector_store_type == "faiss":
                self._load_faiss_index()
            elif self.vector_store_type == "chroma":
                self._load_chroma_index()
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
    
    def _load_faiss_index(self):
        """Load FAISS index from disk"""
        index_path = config.VECTOR_STORE_DIR / "index.faiss"
        metadata_path = config.VECTOR_STORE_DIR / "index_metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            self.index = faiss.read_index(str(index_path))
            
            with open(metadata_path, 'r') as f:
                self.metadata_store = json.load(f)
            
            logger.info("FAISS index loaded successfully")
    
    def _load_chroma_index(self):
        """Load ChromaDB index"""
        try:
            self.chroma_client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))
            self.collection = self.chroma_client.get_collection(config.CHROMA_COLLECTION_NAME)
            logger.info("ChromaDB index loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ChromaDB index: {e}")
    
    def add_documents(self, documents: List[DocumentChunk]) -> bool:
        """
        Add documents to the retrieval system
        
        Args:
            documents: List of DocumentChunk objects to add
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            return False
        
        try:
            if self.vector_store_type == "faiss":
                return self._add_to_faiss(documents)
            elif self.vector_store_type == "chroma":
                return self._add_to_chroma(documents)
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def _add_to_faiss(self, documents: List[DocumentChunk]) -> bool:
        """Add documents to FAISS index"""
        embeddings = []
        
        for doc in documents:
            if doc.embedding is not None:
                embeddings.append(doc.embedding)
                self.metadata_store[doc.id] = doc.to_dict()
        
        if not embeddings:
            return False
        
        embeddings = np.array(embeddings)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        if self.index is None:
            # Create new index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        
        self.index.add(embeddings)
        
        # Save index
        self._save_faiss_index()
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
        return True
    
    def _add_to_chroma(self, documents: List[DocumentChunk]) -> bool:
        """Add documents to ChromaDB"""
        if self.chroma_client is None:
            self.chroma_client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))
        
        if self.collection is None:
            self.collection = self.chroma_client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME
            )
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        embeddings = []
        metadatas = []
        
        for doc in documents:
            if doc.embedding is not None:
                ids.append(doc.id)
                texts.append(doc.content)
                embeddings.append(doc.embedding.tolist())
                metadatas.append(doc.metadata)
        
        if not ids:
            return False
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
        return True
    
    def search(self, query: Union[str, SearchQuery]) -> List[SearchResult]:
        """
        Search for relevant documents
        
        Args:
            query: Search query string or SearchQuery object
            
        Returns:
            List of SearchResult objects
        """
        if isinstance(query, str):
            query = SearchQuery(query)
        
        try:
            if self.vector_store_type == "faiss":
                return self._search_faiss(query)
            elif self.vector_store_type == "chroma":
                return self._search_chroma(query)
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _search_faiss(self, query: SearchQuery) -> List[SearchResult]:
        """Search using FAISS index"""
        if self.index is None or not self.metadata_store:
            logger.warning("No index available for search")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_engine.embed_query(query.query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, query.top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            # Get document metadata
            doc_id = list(self.metadata_store.keys())[idx]
            doc_data = self.metadata_store[doc_id]
            
            # Apply similarity threshold
            if score < query.similarity_threshold:
                continue
            
            # Apply filters
            if self._apply_filters(doc_data, query.filters):
                result = SearchResult(
                    content=doc_data["content"],
                    score=float(score),
                    metadata=doc_data["metadata"],
                    chunk_id=doc_id,
                    source=self._get_source_name(doc_data["metadata"])
                )
                results.append(result)
        
        return results
    
    def _search_chroma(self, query: SearchQuery) -> List[SearchResult]:
        """Search using ChromaDB"""
        if self.collection is None:
            logger.warning("No ChromaDB collection available for search")
            return []
        
        # Prepare where clause for filters
        where_clause = self._build_chroma_where_clause(query.filters)
        
        # Search
        results = self.collection.query(
            query_texts=[query.query],
            n_results=query.top_k,
            where=where_clause if where_clause else None
        )
        
        search_results = []
        
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score
                score = 1 - distance
                
                if score < query.similarity_threshold:
                    continue
                
                result = SearchResult(
                    content=doc,
                    score=score,
                    metadata=metadata,
                    chunk_id=results["ids"][0][i],
                    source=self._get_source_name(metadata)
                )
                search_results.append(result)
        
        return search_results
    
    def _apply_filters(self, doc_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to document data"""
        if not filters:
            return True
        
        metadata = doc_data.get("metadata", {})
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def _build_chroma_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build where clause for ChromaDB query"""
        if not filters:
            return None
        
        where_clause = {}
        for key, value in filters.items():
            if isinstance(value, list):
                where_clause[key] = {"$in": value}
            else:
                where_clause[key] = value
        
        return where_clause
    
    def _get_source_name(self, metadata: Dict[str, Any]) -> str:
        """Extract source name from metadata"""
        if "file_name" in metadata:
            return metadata["file_name"]
        elif "url" in metadata:
            return metadata["url"]
        elif "title" in metadata:
            return metadata["title"]
        else:
            return "Unknown Source"
    
    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        if self.index is not None:
            index_path = config.VECTOR_STORE_DIR / "index.faiss"
            metadata_path = config.VECTOR_STORE_DIR / "index_metadata.json"
            
            faiss.write_index(self.index, str(index_path))
            
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata_store, f, indent=2)
    
    def get_document_count(self) -> int:
        """Get total number of documents in the index"""
        if self.vector_store_type == "faiss":
            return len(self.metadata_store)
        elif self.vector_store_type == "chroma" and self.collection:
            return self.collection.count()
        return 0
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        stats = {
            "vector_store_type": self.vector_store_type,
            "document_count": self.get_document_count(),
            "embedding_dimension": self.embedding_engine.embedding_dim,
            "model_name": self.embedding_engine.model_name
        }
        
        if self.vector_store_type == "faiss" and self.index:
            stats["index_size"] = self.index.ntotal
            stats["index_type"] = type(self.index).__name__
        
        return stats
    
    def clear_index(self):
        """Clear the entire index"""
        if self.vector_store_type == "faiss":
            self.index = None
            self.metadata_store = {}
            
            # Remove index files
            index_path = config.VECTOR_STORE_DIR / "index.faiss"
            metadata_path = config.VECTOR_STORE_DIR / "index_metadata.json"
            
            if index_path.exists():
                index_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
        
        elif self.vector_store_type == "chroma" and self.collection:
            # Delete the collection
            self.chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
            self.collection = None
        
        logger.info("Index cleared")


class HybridRetrievalSystem:
    """Combines multiple retrieval strategies for better results"""
    
    def __init__(self, retrieval_systems: List[RetrievalSystem] = None):
        self.retrieval_systems = retrieval_systems or [
            RetrievalSystem("faiss"),
            RetrievalSystem("chroma")
        ]
        self.embedding_engine = embedding_manager.get_default_engine()
    
    def add_documents(self, documents: List[DocumentChunk]) -> bool:
        """Add documents to all retrieval systems"""
        success = True
        for system in self.retrieval_systems:
            if not system.add_documents(documents):
                success = False
        return success
    
    def search(self, query: Union[str, SearchQuery]) -> List[SearchResult]:
        """Search across all retrieval systems and combine results"""
        if isinstance(query, str):
            query = SearchQuery(query)
        
        all_results = []
        
        # Search each system
        for system in self.retrieval_systems:
            results = system.search(query)
            all_results.extend(results)
        
        # Remove duplicates and rerank
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rerank_results(unique_results, query.query)
        
        # Return top results
        return ranked_results[:query.top_k]
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on content similarity"""
        unique_results = []
        seen_contents = set()
        
        for result in results:
            # Simple deduplication based on content hash
            content_hash = hash(result.content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _rerank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rerank results using query-document similarity"""
        if not results:
            return results
        
        # Generate query embedding
        query_embedding = self.embedding_engine.embed_query(query)
        
        # Calculate similarity scores
        for result in results:
            # Use the existing embedding if available, otherwise generate new one
            if hasattr(result, 'embedding') and result.embedding is not None:
                doc_embedding = result.embedding
            else:
                doc_embedding = self.embedding_engine.embed_text(result.content)
            
            # Calculate cosine similarity
            similarity = self.embedding_engine.compute_similarity(query_embedding, doc_embedding)
            result.score = similarity
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results


# Global retrieval system instance
retrieval_system = RetrievalSystem()
