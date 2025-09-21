"""
Vector storage and retrieval system using FAISS for efficient similarity search.
Handles document indexing, embedding storage, and retrieval operations.
"""

import numpy as np
import faiss
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
from datetime import datetime

from .document_processor import DocumentChunk
from .embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class VectorStore:
    """Efficient vector storage and retrieval system."""
    
    def __init__(self, storage_dir: str = "./data/vector_store", 
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize vector store.
        
        Args:
            storage_dir: Directory to store vector data
            embedding_generator: Embedding generator instance
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.embedding_dim = self.embedding_generator.get_embedding_dimension()
        
        # FAISS index
        self.index = None
        self.index_file = self.storage_dir / "faiss_index.bin"
        
        # Metadata storage
        self.metadata_db = self.storage_dir / "metadata.db"
        self.chunks_db = self.storage_dir / "chunks.db"
        
        # Load existing data
        self._load_index()
        self._init_databases()
    
    def _init_databases(self):
        """Initialize SQLite databases for metadata and chunks."""
        # Metadata database
        conn = sqlite3.connect(self.metadata_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                doc_id TEXT PRIMARY KEY,
                source_type TEXT,
                source_path TEXT,
                title TEXT,
                description TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                chunk_count INTEGER,
                total_length INTEGER
            )
        """)
        conn.commit()
        conn.close()
        
        # Chunks database
        conn = sqlite3.connect(self.chunks_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                content TEXT,
                metadata TEXT,
                embedding_index INTEGER,
                created_at TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def _load_index(self):
        """Load existing FAISS index."""
        if self.index_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")
    
    def _save_index(self):
        """Save FAISS index to disk."""
        try:
            faiss.write_index(self.index, str(self.index_file))
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def add_document(self, chunks: List[DocumentChunk], 
                    doc_id: Optional[str] = None) -> str:
        """
        Add a document (as chunks) to the vector store.
        
        Args:
            chunks: List of document chunks
            doc_id: Optional document ID
            
        Returns:
            Document ID
        """
        if not chunks:
            raise ValueError("No chunks provided")
        
        # Generate document ID if not provided
        if doc_id is None:
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(chunks)}"
        
        # Generate embeddings for all chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks in database
        conn = sqlite3.connect(self.chunks_db)
        for i, chunk in enumerate(chunks):
            embedding_index = self.index.ntotal - len(chunks) + i
            conn.execute("""
                INSERT OR REPLACE INTO chunks 
                (chunk_id, doc_id, content, metadata, embedding_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                doc_id,
                chunk.content,
                json.dumps(chunk.metadata),
                embedding_index,
                datetime.now().isoformat()
            ))
        conn.commit()
        conn.close()
        
        # Store document metadata
        self._store_document_metadata(doc_id, chunks)
        
        # Save index
        self._save_index()
        
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
        return doc_id
    
    def _store_document_metadata(self, doc_id: str, chunks: List[DocumentChunk]):
        """Store document metadata in database."""
        if not chunks:
            return
        
        # Extract metadata from first chunk
        first_chunk = chunks[0]
        metadata = first_chunk.metadata
        
        conn = sqlite3.connect(self.metadata_db)
        conn.execute("""
            INSERT OR REPLACE INTO document_metadata 
            (doc_id, source_type, source_path, title, description, 
             created_at, updated_at, chunk_count, total_length)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            metadata.get('source_type', 'unknown'),
            metadata.get('file_path', metadata.get('url', '')),
            metadata.get('file_name', metadata.get('title', '')),
            f"Document with {len(chunks)} chunks",
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            len(chunks),
            sum(len(chunk.content) for chunk in chunks)
        ))
        conn.commit()
        conn.close()
    
    def search(self, query: str, top_k: int = 10, 
              min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results with chunks and metadata
        """
        if self.index.ntotal == 0:
            logger.warning("No documents in index")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Retrieve chunks and metadata
        results = []
        conn = sqlite3.connect(self.chunks_db)
        
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity < min_similarity:
                continue
            
            # Get chunk data
            cursor = conn.execute("""
                SELECT chunk_id, doc_id, content, metadata 
                FROM chunks WHERE embedding_index = ?
            """, (int(idx),))
            
            row = cursor.fetchone()
            if row:
                chunk_id, doc_id, content, metadata_json = row
                metadata = json.loads(metadata_json)
                
                results.append({
                    'chunk_id': chunk_id,
                    'doc_id': doc_id,
                    'content': content,
                    'metadata': metadata,
                    'similarity': float(similarity)
                })
        
        conn.close()
        
        logger.info(f"Found {len(results)} results for query: {query[:50]}...")
        return results
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunks with metadata
        """
        conn = sqlite3.connect(self.chunks_db)
        cursor = conn.execute("""
            SELECT chunk_id, content, metadata 
            FROM chunks WHERE doc_id = ? 
            ORDER BY embedding_index
        """, (doc_id,))
        
        chunks = []
        for row in cursor.fetchall():
            chunk_id, content, metadata_json = row
            metadata = json.loads(metadata_json)
            chunks.append({
                'chunk_id': chunk_id,
                'content': content,
                'metadata': metadata
            })
        
        conn.close()
        return chunks
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.execute("""
            SELECT * FROM document_metadata WHERE doc_id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, row))
        
        return None
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the store."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.execute("""
            SELECT * FROM document_metadata ORDER BY created_at DESC
        """)
        
        documents = []
        for row in cursor.fetchall():
            columns = [description[0] for description in cursor.description]
            documents.append(dict(zip(columns, row)))
        
        conn.close()
        return documents
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks from the store.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get chunk indices to remove from FAISS
            conn = sqlite3.connect(self.chunks_db)
            cursor = conn.execute("""
                SELECT embedding_index FROM chunks WHERE doc_id = ?
                ORDER BY embedding_index DESC
            """, (doc_id,))
            
            indices_to_remove = [row[0] for row in cursor.fetchall()]
            
            if not indices_to_remove:
                conn.close()
                return False
            
            # Remove from FAISS index (remove in reverse order to maintain indices)
            for idx in indices_to_remove:
                self.index.remove_ids(np.array([idx]))
            
            # Remove from chunks database
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.commit()
            conn.close()
            
            # Remove from metadata database
            conn = sqlite3.connect(self.metadata_db)
            conn.execute("DELETE FROM document_metadata WHERE doc_id = ?", (doc_id,))
            conn.commit()
            conn.close()
            
            # Save updated index
            self._save_index()
            
            logger.info(f"Deleted document {doc_id} with {len(indices_to_remove)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        conn_meta = sqlite3.connect(self.metadata_db)
        conn_chunks = sqlite3.connect(self.chunks_db)
        
        # Document count
        doc_count = conn_meta.execute("SELECT COUNT(*) FROM document_metadata").fetchone()[0]
        
        # Chunk count
        chunk_count = conn_chunks.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        
        # Total storage size
        total_size = sum(
            self.storage_dir.glob("**/*")
            for f in self.storage_dir.glob("**/*")
            if f.is_file()
        )
        
        conn_meta.close()
        conn_chunks.close()
        
        return {
            'total_documents': doc_count,
            'total_chunks': chunk_count,
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.embedding_dim,
            'storage_size_mb': total_size / (1024 * 1024),
            'index_type': 'FAISS IndexFlatIP'
        }

