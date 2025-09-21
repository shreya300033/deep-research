"""
Summarization engine for the Deep Researcher Agent
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Text processing
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Our modules
import config
from embedding_engine import embedding_manager
from retrieval_system import SearchResult

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    import nltk
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    import nltk
    nltk.download('stopwords')


@dataclass
class SummaryResult:
    """Represents a summarization result"""
    original_text: str
    summary: str
    compression_ratio: float
    key_sentences: List[str]
    key_phrases: List[str]
    confidence: float
    method: str
    timestamp: datetime
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_text": self.original_text,
            "summary": self.summary,
            "compression_ratio": self.compression_ratio,
            "key_sentences": self.key_sentences,
            "key_phrases": self.key_phrases,
            "confidence": self.confidence,
            "method": self.method,
            "timestamp": self.timestamp.isoformat()
        }


class SummarizationEngine:
    """Handles text summarization using multiple methods"""
    
    def __init__(self):
        self.embedding_engine = embedding_manager.get_default_engine()
        self.stop_words = set(stopwords.words('english'))
        
    def summarize_text(self, 
                      text: str, 
                      method: str = "extractive",
                      max_sentences: int = 5,
                      max_words: int = 100) -> SummaryResult:
        """
        Summarize text using specified method
        
        Args:
            text: Text to summarize
            method: Summarization method ("extractive", "abstractive", "hybrid")
            max_sentences: Maximum number of sentences in summary
            max_words: Maximum number of words in summary
            
        Returns:
            SummaryResult object
        """
        if not text or not text.strip():
            return SummaryResult(
                original_text=text,
                summary="",
                compression_ratio=0.0,
                key_sentences=[],
                key_phrases=[],
                confidence=0.0,
                method=method
            )
        
        logger.info(f"Summarizing text using {method} method")
        
        if method == "extractive":
            return self._extractive_summarization(text, max_sentences, max_words)
        elif method == "abstractive":
            return self._abstractive_summarization(text, max_sentences, max_words)
        elif method == "hybrid":
            return self._hybrid_summarization(text, max_sentences, max_words)
        else:
            raise ValueError(f"Unsupported summarization method: {method}")
    
    def summarize_search_results(self, 
                                search_results: List[SearchResult],
                                method: str = "extractive",
                                max_sentences: int = 10,
                                max_words: int = 200) -> SummaryResult:
        """
        Summarize multiple search results
        
        Args:
            search_results: List of SearchResult objects
            method: Summarization method
            max_sentences: Maximum number of sentences in summary
            max_words: Maximum number of words in summary
            
        Returns:
            SummaryResult object
        """
        if not search_results:
            return SummaryResult(
                original_text="",
                summary="No search results to summarize",
                compression_ratio=0.0,
                key_sentences=[],
                key_phrases=[],
                confidence=0.0,
                method=method
            )
        
        # Combine all search results
        combined_text = self._combine_search_results(search_results)
        
        # Summarize the combined text
        return self.summarize_text(combined_text, method, max_sentences, max_words)
    
    def _extractive_summarization(self, text: str, max_sentences: int, max_words: int) -> SummaryResult:
        """Extractive summarization using sentence ranking"""
        # Split into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            # Text is already short enough
            return SummaryResult(
                original_text=text,
                summary=text,
                compression_ratio=1.0,
                key_sentences=sentences,
                key_phrases=self._extract_key_phrases(text),
                confidence=1.0,
                method="extractive"
            )
        
        # Calculate sentence scores
        sentence_scores = self._calculate_sentence_scores(sentences, text)
        
        # Select top sentences
        ranked_sentences = sorted(
            zip(sentences, sentence_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        selected_sentences = []
        word_count = 0
        
        for sentence, score in ranked_sentences:
            sentence_words = len(word_tokenize(sentence))
            if (len(selected_sentences) < max_sentences and 
                word_count + sentence_words <= max_words):
                selected_sentences.append(sentence)
                word_count += sentence_words
            else:
                break
        
        # Sort selected sentences by original order
        selected_sentences = [s for s in sentences if s in selected_sentences]
        
        summary = " ".join(selected_sentences)
        compression_ratio = len(summary) / len(text) if text else 0
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            compression_ratio=compression_ratio,
            key_sentences=selected_sentences,
            key_phrases=self._extract_key_phrases(summary),
            confidence=0.8,
            method="extractive"
        )
    
    def _abstractive_summarization(self, text: str, max_sentences: int, max_words: int) -> SummaryResult:
        """Abstractive summarization using embedding-based clustering"""
        # Split into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return SummaryResult(
                original_text=text,
                summary=text,
                compression_ratio=1.0,
                key_sentences=sentences,
                key_phrases=self._extract_key_phrases(text),
                confidence=1.0,
                method="abstractive"
            )
        
        # Generate embeddings for sentences
        sentence_embeddings = self.embedding_engine.embed_texts(sentences)
        
        # Cluster sentences
        clusters = self._cluster_sentences(sentence_embeddings, max_sentences)
        
        # Select representative sentences from each cluster
        selected_sentences = []
        for cluster in clusters:
            if cluster:
                # Select the sentence closest to cluster centroid
                cluster_embeddings = sentence_embeddings[cluster]
                centroid = np.mean(cluster_embeddings, axis=0)
                
                similarities = [
                    self.embedding_engine.compute_similarity(centroid, emb)
                    for emb in cluster_embeddings
                ]
                
                best_idx = cluster[np.argmax(similarities)]
                selected_sentences.append(sentences[best_idx])
        
        # Limit by word count
        final_sentences = []
        word_count = 0
        
        for sentence in selected_sentences:
            sentence_words = len(word_tokenize(sentence))
            if word_count + sentence_words <= max_words:
                final_sentences.append(sentence)
                word_count += sentence_words
            else:
                break
        
        summary = " ".join(final_sentences)
        compression_ratio = len(summary) / len(text) if text else 0
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            compression_ratio=compression_ratio,
            key_sentences=final_sentences,
            key_phrases=self._extract_key_phrases(summary),
            confidence=0.7,
            method="abstractive"
        )
    
    def _hybrid_summarization(self, text: str, max_sentences: int, max_words: int) -> SummaryResult:
        """Hybrid summarization combining extractive and abstractive methods"""
        # Get extractive summary
        extractive_result = self._extractive_summarization(text, max_sentences, max_words)
        
        # Get abstractive summary
        abstractive_result = self._abstractive_summarization(text, max_sentences, max_words)
        
        # Combine the summaries
        combined_sentences = list(set(extractive_result.key_sentences + abstractive_result.key_phrases))
        
        # Remove duplicates and limit length
        final_sentences = []
        word_count = 0
        
        for sentence in combined_sentences:
            sentence_words = len(word_tokenize(sentence))
            if (len(final_sentences) < max_sentences and 
                word_count + sentence_words <= max_words):
                final_sentences.append(sentence)
                word_count += sentence_words
            else:
                break
        
        summary = " ".join(final_sentences)
        compression_ratio = len(summary) / len(text) if text else 0
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            compression_ratio=compression_ratio,
            key_sentences=final_sentences,
            key_phrases=self._extract_key_phrases(summary),
            confidence=(extractive_result.confidence + abstractive_result.confidence) / 2,
            method="hybrid"
        )
    
    def _calculate_sentence_scores(self, sentences: List[str], full_text: str) -> List[float]:
        """Calculate importance scores for sentences"""
        if not sentences:
            return []
        
        # Method 1: TF-IDF based scoring
        tfidf_scores = self._calculate_tfidf_scores(sentences)
        
        # Method 2: Position-based scoring (first and last sentences are important)
        position_scores = self._calculate_position_scores(sentences)
        
        # Method 3: Length-based scoring (very short or very long sentences are less important)
        length_scores = self._calculate_length_scores(sentences)
        
        # Method 4: Keyword-based scoring
        keyword_scores = self._calculate_keyword_scores(sentences, full_text)
        
        # Combine scores
        combined_scores = []
        for i in range(len(sentences)):
            score = (
                tfidf_scores[i] * 0.4 +
                position_scores[i] * 0.2 +
                length_scores[i] * 0.2 +
                keyword_scores[i] * 0.2
            )
            combined_scores.append(score)
        
        return combined_scores
    
    def _calculate_tfidf_scores(self, sentences: List[str]) -> List[float]:
        """Calculate TF-IDF scores for sentences"""
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores as sum of TF-IDF values
            scores = tfidf_matrix.sum(axis=1).A1
            
            # Normalize scores
            if scores.max() > 0:
                scores = scores / scores.max()
            
            return scores.tolist()
            
        except Exception as e:
            logger.warning(f"TF-IDF calculation failed: {e}")
            return [0.5] * len(sentences)
    
    def _calculate_position_scores(self, sentences: List[str]) -> List[float]:
        """Calculate position-based scores for sentences"""
        scores = []
        n = len(sentences)
        
        for i in range(n):
            # Higher scores for first and last sentences
            if i == 0 or i == n - 1:
                score = 1.0
            else:
                # Gradually decrease score towards middle
                distance_from_edge = min(i, n - 1 - i)
                score = 1.0 - (distance_from_edge / (n / 2))
            
            scores.append(score)
        
        return scores
    
    def _calculate_length_scores(self, sentences: List[str]) -> List[float]:
        """Calculate length-based scores for sentences"""
        scores = []
        word_counts = [len(word_tokenize(sentence)) for sentence in sentences]
        
        if not word_counts:
            return []
        
        avg_length = sum(word_counts) / len(word_counts)
        
        for word_count in word_counts:
            # Optimal length is around average
            if word_count < 5:  # Too short
                score = 0.3
            elif word_count > 30:  # Too long
                score = 0.3
            else:
                # Score based on how close to average
                score = 1.0 - abs(word_count - avg_length) / avg_length
                score = max(0.1, min(1.0, score))
            
            scores.append(score)
        
        return scores
    
    def _calculate_keyword_scores(self, sentences: List[str], full_text: str) -> List[float]:
        """Calculate keyword-based scores for sentences"""
        # Extract keywords from full text
        words = word_tokenize(full_text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        top_keywords = set([word for word, freq in top_keywords])
        
        # Score sentences based on keyword presence
        scores = []
        for sentence in sentences:
            sentence_words = set(word_tokenize(sentence.lower()))
            sentence_words = set([word for word in sentence_words if word.isalpha()])
            
            keyword_count = len(sentence_words.intersection(top_keywords))
            score = min(1.0, keyword_count / 5.0)  # Normalize to 0-1
            scores.append(score)
        
        return scores
    
    def _cluster_sentences(self, embeddings: np.ndarray, n_clusters: int) -> List[List[int]]:
        """Cluster sentences using K-means"""
        try:
            from sklearn.cluster import KMeans
            
            n_clusters = min(n_clusters, len(embeddings))
            
            if n_clusters <= 1:
                return [list(range(len(embeddings)))]
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Group sentences by cluster
            clusters = [[] for _ in range(n_clusters)]
            for i, label in enumerate(cluster_labels):
                clusters[label].append(i)
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            # Fallback: return equal-sized groups
            n_sentences = len(embeddings)
            group_size = max(1, n_sentences // n_clusters)
            clusters = []
            for i in range(0, n_sentences, group_size):
                clusters.append(list(range(i, min(i + group_size, n_sentences))))
            return clusters
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple key phrase extraction using n-grams
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Extract bigrams and trigrams
        phrases = []
        
        # Bigrams
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            phrases.append(phrase)
        
        # Trigrams
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            phrases.append(phrase)
        
        # Count phrase frequencies
        phrase_freq = {}
        for phrase in phrases:
            phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
        
        # Return top phrases
        top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [phrase for phrase, freq in top_phrases]
    
    def _combine_search_results(self, search_results: List[SearchResult]) -> str:
        """Combine multiple search results into a single text"""
        # Sort by relevance score
        sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)
        
        # Combine content
        combined_parts = []
        for result in sorted_results:
            # Add source information
            source_info = f"[Source: {result.source}]"
            combined_parts.append(f"{source_info}\n{result.content}")
        
        return "\n\n".join(combined_parts)
    
    def generate_executive_summary(self, 
                                  search_results: List[SearchResult],
                                  max_sentences: int = 5) -> str:
        """Generate an executive summary from search results"""
        if not search_results:
            return "No information available for summary."
        
        # Get a concise summary
        summary_result = self.summarize_search_results(
            search_results, 
            method="extractive", 
            max_sentences=max_sentences,
            max_words=150
        )
        
        # Format as executive summary
        executive_summary = f"Executive Summary:\n\n{summary_result.summary}"
        
        # Add key insights
        if summary_result.key_phrases:
            executive_summary += f"\n\nKey Points:\n"
            for phrase in summary_result.key_phrases[:5]:
                executive_summary += f"• {phrase}\n"
        
        return executive_summary


# Global summarization engine instance
summarization_engine = SummarizationEngine()
