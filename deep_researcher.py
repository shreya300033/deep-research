"""
Main Deep Researcher Agent - Query handling and response generation
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Our modules
import config
from embedding_engine import embedding_manager
from document_processor import DocumentProcessor, DocumentChunk, document_processor
from retrieval_system import RetrievalSystem, SearchResult, retrieval_system
from reasoning_engine import ReasoningEngine, ReasoningPlan, reasoning_engine

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Represents the result of a research query"""
    query: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    reasoning_steps: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "reasoning_steps": self.reasoning_steps,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_markdown(self) -> str:
        """Convert to Markdown format"""
        md = f"# Research Report\n\n"
        md += f"**Query:** {self.query}\n\n"
        md += f"**Confidence:** {self.confidence:.2f}\n\n"
        md += f"**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        md += f"## Answer\n\n{self.answer}\n\n"
        
        if self.reasoning_steps:
            md += f"## Reasoning Process\n\n"
            for i, step in enumerate(self.reasoning_steps, 1):
                md += f"### Step {i}: {step.get('step_type', 'Unknown')}\n\n"
                md += f"**Description:** {step.get('description', '')}\n\n"
                if step.get('reasoning'):
                    md += f"**Reasoning:** {step['reasoning']}\n\n"
                if step.get('confidence'):
                    md += f"**Confidence:** {step['confidence']:.2f}\n\n"
        
        if self.sources:
            md += f"## Sources\n\n"
            for i, source in enumerate(self.sources, 1):
                md += f"{i}. **{source.get('source', 'Unknown')}**\n"
                if source.get('score'):
                    md += f"   - Relevance Score: {source['score']:.2f}\n"
                if source.get('metadata', {}).get('url'):
                    md += f"   - URL: {source['metadata']['url']}\n"
                md += f"   - Content: {source.get('content', '')[:200]}...\n\n"
        
        return md


class DeepResearcher:
    """Main Deep Researcher Agent class"""
    
    def __init__(self, 
                 embedding_model: str = None,
                 vector_store_type: str = None,
                 reasoning_model: str = None):
        """
        Initialize the Deep Researcher Agent
        
        Args:
            embedding_model: Name of the embedding model to use
            vector_store_type: Type of vector store ("faiss" or "chroma")
            reasoning_model: Type of reasoning model ("local", "openai", "anthropic")
        """
        self.embedding_engine = embedding_manager.get_engine(embedding_model)
        self.retrieval_system = RetrievalSystem(vector_store_type)
        self.reasoning_engine = ReasoningEngine(reasoning_model)
        self.document_processor = DocumentProcessor()
        
        # Statistics
        self.stats = {
            "queries_processed": 0,
            "documents_indexed": 0,
            "total_research_time": 0.0
        }
        
        logger.info("Deep Researcher Agent initialized")
    
    def add_documents(self, 
                     file_paths: List[Union[str, Path]] = None,
                     urls: List[str] = None,
                     texts: List[str] = None) -> Dict[str, Any]:
        """
        Add documents to the research system
        
        Args:
            file_paths: List of file paths to process
            urls: List of URLs to process
            texts: List of raw texts to process
            
        Returns:
            Dictionary with processing results
        """
        all_chunks = []
        results = {
            "files_processed": 0,
            "urls_processed": 0,
            "texts_processed": 0,
            "chunks_created": 0,
            "errors": []
        }
        
        # Process files
        if file_paths:
            for file_path in file_paths:
                try:
                    chunks = self.document_processor.process_file(file_path)
                    all_chunks.extend(chunks)
                    results["files_processed"] += 1
                    results["chunks_created"] += len(chunks)
                except Exception as e:
                    error_msg = f"Failed to process file {file_path}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
        
        # Process URLs
        if urls:
            for url in urls:
                try:
                    chunks = self.document_processor.process_url(url)
                    all_chunks.extend(chunks)
                    results["urls_processed"] += 1
                    results["chunks_created"] += len(chunks)
                except Exception as e:
                    error_msg = f"Failed to process URL {url}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
        
        # Process raw texts
        if texts:
            for i, text in enumerate(texts):
                try:
                    # Create a document chunk from raw text
                    metadata = {
                        "source_type": "text",
                        "text_index": i,
                        "processed_at": datetime.now().isoformat()
                    }
                    chunk = DocumentChunk(text, metadata)
                    chunk.embedding = self.embedding_engine.embed_text(text)
                    all_chunks.append(chunk)
                    results["texts_processed"] += 1
                    results["chunks_created"] += 1
                except Exception as e:
                    error_msg = f"Failed to process text {i}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
        
        # Index all chunks
        if all_chunks:
            success = self.retrieval_system.add_documents(all_chunks)
            if success:
                self.stats["documents_indexed"] += len(all_chunks)
                logger.info(f"Successfully indexed {len(all_chunks)} document chunks")
            else:
                results["errors"].append("Failed to index documents")
        
        return results
    
    def research(self, 
                query: str,
                max_sources: int = None,
                research_depth: str = None,
                include_reasoning: bool = True) -> ResearchResult:
        """
        Perform research on a given query
        
        Args:
            query: The research query
            max_sources: Maximum number of sources to use
            research_depth: Depth of research ("shallow", "medium", "deep")
            include_reasoning: Whether to include reasoning steps
            
        Returns:
            ResearchResult object
        """
        start_time = datetime.now()
        logger.info(f"Starting research for query: {query}")
        
        # Update statistics
        self.stats["queries_processed"] += 1
        
        # Set parameters
        max_sources = max_sources or config.MAX_SOURCES_PER_QUERY
        research_depth = research_depth or config.DEFAULT_RESEARCH_DEPTH
        
        try:
            # Create and execute reasoning plan
            if include_reasoning:
                plan = self.reasoning_engine.create_reasoning_plan(query)
                executed_plan = self.reasoning_engine.execute_reasoning_plan(plan)
                reasoning_steps = [step.to_dict() for step in executed_plan.steps]
                confidence = executed_plan.confidence
            else:
                reasoning_steps = []
                confidence = 0.0
            
            # Extract search results from reasoning plan
            search_results = []
            if include_reasoning and executed_plan.steps:
                for step in executed_plan.steps:
                    if (step.step_type.value == "information_gathering" and 
                        step.output_data and "search_results" in step.output_data):
                        search_results = step.output_data["search_results"]
                        break
            
            # If no search results from reasoning, do direct search
            if not search_results:
                search_results = self.retrieval_system.search(query)
            
            # Limit sources based on research depth
            if research_depth == "shallow":
                search_results = search_results[:5]
            elif research_depth == "medium":
                search_results = search_results[:10]
            else:  # deep
                search_results = search_results[:max_sources]
            
            # Generate answer
            if include_reasoning and executed_plan.steps:
                # Extract answer from reasoning plan
                answer = self._extract_answer_from_plan(executed_plan)
            else:
                # Generate simple answer from search results
                answer = self._generate_simple_answer(search_results, query)
            
            # Prepare sources
            sources = [result.to_dict() for result in search_results]
            
            # Calculate final confidence
            if not include_reasoning:
                confidence = self._calculate_confidence(search_results)
            
            # Create result
            result = ResearchResult(
                query=query,
                answer=answer,
                confidence=confidence,
                sources=sources,
                reasoning_steps=reasoning_steps,
                metadata={
                    "research_depth": research_depth,
                    "max_sources": max_sources,
                    "sources_used": len(sources),
                    "processing_time": (datetime.now() - start_time).total_seconds()
                },
                timestamp=start_time
            )
            
            # Update statistics
            self.stats["total_research_time"] += result.metadata["processing_time"]
            
            logger.info(f"Research completed in {result.metadata['processing_time']:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            # Return error result
            return ResearchResult(
                query=query,
                answer=f"Research failed: {str(e)}",
                confidence=0.0,
                sources=[],
                reasoning_steps=[],
                metadata={"error": str(e)},
                timestamp=start_time
            )
    
    def _extract_answer_from_plan(self, plan: ReasoningPlan) -> str:
        """Extract the final answer from a reasoning plan"""
        # Look for conclusion step
        for step in reversed(plan.steps):
            if (step.step_type.value == "conclusion" and 
                step.output_data and "conclusion" in step.output_data):
                conclusion = step.output_data["conclusion"]
                return conclusion.get("answer", "")
        
        # Look for synthesis step
        for step in reversed(plan.steps):
            if (step.step_type.value == "synthesis" and 
                step.output_data and "synthesis" in step.output_data):
                synthesis = step.output_data["synthesis"]
                return synthesis.get("summary", "")
        
        return "Unable to generate answer from reasoning plan."
    
    def _generate_simple_answer(self, search_results: List[SearchResult], query: str) -> str:
        """Generate a simple answer from search results"""
        if not search_results:
            return "No relevant information found for this query."
        
        # Combine top results
        top_results = search_results[:3]  # Use top 3 results
        answer_parts = []
        
        for i, result in enumerate(top_results, 1):
            # Extract key sentences from the result
            sentences = result.content.split('.')
            key_sentences = sentences[:2]  # First 2 sentences
            answer_parts.append(f"{'. '.join(key_sentences)}.")
        
        # Combine and clean up
        answer = " ".join(answer_parts)
        
        # Add context
        answer = f"Based on the available information, {answer.lower()}"
        
        return answer
    
    def _calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """Calculate confidence based on search results"""
        if not search_results:
            return 0.0
        
        # Calculate average score
        scores = [result.score for result in search_results]
        avg_score = sum(scores) / len(scores)
        
        # Adjust based on number of sources
        source_factor = min(len(search_results) / 5, 1.0)  # Cap at 1.0
        
        return avg_score * source_factor
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = self.stats.copy()
        stats.update({
            "embedding_model": self.embedding_engine.model_name,
            "vector_store_type": self.retrieval_system.vector_store_type,
            "indexed_documents": self.retrieval_system.get_document_count(),
            "index_stats": self.retrieval_system.get_index_stats()
        })
        return stats
    
    def clear_index(self):
        """Clear the document index"""
        self.retrieval_system.clear_index()
        self.stats["documents_indexed"] = 0
        logger.info("Document index cleared")
    
    def export_result(self, result: ResearchResult, format: str = None) -> str:
        """
        Export research result in specified format
        
        Args:
            result: ResearchResult to export
            format: Export format ("markdown", "json", "text")
            
        Returns:
            Exported content as string
        """
        format = format or config.DEFAULT_EXPORT_FORMAT
        
        if format == "markdown":
            return result.to_markdown()
        elif format == "json":
            return json.dumps(result.to_dict(), indent=2)
        elif format == "text":
            return f"Query: {result.query}\n\nAnswer: {result.answer}\n\nConfidence: {result.confidence:.2f}"
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def save_result(self, result: ResearchResult, file_path: str, format: str = None):
        """
        Save research result to file
        
        Args:
            result: ResearchResult to save
            file_path: Path to save the file
            format: Export format
        """
        content = self.export_result(result, format)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Research result saved to {file_path}")


# Global instance
deep_researcher = DeepResearcher()
