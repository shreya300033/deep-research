"""
Main Deep Researcher Agent - orchestrates all components
"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from embeddings.embedding_generator import EmbeddingGenerator
from retrieval.vector_store import VectorStore
from reasoning.reasoning_engine import ReasoningEngine
from reasoning.enhanced_reasoning_engine import EnhancedReasoningEngine
from reasoning.query_analyzer import QueryType
from export.pdf_exporter import PDFExporter
from config import DATA_DIR, REPORTS_DIR


class DeepResearcherAgent:
    """Main agent that orchestrates the research process"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", use_enhanced_reasoning: bool = True):
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_store = VectorStore()
        self.reasoning_engine = ReasoningEngine(self.embedding_generator, self.vector_store)
        self.enhanced_reasoning_engine = EnhancedReasoningEngine(self.embedding_generator, self.vector_store)
        self.use_enhanced_reasoning = use_enhanced_reasoning
        self.pdf_exporter = PDFExporter()
        self.research_history = []
        self.current_session = None
        
    def start_research_session(self, session_name: Optional[str] = None) -> str:
        """Start a new research session"""
        if session_name is None:
            session_name = f"research_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = {
            'session_id': session_name,
            'start_time': datetime.now().isoformat(),
            'queries': [],
            'documents_indexed': 0
        }
        
        return session_name
    
    def index_documents(self, documents: List[Dict[str, Any]], 
                       index_name: str = "default") -> Dict[str, Any]:
        """Index a collection of documents for research"""
        if not documents:
            return {"error": "No documents provided"}
        
        print(f"Indexing {len(documents)} documents...")
        
        all_embeddings = []
        all_metadata = []
        
        for i, doc in enumerate(documents):
            try:
                embeddings, metadata = self.embedding_generator.process_document(doc)
                all_embeddings.extend(embeddings)
                all_metadata.extend(metadata)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                print(f"Error processing document {i}: {e}")
                continue
        
        if all_embeddings:
            # Convert to numpy array
            import numpy as np
            embeddings_array = np.array(all_embeddings)
            
            # Add to vector store
            self.vector_store.add_embeddings(embeddings_array, all_metadata)
            
            # Save index
            self.vector_store.save_index(index_name)
            
            # Update session info
            if self.current_session:
                self.current_session['documents_indexed'] += len(documents)
            
            return {
                "success": True,
                "documents_indexed": len(documents),
                "chunks_created": len(all_metadata),
                "index_name": index_name
            }
        else:
            return {"error": "No valid documents could be processed"}
    
    def load_index(self, index_name: str) -> bool:
        """Load an existing index"""
        try:
            self.vector_store.load_index(index_name)
            return True
        except Exception as e:
            print(f"Error loading index {index_name}: {e}")
            return False
    
    def research_query(self, query: str, max_reasoning_steps: int = 5) -> Dict[str, Any]:
        """Process a research query using advanced multi-step reasoning"""
        if not self.current_session:
            self.start_research_session()
        
        print(f"Processing research query: {query}")
        
        # Choose reasoning engine based on configuration
        if self.use_enhanced_reasoning:
            # Use enhanced reasoning engine for advanced analysis
            result = self.enhanced_reasoning_engine.process_query_advanced(query, max_reasoning_steps)
        else:
            # Use standard reasoning engine
            result = self.reasoning_engine.process_query(query, max_reasoning_steps)
        
        # Add to research history
        research_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'result': result,
            'session_id': self.current_session['session_id'] if self.current_session else None,
            'reasoning_type': 'enhanced' if self.use_enhanced_reasoning else 'standard'
        }
        
        self.research_history.append(research_entry)
        
        if self.current_session:
            self.current_session['queries'].append(research_entry)
        
        return result
    
    def refine_query(self, original_query: str, refinement: str) -> Dict[str, Any]:
        """Refine a previous query with additional context"""
        # Combine original query with refinement
        refined_query = f"{original_query}. {refinement}"
        
        # Process the refined query
        result = self.research_query(refined_query)
        
        # Mark as refinement
        result['is_refinement'] = True
        result['original_query'] = original_query
        result['refinement'] = refinement
        
        return result
    
    def get_research_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of research conducted"""
        if session_id:
            session_queries = [q for q in self.research_history 
                             if q.get('session_id') == session_id]
        else:
            session_queries = self.research_history
        
        if not session_queries:
            return {"error": "No research queries found"}
        
        # Analyze research patterns
        query_types = [q['result'].get('query_type', 'unknown') for q in session_queries]
        
        # Get unique sources
        all_sources = set()
        for query in session_queries:
            for step in query['result'].get('reasoning_steps', []):
                all_sources.update(step.get('sources', []))
        
        # Count query types
        from collections import Counter
        query_type_counts = Counter(query_types)
        
        return {
            "session_id": session_id,
            "total_queries": len(session_queries),
            "query_types": dict(query_type_counts),
            "total_sources_accessed": len(all_sources),
            "research_timeline": [
                {
                    "timestamp": q['timestamp'],
                    "query": q['query'],
                    "type": q['result'].get('query_type', 'unknown')
                }
                for q in session_queries
            ]
        }
    
    def export_research_report(self, session_id: Optional[str] = None, 
                             format: str = "markdown") -> str:
        """Export research results to a structured format"""
        if session_id:
            session_queries = [q for q in self.research_history 
                             if q.get('session_id') == session_id]
        else:
            session_queries = self.research_history
        
        if not session_queries:
            return "No research data to export"
        
        # Generate report content
        if format.lower() == "markdown":
            return self._generate_markdown_report(session_queries, session_id)
        elif format.lower() == "json":
            return self._generate_json_report(session_queries, session_id)
        elif format.lower() == "pdf":
            return self._generate_pdf_report(session_queries, session_id)
        else:
            return "Unsupported format. Use 'markdown', 'json', or 'pdf'"
    
    def _generate_markdown_report(self, queries: List[Dict[str, Any]], 
                                session_id: Optional[str]) -> str:
        """Generate a markdown research report"""
        report = f"# Research Report\n\n"
        
        if session_id:
            report += f"**Session ID:** {session_id}\n\n"
        
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Total Queries:** {len(queries)}\n\n"
        
        report += "## Research Summary\n\n"
        
        # Add summary statistics
        query_types = [q['result'].get('query_type', 'unknown') for q in queries]
        unique_types = list(set(query_types))
        report += f"**Query Types:** {', '.join(unique_types)}\n\n"
        
        # Add each research query
        for i, query_data in enumerate(queries, 1):
            result = query_data['result']
            
            report += f"## Query {i}: {query_data['query']}\n\n"
            report += f"**Type:** {result.get('query_type', 'unknown')}\n\n"
            report += f"**Reasoning Quality:** {result.get('reasoning_quality', 'unknown')}\n\n"
            
            # Add reasoning steps
            report += "### Reasoning Steps\n\n"
            for step in result.get('reasoning_steps', []):
                report += f"**{step['description']}**\n\n"
                report += f"{step['answer']}\n\n"
                report += f"*Confidence: {step['confidence']:.2f}*\n\n"
            
            # Add final answer
            report += "### Final Answer\n\n"
            report += f"{result.get('final_answer', 'No answer generated')}\n\n"
            
            # Add follow-up questions
            follow_ups = result.get('follow_up_questions', [])
            if follow_ups:
                report += "### Suggested Follow-up Questions\n\n"
                for j, follow_up in enumerate(follow_ups, 1):
                    report += f"{j}. {follow_up}\n"
                report += "\n"
            
            report += "---\n\n"
        
        return report
    
    def _generate_json_report(self, queries: List[Dict[str, Any]], 
                            session_id: Optional[str]) -> str:
        """Generate a JSON research report"""
        report_data = {
            "session_id": session_id,
            "generated_at": datetime.now().isoformat(),
            "total_queries": len(queries),
            "queries": queries
        }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def _generate_pdf_report(self, queries: List[Dict[str, Any]], 
                           session_id: Optional[str]) -> str:
        """Generate a PDF research report"""
        # Prepare data for PDF export
        research_data = {
            'session_id': session_id,
            'queries': queries,
            'total_sources_accessed': len(set(
                source for q in queries 
                for step in q['result'].get('reasoning_steps', [])
                for source in step.get('sources', [])
            ))
        }
        
        # Create temporary PDF file
        temp_pdf_path = REPORTS_DIR / f"temp_report_{session_id}.pdf"
        
        try:
            # Export to PDF
            self.pdf_exporter.export_research_report(research_data, str(temp_pdf_path))
            
            # Read the PDF content (for consistency with other export methods)
            with open(temp_pdf_path, 'rb') as f:
                pdf_content = f.read()
            
            # Clean up temp file
            temp_pdf_path.unlink()
            
            return f"PDF report generated successfully. Use save_report() to save to file."
            
        except Exception as e:
            return f"Error generating PDF report: {str(e)}"
    
    def save_report(self, content: str, filename: str, format: str = "markdown") -> str:
        """Save a research report to file"""
        if format.lower() == "markdown":
            file_path = REPORTS_DIR / f"{filename}.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        elif format.lower() == "json":
            file_path = REPORTS_DIR / f"{filename}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        elif format.lower() == "pdf":
            file_path = REPORTS_DIR / f"{filename}.pdf"
            self.pdf_exporter.export_simple_report(content, str(file_path), filename)
        else:
            return "Unsupported format"
        
        return str(file_path)
    
    def save_pdf_report(self, session_id: Optional[str] = None, 
                       filename: Optional[str] = None) -> str:
        """Save research report directly as PDF"""
        if session_id:
            session_queries = [q for q in self.research_history 
                             if q.get('session_id') == session_id]
        else:
            session_queries = self.research_history
        
        if not session_queries:
            return "No research data to export"
        
        if filename is None:
            filename = f"research_report_{session_id or 'all_sessions'}"
        
        file_path = REPORTS_DIR / f"{filename}.pdf"
        
        # Prepare data for PDF export
        research_data = {
            'session_id': session_id,
            'queries': session_queries,
            'total_sources_accessed': len(set(
                source for q in session_queries 
                for step in q['result'].get('reasoning_steps', [])
                for source in step.get('sources', [])
            ))
        }
        
        try:
            self.pdf_exporter.export_research_report(research_data, str(file_path))
            return str(file_path)
        except Exception as e:
            return f"Error creating PDF report: {str(e)}"
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the current vector store"""
        return self.vector_store.get_stats()
    
    def clear_research_history(self) -> None:
        """Clear the research history"""
        self.research_history = []
        self.current_session = None
    
    def get_available_indices(self) -> List[str]:
        """Get list of available saved indices"""
        index_files = list(Path("index").glob("*.index"))
        return [f.stem for f in index_files]
