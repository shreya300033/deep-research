"""
Multi-step reasoning engine for complex query processing
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from reasoning.query_analyzer import QueryAnalyzer, ReasoningStep, QueryType
from embeddings.embedding_generator import EmbeddingGenerator
from retrieval.vector_store import VectorStore
import json


@dataclass
class ReasoningResult:
    """Result of a reasoning step"""
    step: ReasoningStep
    query_embedding: Any
    retrieved_documents: List[Dict[str, Any]]
    synthesized_answer: str
    confidence_score: float
    sources: List[str]


class ReasoningEngine:
    """Handles multi-step reasoning and query processing"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: VectorStore):
        self.query_analyzer = QueryAnalyzer()
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.reasoning_history = []
    
    def process_query(self, query: str, max_steps: int = 5) -> Dict[str, Any]:
        """Process a complex query through multi-step reasoning"""
        # Decompose the query into reasoning steps
        reasoning_steps = self.query_analyzer.decompose_query(query)
        
        # Limit the number of steps
        reasoning_steps = reasoning_steps[:max_steps]
        
        # Process each step
        results = []
        context = {}
        
        for step in reasoning_steps:
            result = self._process_reasoning_step(step, context, query)
            results.append(result)
            
            # Update context with results from this step
            context[f"step_{step.step_id}"] = {
                'answer': result.synthesized_answer,
                'sources': result.sources,
                'confidence': result.confidence_score
            }
        
        # Generate final synthesis
        final_answer = self._synthesize_final_answer(query, results, context)
        
        # Generate follow-up questions
        follow_ups = self.query_analyzer.create_follow_up_questions(
            query, [r.retrieved_documents for r in results]
        )
        
        return {
            'original_query': query,
            'query_type': self.query_analyzer.classify_query(query).value,
            'reasoning_steps': [
                {
                    'step_id': r.step.step_id,
                    'description': r.step.description,
                    'answer': r.synthesized_answer,
                    'confidence': r.confidence_score,
                    'sources': r.sources
                }
                for r in results
            ],
            'final_answer': final_answer,
            'follow_up_questions': follow_ups,
            'total_sources': len(set(source for r in results for source in r.sources)),
            'reasoning_quality': self._assess_reasoning_quality(results)
        }
    
    def _process_reasoning_step(self, step: ReasoningStep, context: Dict[str, Any], 
                              original_query: str) -> ReasoningResult:
        """Process a single reasoning step"""
        # Generate embedding for the step query
        query_embedding = self.embedding_generator.generate_embedding(step.query)
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query_embedding)
        
        # Synthesize answer based on retrieved documents and context
        synthesized_answer = self._synthesize_step_answer(
            step, retrieved_docs, context, original_query
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence(retrieved_docs, synthesized_answer)
        
        # Extract sources
        sources = list(set(doc.get('source', 'Unknown') for doc in retrieved_docs))
        
        return ReasoningResult(
            step=step,
            query_embedding=query_embedding,
            retrieved_documents=retrieved_docs,
            synthesized_answer=synthesized_answer,
            confidence_score=confidence,
            sources=sources
        )
    
    def _synthesize_step_answer(self, step: ReasoningStep, retrieved_docs: List[Dict[str, Any]], 
                              context: Dict[str, Any], original_query: str) -> str:
        """Synthesize an answer for a reasoning step with enhanced reasoning"""
        if not retrieved_docs:
            return f"No relevant information found for: {step.description}"
        
        # Extract relevant text chunks with similarity scores
        relevant_chunks = []
        for doc in retrieved_docs[:8]:  # Increased from 5 to 8 for more context
            chunk_text = doc.get('chunk_text', '')
            similarity = doc.get('similarity_score', 0)
            if chunk_text and similarity > 0.2:  # Filter out very low similarity
                relevant_chunks.append({
                    'text': chunk_text,
                    'similarity': similarity,
                    'source': doc.get('source', 'Unknown')
                })
        
        if not relevant_chunks:
            return f"No sufficiently relevant information found for: {step.description}"
        
        # Build comprehensive context
        context_info = self._build_context_info(context, original_query)
        
        # Enhanced synthesis based on step type with context awareness
        if step.description.startswith("Identify"):
            return self._synthesize_identification_enhanced(step, relevant_chunks, context_info, original_query)
        elif step.description.startswith("Gather"):
            return self._synthesize_information_gathering_enhanced(step, relevant_chunks, context_info, original_query)
        elif step.description.startswith("Analyze") or step.description.startswith("Evaluate"):
            return self._synthesize_analysis_enhanced(step, relevant_chunks, context_info, original_query)
        elif step.description.startswith("Compare"):
            return self._synthesize_comparison_enhanced(step, relevant_chunks, context_info, original_query)
        elif step.description.startswith("Explain"):
            return self._synthesize_explanation_enhanced(step, relevant_chunks, context_info, original_query)
        elif step.description.startswith("Synthesize"):
            return self._synthesize_synthesis_enhanced(step, relevant_chunks, context_info, original_query)
        else:
            return self._synthesize_general_enhanced(step, relevant_chunks, context_info, original_query)
    
    def _synthesize_identification(self, step: ReasoningStep, chunks: List[str], 
                                 context: str) -> str:
        """Synthesize identification-type answers with query-specific focus"""
        if not chunks:
            # Extract the original query from context if available
            original_query = ""
            if "Original Query:" in context:
                original_query = context.split("Original Query:")[1].split("\n")[0].strip()
            
            # Provide a helpful fallback response
            if original_query:
                return f"Based on the available knowledge base, no specific information was found for '{original_query}'. The knowledge base contains information about artificial intelligence, machine learning, computer vision, AI ethics, and large language models. You may want to try rephrasing your question or asking about one of these topics."
            else:
                return f"No relevant information found for: {step.description}. The knowledge base may not contain information about this specific topic."
        
        # Extract the original query from context if available
        original_query = ""
        if "Original Query:" in context:
            original_query = context.split("Original Query:")[1].split("\n")[0].strip()
        
        # Use the most relevant chunk based on content matching
        best_chunk = chunks[0]  # Default to first chunk
        
        if original_query:
            query_words = original_query.lower().split()
            best_score = 0
            for chunk in chunks:
                chunk_lower = chunk.lower()
                score = sum(1 for word in query_words if word in chunk_lower)
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
        
        # Extract query-specific information
        if original_query:
            query_lower = original_query.lower()
            
            if 'what is' in query_lower or 'define' in query_lower:
                # Look for definitions
                import re
                definitions = re.findall(r'([A-Z][^.!?]*(?:is|are|refers to|means)[^.!?]*)', best_chunk)
                if definitions:
                    return f"Based on the available information: {definitions[0]}. Additional context: {best_chunk[:400]}..."
            
            elif 'how' in query_lower:
                # Look for processes and explanations
                processes = re.findall(r'([A-Z][^.!?]*(?:works|functions|operates|processes)[^.!?]*)', best_chunk)
                if processes:
                    return f"Based on the available information: {processes[0]}. Additional context: {best_chunk[:400]}..."
            
            elif 'compare' in query_lower or 'difference' in query_lower:
                # Look for comparisons
                comparisons = re.findall(r'([A-Z][^.!?]*(?:versus|compared to|different from|similar to)[^.!?]*)', best_chunk)
                if comparisons:
                    return f"Based on the available information: {comparisons[0]}. Additional context: {best_chunk[:400]}..."
        
        # Fallback to general concept extraction
        key_terms = []
        words = best_chunk.split()
        for word in words:
            if word.istitle() and len(word) > 3:
                key_terms.append(word)
        
        unique_terms = list(set(key_terms))[:8]
        
        if unique_terms:
            return f"Based on the available information, the key elements for '{step.description}' are: {', '.join(unique_terms)}. " + \
                   f"Additional context: {best_chunk[:400]}..."
        else:
            return f"Based on the available information: {best_chunk[:600]}..."
    
    def _synthesize_information_gathering(self, step: ReasoningStep, chunks: List[str], 
                                        context: str) -> str:
        """Synthesize information gathering answers with query-specific focus"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Extract the original query from context if available
        original_query = ""
        if "Original Query:" in context:
            original_query = context.split("Original Query:")[1].split("\n")[0].strip()
        
        # Use query-specific information gathering
        if original_query:
            query_lower = original_query.lower()
            
            # Find the most relevant chunks for the specific query
            relevant_chunks = []
            for chunk in chunks:
                chunk_lower = chunk.lower()
                # Check if chunk contains query-related terms
                query_words = query_lower.split()
                relevance_score = sum(1 for word in query_words if word in chunk_lower)
                if relevance_score > 0:
                    relevant_chunks.append((chunk, relevance_score))
            
            # Sort by relevance and use the most relevant content
            if relevant_chunks:
                relevant_chunks.sort(key=lambda x: x[1], reverse=True)
                best_chunks = [chunk for chunk, score in relevant_chunks[:2]]
                combined_text = " ".join(best_chunks)
            else:
                combined_text = chunks[0]  # Fallback to first chunk
        else:
            combined_text = " ".join(chunks[:2])  # Use first two chunks
        
        # Summarize the gathered information
        summary = combined_text[:600] + "..." if len(combined_text) > 600 else combined_text
        
        return f"Information gathered for '{step.description}': {summary}"
    
    def _synthesize_analysis(self, step: ReasoningStep, chunks: List[str], 
                           context: str) -> str:
        """Synthesize analytical answers"""
        combined_text = " ".join(chunks)
        
        # Look for analytical indicators
        analytical_indicators = ['however', 'therefore', 'consequently', 'furthermore', 
                               'moreover', 'in contrast', 'on the other hand']
        
        analysis_parts = []
        for indicator in analytical_indicators:
            if indicator in combined_text.lower():
                # Find sentences containing the indicator
                sentences = combined_text.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        analysis_parts.append(sentence.strip())
        
        if analysis_parts:
            analysis = ". ".join(analysis_parts[:3])
        else:
            analysis = combined_text[:600] + "..." if len(combined_text) > 600 else combined_text
        
        return f"Analysis for '{step.description}': {analysis}"
    
    def _synthesize_comparison(self, step: ReasoningStep, chunks: List[str], 
                             context: str) -> str:
        """Synthesize comparison answers"""
        combined_text = " ".join(chunks)
        
        # Look for comparison indicators
        comparison_indicators = ['versus', 'compared to', 'in contrast', 'similarly', 
                               'differently', 'better', 'worse', 'advantage', 'disadvantage']
        
        comparison_parts = []
        for indicator in comparison_indicators:
            if indicator in combined_text.lower():
                sentences = combined_text.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        comparison_parts.append(sentence.strip())
        
        if comparison_parts:
            comparison = ". ".join(comparison_parts[:3])
        else:
            comparison = combined_text[:600] + "..." if len(combined_text) > 600 else combined_text
        
        return f"Comparison for '{step.description}': {comparison}"
    
    def _synthesize_explanation(self, step: ReasoningStep, chunks: List[str], 
                              context: str) -> str:
        """Synthesize explanatory answers"""
        combined_text = " ".join(chunks)
        
        # Look for explanatory indicators
        explanatory_indicators = ['because', 'due to', 'as a result', 'leads to', 
                                'causes', 'results in', 'explains', 'means that']
        
        explanation_parts = []
        for indicator in explanatory_indicators:
            if indicator in combined_text.lower():
                sentences = combined_text.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        explanation_parts.append(sentence.strip())
        
        if explanation_parts:
            explanation = ". ".join(explanation_parts[:3])
        else:
            explanation = combined_text[:600] + "..." if len(combined_text) > 600 else combined_text
        
        return f"Explanation for '{step.description}': {explanation}"
    
    def _synthesize_general(self, step: ReasoningStep, chunks: List[str], 
                          context: str) -> str:
        """Synthesize general answers"""
        combined_text = " ".join(chunks)
        summary = combined_text[:700] + "..." if len(combined_text) > 700 else combined_text
        
        return f"Answer for '{step.description}': {summary}"
    
    def _synthesize_final_answer(self, original_query: str, results: List[ReasoningResult], 
                               context: Dict[str, Any]) -> str:
        """Synthesize the final comprehensive answer with query-specific focus"""
        if not results:
            return "No reasoning steps completed"
        
        # Extract query type for tailored response
        query_lower = original_query.lower()
        
        # Create query-specific summary
        if 'what is' in query_lower or 'define' in query_lower:
            summary = f"Definition and explanation of '{original_query}':\n\n"
        elif 'how' in query_lower:
            summary = f"Process and mechanism explanation for '{original_query}':\n\n"
        elif 'compare' in query_lower or 'difference' in query_lower:
            summary = f"Comparison and analysis of '{original_query}':\n\n"
        elif 'why' in query_lower:
            summary = f"Causal analysis and explanation for '{original_query}':\n\n"
        else:
            summary = f"Comprehensive analysis of '{original_query}':\n\n"
        
        # Present insights in a structured way
        for i, result in enumerate(results, 1):
            # Extract the most relevant part of each answer
            answer = result.synthesized_answer
            if "Based on the available information:" in answer:
                # Extract the main content after the prefix
                main_content = answer.split("Based on the available information:")[1].strip()
                summary += f"{i}. {result.step.description}: {main_content}\n\n"
            else:
                summary += f"{i}. {result.step.description}: {answer}\n\n"
        
        # Add query-specific conclusion
        if 'what is' in query_lower or 'define' in query_lower:
            summary += f"In summary, this analysis provides a clear definition and understanding of '{original_query}' based on {len(results)} systematic reasoning steps."
        elif 'how' in query_lower:
            summary += f"In summary, this analysis explains how '{original_query}' works through {len(results)} systematic reasoning steps."
        elif 'compare' in query_lower or 'difference' in query_lower:
            summary += f"In summary, this analysis provides a comprehensive comparison of '{original_query}' through {len(results)} systematic reasoning steps."
        else:
            summary += f"In summary, this analysis provides a comprehensive understanding of '{original_query}' through {len(results)} systematic reasoning steps."
        
        return summary
    
    def _calculate_confidence(self, retrieved_docs: List[Dict[str, Any]], 
                            synthesized_answer: str) -> float:
        """Calculate confidence score for a reasoning step"""
        if not retrieved_docs:
            return 0.0
        
        # Base confidence on number of sources and similarity scores
        num_sources = len(retrieved_docs)
        avg_similarity = sum(doc.get('similarity_score', 0) for doc in retrieved_docs) / num_sources
        
        # Adjust based on answer quality (length and coherence)
        answer_quality = min(len(synthesized_answer) / 300, 1.0)  # Lower threshold for better scores
        
        # Boost confidence if we have good similarity scores
        if avg_similarity > 0.7:
            confidence = (avg_similarity * 0.7 + answer_quality * 0.3)
        elif avg_similarity > 0.5:
            confidence = (avg_similarity * 0.6 + answer_quality * 0.4)
        else:
            confidence = (avg_similarity * 0.5 + answer_quality * 0.5)
        
        # Minimum confidence boost for any retrieved content
        if num_sources > 0 and len(synthesized_answer) > 50:
            confidence = max(confidence, 0.3)
        
        return min(confidence, 1.0)
    
    def _build_context_info(self, context: Dict[str, Any], original_query: str) -> str:
        """Build comprehensive context information for reasoning"""
        context_parts = []
        
        if context:
            context_parts.append("Previous reasoning context:")
            for key, value in context.items():
                if isinstance(value, dict) and 'answer' in value:
                    context_parts.append(f"- {value['answer'][:200]}...")
        
        context_parts.append(f"Original query: {original_query}")
        
        return "\n".join(context_parts)
    
    def _synthesize_identification_enhanced(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                         context: str, original_query: str) -> str:
        """Enhanced identification synthesis with better analysis"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Extract key concepts with better analysis
        key_concepts = []
        important_phrases = []
        
        for chunk in chunks:
            text = chunk['text']
            similarity = chunk['similarity']
            
            # Extract capitalized terms (proper nouns, concepts)
            words = text.split()
            for word in words:
                if word.istitle() and len(word) > 3 and word.isalpha():
                    key_concepts.append(word)
            
            # Extract important phrases (sentences with key terms)
            sentences = text.split('.')
            for sentence in sentences:
                if any(term.lower() in sentence.lower() for term in original_query.split()):
                    important_phrases.append(sentence.strip())
        
        # Remove duplicates and rank by frequency
        concept_counts = {}
        for concept in key_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        
        # Build comprehensive answer
        answer_parts = []
        
        if top_concepts:
            concepts_text = ", ".join([concept for concept, count in top_concepts])
            answer_parts.append(f"Key concepts identified: {concepts_text}")
        
        if important_phrases:
            # Select most relevant phrases
            relevant_phrases = important_phrases[:3]
            answer_parts.append("Relevant information:")
            for phrase in relevant_phrases:
                if len(phrase) > 20:
                    answer_parts.append(f"- {phrase}")
        
        # Add context from highest similarity chunks
        high_similarity_chunks = [c for c in chunks if c['similarity'] > 0.6]
        if high_similarity_chunks:
            best_chunk = high_similarity_chunks[0]
            answer_parts.append(f"Primary source context: {best_chunk['text'][:300]}...")
        
        return " ".join(answer_parts) if answer_parts else f"Based on available information: {chunks[0]['text'][:400]}..."
    
    def _synthesize_information_gathering_enhanced(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                                 context: str, original_query: str) -> str:
        """Enhanced information gathering with better organization"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Organize information by relevance and source
        organized_info = []
        
        for chunk in chunks:
            text = chunk['text']
            similarity = chunk['similarity']
            source = chunk['source']
            
            # Extract key sentences that relate to the query
            sentences = text.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    # Check if sentence contains query-related terms
                    query_terms = [term.lower() for term in original_query.split() if len(term) > 3]
                    if any(term in sentence.lower() for term in query_terms):
                        relevant_sentences.append(sentence)
            
            if relevant_sentences:
                organized_info.append({
                    'sentences': relevant_sentences[:2],  # Top 2 relevant sentences
                    'similarity': similarity,
                    'source': source
                })
        
        # Sort by similarity and build answer
        organized_info.sort(key=lambda x: x['similarity'], reverse=True)
        
        answer_parts = []
        for info in organized_info[:4]:  # Top 4 most relevant sources
            for sentence in info['sentences']:
                answer_parts.append(f"- {sentence}")
        
        if answer_parts:
            return f"Information gathered from {len(organized_info)} sources:\n" + "\n".join(answer_parts)
        else:
            # Fallback to general information
            combined_text = " ".join([chunk['text'] for chunk in chunks[:3]])
            return f"Information gathered: {combined_text[:500]}..."
    
    def _synthesize_analysis_enhanced(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                    context: str, original_query: str) -> str:
        """Enhanced analysis with deeper reasoning"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Look for analytical indicators and patterns
        analytical_indicators = {
            'causal': ['because', 'due to', 'as a result', 'leads to', 'causes', 'results in'],
            'comparative': ['however', 'in contrast', 'on the other hand', 'similarly', 'differently'],
            'evaluative': ['effective', 'successful', 'important', 'significant', 'critical', 'beneficial'],
            'temporal': ['first', 'then', 'finally', 'initially', 'subsequently', 'eventually']
        }
        
        analysis_parts = []
        
        for chunk in chunks:
            text = chunk['text']
            similarity = chunk['similarity']
            
            # Find sentences with analytical indicators
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 30:
                    for category, indicators in analytical_indicators.items():
                        if any(indicator in sentence.lower() for indicator in indicators):
                            analysis_parts.append({
                                'sentence': sentence,
                                'category': category,
                                'similarity': similarity
                            })
        
        # Organize analysis by category
        if analysis_parts:
            analysis_parts.sort(key=lambda x: x['similarity'], reverse=True)
            
            answer_parts = []
            categories_found = set()
            
            for analysis in analysis_parts[:6]:  # Top 6 analytical insights
                if analysis['category'] not in categories_found:
                    answer_parts.append(f"{analysis['category'].title()} analysis: {analysis['sentence']}")
                    categories_found.add(analysis['category'])
            
            if answer_parts:
                return "Analysis findings:\n" + "\n".join(answer_parts)
        
        # Fallback to general analysis
        combined_text = " ".join([chunk['text'] for chunk in chunks[:3]])
        return f"Analysis based on available information: {combined_text[:600]}..."
    
    def _synthesize_comparison_enhanced(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                      context: str, original_query: str) -> str:
        """Enhanced comparison with structured analysis"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Extract comparison subjects from the query
        query_lower = original_query.lower()
        comparison_terms = []
        
        # Look for comparison indicators
        comparison_indicators = ['vs', 'versus', 'compare', 'contrast', 'difference', 'similarity']
        for indicator in comparison_indicators:
            if indicator in query_lower:
                # Extract terms around the indicator
                parts = query_lower.split(indicator)
                if len(parts) > 1:
                    comparison_terms.extend([part.strip() for part in parts])
        
        # Find comparative information
        comparative_info = []
        
        for chunk in chunks:
            text = chunk['text']
            similarity = chunk['similarity']
            
            # Look for comparative language
            comparative_words = ['better', 'worse', 'advantage', 'disadvantage', 'superior', 'inferior', 
                               'more', 'less', 'different', 'similar', 'versus', 'compared to']
            
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(word in sentence.lower() for word in comparative_words):
                    comparative_info.append({
                        'sentence': sentence,
                        'similarity': similarity
                    })
        
        if comparative_info:
            comparative_info.sort(key=lambda x: x['similarity'], reverse=True)
            
            answer_parts = []
            answer_parts.append("Comparative analysis:")
            
            for info in comparative_info[:4]:  # Top 4 comparative insights
                answer_parts.append(f"- {info['sentence']}")
            
            return "\n".join(answer_parts)
        
        # Fallback to general comparison
        combined_text = " ".join([chunk['text'] for chunk in chunks[:3]])
        return f"Comparison based on available information: {combined_text[:600]}..."
    
    def _synthesize_explanation_enhanced(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                       context: str, original_query: str) -> str:
        """Enhanced explanation with causal reasoning"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Look for explanatory patterns
        explanatory_indicators = ['because', 'due to', 'as a result', 'leads to', 'causes', 'results in', 
                                'explains', 'means that', 'therefore', 'consequently']
        
        explanations = []
        
        for chunk in chunks:
            text = chunk['text']
            similarity = chunk['similarity']
            
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 30:
                    if any(indicator in sentence.lower() for indicator in explanatory_indicators):
                        explanations.append({
                            'sentence': sentence,
                            'similarity': similarity
                        })
        
        if explanations:
            explanations.sort(key=lambda x: x['similarity'], reverse=True)
            
            answer_parts = []
            answer_parts.append("Explanatory analysis:")
            
            for explanation in explanations[:4]:  # Top 4 explanations
                answer_parts.append(f"- {explanation['sentence']}")
            
            return "\n".join(answer_parts)
        
        # Fallback to general explanation
        combined_text = " ".join([chunk['text'] for chunk in chunks[:3]])
        return f"Explanation based on available information: {combined_text[:600]}..."
    
    def _synthesize_synthesis_enhanced(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                     context: str, original_query: str) -> str:
        """Enhanced synthesis with comprehensive integration"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Integrate information from multiple sources
        synthesis_parts = []
        
        # Group chunks by similarity for better organization
        high_similarity = [c for c in chunks if c['similarity'] > 0.6]
        medium_similarity = [c for c in chunks if 0.4 <= c['similarity'] <= 0.6]
        low_similarity = [c for c in chunks if c['similarity'] < 0.4]
        
        if high_similarity:
            synthesis_parts.append("Primary findings:")
            for chunk in high_similarity[:3]:
                synthesis_parts.append(f"- {chunk['text'][:200]}...")
        
        if medium_similarity:
            synthesis_parts.append("Supporting information:")
            for chunk in medium_similarity[:2]:
                synthesis_parts.append(f"- {chunk['text'][:150]}...")
        
        if synthesis_parts:
            return "\n".join(synthesis_parts)
        
        # Fallback to general synthesis
        combined_text = " ".join([chunk['text'] for chunk in chunks[:4]])
        return f"Comprehensive synthesis: {combined_text[:700]}..."
    
    def _synthesize_general_enhanced(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                   context: str, original_query: str) -> str:
        """Enhanced general synthesis with better organization"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Organize by relevance and build comprehensive answer
        sorted_chunks = sorted(chunks, key=lambda x: x['similarity'], reverse=True)
        
        answer_parts = []
        
        # Primary information (highest similarity)
        primary_chunks = [c for c in sorted_chunks if c['similarity'] > 0.5]
        if primary_chunks:
            answer_parts.append("Primary information:")
            for chunk in primary_chunks[:2]:
                answer_parts.append(f"- {chunk['text'][:250]}...")
        
        # Secondary information
        secondary_chunks = [c for c in sorted_chunks if 0.3 <= c['similarity'] <= 0.5]
        if secondary_chunks:
            answer_parts.append("Additional context:")
            for chunk in secondary_chunks[:2]:
                answer_parts.append(f"- {chunk['text'][:200]}...")
        
        if answer_parts:
            return "\n".join(answer_parts)
        
        # Fallback
        combined_text = " ".join([chunk['text'] for chunk in sorted_chunks[:3]])
        return f"Information synthesis: {combined_text[:600]}..."
    
    def _assess_reasoning_quality(self, results: List[ReasoningResult]) -> str:
        """Assess the overall quality of the reasoning process"""
        if not results:
            return "No reasoning steps completed"
        
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        total_sources = len(set(source for r in results for source in r.sources))
        
        # Count steps with meaningful content
        meaningful_steps = sum(1 for r in results if len(r.synthesized_answer) > 100)
        
        if avg_confidence > 0.7 and total_sources > 2 and meaningful_steps >= len(results) * 0.6:
            return "High quality reasoning with strong evidence"
        elif avg_confidence > 0.6 and total_sources > 1 and meaningful_steps >= len(results) * 0.4:
            return "Good quality reasoning with adequate evidence"
        elif avg_confidence > 0.4 and total_sources > 0:
            return "Moderate quality reasoning with limited evidence"
        else:
            return "Low quality reasoning with insufficient evidence"
