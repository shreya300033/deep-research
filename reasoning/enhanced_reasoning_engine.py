"""
Enhanced Reasoning Engine with Advanced AI Reasoning Patterns
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from reasoning.query_analyzer import QueryAnalyzer, ReasoningStep, QueryType
from embeddings.embedding_generator import EmbeddingGenerator
from retrieval.vector_store import VectorStore
import json


@dataclass
class EnhancedReasoningResult:
    """Enhanced result of a reasoning step with detailed analysis"""
    step: ReasoningStep
    query_embedding: Any
    retrieved_documents: List[Dict[str, Any]]
    synthesized_answer: str
    confidence_score: float
    sources: List[str]
    reasoning_pattern: str
    key_insights: List[str]
    supporting_evidence: List[str]
    logical_connections: List[str]


class EnhancedReasoningEngine:
    """Advanced reasoning engine with sophisticated AI reasoning patterns"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: VectorStore):
        self.query_analyzer = QueryAnalyzer()
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.reasoning_history = []
        
        # Advanced reasoning patterns
        self.reasoning_patterns = {
            'deductive': ['therefore', 'thus', 'hence', 'consequently', 'it follows'],
            'inductive': ['suggests', 'indicates', 'implies', 'points to', 'evidence shows'],
            'abductive': ['likely', 'probably', 'best explanation', 'most plausible'],
            'causal': ['causes', 'leads to', 'results in', 'due to', 'because of'],
            'comparative': ['compared to', 'versus', 'in contrast', 'similarly', 'differently'],
            'analogical': ['like', 'similar to', 'analogous', 'comparable', 'resembles']
        }
    
    def process_query_advanced(self, query: str, max_steps: int = 6) -> Dict[str, Any]:
        """Process a complex query through advanced multi-step reasoning"""
        # Decompose the query into reasoning steps
        reasoning_steps = self.query_analyzer.decompose_query(query)
        reasoning_steps = reasoning_steps[:max_steps]
        
        # Process each step with advanced reasoning
        results = []
        context = {}
        reasoning_chain = []
        
        for step in reasoning_steps:
            result = self._process_reasoning_step_advanced(step, context, query, reasoning_chain)
            results.append(result)
            
            # Update context with enhanced results
            context[f"step_{step.step_id}"] = {
                'answer': result.synthesized_answer,
                'insights': result.key_insights,
                'evidence': result.supporting_evidence,
                'pattern': result.reasoning_pattern,
                'confidence': result.confidence_score
            }
            
            reasoning_chain.append({
                'step': step.step_id,
                'pattern': result.reasoning_pattern,
                'insights': result.key_insights
            })
        
        # Generate advanced final synthesis
        final_answer = self._synthesize_final_answer_advanced(query, results, context, reasoning_chain)
        
        # Generate intelligent follow-up questions
        follow_ups = self._generate_intelligent_follow_ups(query, results, reasoning_chain)
        
        # Assess reasoning quality with advanced metrics
        reasoning_quality = self._assess_advanced_reasoning_quality(results, reasoning_chain)
        
        return {
            'original_query': query,
            'query_type': self.query_analyzer.classify_query(query).value,
            'reasoning_steps': [
                {
                    'step_id': r.step.step_id,
                    'description': r.step.description,
                    'answer': r.synthesized_answer,
                    'confidence': r.confidence_score,
                    'sources': r.sources,
                    'reasoning_pattern': r.reasoning_pattern,
                    'key_insights': r.key_insights,
                    'supporting_evidence': r.supporting_evidence,
                    'logical_connections': r.logical_connections
                }
                for r in results
            ],
            'final_answer': final_answer,
            'follow_up_questions': follow_ups,
            'total_sources': len(set(source for r in results for source in r.sources)),
            'reasoning_quality': reasoning_quality,
            'reasoning_chain': reasoning_chain,
            'advanced_metrics': self._calculate_advanced_metrics(results)
        }
    
    def _process_reasoning_step_advanced(self, step: ReasoningStep, context: Dict[str, Any], 
                                       original_query: str, reasoning_chain: List[Dict]) -> EnhancedReasoningResult:
        """Process a single reasoning step with advanced reasoning patterns"""
        # Generate embedding for the step query
        query_embedding = self.embedding_generator.generate_embedding(step.query)
        
        # Retrieve relevant documents with enhanced search
        retrieved_docs = self.vector_store.search(query_embedding, k=12)  # Increased for better context
        
        # Apply advanced reasoning patterns
        reasoning_pattern = self._identify_reasoning_pattern(step, retrieved_docs, context)
        
        # Synthesize answer with advanced reasoning
        synthesized_answer = self._synthesize_step_answer_advanced(
            step, retrieved_docs, context, original_query, reasoning_pattern
        )
        
        # Extract key insights and evidence
        key_insights = self._extract_key_insights(retrieved_docs, synthesized_answer)
        supporting_evidence = self._extract_supporting_evidence(retrieved_docs)
        logical_connections = self._identify_logical_connections(step, retrieved_docs, context)
        
        # Calculate enhanced confidence score
        confidence = self._calculate_enhanced_confidence(
            retrieved_docs, synthesized_answer, key_insights, reasoning_pattern
        )
        
        # Extract sources
        sources = list(set(doc.get('source', 'Unknown') for doc in retrieved_docs))
        
        return EnhancedReasoningResult(
            step=step,
            query_embedding=query_embedding,
            retrieved_documents=retrieved_docs,
            synthesized_answer=synthesized_answer,
            confidence_score=confidence,
            sources=sources,
            reasoning_pattern=reasoning_pattern,
            key_insights=key_insights,
            supporting_evidence=supporting_evidence,
            logical_connections=logical_connections
        )
    
    def _identify_reasoning_pattern(self, step: ReasoningStep, retrieved_docs: List[Dict[str, Any]], 
                                  context: Dict[str, Any]) -> str:
        """Identify the reasoning pattern being used"""
        combined_text = " ".join([doc.get('chunk_text', '') for doc in retrieved_docs[:5]])
        combined_text = combined_text.lower()
        
        pattern_scores = {}
        for pattern_name, indicators in self.reasoning_patterns.items():
            score = sum(1 for indicator in indicators if indicator in combined_text)
            pattern_scores[pattern_name] = score
        
        # Also consider step description
        step_text = step.description.lower()
        for pattern_name, indicators in self.reasoning_patterns.items():
            for indicator in indicators:
                if indicator in step_text:
                    pattern_scores[pattern_name] = pattern_scores.get(pattern_name, 0) + 2
        
        if pattern_scores:
            return max(pattern_scores.items(), key=lambda x: x[1])[0]
        else:
            return "analytical"
    
    def _synthesize_step_answer_advanced(self, step: ReasoningStep, retrieved_docs: List[Dict[str, Any]], 
                                       context: Dict[str, Any], original_query: str, 
                                       reasoning_pattern: str) -> str:
        """Advanced synthesis with pattern-specific reasoning"""
        if not retrieved_docs:
            return f"No relevant information found for: {step.description}"
        
        # Extract and organize information
        relevant_chunks = []
        for doc in retrieved_docs[:8]:
            chunk_info = {
                'text': doc.get('chunk_text', ''),
                'source': doc.get('source', 'Unknown'),
                'similarity': doc.get('similarity_score', 0),
                'title': doc.get('title', 'Unknown')
            }
            relevant_chunks.append(chunk_info)
        
        # Build comprehensive context
        context_info = self._build_advanced_context(context, original_query)
        
        # Apply pattern-specific reasoning
        if reasoning_pattern == 'deductive':
            return self._synthesize_deductive_reasoning(step, relevant_chunks, context_info)
        elif reasoning_pattern == 'inductive':
            return self._synthesize_inductive_reasoning(step, relevant_chunks, context_info)
        elif reasoning_pattern == 'causal':
            return self._synthesize_causal_reasoning(step, relevant_chunks, context_info)
        elif reasoning_pattern == 'comparative':
            return self._synthesize_comparative_reasoning(step, relevant_chunks, context_info)
        elif reasoning_pattern == 'analogical':
            return self._synthesize_analogical_reasoning(step, relevant_chunks, context_info)
        else:
            return self._synthesize_analytical_reasoning(step, relevant_chunks, context_info)
    
    def _synthesize_deductive_reasoning(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                      context: str) -> str:
        """Deductive reasoning: general to specific"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Look for general principles and specific conclusions
        general_principles = []
        specific_conclusions = []
        
        for chunk in chunks:
            text = chunk['text']
            # Look for general statements
            if any(word in text.lower() for word in ['principle', 'rule', 'law', 'theory', 'general']):
                general_principles.append(text[:200])
            # Look for specific conclusions
            if any(word in text.lower() for word in ['therefore', 'thus', 'hence', 'consequently']):
                specific_conclusions.append(text[:200])
        
        if general_principles and specific_conclusions:
            return f"Deductive reasoning for '{step.description}': " + \
                   f"General principles: {'; '.join(general_principles[:2])} " + \
                   f"Therefore: {'; '.join(specific_conclusions[:2])}"
        else:
            return f"Deductive analysis for '{step.description}': {chunks[0]['text'][:600]}..."
    
    def _synthesize_inductive_reasoning(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                      context: str) -> str:
        """Inductive reasoning: specific to general"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Look for patterns and evidence
        evidence_patterns = []
        general_patterns = []
        
        for chunk in chunks:
            text = chunk['text']
            # Look for evidence
            if any(word in text.lower() for word in ['evidence', 'data', 'study', 'research', 'findings']):
                evidence_patterns.append(text[:200])
            # Look for general patterns
            if any(word in text.lower() for word in ['suggests', 'indicates', 'implies', 'tends to']):
                general_patterns.append(text[:200])
        
        if evidence_patterns and general_patterns:
            return f"Inductive reasoning for '{step.description}': " + \
                   f"Evidence: {'; '.join(evidence_patterns[:2])} " + \
                   f"This suggests: {'; '.join(general_patterns[:2])}"
        else:
            return f"Inductive analysis for '{step.description}': {chunks[0]['text'][:600]}..."
    
    def _synthesize_causal_reasoning(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                   context: str) -> str:
        """Causal reasoning: cause and effect relationships"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Extract cause-effect relationships
        causes = []
        effects = []
        
        for chunk in chunks:
            text = chunk['text']
            # Look for causes
            if any(word in text.lower() for word in ['causes', 'leads to', 'results in', 'due to', 'because']):
                causes.append(text[:200])
            # Look for effects
            if any(word in text.lower() for word in ['effect', 'outcome', 'consequence', 'impact']):
                effects.append(text[:200])
        
        if causes and effects:
            return f"Causal reasoning for '{step.description}': " + \
                   f"Causes: {'; '.join(causes[:2])} " + \
                   f"Effects: {'; '.join(effects[:2])}"
        else:
            return f"Causal analysis for '{step.description}': {chunks[0]['text'][:600]}..."
    
    def _synthesize_comparative_reasoning(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                        context: str) -> str:
        """Comparative reasoning: similarities and differences"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Extract comparison elements
        similarities = []
        differences = []
        
        for chunk in chunks:
            text = chunk['text']
            # Look for similarities
            if any(word in text.lower() for word in ['similar', 'alike', 'both', 'common', 'shared']):
                similarities.append(text[:200])
            # Look for differences
            if any(word in text.lower() for word in ['different', 'unlike', 'versus', 'contrast', 'distinct']):
                differences.append(text[:200])
        
        if similarities and differences:
            return f"Comparative reasoning for '{step.description}': " + \
                   f"Similarities: {'; '.join(similarities[:2])} " + \
                   f"Differences: {'; '.join(differences[:2])}"
        else:
            return f"Comparative analysis for '{step.description}': {chunks[0]['text'][:600]}..."
    
    def _synthesize_analogical_reasoning(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                       context: str) -> str:
        """Analogical reasoning: comparisons and analogies"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Look for analogies and comparisons
        analogies = []
        
        for chunk in chunks:
            text = chunk['text']
            if any(word in text.lower() for word in ['like', 'similar to', 'analogous', 'comparable', 'resembles']):
                analogies.append(text[:200])
        
        if analogies:
            return f"Analogical reasoning for '{step.description}': " + \
                   f"Analogies found: {'; '.join(analogies[:2])}"
        else:
            return f"Analogical analysis for '{step.description}': {chunks[0]['text'][:600]}..."
    
    def _synthesize_analytical_reasoning(self, step: ReasoningStep, chunks: List[Dict[str, Any]], 
                                       context: str) -> str:
        """General analytical reasoning"""
        if not chunks:
            return f"No relevant information found for: {step.description}"
        
        # Comprehensive analysis
        high_quality_chunks = [chunk for chunk in chunks if chunk['similarity'] > 0.4]
        
        if high_quality_chunks:
            best_chunks = sorted(high_quality_chunks, key=lambda x: x['similarity'], reverse=True)[:3]
            combined_analysis = " ".join([chunk['text'] for chunk in best_chunks])
        else:
            combined_analysis = " ".join([chunk['text'] for chunk in chunks[:3]])
        
        return f"Analytical reasoning for '{step.description}': {combined_analysis[:600]}..."
    
    def _build_advanced_context(self, context: Dict[str, Any], original_query: str) -> str:
        """Build advanced context information"""
        context_info = f"Original Query: {original_query}\n\n"
        
        if context:
            context_info += "Previous Reasoning Context:\n"
            for key, value in context.items():
                if isinstance(value, dict):
                    context_info += f"- {value.get('answer', '')[:150]}...\n"
                    if 'insights' in value:
                        context_info += f"  Key insights: {', '.join(value['insights'][:2])}\n"
        
        return context_info
    
    def _extract_key_insights(self, retrieved_docs: List[Dict[str, Any]], 
                            synthesized_answer: str) -> List[str]:
        """Extract key insights from retrieved documents"""
        insights = []
        
        # Extract insights from high-similarity documents
        high_sim_docs = [doc for doc in retrieved_docs if doc.get('similarity_score', 0) > 0.6]
        
        for doc in high_sim_docs[:3]:
            text = doc.get('chunk_text', '')
            # Look for insight indicators
            sentences = text.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['insight', 'key', 'important', 'significant', 'critical']):
                    if len(sentence.strip()) > 20:
                        insights.append(sentence.strip()[:100])
        
        return insights[:3]
    
    def _extract_supporting_evidence(self, retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """Extract supporting evidence from documents"""
        evidence = []
        
        for doc in retrieved_docs[:5]:
            text = doc.get('chunk_text', '')
            similarity = doc.get('similarity_score', 0)
            
            if similarity > 0.5:
                # Extract evidence indicators
                if any(word in text.lower() for word in ['evidence', 'data', 'study', 'research', 'findings']):
                    evidence.append(f"{text[:150]}... (similarity: {similarity:.2f})")
        
        return evidence[:3]
    
    def _identify_logical_connections(self, step: ReasoningStep, retrieved_docs: List[Dict[str, Any]], 
                                    context: Dict[str, Any]) -> List[str]:
        """Identify logical connections between information"""
        connections = []
        
        # Look for logical connectors
        for doc in retrieved_docs[:3]:
            text = doc.get('chunk_text', '')
            if any(word in text.lower() for word in ['therefore', 'thus', 'hence', 'because', 'since', 'as a result']):
                connections.append(f"Logical connection: {text[:100]}...")
        
        return connections[:2]
    
    def _calculate_enhanced_confidence(self, retrieved_docs: List[Dict[str, Any]], 
                                     synthesized_answer: str, key_insights: List[str], 
                                     reasoning_pattern: str) -> float:
        """Calculate enhanced confidence score"""
        if not retrieved_docs:
            return 0.0
        
        # Base confidence from similarity scores
        num_sources = len(retrieved_docs)
        avg_similarity = sum(doc.get('similarity_score', 0) for doc in retrieved_docs) / num_sources
        
        # Boost for answer quality
        answer_quality = min(len(synthesized_answer) / 400, 1.0)
        
        # Boost for insights
        insight_boost = min(len(key_insights) * 0.1, 0.2)
        
        # Boost for reasoning pattern
        pattern_boost = 0.1 if reasoning_pattern in ['deductive', 'inductive', 'causal'] else 0.05
        
        # Calculate final confidence
        confidence = (avg_similarity * 0.6 + answer_quality * 0.3 + insight_boost + pattern_boost)
        
        # Minimum confidence for any retrieved content
        if num_sources > 0 and len(synthesized_answer) > 100:
            confidence = max(confidence, 0.4)
        
        return min(confidence, 1.0)
    
    def _synthesize_final_answer_advanced(self, original_query: str, results: List[EnhancedReasoningResult], 
                                        context: Dict[str, Any], reasoning_chain: List[Dict]) -> str:
        """Generate advanced final synthesis"""
        # Collect all insights and evidence
        all_insights = []
        all_evidence = []
        reasoning_patterns = []
        
        for result in results:
            all_insights.extend(result.key_insights)
            all_evidence.extend(result.supporting_evidence)
            reasoning_patterns.append(result.reasoning_pattern)
        
        # Create comprehensive synthesis
        synthesis = f"Based on advanced multi-step reasoning analysis of '{original_query}', here are the comprehensive findings:\n\n"
        
        # Add reasoning chain summary
        synthesis += f"Reasoning approach: {' → '.join(set(reasoning_patterns))}\n\n"
        
        # Add step-by-step analysis
        for i, result in enumerate(results, 1):
            synthesis += f"{i}. {result.step.description} ({result.reasoning_pattern} reasoning):\n"
            synthesis += f"   {result.synthesized_answer}\n"
            if result.key_insights:
                synthesis += f"   Key insights: {'; '.join(result.key_insights)}\n"
            synthesis += "\n"
        
        # Add integrated insights
        if all_insights:
            synthesis += f"Integrated insights: {'; '.join(set(all_insights))}\n\n"
        
        # Add supporting evidence
        if all_evidence:
            synthesis += f"Supporting evidence: {'; '.join(set(all_evidence))}\n\n"
        
        synthesis += f"Overall, this advanced reasoning analysis provides a comprehensive understanding of '{original_query}' through systematic application of {len(set(reasoning_patterns))} different reasoning patterns across {len(results)} analytical steps."
        
        return synthesis
    
    def _generate_intelligent_follow_ups(self, original_query: str, results: List[EnhancedReasoningResult], 
                                       reasoning_chain: List[Dict]) -> List[str]:
        """Generate intelligent follow-up questions"""
        follow_ups = []
        
        # Generate follow-ups based on reasoning patterns used
        patterns_used = set(result.reasoning_pattern for result in results)
        
        if 'causal' in patterns_used:
            follow_ups.append("What are the underlying mechanisms behind these causal relationships?")
        
        if 'comparative' in patterns_used:
            follow_ups.append("What are the practical implications of these differences?")
        
        if 'inductive' in patterns_used:
            follow_ups.append("What additional evidence would strengthen these conclusions?")
        
        # Generate follow-ups based on insights
        all_insights = []
        for result in results:
            all_insights.extend(result.key_insights)
        
        if all_insights:
            follow_ups.append("How do these insights apply to real-world scenarios?")
        
        # Generate follow-ups based on query type
        query_type = self.query_analyzer.classify_query(original_query)
        if query_type == QueryType.ANALYTICAL:
            follow_ups.append("What are the limitations of this analysis?")
        elif query_type == QueryType.COMPARATIVE:
            follow_ups.append("Which approach would be better for specific use cases?")
        
        return follow_ups[:3]
    
    def _assess_advanced_reasoning_quality(self, results: List[EnhancedReasoningResult], 
                                         reasoning_chain: List[Dict]) -> str:
        """Assess advanced reasoning quality"""
        if not results:
            return "No reasoning steps completed"
        
        # Calculate advanced metrics
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        total_sources = len(set(source for r in results for source in r.sources))
        total_insights = sum(len(r.key_insights) for r in results)
        reasoning_diversity = len(set(r.reasoning_pattern for r in results))
        
        # Assess quality based on multiple factors
        if avg_confidence > 0.7 and total_sources > 3 and total_insights > 5 and reasoning_diversity > 2:
            return "Exceptional reasoning quality with diverse patterns and strong evidence"
        elif avg_confidence > 0.6 and total_sources > 2 and total_insights > 3 and reasoning_diversity > 1:
            return "High quality reasoning with multiple patterns and good evidence"
        elif avg_confidence > 0.5 and total_sources > 1 and total_insights > 2:
            return "Good quality reasoning with adequate evidence and insights"
        elif avg_confidence > 0.4 and total_sources > 0:
            return "Moderate quality reasoning with limited evidence"
        else:
            return "Low quality reasoning with insufficient evidence"
    
    def _calculate_advanced_metrics(self, results: List[EnhancedReasoningResult]) -> Dict[str, Any]:
        """Calculate advanced reasoning metrics"""
        if not results:
            return {}
        
        total_insights = sum(len(r.key_insights) for r in results)
        total_evidence = sum(len(r.supporting_evidence) for r in results)
        reasoning_patterns = set(r.reasoning_pattern for r in results)
        
        return {
            'total_insights': total_insights,
            'total_evidence': total_evidence,
            'reasoning_patterns_used': list(reasoning_patterns),
            'pattern_diversity': len(reasoning_patterns),
            'average_confidence': sum(r.confidence_score for r in results) / len(results),
            'total_sources': len(set(source for r in results for source in r.sources))
        }