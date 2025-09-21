"""
Multi-step reasoning and query decomposition
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process"""
    step_id: int
    description: str
    query: str
    expected_output: str
    dependencies: List[int] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class QueryAnalyzer:
    """Analyzes queries and breaks them down into reasoning steps"""
    
    def __init__(self):
        self.query_patterns = {
            QueryType.FACTUAL: [
                r"what is",
                r"who is",
                r"when did",
                r"where is",
                r"define",
                r"explain what"
            ],
            QueryType.ANALYTICAL: [
                r"analyze",
                r"examine",
                r"evaluate",
                r"assess",
                r"critique"
            ],
            QueryType.COMPARATIVE: [
                r"compare",
                r"contrast",
                r"difference between",
                r"similarities",
                r"versus"
            ],
            QueryType.CAUSAL: [
                r"why",
                r"cause",
                r"effect",
                r"result in",
                r"lead to",
                r"because of"
            ],
            QueryType.PROCEDURAL: [
                r"how to",
                r"steps to",
                r"process",
                r"method",
                r"procedure"
            ],
            QueryType.CONCEPTUAL: [
                r"concept of",
                r"theory",
                r"principle",
                r"framework",
                r"model"
            ]
        }
    
    def classify_query(self, query: str) -> QueryType:
        """Classify the type of query"""
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        return QueryType.FACTUAL  # Default to factual
    
    def decompose_query(self, query: str) -> List[ReasoningStep]:
        """Break down a complex query into reasoning steps"""
        query_type = self.classify_query(query)
        
        if query_type == QueryType.FACTUAL:
            return self._decompose_factual_query(query)
        elif query_type == QueryType.ANALYTICAL:
            return self._decompose_analytical_query(query)
        elif query_type == QueryType.COMPARATIVE:
            return self._decompose_comparative_query(query)
        elif query_type == QueryType.CAUSAL:
            return self._decompose_causal_query(query)
        elif query_type == QueryType.PROCEDURAL:
            return self._decompose_procedural_query(query)
        elif query_type == QueryType.CONCEPTUAL:
            return self._decompose_conceptual_query(query)
        
        return [ReasoningStep(1, "Answer query", query, "Comprehensive answer")]
    
    def _decompose_factual_query(self, query: str) -> List[ReasoningStep]:
        """Decompose factual queries with enhanced reasoning"""
        return [
            ReasoningStep(
                1,
                "Identify key concepts and entities",
                f"What are the key concepts, entities, and important terms in: {query}",
                "Comprehensive list of key concepts, entities, and terminology"
            ),
            ReasoningStep(
                2,
                "Gather comprehensive factual information",
                f"Find detailed factual information, definitions, and explanations for: {query}",
                "Detailed factual information with context and examples"
            ),
            ReasoningStep(
                3,
                "Analyze relationships and connections",
                f"What are the relationships, connections, and implications related to: {query}",
                "Analysis of relationships and contextual connections"
            ),
            ReasoningStep(
                4,
                "Synthesize comprehensive findings",
                f"Create a comprehensive synthesis integrating all information about: {query}",
                "Well-structured, comprehensive factual answer"
            )
        ]
    
    def _decompose_analytical_query(self, query: str) -> List[ReasoningStep]:
        """Decompose analytical queries with enhanced reasoning"""
        return [
            ReasoningStep(
                1,
                "Define analytical scope and framework",
                f"What are the key components, scope, and analytical framework for: {query}",
                "Clear analytical scope and methodological framework"
            ),
            ReasoningStep(
                2,
                "Gather comprehensive evidence and data",
                f"Collect all relevant evidence, data, and supporting information for: {query}",
                "Comprehensive evidence base and supporting data"
            ),
            ReasoningStep(
                3,
                "Apply systematic analytical methods",
                f"Apply systematic analysis, evaluation, and critical examination to: {query}",
                "Detailed analytical findings and evaluations"
            ),
            ReasoningStep(
                4,
                "Identify patterns and insights",
                f"What patterns, trends, and key insights emerge from analyzing: {query}",
                "Patterns, trends, and analytical insights"
            ),
            ReasoningStep(
                5,
                "Synthesize analytical conclusions",
                f"Draw comprehensive conclusions and implications from the analysis of: {query}",
                "Well-reasoned analytical conclusions and implications"
            )
        ]
    
    def _decompose_comparative_query(self, query: str) -> List[ReasoningStep]:
        """Decompose comparative queries with enhanced reasoning"""
        return [
            ReasoningStep(
                1,
                "Identify and define comparison subjects",
                f"What are the specific subjects, concepts, or entities being compared in: {query}",
                "Clear identification and definition of comparison subjects"
            ),
            ReasoningStep(
                2,
                "Gather comprehensive information about each subject",
                f"Collect detailed information, characteristics, and context for each subject in: {query}",
                "Comprehensive information about each comparison subject"
            ),
            ReasoningStep(
                3,
                "Establish comparison framework and criteria",
                f"What are the appropriate criteria, dimensions, and framework for comparing subjects in: {query}",
                "Systematic comparison framework and evaluation criteria"
            ),
            ReasoningStep(
                4,
                "Perform systematic comparative analysis",
                f"Conduct detailed comparative analysis across all relevant dimensions for: {query}",
                "Systematic comparative analysis and evaluation"
            ),
            ReasoningStep(
                5,
                "Identify key differences and similarities",
                f"What are the most significant differences, similarities, and patterns in: {query}",
                "Key differences, similarities, and comparative patterns"
            ),
            ReasoningStep(
                6,
                "Synthesize comparative insights and implications",
                f"What insights, implications, and conclusions emerge from comparing subjects in: {query}",
                "Comprehensive comparative insights and implications"
            )
        ]
    
    def _decompose_causal_query(self, query: str) -> List[ReasoningStep]:
        """Decompose causal queries"""
        return [
            ReasoningStep(
                1,
                "Identify the phenomenon",
                f"What is the main phenomenon in: {query}",
                "Description of the phenomenon"
            ),
            ReasoningStep(
                2,
                "Identify potential causes",
                f"What are potential causes for: {query}",
                "List of potential causes"
            ),
            ReasoningStep(
                3,
                "Gather evidence for each cause",
                f"Find evidence supporting each potential cause for: {query}",
                "Evidence for each potential cause"
            ),
            ReasoningStep(
                4,
                "Evaluate causal relationships",
                f"Evaluate which causes are most likely for: {query}",
                "Evaluation of causal relationships"
            ),
            ReasoningStep(
                5,
                "Explain the causal mechanism",
                f"Explain how the causes lead to: {query}",
                "Causal mechanism explanation"
            )
        ]
    
    def _decompose_procedural_query(self, query: str) -> List[ReasoningStep]:
        """Decompose procedural queries"""
        return [
            ReasoningStep(
                1,
                "Identify the goal",
                f"What is the goal in: {query}",
                "Clear statement of the goal"
            ),
            ReasoningStep(
                2,
                "Identify prerequisites",
                f"What are the prerequisites for: {query}",
                "List of prerequisites and requirements"
            ),
            ReasoningStep(
                3,
                "Break down into steps",
                f"What are the main steps for: {query}",
                "List of main procedural steps"
            ),
            ReasoningStep(
                4,
                "Detail each step",
                f"Provide detailed instructions for each step in: {query}",
                "Detailed step-by-step instructions"
            ),
            ReasoningStep(
                5,
                "Identify potential issues",
                f"What are potential issues or considerations for: {query}",
                "Potential issues and considerations"
            )
        ]
    
    def _decompose_conceptual_query(self, query: str) -> List[ReasoningStep]:
        """Decompose conceptual queries"""
        return [
            ReasoningStep(
                1,
                "Define the concept",
                f"What is the definition of the concept in: {query}",
                "Clear definition of the concept"
            ),
            ReasoningStep(
                2,
                "Identify key components",
                f"What are the key components of the concept in: {query}",
                "Key components and elements"
            ),
            ReasoningStep(
                3,
                "Provide examples",
                f"What are examples of the concept in: {query}",
                "Relevant examples and applications"
            ),
            ReasoningStep(
                4,
                "Explain relationships",
                f"How does this concept relate to other concepts in: {query}",
                "Relationships with other concepts"
            ),
            ReasoningStep(
                5,
                "Discuss implications",
                f"What are the implications of the concept in: {query}",
                "Implications and significance"
            )
        ]
    
    def create_follow_up_questions(self, original_query: str, 
                                 initial_results: List[Dict[str, Any]]) -> List[str]:
        """Generate follow-up questions based on initial results"""
        follow_ups = []
        
        # Check for gaps in information
        if len(initial_results) < 3:
            follow_ups.append(f"Can you provide more detailed information about {original_query}?")
        
        # Generate specific follow-ups based on query type
        query_type = self.classify_query(original_query)
        
        if query_type == QueryType.COMPARATIVE:
            follow_ups.append("What are the practical implications of these differences?")
            follow_ups.append("Which option would be better for specific use cases?")
        
        elif query_type == QueryType.CAUSAL:
            follow_ups.append("What are the underlying mechanisms behind this relationship?")
            follow_ups.append("Are there any confounding factors to consider?")
        
        elif query_type == QueryType.ANALYTICAL:
            follow_ups.append("What are the limitations of this analysis?")
            follow_ups.append("How reliable are these findings?")
        
        return follow_ups[:3]  # Return top 3 follow-up questions
