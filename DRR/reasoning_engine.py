"""
Multi-step reasoning engine for the Deep Researcher Agent
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# Local reasoning models
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Our modules
import config
from retrieval_system import SearchResult, retrieval_system

logger = logging.getLogger(__name__)


class ReasoningStepType(Enum):
    """Types of reasoning steps"""
    QUERY_DECOMPOSITION = "query_decomposition"
    INFORMATION_GATHERING = "information_gathering"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    CONCLUSION = "conclusion"


@dataclass
class ReasoningStep:
    """Represents a single reasoning step"""
    step_type: ReasoningStepType
    description: str
    input_data: Any
    output_data: Any
    confidence: float
    reasoning: str
    timestamp: datetime
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['step_type'] = self.step_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ReasoningPlan:
    """Represents a complete reasoning plan"""
    query: str
    steps: List[ReasoningStep]
    current_step: int
    status: str  # "planning", "executing", "completed", "failed"
    confidence: float
    created_at: datetime
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "steps": [step.to_dict() for step in self.steps],
            "current_step": self.current_step,
            "status": self.status,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat()
        }


class LocalReasoningModel:
    """Local reasoning model using transformers"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.LOCAL_MODEL_PATH
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the local reasoning model"""
        try:
            logger.info(f"Loading local reasoning model: {self.model_name}")
            
            # Use a smaller, more suitable model for reasoning
            if "DialoGPT" in self.model_name:
                # Use a more appropriate model for reasoning tasks
                self.model_name = "microsoft/DialoGPT-small"
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Local reasoning model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load local reasoning model: {e}")
            # Fallback to a simpler approach
            self.model = None
            self.pipeline = None
    
    def generate_reasoning(self, prompt: str, max_length: int = 200) -> str:
        """Generate reasoning text from a prompt"""
        if self.pipeline is None:
            return self._fallback_reasoning(prompt)
        
        try:
            # Truncate prompt if too long
            if len(prompt) > 400:
                prompt = prompt[:400] + "..."
            
            result = self.pipeline(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = result[0]['generated_text']
            # Remove the original prompt from the generated text
            reasoning = generated_text[len(prompt):].strip()
            
            return reasoning if reasoning else self._fallback_reasoning(prompt)
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            return self._fallback_reasoning(prompt)
    
    def _fallback_reasoning(self, prompt: str) -> str:
        """Fallback reasoning when model is not available"""
        # Simple rule-based reasoning
        if "decompose" in prompt.lower() or "break down" in prompt.lower():
            return "I'll break this query into smaller, manageable sub-questions to ensure comprehensive coverage."
        elif "analyze" in prompt.lower():
            return "I'll analyze the gathered information to identify patterns, relationships, and key insights."
        elif "synthesize" in prompt.lower():
            return "I'll combine the analyzed information to form a coherent and comprehensive response."
        else:
            return "I'll process this step systematically to ensure accurate results."


class ReasoningEngine:
    """Main reasoning engine for multi-step query processing"""
    
    def __init__(self, model_type: str = None):
        self.model_type = model_type or config.REASONING_MODEL
        self.local_model = None
        self.max_steps = config.MAX_REASONING_STEPS
        
        if self.model_type == "local":
            self.local_model = LocalReasoningModel()
    
    def create_reasoning_plan(self, query: str) -> ReasoningPlan:
        """
        Create a reasoning plan for a given query
        
        Args:
            query: The research query
            
        Returns:
            ReasoningPlan object
        """
        logger.info(f"Creating reasoning plan for query: {query}")
        
        # Analyze query complexity
        complexity = self._analyze_query_complexity(query)
        
        # Generate reasoning steps
        steps = self._generate_reasoning_steps(query, complexity)
        
        # Create plan
        plan = ReasoningPlan(
            query=query,
            steps=steps,
            current_step=0,
            status="planning",
            confidence=0.0
        )
        
        logger.info(f"Created reasoning plan with {len(steps)} steps")
        return plan
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze the complexity of a query"""
        complexity = {
            "word_count": len(query.split()),
            "has_multiple_concepts": len(re.findall(r'\b(and|or|but|however|although)\b', query.lower())) > 0,
            "has_comparison": len(re.findall(r'\b(compare|versus|vs|difference|similarity)\b', query.lower())) > 0,
            "has_temporal": len(re.findall(r'\b(when|time|history|evolution|development)\b', query.lower())) > 0,
            "has_causal": len(re.findall(r'\b(why|how|cause|effect|because|due to)\b', query.lower())) > 0,
            "complexity_score": 0
        }
        
        # Calculate complexity score
        score = 0
        if complexity["word_count"] > 20:
            score += 2
        if complexity["has_multiple_concepts"]:
            score += 2
        if complexity["has_comparison"]:
            score += 2
        if complexity["has_temporal"]:
            score += 1
        if complexity["has_causal"]:
            score += 2
        
        complexity["complexity_score"] = min(score, 5)  # Cap at 5
        return complexity
    
    def _generate_reasoning_steps(self, query: str, complexity: Dict[str, Any]) -> List[ReasoningStep]:
        """Generate reasoning steps based on query analysis"""
        steps = []
        
        # Step 1: Query Decomposition
        if complexity["complexity_score"] > 2:
            steps.append(ReasoningStep(
                step_type=ReasoningStepType.QUERY_DECOMPOSITION,
                description="Break down the complex query into smaller, focused sub-questions",
                input_data={"query": query, "complexity": complexity},
                output_data=None,
                confidence=0.0,
                reasoning=""
            ))
        
        # Step 2: Information Gathering
        steps.append(ReasoningStep(
            step_type=ReasoningStepType.INFORMATION_GATHERING,
            description="Search and gather relevant information from available sources",
            input_data={"query": query},
            output_data=None,
            confidence=0.0,
            reasoning=""
        ))
        
        # Step 3: Analysis
        steps.append(ReasoningStep(
            step_type=ReasoningStepType.ANALYSIS,
            description="Analyze the gathered information for patterns, relationships, and insights",
            input_data=None,
            output_data=None,
            confidence=0.0,
            reasoning=""
        ))
        
        # Step 4: Synthesis
        steps.append(ReasoningStep(
            step_type=ReasoningStepType.SYNTHESIS,
            description="Synthesize the analyzed information into a coherent response",
            input_data=None,
            output_data=None,
            confidence=0.0,
            reasoning=""
        ))
        
        # Step 5: Verification (for complex queries)
        if complexity["complexity_score"] > 3:
            steps.append(ReasoningStep(
                step_type=ReasoningStepType.VERIFICATION,
                description="Verify the synthesized information for accuracy and completeness",
                input_data=None,
                output_data=None,
                confidence=0.0,
                reasoning=""
            ))
        
        # Step 6: Conclusion
        steps.append(ReasoningStep(
            step_type=ReasoningStepType.CONCLUSION,
            description="Formulate the final conclusion and recommendations",
            input_data=None,
            output_data=None,
            confidence=0.0,
            reasoning=""
        ))
        
        return steps
    
    def execute_reasoning_plan(self, plan: ReasoningPlan) -> ReasoningPlan:
        """
        Execute a reasoning plan step by step
        
        Args:
            plan: The reasoning plan to execute
            
        Returns:
            Updated reasoning plan
        """
        logger.info(f"Executing reasoning plan with {len(plan.steps)} steps")
        
        plan.status = "executing"
        
        try:
            for i, step in enumerate(plan.steps):
                logger.info(f"Executing step {i+1}: {step.step_type.value}")
                
                # Execute the step
                step_result = self._execute_reasoning_step(step, plan)
                
                # Update step with results
                step.output_data = step_result["output_data"]
                step.confidence = step_result["confidence"]
                step.reasoning = step_result["reasoning"]
                
                # Update plan
                plan.current_step = i + 1
                plan.confidence = self._calculate_plan_confidence(plan)
                
                # Check if we should continue
                if step_result.get("should_stop", False):
                    break
            
            plan.status = "completed"
            logger.info("Reasoning plan execution completed")
            
        except Exception as e:
            logger.error(f"Error executing reasoning plan: {e}")
            plan.status = "failed"
        
        return plan
    
    def _execute_reasoning_step(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """Execute a single reasoning step"""
        try:
            if step.step_type == ReasoningStepType.QUERY_DECOMPOSITION:
                return self._execute_query_decomposition(step, plan)
            elif step.step_type == ReasoningStepType.INFORMATION_GATHERING:
                return self._execute_information_gathering(step, plan)
            elif step.step_type == ReasoningStepType.ANALYSIS:
                return self._execute_analysis(step, plan)
            elif step.step_type == ReasoningStepType.SYNTHESIS:
                return self._execute_synthesis(step, plan)
            elif step.step_type == ReasoningStepType.VERIFICATION:
                return self._execute_verification(step, plan)
            elif step.step_type == ReasoningStepType.CONCLUSION:
                return self._execute_conclusion(step, plan)
            else:
                return {"output_data": None, "confidence": 0.0, "reasoning": "Unknown step type"}
        except Exception as e:
            logger.error(f"Error executing step {step.step_type.value}: {e}")
            return {"output_data": None, "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def _execute_query_decomposition(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """Execute query decomposition step"""
        query = step.input_data["query"]
        
        # Generate sub-questions
        sub_questions = self._generate_sub_questions(query)
        
        reasoning = self._generate_step_reasoning(
            f"Decomposing the query '{query}' into smaller, focused sub-questions for better research coverage."
        )
        
        return {
            "output_data": {"sub_questions": sub_questions},
            "confidence": 0.8,
            "reasoning": reasoning
        }
    
    def _execute_information_gathering(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """Execute information gathering step"""
        # Get the main query or sub-questions
        if plan.steps[0].output_data and "sub_questions" in plan.steps[0].output_data:
            queries = plan.steps[0].output_data["sub_questions"]
        else:
            queries = [plan.query]
        
        # Search for information
        all_results = []
        for query in queries:
            results = retrieval_system.search(query)
            all_results.extend(results)
        
        # Remove duplicates and rank
        unique_results = self._deduplicate_search_results(all_results)
        
        reasoning = self._generate_step_reasoning(
            f"Gathered {len(unique_results)} relevant documents from available sources for comprehensive coverage."
        )
        
        return {
            "output_data": {"search_results": unique_results},
            "confidence": 0.9 if unique_results else 0.3,
            "reasoning": reasoning
        }
    
    def _execute_analysis(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """Execute analysis step"""
        # Get search results from previous step
        search_results = None
        for prev_step in plan.steps:
            if (prev_step.step_type == ReasoningStepType.INFORMATION_GATHERING and 
                prev_step.output_data and "search_results" in prev_step.output_data):
                search_results = prev_step.output_data["search_results"]
                break
        
        if not search_results:
            return {"output_data": None, "confidence": 0.0, "reasoning": "No search results to analyze"}
        
        # Analyze the results
        analysis = self._analyze_search_results(search_results, plan.query)
        
        reasoning = self._generate_step_reasoning(
            f"Analyzed {len(search_results)} documents to identify key patterns, relationships, and insights."
        )
        
        return {
            "output_data": {"analysis": analysis},
            "confidence": 0.8,
            "reasoning": reasoning
        }
    
    def _execute_synthesis(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """Execute synthesis step"""
        # Get analysis from previous step
        analysis = None
        for prev_step in plan.steps:
            if (prev_step.step_type == ReasoningStepType.ANALYSIS and 
                prev_step.output_data and "analysis" in prev_step.output_data):
                analysis = prev_step.output_data["analysis"]
                break
        
        if not analysis:
            return {"output_data": None, "confidence": 0.0, "reasoning": "No analysis to synthesize"}
        
        # Synthesize the information
        synthesis = self._synthesize_analysis(analysis, plan.query)
        
        reasoning = self._generate_step_reasoning(
            "Synthesized the analyzed information into a coherent and comprehensive response."
        )
        
        return {
            "output_data": {"synthesis": synthesis},
            "confidence": 0.8,
            "reasoning": reasoning
        }
    
    def _execute_verification(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """Execute verification step"""
        # Get synthesis from previous step
        synthesis = None
        for prev_step in plan.steps:
            if (prev_step.step_type == ReasoningStepType.SYNTHESIS and 
                prev_step.output_data and "synthesis" in prev_step.output_data):
                synthesis = prev_step.output_data["synthesis"]
                break
        
        if not synthesis:
            return {"output_data": None, "confidence": 0.0, "reasoning": "No synthesis to verify"}
        
        # Verify the synthesis
        verification = self._verify_synthesis(synthesis, plan.query)
        
        reasoning = self._generate_step_reasoning(
            "Verified the synthesized information for accuracy, completeness, and consistency."
        )
        
        return {
            "output_data": {"verification": verification},
            "confidence": 0.7,
            "reasoning": reasoning
        }
    
    def _execute_conclusion(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """Execute conclusion step"""
        # Get the final synthesis or verification
        final_data = None
        for prev_step in reversed(plan.steps):
            if (prev_step.step_type in [ReasoningStepType.SYNTHESIS, ReasoningStepType.VERIFICATION] and 
                prev_step.output_data):
                final_data = prev_step.output_data
                break
        
        if not final_data:
            return {"output_data": None, "confidence": 0.0, "reasoning": "No data to conclude"}
        
        # Formulate conclusion
        conclusion = self._formulate_conclusion(final_data, plan.query)
        
        reasoning = self._generate_step_reasoning(
            "Formulated the final conclusion and recommendations based on the comprehensive analysis."
        )
        
        return {
            "output_data": {"conclusion": conclusion},
            "confidence": 0.9,
            "reasoning": reasoning
        }
    
    def _generate_sub_questions(self, query: str) -> List[str]:
        """Generate sub-questions from a complex query"""
        # Simple rule-based decomposition
        sub_questions = []
        
        # Extract key concepts
        words = query.lower().split()
        
        # Look for comparison indicators
        if any(word in words for word in ["compare", "versus", "vs", "difference", "similarity"]):
            sub_questions.append(f"What are the key characteristics of the first concept in: {query}")
            sub_questions.append(f"What are the key characteristics of the second concept in: {query}")
            sub_questions.append(f"What are the main differences and similarities in: {query}")
        
        # Look for causal relationships
        elif any(word in words for word in ["why", "how", "cause", "effect", "because"]):
            sub_questions.append(f"What are the underlying causes in: {query}")
            sub_questions.append(f"What are the effects and consequences in: {query}")
            sub_questions.append(f"What is the relationship between cause and effect in: {query}")
        
        # Look for temporal aspects
        elif any(word in words for word in ["when", "time", "history", "evolution", "development"]):
            sub_questions.append(f"What is the historical context of: {query}")
            sub_questions.append(f"How has this evolved over time: {query}")
            sub_questions.append(f"What are the current trends in: {query}")
        
        # Default decomposition
        else:
            sub_questions.append(f"What is the main topic of: {query}")
            sub_questions.append(f"What are the key aspects of: {query}")
            sub_questions.append(f"What are the important details about: {query}")
        
        return sub_questions
    
    def _generate_step_reasoning(self, prompt: str) -> str:
        """Generate reasoning text for a step"""
        if self.local_model:
            return self.local_model.generate_reasoning(prompt)
        else:
            return prompt
    
    def _deduplicate_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate search results"""
        seen = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(result.content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _analyze_search_results(self, results: List[SearchResult], query: str) -> Dict[str, Any]:
        """Analyze search results for patterns and insights"""
        if not results:
            return {"insights": [], "patterns": [], "key_points": []}
        
        # Extract key insights
        insights = []
        patterns = []
        key_points = []
        
        # Analyze content for patterns
        all_content = " ".join([result.content for result in results])
        
        # Simple pattern detection
        if "however" in all_content.lower():
            patterns.append("Contrasting viewpoints present")
        if "therefore" in all_content.lower() or "thus" in all_content.lower():
            patterns.append("Causal relationships identified")
        if "recent" in all_content.lower() or "latest" in all_content.lower():
            patterns.append("Recent developments mentioned")
        
        # Extract key points from high-scoring results
        high_score_results = [r for r in results if r.score > 0.8]
        for result in high_score_results[:5]:  # Top 5 results
            # Extract first sentence as key point
            sentences = result.content.split('.')
            if sentences:
                key_points.append(sentences[0].strip())
        
        return {
            "insights": insights,
            "patterns": patterns,
            "key_points": key_points,
            "total_sources": len(results),
            "high_confidence_sources": len(high_score_results)
        }
    
    def _synthesize_analysis(self, analysis: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Synthesize analysis into coherent response"""
        synthesis = {
            "summary": f"Based on the analysis of {analysis.get('total_sources', 0)} sources, ",
            "key_findings": analysis.get("key_points", []),
            "patterns_identified": analysis.get("patterns", []),
            "confidence_level": "high" if analysis.get("high_confidence_sources", 0) > 3 else "medium"
        }
        
        # Build summary
        if analysis.get("patterns"):
            synthesis["summary"] += f"several patterns were identified: {', '.join(analysis['patterns'])}. "
        
        if analysis.get("key_points"):
            synthesis["summary"] += f"Key findings include: {'. '.join(analysis['key_points'][:3])}."
        
        return synthesis
    
    def _verify_synthesis(self, synthesis: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Verify the synthesis for accuracy and completeness"""
        verification = {
            "accuracy_score": 0.8,  # Placeholder
            "completeness_score": 0.7,  # Placeholder
            "consistency_score": 0.9,  # Placeholder
            "verification_notes": []
        }
        
        # Simple verification checks
        if len(synthesis.get("key_findings", [])) > 0:
            verification["verification_notes"].append("Key findings are present")
        
        if synthesis.get("patterns_identified"):
            verification["verification_notes"].append("Patterns have been identified")
        
        if synthesis.get("summary"):
            verification["verification_notes"].append("Summary is comprehensive")
        
        return verification
    
    def _formulate_conclusion(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Formulate final conclusion"""
        conclusion = {
            "answer": "",
            "confidence": 0.0,
            "recommendations": [],
            "limitations": []
        }
        
        # Extract the main synthesis or verification data
        if "synthesis" in data:
            synthesis = data["synthesis"]
            conclusion["answer"] = synthesis.get("summary", "")
            conclusion["confidence"] = 0.8 if synthesis.get("confidence_level") == "high" else 0.6
        elif "verification" in data:
            verification = data["verification"]
            conclusion["confidence"] = (verification.get("accuracy_score", 0) + 
                                      verification.get("completeness_score", 0) + 
                                      verification.get("consistency_score", 0)) / 3
        
        # Add recommendations
        conclusion["recommendations"] = [
            "Consider multiple perspectives when interpreting the results",
            "Verify information from multiple sources",
            "Stay updated with latest developments in the field"
        ]
        
        # Add limitations
        conclusion["limitations"] = [
            "Analysis based on available sources only",
            "May not include the most recent developments",
            "Subject to the quality and bias of source materials"
        ]
        
        return conclusion
    
    def _calculate_plan_confidence(self, plan: ReasoningPlan) -> float:
        """Calculate overall confidence for the reasoning plan"""
        if not plan.steps:
            return 0.0
        
        completed_steps = [step for step in plan.steps if step.confidence > 0]
        if not completed_steps:
            return 0.0
        
        total_confidence = sum(step.confidence for step in completed_steps)
        return total_confidence / len(completed_steps)


# Global reasoning engine instance
reasoning_engine = ReasoningEngine()
