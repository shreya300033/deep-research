"""
Multi-step reasoning engine for query decomposition and analysis.
Breaks down complex queries into smaller, manageable tasks.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of reasoning tasks."""
    SEARCH = "search"
    ANALYZE = "analyze"
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    SYNTHESIZE = "synthesize"
    CLARIFY = "clarify"


@dataclass
class ReasoningTask:
    """Represents a single reasoning task."""
    task_id: str
    task_type: TaskType
    description: str
    query: str
    context: Optional[Dict[str, Any]] = None
    dependencies: List[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None
    confidence: float = 0.0
    created_at: str = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class ReasoningEngine:
    """Multi-step reasoning engine for complex query analysis."""
    
    def __init__(self):
        """Initialize the reasoning engine."""
        self.task_templates = self._load_task_templates()
        self.query_patterns = self._load_query_patterns()
    
    def _load_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load templates for different types of reasoning tasks."""
        return {
            "search": {
                "description": "Search for information about {topic}",
                "query_template": "Find information about {topic} including key facts, definitions, and examples",
                "keywords": ["find", "search", "locate", "discover", "what is", "tell me about"]
            },
            "analyze": {
                "description": "Analyze {topic} in detail",
                "query_template": "Analyze {topic} including structure, components, relationships, and implications",
                "keywords": ["analyze", "examine", "study", "investigate", "break down", "explain how"]
            },
            "compare": {
                "description": "Compare {items}",
                "query_template": "Compare {items} highlighting similarities, differences, advantages, and disadvantages",
                "keywords": ["compare", "contrast", "versus", "vs", "difference between", "similarities"]
            },
            "summarize": {
                "description": "Summarize information about {topic}",
                "query_template": "Provide a comprehensive summary of {topic} including key points and main conclusions",
                "keywords": ["summarize", "overview", "brief", "summary", "key points", "main points"]
            },
            "synthesize": {
                "description": "Synthesize information from multiple sources about {topic}",
                "query_template": "Synthesize information about {topic} from multiple perspectives and sources",
                "keywords": ["synthesize", "combine", "integrate", "merge", "unify", "consolidate"]
            },
            "clarify": {
                "description": "Clarify aspects of {topic}",
                "query_template": "Clarify {topic} by providing detailed explanations, examples, and context",
                "keywords": ["clarify", "explain", "define", "elaborate", "detail", "expand on"]
            }
        }
    
    def _load_query_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying query types."""
        return {
            "multi_part": [
                r"(.+?)\s+(?:and|&)\s+(.+)",
                r"(.+?)\s+(?:or|/)\s+(.+)",
                r"(.+?)\s+(?:versus|vs|compared to)\s+(.+)",
                r"(.+?)\s+(?:including|such as|like)\s+(.+)"
            ],
            "conditional": [
                r"if\s+(.+?)\s+then\s+(.+)",
                r"when\s+(.+?)\s+(.+)",
                r"given\s+(.+?)\s+(.+)"
            ],
            "hierarchical": [
                r"(.+?)\s+(?:overview|summary|introduction)\s+(.+)",
                r"(.+?)\s+(?:details|specifics|examples)\s+(.+)",
                r"(.+?)\s+(?:pros and cons|advantages and disadvantages)\s+(.+)"
            ]
        }
    
    def decompose_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[ReasoningTask]:
        """
        Decompose a complex query into smaller reasoning tasks.
        
        Args:
            query: The original query
            context: Optional context information
            
        Returns:
            List of reasoning tasks
        """
        logger.info(f"Decomposing query: {query}")
        
        # Clean and normalize query
        normalized_query = self._normalize_query(query)
        
        # Identify query type and complexity
        query_type = self._identify_query_type(normalized_query)
        complexity = self._assess_complexity(normalized_query)
        
        # Generate tasks based on complexity
        if complexity == "simple":
            tasks = self._create_simple_tasks(normalized_query, query_type, context)
        elif complexity == "moderate":
            tasks = self._create_moderate_tasks(normalized_query, query_type, context)
        else:  # complex
            tasks = self._create_complex_tasks(normalized_query, query_type, context)
        
        # Add dependencies between tasks
        tasks = self._add_task_dependencies(tasks)
        
        logger.info(f"Created {len(tasks)} reasoning tasks")
        return tasks
    
    def _normalize_query(self, query: str) -> str:
        """Normalize and clean the query."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase for pattern matching
        normalized = query.lower()
        
        return normalized
    
    def _identify_query_type(self, query: str) -> str:
        """Identify the primary type of the query."""
        query_lower = query.lower()
        
        # Check for multi-part queries
        for pattern in self.query_patterns["multi_part"]:
            if re.search(pattern, query_lower):
                return "multi_part"
        
        # Check for conditional queries
        for pattern in self.query_patterns["conditional"]:
            if re.search(pattern, query_lower):
                return "conditional"
        
        # Check for hierarchical queries
        for pattern in self.query_patterns["hierarchical"]:
            if re.search(pattern, query_lower):
                return "hierarchical"
        
        # Check task templates
        for task_type, template in self.task_templates.items():
            for keyword in template["keywords"]:
                if keyword in query_lower:
                    return task_type
        
        return "search"  # Default to search
    
    def _assess_complexity(self, query: str) -> str:
        """Assess the complexity of the query."""
        # Count question words and conjunctions
        question_words = len(re.findall(r'\b(what|how|why|when|where|which|who)\b', query))
        conjunctions = len(re.findall(r'\b(and|or|but|however|although|because|since|if|when)\b', query))
        clauses = len(re.findall(r'[,;]', query))
        
        complexity_score = question_words + conjunctions + clauses
        
        if complexity_score <= 2:
            return "simple"
        elif complexity_score <= 5:
            return "moderate"
        else:
            return "complex"
    
    def _create_simple_tasks(self, query: str, query_type: str, context: Optional[Dict[str, Any]]) -> List[ReasoningTask]:
        """Create tasks for simple queries."""
        tasks = []
        
        # Extract main topic
        topic = self._extract_topic(query)
        
        # Create primary task
        task_type = TaskType(query_type) if query_type in [t.value for t in TaskType] else TaskType.SEARCH
        primary_task = ReasoningTask(
            task_id=f"task_1",
            task_type=task_type,
            description=f"Find information about {topic}",
            query=query,
            context=context
        )
        tasks.append(primary_task)
        
        return tasks
    
    def _create_moderate_tasks(self, query: str, query_type: str, context: Optional[Dict[str, Any]]) -> List[ReasoningTask]:
        """Create tasks for moderate complexity queries."""
        tasks = []
        
        if query_type == "multi_part":
            # Split into multiple parts
            parts = self._split_multi_part_query(query)
            for i, part in enumerate(parts):
                topic = self._extract_topic(part)
                task = ReasoningTask(
                    task_id=f"task_{i+1}",
                    task_type=TaskType.SEARCH,
                    description=f"Search for information about {topic}",
                    query=part.strip(),
                    context=context
                )
                tasks.append(task)
            
            # Add synthesis task
            synthesis_task = ReasoningTask(
                task_id=f"task_{len(parts)+1}",
                task_type=TaskType.SYNTHESIZE,
                description="Synthesize information from multiple parts",
                query=f"Synthesize findings about: {', '.join(parts)}",
                context=context,
                dependencies=[f"task_{i+1}" for i in range(len(parts))]
            )
            tasks.append(synthesis_task)
        
        else:
            # Create search and analysis tasks
            topic = self._extract_topic(query)
            
            search_task = ReasoningTask(
                task_id="task_1",
                task_type=TaskType.SEARCH,
                description=f"Search for information about {topic}",
                query=f"Find comprehensive information about {topic}",
                context=context
            )
            tasks.append(search_task)
            
            analyze_task = ReasoningTask(
                task_id="task_2",
                task_type=TaskType.ANALYZE,
                description=f"Analyze information about {topic}",
                query=f"Analyze and explain {topic} in detail",
                context=context,
                dependencies=["task_1"]
            )
            tasks.append(analyze_task)
        
        return tasks
    
    def _create_complex_tasks(self, query: str, query_type: str, context: Optional[Dict[str, Any]]) -> List[ReasoningTask]:
        """Create tasks for complex queries."""
        tasks = []
        
        # Extract main topics and subtopics
        topics = self._extract_topics(query)
        
        # Create search tasks for each topic
        for i, topic in enumerate(topics):
            search_task = ReasoningTask(
                task_id=f"search_{i+1}",
                task_type=TaskType.SEARCH,
                description=f"Search for information about {topic}",
                query=f"Find detailed information about {topic}",
                context=context
            )
            tasks.append(search_task)
        
        # Create analysis tasks
        for i, topic in enumerate(topics):
            analyze_task = ReasoningTask(
                task_id=f"analyze_{i+1}",
                task_type=TaskType.ANALYZE,
                description=f"Analyze information about {topic}",
                query=f"Analyze {topic} including key aspects and implications",
                context=context,
                dependencies=[f"search_{i+1}"]
            )
            tasks.append(analyze_task)
        
        # Create comparison task if multiple topics
        if len(topics) > 1:
            compare_task = ReasoningTask(
                task_id="compare",
                task_type=TaskType.COMPARE,
                description=f"Compare {', '.join(topics)}",
                query=f"Compare and contrast {', '.join(topics)}",
                context=context,
                dependencies=[f"analyze_{i+1}" for i in range(len(topics))]
            )
            tasks.append(compare_task)
        
        # Create synthesis task
        synthesis_task = ReasoningTask(
            task_id="synthesize",
            task_type=TaskType.SYNTHESIZE,
            description="Synthesize all findings",
            query=f"Synthesize comprehensive information about {', '.join(topics)}",
            context=context,
            dependencies=[f"analyze_{i+1}" for i in range(len(topics))] + 
                        (["compare"] if len(topics) > 1 else [])
        )
        tasks.append(synthesis_task)
        
        return tasks
    
    def _extract_topic(self, query: str) -> str:
        """Extract the main topic from a query."""
        # Remove common question words
        topic = re.sub(r'\b(what|how|why|when|where|which|who|is|are|can|could|would|should)\b', '', query)
        
        # Remove common verbs
        topic = re.sub(r'\b(find|search|locate|discover|explain|analyze|compare|summarize)\b', '', topic)
        
        # Clean up
        topic = re.sub(r'\s+', ' ', topic.strip())
        
        return topic if topic else query
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract multiple topics from a complex query."""
        topics = []
        
        # Try to split by conjunctions
        parts = re.split(r'\b(and|or|&|versus|vs|compared to)\b', query)
        
        for part in parts:
            part = part.strip()
            if part and part.lower() not in ['and', 'or', '&', 'versus', 'vs', 'compared to']:
                topic = self._extract_topic(part)
                if topic:
                    topics.append(topic)
        
        # If no clear split, return the main topic
        if not topics:
            topics.append(self._extract_topic(query))
        
        return topics
    
    def _split_multi_part_query(self, query: str) -> List[str]:
        """Split a multi-part query into individual parts."""
        parts = []
        
        # Try different splitting patterns
        for pattern in self.query_patterns["multi_part"]:
            match = re.search(pattern, query)
            if match:
                parts.extend([match.group(1).strip(), match.group(2).strip()])
                break
        
        # If no pattern matched, try splitting by commas
        if not parts:
            parts = [part.strip() for part in query.split(',') if part.strip()]
        
        return parts if parts else [query]
    
    def _add_task_dependencies(self, tasks: List[ReasoningTask]) -> List[ReasoningTask]:
        """Add dependencies between tasks based on their types."""
        # This is already handled in task creation, but could be enhanced
        return tasks
    
    def execute_task(self, task: ReasoningTask, vector_store, search_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Execute a single reasoning task.
        
        Args:
            task: The reasoning task to execute
            vector_store: Vector store for searching
            search_results: Optional pre-computed search results
            
        Returns:
            Task execution result
        """
        logger.info(f"Executing task {task.task_id}: {task.description}")
        
        try:
            task.status = "in_progress"
            
            if task.task_type == TaskType.SEARCH:
                result = self._execute_search_task(task, vector_store)
            elif task.task_type == TaskType.ANALYZE:
                result = self._execute_analyze_task(task, vector_store, search_results)
            elif task.task_type == TaskType.COMPARE:
                result = self._execute_compare_task(task, vector_store, search_results)
            elif task.task_type == TaskType.SUMMARIZE:
                result = self._execute_summarize_task(task, vector_store, search_results)
            elif task.task_type == TaskType.SYNTHESIZE:
                result = self._execute_synthesize_task(task, vector_store, search_results)
            elif task.task_type == TaskType.CLARIFY:
                result = self._execute_clarify_task(task, vector_store, search_results)
            else:
                result = {"error": f"Unknown task type: {task.task_type}"}
            
            task.result = result
            task.status = "completed"
            task.confidence = result.get("confidence", 0.8)
            
            logger.info(f"Completed task {task.task_id} with confidence {task.confidence}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            task.status = "failed"
            task.result = {"error": str(e)}
            return task.result
    
    def _execute_search_task(self, task: ReasoningTask, vector_store) -> Dict[str, Any]:
        """Execute a search task."""
        results = vector_store.search(task.query, top_k=10)
        
        return {
            "type": "search",
            "query": task.query,
            "results": results,
            "result_count": len(results),
            "confidence": 0.9 if results else 0.1
        }
    
    def _execute_analyze_task(self, task: ReasoningTask, vector_store, search_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Execute an analysis task."""
        if not search_results:
            search_results = vector_store.search(task.query, top_k=15)
        
        # Analyze the content
        analysis = {
            "type": "analysis",
            "query": task.query,
            "key_points": [],
            "themes": [],
            "relationships": [],
            "implications": []
        }
        
        # Extract key points from results
        for result in search_results[:10]:
            content = result["content"]
            # Simple key point extraction (could be enhanced with NLP)
            sentences = content.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    analysis["key_points"].append(sentence.strip())
        
        analysis["confidence"] = 0.8 if analysis["key_points"] else 0.3
        return analysis
    
    def _execute_compare_task(self, task: ReasoningTask, vector_store, search_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Execute a comparison task."""
        if not search_results:
            search_results = vector_store.search(task.query, top_k=15)
        
        comparison = {
            "type": "comparison",
            "query": task.query,
            "similarities": [],
            "differences": [],
            "advantages": [],
            "disadvantages": []
        }
        
        # Simple comparison logic (could be enhanced)
        for result in search_results[:10]:
            content = result["content"].lower()
            if "similar" in content or "same" in content:
                comparison["similarities"].append(result["content"])
            elif "different" in content or "versus" in content:
                comparison["differences"].append(result["content"])
        
        comparison["confidence"] = 0.7 if comparison["similarities"] or comparison["differences"] else 0.3
        return comparison
    
    def _execute_summarize_task(self, task: ReasoningTask, vector_store, search_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Execute a summarization task."""
        if not search_results:
            search_results = vector_store.search(task.query, top_k=20)
        
        # Combine content for summarization
        combined_content = " ".join([result["content"] for result in search_results[:15]])
        
        summary = {
            "type": "summary",
            "query": task.query,
            "summary": combined_content[:1000] + "..." if len(combined_content) > 1000 else combined_content,
            "source_count": len(search_results),
            "confidence": 0.8 if combined_content else 0.2
        }
        
        return summary
    
    def _execute_synthesize_task(self, task: ReasoningTask, vector_store, search_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Execute a synthesis task."""
        if not search_results:
            search_results = vector_store.search(task.query, top_k=25)
        
        synthesis = {
            "type": "synthesis",
            "query": task.query,
            "synthesis": "Synthesized information from multiple sources...",
            "source_count": len(search_results),
            "key_insights": [],
            "conclusions": []
        }
        
        # Extract insights from results
        for result in search_results[:20]:
            if result["similarity"] > 0.7:
                synthesis["key_insights"].append(result["content"][:200])
        
        synthesis["confidence"] = 0.8 if synthesis["key_insights"] else 0.4
        return synthesis
    
    def _execute_clarify_task(self, task: ReasoningTask, vector_store, search_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Execute a clarification task."""
        if not search_results:
            search_results = vector_store.search(task.query, top_k=10)
        
        clarification = {
            "type": "clarification",
            "query": task.query,
            "explanations": [],
            "examples": [],
            "definitions": []
        }
        
        for result in search_results:
            content = result["content"]
            clarification["explanations"].append(content)
        
        clarification["confidence"] = 0.8 if clarification["explanations"] else 0.3
        return clarification

