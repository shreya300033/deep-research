"""
Example usage of the Deep Researcher Agent
"""
from researcher_agent import DeepResearcherAgent
from utils.document_processor import DocumentProcessor
import json


def main():
    """Demonstrate the Deep Researcher Agent capabilities"""
    
    print("=== Deep Researcher Agent Demo ===\n")
    
    # Initialize the agent
    print("1. Initializing Deep Researcher Agent...")
    agent = DeepResearcherAgent()
    
    # Start a research session
    session_id = agent.start_research_session("AI_Research_Demo")
    print(f"Started research session: {session_id}\n")
    
    # Create and index sample documents
    print("2. Creating and indexing sample documents...")
    sample_docs = DocumentProcessor.create_sample_documents()
    index_result = agent.index_documents(sample_docs, "ai_knowledge_base")
    print(f"Indexing result: {index_result}\n")
    
    # Get vector store statistics
    stats = agent.get_vector_store_stats()
    print(f"Vector store stats: {stats}\n")
    
    # Example research queries
    research_queries = [
        "What is artificial intelligence and how does it work?",
        "Compare supervised and unsupervised learning approaches",
        "What are the ethical implications of AI development?",
        "How do neural networks process information?"
    ]
    
    print("3. Processing research queries...\n")
    
    for i, query in enumerate(research_queries, 1):
        print(f"--- Query {i}: {query} ---")
        
        # Process the query
        result = agent.research_query(query)
        
        # Display results
        print(f"Query Type: {result['query_type']}")
        print(f"Reasoning Quality: {result['reasoning_quality']}")
        print(f"Total Sources: {result['total_sources']}")
        
        print("\nReasoning Steps:")
        for step in result['reasoning_steps']:
            print(f"  - {step['description']}: {step['answer'][:100]}...")
            print(f"    Confidence: {step['confidence']:.2f}")
        
        print(f"\nFinal Answer: {result['final_answer'][:200]}...")
        
        if result['follow_up_questions']:
            print("\nSuggested Follow-up Questions:")
            for j, follow_up in enumerate(result['follow_up_questions'], 1):
                print(f"  {j}. {follow_up}")
        
        print("\n" + "="*80 + "\n")
    
    # Demonstrate query refinement
    print("4. Demonstrating query refinement...")
    original_query = "What is machine learning?"
    refinement = "Focus specifically on deep learning and neural networks"
    
    refined_result = agent.refine_query(original_query, refinement)
    print(f"Refined query result: {refined_result['final_answer'][:200]}...\n")
    
    # Get research summary
    print("5. Generating research summary...")
    summary = agent.get_research_summary(session_id)
    print(f"Research Summary: {json.dumps(summary, indent=2)}\n")
    
    # Export research report
    print("6. Exporting research report...")
    report_content = agent.export_research_report(session_id, "markdown")
    report_path = agent.save_report(report_content, f"research_report_{session_id}", "markdown")
    print(f"Report saved to: {report_path}\n")
    
    # Demonstrate interactive features
    print("7. Interactive query refinement demo...")
    interactive_demo(agent)
    
    print("=== Demo Complete ===")


def interactive_demo(agent):
    """Demonstrate interactive query refinement"""
    print("Interactive Query Refinement Demo")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("Enter your research query: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        print(f"\nProcessing: {query}")
        result = agent.research_query(query)
        
        print(f"\nAnswer: {result['final_answer']}")
        
        if result['follow_up_questions']:
            print("\nSuggested follow-up questions:")
            for i, follow_up in enumerate(result['follow_up_questions'], 1):
                print(f"{i}. {follow_up}")
        
        # Ask for refinement
        refinement = input("\nWould you like to refine this query? (y/n): ").strip().lower()
        if refinement == 'y':
            refinement_text = input("Enter your refinement: ").strip()
            if refinement_text:
                refined_result = agent.refine_query(query, refinement_text)
                print(f"\nRefined Answer: {refined_result['final_answer']}")
        
        print("\n" + "-"*50 + "\n")


def demonstrate_advanced_features():
    """Demonstrate advanced features of the system"""
    print("=== Advanced Features Demo ===\n")
    
    agent = DeepResearcherAgent()
    session_id = agent.start_research_session("Advanced_Demo")
    
    # Index sample documents
    sample_docs = DocumentProcessor.create_sample_documents()
    agent.index_documents(sample_docs, "advanced_demo")
    
    # Complex analytical query
    complex_query = "Analyze the relationship between machine learning algorithms and their applications in computer vision"
    print(f"Complex Query: {complex_query}")
    
    result = agent.research_query(complex_query, max_reasoning_steps=6)
    
    print(f"Query Type: {result['query_type']}")
    print(f"Number of Reasoning Steps: {len(result['reasoning_steps'])}")
    print(f"Reasoning Quality: {result['reasoning_quality']}")
    
    print("\nDetailed Reasoning Process:")
    for i, step in enumerate(result['reasoning_steps'], 1):
        print(f"\nStep {i}: {step['description']}")
        print(f"Answer: {step['answer']}")
        print(f"Confidence: {step['confidence']:.2f}")
        print(f"Sources: {', '.join(step['sources'])}")
    
    print(f"\nFinal Comprehensive Answer:\n{result['final_answer']}")
    
    # Export detailed report
    report = agent.export_research_report(session_id, "markdown")
    agent.save_report(report, "advanced_demo_report", "markdown")
    
    print("\nAdvanced demo complete!")


if __name__ == "__main__":
    # Run the main demo
    main()
    
    # Uncomment to run advanced features demo
    # demonstrate_advanced_features()
