#!/usr/bin/env python3
"""
Quick demonstration of the Deep Researcher Agent
"""
import sys
import os

def main():
    print("🚀 Deep Researcher Agent - Quick Demo")
    print("=" * 50)
    
    try:
        # Import the main components
        print("📦 Importing components...")
        from researcher_agent import DeepResearcherAgent
        from utils.document_processor import DocumentProcessor
        print("✅ Components imported successfully!")
        
        # Initialize the agent
        print("\n🤖 Initializing Deep Researcher Agent...")
        agent = DeepResearcherAgent()
        print("✅ Agent initialized!")
        
        # Start a research session
        print("\n📝 Starting research session...")
        session_id = agent.start_research_session("Quick Demo")
        print(f"✅ Session started: {session_id}")
        
        # Create and index sample documents
        print("\n📚 Creating sample knowledge base...")
        sample_docs = DocumentProcessor.create_sample_documents()
        print(f"✅ Created {len(sample_docs)} sample documents")
        
        print("\n🔍 Indexing documents...")
        index_result = agent.index_documents(sample_docs, "demo_knowledge_base")
        print(f"✅ Documents indexed: {index_result}")
        
        # Process some research queries
        queries = [
            "What is artificial intelligence?",
            "How do machine learning algorithms work?",
            "What are the ethical implications of AI?"
        ]
        
        print("\n🔬 Processing research queries...")
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            result = agent.research_query(query)
            
            print(f"📊 Query Type: {result.get('query_type', 'unknown')}")
            print(f"🎯 Reasoning Quality: {result.get('reasoning_quality', 'unknown')}")
            print(f"📖 Sources Used: {result.get('total_sources', 0)}")
            
            # Show reasoning steps
            steps = result.get('reasoning_steps', [])
            print(f"🧠 Reasoning Steps: {len(steps)}")
            for j, step in enumerate(steps[:2], 1):  # Show first 2 steps
                print(f"  {j}. {step.get('description', 'No description')}")
                print(f"     Confidence: {step.get('confidence', 0):.2f}")
            
            # Show final answer preview
            final_answer = result.get('final_answer', 'No answer generated')
            preview = final_answer[:150] + "..." if len(final_answer) > 150 else final_answer
            print(f"💡 Answer Preview: {preview}")
        
        # Export results
        print(f"\n📄 Exporting research report...")
        report_path = agent.save_pdf_report(session_id, "quick_demo_report")
        print(f"✅ Report saved to: {report_path}")
        
        # Show session summary
        print(f"\n📈 Research Session Summary:")
        summary = agent.get_research_summary(session_id)
        print(f"  - Total Queries: {summary.get('total_queries', 0)}")
        print(f"  - Query Types: {list(summary.get('query_types', {}).keys())}")
        print(f"  - Sources Accessed: {summary.get('total_sources_accessed', 0)}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"\n📁 Check the 'reports' folder for the generated PDF report")
        print(f"🔧 Try running 'python cli.py demo --interactive' for interactive mode")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
