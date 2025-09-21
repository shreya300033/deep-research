#!/usr/bin/env python3
"""
Test script to verify the confidence and reasoning improvements
"""
import sys
import os

def test_improved_system():
    """Test the improved system with better confidence scoring"""
    print("🔧 Testing Improved Deep Researcher Agent")
    print("=" * 50)
    
    try:
        from researcher_agent import DeepResearcherAgent
        from utils.document_processor import DocumentProcessor
        
        # Initialize agent
        print("🤖 Initializing agent...")
        agent = DeepResearcherAgent()
        session_id = agent.start_research_session("improvement_test")
        print(f"✅ Session started: {session_id}")
        
        # Load sample documents
        print("📚 Loading sample documents...")
        sample_docs = DocumentProcessor.create_sample_documents()
        result = agent.index_documents(sample_docs, "improvement_test")
        print(f"✅ Indexed {result['documents_indexed']} documents")
        
        # Test queries with different complexity
        test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the ethical implications of AI?",
            "Compare supervised and unsupervised learning"
        ]
        
        print("\n🔍 Testing improved reasoning...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}: {query} ---")
            
            result = agent.research_query(query, max_reasoning_steps=3)
            
            print(f"📊 Query Type: {result.get('query_type', 'unknown')}")
            print(f"🎯 Reasoning Quality: {result.get('reasoning_quality', 'unknown')}")
            print(f"📖 Total Sources: {result.get('total_sources', 0)}")
            
            # Show confidence scores
            steps = result.get('reasoning_steps', [])
            print(f"🧠 Reasoning Steps: {len(steps)}")
            for j, step in enumerate(steps, 1):
                confidence = step.get('confidence', 0)
                answer_length = len(step.get('answer', ''))
                print(f"  {j}. {step.get('description', 'No description')}")
                print(f"     Confidence: {confidence:.2f}")
                print(f"     Answer Length: {answer_length} chars")
            
            # Show final answer preview
            final_answer = result.get('final_answer', 'No answer')
            preview = final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
            print(f"💡 Final Answer Preview: {preview}")
        
        print(f"\n🎉 Improvement test completed!")
        print(f"📈 Check the confidence scores and reasoning quality above")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the improvement test"""
    success = test_improved_system()
    
    if success:
        print("\n✅ System improvements are working!")
        print("The confidence scoring and reasoning quality should be better now.")
    else:
        print("\n❌ System improvements need more work.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
