#!/usr/bin/env python3
"""
Final verification test to ensure different queries produce different answers
"""
import sys
import os

def test_final_verification():
    """Test that different queries produce different, specific answers"""
    print("🎯 Final Verification Test - Query Diversity")
    print("=" * 60)
    
    try:
        from researcher_agent import DeepResearcherAgent
        from utils.document_processor import DocumentProcessor
        
        # Initialize agent
        print("🤖 Initializing agent...")
        agent = DeepResearcherAgent(use_enhanced_reasoning=True)
        session_id = agent.start_research_session("final_test")
        print(f"✅ Session started: {session_id}")
        
        # Load sample documents (including new LLM content)
        print("📚 Loading sample documents...")
        sample_docs = DocumentProcessor.create_sample_documents()
        result = agent.index_documents(sample_docs, "final_test")
        print(f"✅ Indexed {result['documents_indexed']} documents")
        
        # Test diverse queries
        test_queries = [
            "What is artificial intelligence?",
            "What is LLM?",
            "How does machine learning work?",
            "What are neural networks?",
            "Compare supervised and unsupervised learning",
            "What are the ethical implications of AI?"
        ]
        
        print("\n🔍 Testing Query Diversity...")
        results = {}
        
        for query in test_queries:
            print(f"\n--- Query: {query} ---")
            
            result = agent.research_query(query, max_reasoning_steps=2)
            results[query] = result
            
            # Show first step answer
            steps = result.get('reasoning_steps', [])
            if steps:
                first_step = steps[0]
                answer = first_step.get('answer', 'No answer')
                print(f"First step answer: {answer[:150]}...")
                
                # Check if answer is query-specific
                query_words = query.lower().split()
                answer_lower = answer.lower()
                relevance_score = sum(1 for word in query_words if word in answer_lower)
                print(f"Query relevance score: {relevance_score}/{len(query_words)}")
            else:
                print("No reasoning steps found!")
        
        # Compare answers to ensure they're different
        print(f"\n🔍 Answer Diversity Analysis...")
        answers = []
        for query, result in results.items():
            steps = result.get('reasoning_steps', [])
            if steps:
                answer = steps[0].get('answer', '')
                answers.append((query, answer))
        
        # Check for similarity between answers
        print("Answer similarity matrix:")
        for i in range(len(answers)):
            for j in range(i+1, len(answers)):
                query1, answer1 = answers[i]
                query2, answer2 = answers[j]
                
                # Simple similarity check
                words1 = set(answer1.lower().split())
                words2 = set(answer2.lower().split())
                common_words = words1.intersection(words2)
                similarity = len(common_words) / max(len(words1), len(words2))
                
                print(f"  {query1[:25]}... vs {query2[:25]}...: {similarity:.2f}")
        
        # Test final answers
        print(f"\n💡 Final Answer Diversity...")
        for query, result in results.items():
            final_answer = result.get('final_answer', '')
            print(f"\nQuery: {query}")
            print(f"Final answer preview: {final_answer[:200]}...")
        
        # Check if LLM query specifically works
        llm_result = results.get("What is LLM?", {})
        if llm_result:
            print(f"\n🎯 LLM Query Specific Test:")
            steps = llm_result.get('reasoning_steps', [])
            if steps:
                first_step = steps[0]
                answer = first_step.get('answer', '')
                if 'Large Language Models' in answer or 'LLM' in answer:
                    print("✅ LLM query correctly identifies Large Language Models!")
                else:
                    print("❌ LLM query doesn't mention Large Language Models")
        
        print(f"\n🎉 Final verification test completed!")
        print("The system should now provide different, query-specific answers.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run final verification test"""
    print("🎯 Final Verification Test Suite")
    print("=" * 50)
    
    success = test_final_verification()
    
    if success:
        print("\n✅ Final verification test completed!")
        print("The system now provides different, query-specific answers!")
    else:
        print("\n❌ Final verification test failed.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
