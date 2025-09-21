#!/usr/bin/env python3
"""
Test script to verify query-specific answers
"""
import sys
import os

def test_query_specificity():
    """Test that different queries produce different, specific answers"""
    print("🎯 Testing Query-Specific Answer Generation")
    print("=" * 60)
    
    try:
        from researcher_agent import DeepResearcherAgent
        from utils.document_processor import DocumentProcessor
        
        # Initialize agent
        print("🤖 Initializing agent...")
        agent = DeepResearcherAgent(use_enhanced_reasoning=False)  # Use standard for testing
        session_id = agent.start_research_session("specificity_test")
        print(f"✅ Session started: {session_id}")
        
        # Load sample documents
        print("📚 Loading sample documents...")
        sample_docs = DocumentProcessor.create_sample_documents()
        result = agent.index_documents(sample_docs, "specificity_test")
        print(f"✅ Indexed {result['documents_indexed']} documents")
        
        # Test different types of queries
        test_queries = [
            ("What is artificial intelligence?", "definition"),
            ("How does machine learning work?", "process"),
            ("What are neural networks?", "definition"),
            ("Compare supervised and unsupervised learning", "comparison"),
            ("Why is deep learning effective?", "causal")
        ]
        
        print("\n🔍 Testing Query-Specific Answers...")
        results = {}
        
        for query, query_type in test_queries:
            print(f"\n--- {query_type.upper()}: {query} ---")
            
            result = agent.research_query(query, max_reasoning_steps=2)
            results[query] = result
            
            # Show first step answer
            steps = result.get('reasoning_steps', [])
            if steps:
                first_step = steps[0]
                answer = first_step.get('answer', 'No answer')
                print(f"First step answer: {answer[:200]}...")
                
                # Check if answer is query-specific
                query_words = query.lower().split()
                answer_lower = answer.lower()
                relevance_score = sum(1 for word in query_words if word in answer_lower)
                print(f"Query relevance score: {relevance_score}/{len(query_words)}")
            else:
                print("No reasoning steps found!")
        
        # Compare answers to ensure they're different
        print(f"\n🔍 Comparing Answer Diversity...")
        answers = []
        for query, result in results.items():
            steps = result.get('reasoning_steps', [])
            if steps:
                answer = steps[0].get('answer', '')
                answers.append((query, answer))
        
        # Check for similarity between answers
        print("Answer similarity analysis:")
        for i in range(len(answers)):
            for j in range(i+1, len(answers)):
                query1, answer1 = answers[i]
                query2, answer2 = answers[j]
                
                # Simple similarity check
                words1 = set(answer1.lower().split())
                words2 = set(answer2.lower().split())
                common_words = words1.intersection(words2)
                similarity = len(common_words) / max(len(words1), len(words2))
                
                print(f"  {query1[:30]}... vs {query2[:30]}...: {similarity:.2f}")
        
        # Test final answers
        print(f"\n💡 Testing Final Answer Diversity...")
        for query, result in results.items():
            final_answer = result.get('final_answer', '')
            print(f"\nQuery: {query}")
            print(f"Final answer preview: {final_answer[:300]}...")
        
        print(f"\n🎉 Query specificity test completed!")
        print("Check the output above to verify that different queries produce different answers.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run query specificity test"""
    print("🎯 Query Specificity Test Suite")
    print("=" * 50)
    
    success = test_query_specificity()
    
    if success:
        print("\n✅ Query specificity test completed!")
        print("The system should now provide different, query-specific answers.")
    else:
        print("\n❌ Query specificity test failed.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
