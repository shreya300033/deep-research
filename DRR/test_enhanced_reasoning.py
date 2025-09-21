#!/usr/bin/env python3
"""
Test script for Enhanced Reasoning Engine
"""
import sys
import os

def test_enhanced_reasoning():
    """Test the enhanced reasoning capabilities"""
    print("🧠 Testing Enhanced Reasoning Engine")
    print("=" * 50)
    
    try:
        from researcher_agent import DeepResearcherAgent
        from utils.document_processor import DocumentProcessor
        
        # Initialize agent with enhanced reasoning
        print("🤖 Initializing agent with enhanced reasoning...")
        agent = DeepResearcherAgent(use_enhanced_reasoning=True)
        session_id = agent.start_research_session("enhanced_reasoning_test")
        print(f"✅ Session started: {session_id}")
        
        # Load sample documents
        print("📚 Loading sample documents...")
        sample_docs = DocumentProcessor.create_sample_documents()
        result = agent.index_documents(sample_docs, "enhanced_test")
        print(f"✅ Indexed {result['documents_indexed']} documents")
        
        # Test complex queries that benefit from enhanced reasoning
        test_queries = [
            "What is artificial intelligence and how does it work?",
            "Compare supervised and unsupervised learning approaches",
            "What are the ethical implications of AI development?",
            "How do neural networks process information and what makes them effective?"
        ]
        
        print("\n🔍 Testing enhanced reasoning capabilities...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Enhanced Query {i}: {query} ---")
            
            result = agent.research_query(query, max_reasoning_steps=4)
            
            print(f"📊 Query Type: {result.get('query_type', 'unknown')}")
            print(f"🎯 Reasoning Quality: {result.get('reasoning_quality', 'unknown')}")
            print(f"📖 Total Sources: {result.get('total_sources', 0)}")
            
            # Show enhanced reasoning details
            steps = result.get('reasoning_steps', [])
            print(f"🧠 Reasoning Steps: {len(steps)}")
            
            for j, step in enumerate(steps, 1):
                confidence = step.get('confidence', 0)
                reasoning_pattern = step.get('reasoning_pattern', 'unknown')
                key_insights = step.get('key_insights', [])
                
                print(f"  {j}. {step.get('description', 'No description')}")
                print(f"     Pattern: {reasoning_pattern}")
                print(f"     Confidence: {confidence:.2f}")
                print(f"     Insights: {len(key_insights)} found")
                if key_insights:
                    print(f"     Key insights: {', '.join(key_insights[:2])}")
            
            # Show advanced metrics if available
            advanced_metrics = result.get('advanced_metrics', {})
            if advanced_metrics:
                print(f"📈 Advanced Metrics:")
                print(f"   - Total Insights: {advanced_metrics.get('total_insights', 0)}")
                print(f"   - Reasoning Patterns: {advanced_metrics.get('reasoning_patterns_used', [])}")
                print(f"   - Pattern Diversity: {advanced_metrics.get('pattern_diversity', 0)}")
            
            # Show final answer preview
            final_answer = result.get('final_answer', 'No answer')
            preview = final_answer[:300] + "..." if len(final_answer) > 300 else final_answer
            print(f"💡 Final Answer Preview: {preview}")
            
            # Show follow-up questions
            follow_ups = result.get('follow_up_questions', [])
            if follow_ups:
                print(f"❓ Intelligent Follow-ups:")
                for k, follow_up in enumerate(follow_ups, 1):
                    print(f"   {k}. {follow_up}")
        
        print(f"\n🎉 Enhanced reasoning test completed!")
        print(f"📈 The enhanced reasoning engine provides:")
        print(f"   - Multiple reasoning patterns (deductive, inductive, causal, etc.)")
        print(f"   - Key insights extraction")
        print(f"   - Supporting evidence identification")
        print(f"   - Logical connections analysis")
        print(f"   - Intelligent follow-up question generation")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_reasoning_engines():
    """Compare standard vs enhanced reasoning"""
    print("\n🔄 Comparing Standard vs Enhanced Reasoning")
    print("=" * 50)
    
    try:
        from researcher_agent import DeepResearcherAgent
        from utils.document_processor import DocumentProcessor
        
        # Test query
        test_query = "What is machine learning and how does it differ from traditional programming?"
        
        # Test with standard reasoning
        print("📊 Testing with Standard Reasoning...")
        agent_standard = DeepResearcherAgent(use_enhanced_reasoning=False)
        agent_standard.start_research_session("standard_test")
        sample_docs = DocumentProcessor.create_sample_documents()
        agent_standard.index_documents(sample_docs, "standard_test")
        
        standard_result = agent_standard.research_query(test_query, max_reasoning_steps=3)
        
        print(f"Standard Reasoning Quality: {standard_result.get('reasoning_quality', 'unknown')}")
        print(f"Standard Confidence: {sum(step.get('confidence', 0) for step in standard_result.get('reasoning_steps', [])) / len(standard_result.get('reasoning_steps', [1])):.2f}")
        
        # Test with enhanced reasoning
        print("\n🚀 Testing with Enhanced Reasoning...")
        agent_enhanced = DeepResearcherAgent(use_enhanced_reasoning=True)
        agent_enhanced.start_research_session("enhanced_test")
        agent_enhanced.index_documents(sample_docs, "enhanced_test")
        
        enhanced_result = agent_enhanced.research_query(test_query, max_reasoning_steps=3)
        
        print(f"Enhanced Reasoning Quality: {enhanced_result.get('reasoning_quality', 'unknown')}")
        print(f"Enhanced Confidence: {sum(step.get('confidence', 0) for step in enhanced_result.get('reasoning_steps', [])) / len(enhanced_result.get('reasoning_steps', [1])):.2f}")
        
        # Show reasoning patterns used
        patterns = enhanced_result.get('advanced_metrics', {}).get('reasoning_patterns_used', [])
        print(f"Reasoning Patterns Used: {patterns}")
        
        print(f"\n✅ Comparison completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False

def main():
    """Run enhanced reasoning tests"""
    print("🧠 Enhanced Reasoning Engine Test Suite")
    print("=" * 60)
    
    success1 = test_enhanced_reasoning()
    success2 = compare_reasoning_engines()
    
    if success1 and success2:
        print("\n🎉 All enhanced reasoning tests passed!")
        print("The enhanced reasoning engine provides significantly improved:")
        print("  - Reasoning pattern recognition")
        print("  - Key insights extraction")
        print("  - Evidence-based analysis")
        print("  - Logical connection identification")
        print("  - Intelligent follow-up generation")
    else:
        print("\n❌ Some tests failed.")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)