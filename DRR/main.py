"""
Streamlit Web Interface for Deep Researcher Agent
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
from pathlib import Path

# Import the Deep Researcher Agent components
from researcher_agent import DeepResearcherAgent
from utils.document_processor import DocumentProcessor

# Page configuration
st.set_page_config(
    page_title="Deep Researcher Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, attractive styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subheader styling */
    .subheader {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }
    
    /* Reasoning step styling */
    .reasoning-step {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .reasoning-step:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Confidence indicators */
    .confidence-high { 
        color: #28a745; 
        font-weight: 600;
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.9rem;
    }
    .confidence-medium { 
        color: #856404; 
        font-weight: 600;
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.9rem;
    }
    .confidence-low { 
        color: #721c24; 
        font-weight: 600;
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.9rem;
    }
    
    /* Button styling - Enhanced selectors */
    .stButton > button,
    .stButton > button:focus,
    .stButton > button:active,
    .stButton > button:hover,
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 25px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        color: white !important;
    }
    
    /* Primary button specific styling */
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Secondary button styling */
    .stButton > button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: #f8f9fa;
        border-radius: 0 0 10px 10px;
        border: 1px solid #dee2e6;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid #e9ecef;
        padding: 0.75rem 1rem;
        font-family: 'Inter', sans-serif;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 15px;
        border: 2px solid #e9ecef;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border-radius: 15px;
        border: 2px dashed #667eea;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #5a6fd8;
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 15px;
        color: #155724;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #f5c6cb;
        border-radius: 15px;
        color: #721c24;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        border-radius: 15px;
        color: #0c5460;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px 15px 0 0;
        border: 1px solid #dee2e6;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Additional button styling for better coverage */
    div[data-testid="stButton"] > button,
    div[data-testid="stButton"] > button:focus,
    div[data-testid="stButton"] > button:active,
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 25px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Force button styling with higher specificity */
    .stApp .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 25px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stApp .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        color: white !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'vector_stats' not in st.session_state:
    st.session_state.vector_stats = {}

def initialize_agent(use_enhanced_reasoning=True):
    """Initialize the Deep Researcher Agent"""
    if st.session_state.agent is None or st.session_state.agent.use_enhanced_reasoning != use_enhanced_reasoning:
        with st.spinner("Initializing Deep Researcher Agent..."):
            st.session_state.agent = DeepResearcherAgent(use_enhanced_reasoning=use_enhanced_reasoning)
            st.session_state.session_id = st.session_state.agent.start_research_session("Streamlit_Session")
        reasoning_type = "Enhanced" if use_enhanced_reasoning else "Standard"
        st.success(f"✅ Agent initialized with {reasoning_type} reasoning!")

def load_sample_documents():
    """Load and index sample documents"""
    if st.session_state.agent:
        with st.spinner("Loading sample documents..."):
            sample_docs = DocumentProcessor.create_sample_documents()
            result = st.session_state.agent.index_documents(sample_docs, "streamlit_knowledge_base")
            st.session_state.vector_stats = st.session_state.agent.get_vector_store_stats()
        st.success(f"✅ Indexed {result['documents_indexed']} documents with {result['chunks_created']} chunks!")

def get_confidence_class(confidence):
    """Get CSS class for confidence score"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🔬 Deep Researcher Agent</h1>', unsafe_allow_html=True)
    
    # Add a beautiful subtitle
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #6c757d; font-family: 'Inter', sans-serif; margin: 0;">
            Advanced AI-powered research assistant with multi-step reasoning and document analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a decorative separator
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <div style="height: 3px; background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent); border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #2c3e50; font-family: 'Inter', sans-serif; margin: 0;">🎛️ Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced reasoning toggle with better styling
        st.markdown("### 🧠 Reasoning Mode")
        use_enhanced_reasoning = st.checkbox(
            "Enhanced Reasoning", 
            value=True,
            help="Use advanced reasoning patterns (deductive, inductive, causal, etc.)"
        )
        
        # Initialize Agent with enhanced button
        st.markdown("### 🚀 Agent Control")
        if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
            initialize_agent(use_enhanced_reasoning)
        
        # Load Sample Documents
        if st.button("📚 Load Sample Documents", use_container_width=True):
            if st.session_state.agent:
                load_sample_documents()
            else:
                st.error("Please initialize the agent first!")
        
        # Session info with better styling
        if st.session_state.session_id:
            st.markdown("### 📋 Session Info")
            st.info(f"**Session ID:** {st.session_state.session_id}")
        
        # Enhanced vector store stats
        if st.session_state.vector_stats:
            st.markdown("### 📊 Knowledge Base")
            stats = st.session_state.vector_stats
            
            # Create metric cards
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Documents", stats.get('unique_documents', 0))
                st.metric("🔢 Vectors", stats.get('total_vectors', 0))
            with col2:
                st.metric("📐 Dimensions", stats.get('dimension', 0))
                st.metric("💾 Chunks", stats.get('total_chunks', 0))
        
        # Enhanced Research History
        if st.session_state.research_history:
            st.markdown("### 📝 Recent Queries")
            for i, query in enumerate(st.session_state.research_history[-5:], 1):
                with st.container():
                    st.markdown(f"**{i}.** {query['query'][:40]}...")
        
        # Add a decorative element
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <div style="height: 2px; background: linear-gradient(90deg, transparent, #667eea, transparent); border-radius: 1px;"></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if st.session_state.agent is None:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin: 2rem 0;">
            <h3 style="color: #2c3e50; font-family: 'Inter', sans-serif; margin-bottom: 1rem;">🚀 Ready to Start Research?</h3>
            <p style="color: #6c757d; font-size: 1.1rem; margin: 0;">Please initialize the agent from the sidebar to begin your research journey!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Enhanced tabs with better styling
    st.markdown("### 🎯 Research Interface")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Research Query", "📊 Analytics", "📄 Reports", "📁 Document Management", "⚙️ Settings"])
    
    with tab1:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #2c3e50; font-family: 'Inter', sans-serif; margin: 0;">🔍 Research Query</h2>
            <p style="color: #6c757d; margin: 0.5rem 0 0 0;">Ask any question and get comprehensive, AI-powered research results</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced query input
        query = st.text_area(
            "💭 Enter your research query:",
            placeholder="e.g., What is artificial intelligence and how does it work?\n\nTry questions like:\n• What are the ethical implications of AI?\n• How do neural networks process information?\n• Compare supervised and unsupervised learning",
            height=120,
            help="Enter any research question. The system will use advanced reasoning to provide comprehensive answers."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            max_steps = st.slider("Max Reasoning Steps", 1, 10, 5)
        
        with col2:
            if st.button("🔬 Process Query", type="primary"):
                if query.strip():
                    with st.spinner("Processing query..."):
                        result = st.session_state.agent.research_query(query, max_steps)
                        st.session_state.research_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'query': query,
                            'result': result
                        })
                        st.rerun()
                else:
                    st.error("Please enter a query!")
        
            # Enhanced results display
        if st.session_state.research_history:
            latest_result = st.session_state.research_history[-1]['result']
            reasoning_type = st.session_state.research_history[-1].get('reasoning_type', 'standard')
            
            # Add a beautiful results header
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; margin: 2rem 0; text-align: center;">
                <h3 style="color: white; margin: 0; font-family: 'Inter', sans-serif;">🎯 Research Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced query info with better styling
            st.markdown("### 📊 Analysis Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("🔍 Query Type", latest_result.get('query_type', 'unknown').title())
            with col2:
                st.metric("🎯 Quality", latest_result.get('reasoning_quality', 'unknown'))
            with col3:
                st.metric("📚 Sources", latest_result.get('total_sources', 0))
            with col4:
                st.metric("🧠 Steps", len(latest_result.get('reasoning_steps', [])))
            with col5:
                st.metric("⚡ Type", reasoning_type.title())
            
            # Show advanced metrics if available
            advanced_metrics = latest_result.get('advanced_metrics', {})
            if advanced_metrics:
                st.subheader("📈 Advanced Reasoning Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Insights", advanced_metrics.get('total_insights', 0))
                with col2:
                    st.metric("Pattern Diversity", advanced_metrics.get('pattern_diversity', 0))
                with col3:
                    st.metric("Total Evidence", advanced_metrics.get('total_evidence', 0))
                with col4:
                    st.metric("Avg Confidence", f"{advanced_metrics.get('average_confidence', 0):.2f}")
                
                # Show reasoning patterns used
                patterns = advanced_metrics.get('reasoning_patterns_used', [])
                if patterns:
                    st.write(f"**Reasoning Patterns Used:** {', '.join(patterns)}")
            
            # Reasoning steps
            st.subheader("🧠 Reasoning Process")
            for i, step in enumerate(latest_result.get('reasoning_steps', []), 1):
                with st.expander(f"Step {i}: {step.get('description', 'No description')}"):
                    st.markdown(f"**Answer:** {step.get('answer', 'No answer')}")
                    confidence = step.get('confidence', 0)
                    confidence_class = get_confidence_class(confidence)
                    st.markdown(f"**Confidence:** <span class='{confidence_class}'>{confidence:.2f}</span>", unsafe_allow_html=True)
                    if step.get('sources'):
                        st.markdown(f"**Sources:** {', '.join(step['sources'])}")
                    
                    # Show enhanced reasoning details if available
                    if 'reasoning_pattern' in step:
                        st.markdown(f"**Reasoning Pattern:** {step['reasoning_pattern']}")
                    
                    if 'key_insights' in step and step['key_insights']:
                        st.markdown("**Key Insights:**")
                        for insight in step['key_insights']:
                            st.markdown(f"- {insight}")
                    
                    if 'supporting_evidence' in step and step['supporting_evidence']:
                        st.markdown("**Supporting Evidence:**")
                        for evidence in step['supporting_evidence']:
                            st.markdown(f"- {evidence}")
                    
                    if 'logical_connections' in step and step['logical_connections']:
                        st.markdown("**Logical Connections:**")
                        for connection in step['logical_connections']:
                            st.markdown(f"- {connection}")
            
            # Final answer
            st.subheader("💡 Final Answer")
            st.markdown(latest_result.get('final_answer', 'No answer generated'))
            
            # Follow-up questions
            follow_ups = latest_result.get('follow_up_questions', [])
            if follow_ups:
                st.subheader("❓ Suggested Follow-up Questions")
                for i, follow_up in enumerate(follow_ups, 1):
                    st.markdown(f"{i}. {follow_up}")
    
    with tab2:
        st.header("📊 Research Analytics")
        
        if st.session_state.research_history:
            # Summary statistics
            total_queries = len(st.session_state.research_history)
            query_types = {}
            total_sources = 0
            
            for query_data in st.session_state.research_history:
                result = query_data['result']
                query_type = result.get('query_type', 'unknown')
                query_types[query_type] = query_types.get(query_type, 0) + 1
                total_sources += result.get('total_sources', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Queries", total_queries)
            with col2:
                st.metric("Query Types", len(query_types))
            with col3:
                st.metric("Total Sources", total_sources)
            
            # Query type distribution
            if query_types:
                st.subheader("📈 Query Type Distribution")
                df_types = pd.DataFrame(list(query_types.items()), columns=['Query Type', 'Count'])
                st.bar_chart(df_types.set_index('Query Type'))
            
            # Research timeline
            st.subheader("⏰ Research Timeline")
            timeline_data = []
            for query_data in st.session_state.research_history:
                timeline_data.append({
                    'Timestamp': query_data['timestamp'][:19],
                    'Query': query_data['query'][:50] + '...',
                    'Type': query_data['result'].get('query_type', 'unknown'),
                    'Sources': query_data['result'].get('total_sources', 0)
                })
            
            if timeline_data:
                df_timeline = pd.DataFrame(timeline_data)
                st.dataframe(df_timeline, use_container_width=True)
        else:
            st.info("No research queries yet. Start by processing a query in the Research Query tab!")
    
    with tab3:
        st.header("📄 Export Reports")
        
        if st.session_state.research_history:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 Markdown Report")
                if st.button("Generate Markdown Report"):
                    with st.spinner("Generating report..."):
                        report_content = st.session_state.agent.export_research_report(
                            st.session_state.session_id, "markdown"
                        )
                        st.download_button(
                            label="Download Markdown Report",
                            data=report_content,
                            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
            
            with col2:
                st.subheader("📄 PDF Report")
                if st.button("Generate PDF Report"):
                    with st.spinner("Generating PDF report..."):
                        try:
                            pdf_path = st.session_state.agent.save_pdf_report(
                                st.session_state.session_id, 
                                f"streamlit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            )
                            if os.path.exists(pdf_path):
                                with open(pdf_path, "rb") as pdf_file:
                                    st.download_button(
                                        label="Download PDF Report",
                                        data=pdf_file.read(),
                                        file_name=os.path.basename(pdf_path),
                                        mime="application/pdf"
                                    )
                        except Exception as e:
                            st.error(f"Error generating PDF: {e}")
            
            # Display report preview
            st.subheader("📖 Report Preview")
            if st.button("Show Report Preview"):
                report_content = st.session_state.agent.export_research_report(
                    st.session_state.session_id, "markdown"
                )
                st.markdown(report_content)
        else:
            st.info("No research data to export. Process some queries first!")
    
    with tab4:
        st.header("📁 Document Management")
        
        st.subheader("📚 Current Knowledge Base")
        if st.session_state.vector_stats:
            stats = st.session_state.vector_stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", stats.get('unique_documents', 0))
            with col2:
                st.metric("Chunks", stats.get('total_chunks', 0))
            with col3:
                st.metric("Vectors", stats.get('total_vectors', 0))
        
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, JSON, MD)",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'json', 'md']
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                documents = []
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == 'application/pdf':
                        # Process PDF file
                        try:
                            pdf_data = uploaded_file.read()
                            doc = DocumentProcessor.process_pdf_data(pdf_data, uploaded_file.name)
                            if doc and 'error' not in doc:
                                documents.append(doc)
                            else:
                                st.error(f"Error processing PDF {uploaded_file.name}: {doc.get('error', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Error processing PDF {uploaded_file.name}: {e}")
                    
                    elif uploaded_file.type == 'application/json':
                        # Process JSON file
                        try:
                            content = uploaded_file.read().decode('utf-8')
                            data = json.loads(content)
                            if isinstance(data, list):
                                for item in data:
                                    documents.append({
                                        'id': item.get('id', f"upload_{len(documents)}"),
                                        'title': item.get('title', uploaded_file.name),
                                        'content': item.get('content', str(item)),
                                        'source': uploaded_file.name,
                                        'type': 'upload'
                                    })
                            else:
                                documents.append({
                                    'id': data.get('id', 'upload_document'),
                                    'title': data.get('title', uploaded_file.name),
                                    'content': data.get('content', str(data)),
                                    'source': uploaded_file.name,
                                    'type': 'upload'
                                })
                        except json.JSONDecodeError:
                            st.error(f"Invalid JSON in {uploaded_file.name}")
                    
                    else:
                        # Process text files (TXT, MD)
                        try:
                            content = uploaded_file.read().decode('utf-8')
                            documents.append({
                                'id': f"upload_{len(documents)}",
                                'title': uploaded_file.name,
                                'content': content,
                                'source': uploaded_file.name,
                                'type': 'upload'
                            })
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                
                if documents:
                    with st.spinner(f"Processing {len(documents)} documents..."):
                        result = st.session_state.agent.index_documents(documents, "uploaded_documents")
                        st.session_state.vector_stats = st.session_state.agent.get_vector_store_stats()
                    st.success(f"✅ Processed {result['documents_indexed']} documents!")
                    
                    # Show document details
                    st.subheader("📋 Processed Documents")
                    for doc in documents:
                        with st.expander(f"📄 {doc.get('title', 'Unknown')}"):
                            st.write(f"**Type:** {doc.get('type', 'unknown')}")
                            st.write(f"**Source:** {doc.get('source', 'unknown')}")
                            if doc.get('page_count'):
                                st.write(f"**Pages:** {doc['page_count']}")
                            if doc.get('parser_used'):
                                st.write(f"**Parser:** {doc['parser_used']}")
                            st.write(f"**Content Length:** {len(doc.get('content', ''))} characters")
                            if doc.get('metadata'):
                                st.write("**Metadata:**")
                                st.json(doc['metadata'])
                else:
                    st.warning("No documents were successfully processed.")
    
    with tab5:
        st.header("⚙️ Settings")
        
        st.subheader("🔧 System Configuration")
        
        # Display current config
        st.json({
            "Embedding Model": "all-MiniLM-L6-v2",
            "Chunk Size": 1000,
            "Chunk Overlap": 200,
            "Top K Results": 10,
            "Similarity Threshold": 0.7
        })
        
        st.subheader("🗑️ Clear Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Research History", type="secondary"):
                st.session_state.research_history = []
                st.rerun()
        
        with col2:
            if st.button("Reset Session", type="secondary"):
                if st.session_state.agent:
                    st.session_state.agent.clear_research_history()
                    st.session_state.session_id = st.session_state.agent.start_research_session("Streamlit_Session")
                    st.session_state.research_history = []
                    st.rerun()
        
        st.subheader("ℹ️ System Information")
        st.info("""
        **Deep Researcher Agent** - A comprehensive research system that:
        - Uses local embedding generation (no external APIs)
        - Performs multi-step reasoning
        - Provides confidence scoring
        - Exports professional reports
        - Supports various document formats
        """)

if __name__ == "__main__":
    main()
