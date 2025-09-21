# 🌐 Deep Researcher Agent - Streamlit Interface

## 🚀 Quick Start

The Streamlit web interface provides a user-friendly GUI for the Deep Researcher Agent. Here's how to use it:

### 1. Launch the Interface

```bash
# Option 1: Direct launch
streamlit run streamlit_app.py

# Option 2: Using the launcher script
python run_streamlit.py
```

### 2. Access the Interface

- **URL**: http://localhost:8501
- The interface will automatically open in your default browser
- If it doesn't open automatically, manually navigate to the URL above

## 🎛️ Interface Overview

### Sidebar - Control Panel
- **🚀 Initialize Agent**: Start the Deep Researcher Agent
- **📚 Load Sample Documents**: Load pre-built AI/ML knowledge base
- **📊 Knowledge Base Stats**: View indexing statistics
- **📝 Recent Queries**: See your latest research queries

### Main Tabs

#### 🔍 Research Query Tab
- **Query Input**: Enter your research questions
- **Reasoning Steps**: Configure how many reasoning steps to use (1-10)
- **Process Query**: Execute the research query
- **Results Display**: 
  - Query type and quality metrics
  - Step-by-step reasoning process
  - Final synthesized answer
  - Suggested follow-up questions

#### 📊 Analytics Tab
- **Summary Statistics**: Total queries, types, sources
- **Query Type Distribution**: Visual chart of query types
- **Research Timeline**: Chronological view of all queries
- **Performance Metrics**: Confidence scores and quality assessment

#### 📄 Reports Tab
- **Markdown Export**: Download research reports in Markdown format
- **PDF Export**: Generate professional PDF reports
- **Report Preview**: View report content before downloading
- **Session-based Reports**: Export specific research sessions

#### 📁 Document Management Tab
- **Knowledge Base Stats**: Current document and chunk counts
- **File Upload**: Upload text, JSON, or Markdown files
- **Document Processing**: Automatic indexing of uploaded documents
- **Format Support**: Handles various document formats

#### ⚙️ Settings Tab
- **System Configuration**: View current settings
- **Clear Data**: Reset research history or entire session
- **System Information**: Learn about the Deep Researcher Agent

## 🎯 Key Features

### 🔬 Multi-step Reasoning
- **Query Classification**: Automatically identifies query types
- **Step Decomposition**: Breaks complex queries into logical steps
- **Confidence Scoring**: Quality assessment for each reasoning step
- **Context Awareness**: Uses previous steps to inform later reasoning

### 📊 Real-time Analytics
- **Live Metrics**: Query counts, types, and performance
- **Visual Charts**: Bar charts for query type distribution
- **Timeline View**: Chronological research history
- **Quality Assessment**: Reasoning quality and confidence scores

### 📄 Professional Reports
- **Multiple Formats**: PDF and Markdown export options
- **Structured Content**: Organized with sections and subsections
- **Source Attribution**: Proper citation of information sources
- **Download Ready**: Direct download functionality

### 📁 Document Management
- **Drag & Drop**: Easy file upload interface
- **Multiple Formats**: Support for TXT, JSON, MD files
- **Automatic Processing**: Instant indexing of uploaded documents
- **Knowledge Base Stats**: Real-time statistics

## 🎨 User Experience

### Visual Design
- **Modern Interface**: Clean, professional design
- **Color-coded Metrics**: Green (high), Yellow (medium), Red (low) confidence
- **Responsive Layout**: Works on different screen sizes
- **Interactive Elements**: Expandable sections and real-time updates

### Navigation
- **Tab-based Interface**: Easy switching between features
- **Sidebar Controls**: Quick access to main functions
- **Status Indicators**: Clear feedback on system state
- **Progress Indicators**: Loading states for long operations

## 🔧 Technical Features

### Performance
- **Local Processing**: No external API dependencies
- **Efficient Indexing**: FAISS-based vector storage
- **Real-time Updates**: Live statistics and metrics
- **Session Management**: Persistent research sessions

### Security
- **Local Data**: All processing happens locally
- **No External Calls**: Complete offline operation
- **Data Privacy**: Your documents stay on your machine
- **Session Isolation**: Separate research sessions

## 📱 Usage Tips

### Getting Started
1. **Initialize Agent**: Click "Initialize Agent" in the sidebar
2. **Load Sample Data**: Use "Load Sample Documents" for quick start
3. **Ask Questions**: Use the Research Query tab to ask questions
4. **Explore Results**: Check the reasoning steps and confidence scores

### Advanced Usage
1. **Upload Documents**: Add your own knowledge base in Document Management
2. **Analyze Patterns**: Use Analytics tab to understand your research patterns
3. **Export Reports**: Generate professional reports for sharing
4. **Refine Queries**: Use follow-up questions to dig deeper

### Best Practices
- **Start Simple**: Begin with basic questions to understand the system
- **Use Follow-ups**: Leverage suggested follow-up questions
- **Check Analytics**: Monitor your research patterns and quality
- **Export Regularly**: Save important research sessions as reports

## 🚨 Troubleshooting

### Common Issues
- **Agent Not Initializing**: Make sure all dependencies are installed
- **No Results**: Try loading sample documents first
- **Slow Performance**: Reduce the number of reasoning steps
- **Upload Errors**: Check file format (TXT, JSON, MD supported)

### Performance Tips
- **Batch Uploads**: Upload multiple documents at once
- **Reasonable Queries**: Keep queries focused and specific
- **Monitor Resources**: Check system resources for large document sets
- **Clear History**: Reset sessions periodically for better performance

## 🎉 Success Indicators

You'll know the system is working well when you see:
- ✅ High confidence scores (>0.7) in reasoning steps
- 📊 Multiple query types in your analytics
- 🎯 Relevant and comprehensive answers
- 📄 Professional-looking exported reports
- 🔄 Smooth interaction with the interface

## 🔗 Integration

The Streamlit interface integrates seamlessly with:
- **CLI Interface**: Use `python cli.py` for command-line access
- **Python API**: Import and use `DeepResearcherAgent` directly
- **Export Formats**: PDF and Markdown reports
- **File Systems**: Local document storage and retrieval

Enjoy exploring the Deep Researcher Agent through this intuitive web interface! 🚀
