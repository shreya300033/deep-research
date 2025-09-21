"""
Command Line Interface for the Deep Researcher Agent
"""
import argparse
import json
import sys
from pathlib import Path

from researcher_agent import DeepResearcherAgent
from utils.document_processor import DocumentProcessor


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Deep Researcher Agent CLI")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Process a research query')
    query_parser.add_argument('query', help='The research query to process')
    query_parser.add_argument('--session', help='Session ID (optional)')
    query_parser.add_argument('--steps', type=int, default=5, help='Max reasoning steps')
    query_parser.add_argument('--output', help='Output file path')
    query_parser.add_argument('--format', choices=['json', 'markdown', 'pdf'], 
                            default='json', help='Output format')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    index_parser.add_argument('path', help='Path to documents or directory')
    index_parser.add_argument('--name', default='default', help='Index name')
    index_parser.add_argument('--type', choices=['file', 'directory'], 
                            default='directory', help='Input type')
    index_parser.add_argument('--formats', nargs='+', 
                            default=['txt', 'json', 'md', 'pdf'],
                            help='File formats to process')
    
    # Session command
    session_parser = subparsers.add_parser('session', help='Manage research sessions')
    session_parser.add_argument('action', choices=['start', 'list', 'summary'], 
                              help='Session action')
    session_parser.add_argument('--name', help='Session name')
    session_parser.add_argument('--id', help='Session ID')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export research results')
    export_parser.add_argument('--session', help='Session ID to export')
    export_parser.add_argument('--format', choices=['json', 'markdown', 'pdf'], 
                             default='markdown', help='Export format')
    export_parser.add_argument('--output', help='Output file path')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    demo_parser.add_argument('--interactive', action='store_true', 
                           help='Run interactive demo')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize agent
    try:
        agent = DeepResearcherAgent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return
    
    # Execute commands
    if args.command == 'query':
        handle_query(agent, args)
    elif args.command == 'index':
        handle_index(agent, args)
    elif args.command == 'session':
        handle_session(agent, args)
    elif args.command == 'export':
        handle_export(agent, args)
    elif args.command == 'demo':
        handle_demo(agent, args)


def handle_query(agent, args):
    """Handle query command"""
    session_id = args.session
    if not session_id:
        session_id = agent.start_research_session()
        print(f"Started new session: {session_id}")
    
    print(f"Processing query: {args.query}")
    result = agent.research_query(args.query, args.steps)
    
    if args.output:
        save_result(result, args.output, args.format)
    else:
        display_result(result, args.format)


def handle_index(agent, args):
    """Handle index command"""
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path {path} does not exist")
        return
    
    documents = []
    
    if args.type == 'file':
        if path.suffix.lower() == '.json':
            documents = DocumentProcessor.process_json_file(str(path))
        elif path.suffix.lower() == '.pdf':
            doc = DocumentProcessor.process_pdf_file(str(path))
            if doc:
                documents = [doc]
        else:
            doc = DocumentProcessor.process_text_file(str(path))
            if doc:
                documents = [doc]
    else:  # directory
        documents = DocumentProcessor.process_directory(str(path), args.formats)
    
    if not documents:
        print("No documents found to index")
        return
    
    print(f"Indexing {len(documents)} documents...")
    result = agent.index_documents(documents, args.name)
    print(f"Indexing result: {result}")


def handle_session(agent, args):
    """Handle session command"""
    if args.action == 'start':
        session_id = agent.start_research_session(args.name)
        print(f"Started session: {session_id}")
    elif args.action == 'list':
        # This would require storing session info, simplified for now
        print("Session listing not implemented in this version")
    elif args.action == 'summary':
        if not args.id:
            print("Error: Session ID required for summary")
            return
        summary = agent.get_research_summary(args.id)
        print(json.dumps(summary, indent=2))


def handle_export(agent, args):
    """Handle export command"""
    if not args.session:
        print("Error: Session ID required for export")
        return
    
    print(f"Exporting session {args.session} in {args.format} format...")
    
    if args.format == 'pdf':
        output_path = agent.save_pdf_report(args.session, args.output)
        print(f"PDF report saved to: {output_path}")
    else:
        content = agent.export_research_report(args.session, args.format)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Report saved to: {args.output}")
        else:
            print(content)


def handle_demo(agent, args):
    """Handle demo command"""
    if args.interactive:
        run_interactive_demo(agent)
    else:
        run_quick_demo(agent)


def run_quick_demo(agent):
    """Run a quick demonstration"""
    print("=== Quick Demo ===")
    
    # Start session
    session_id = agent.start_research_session("demo")
    print(f"Started session: {session_id}")
    
    # Index sample documents
    sample_docs = DocumentProcessor.create_sample_documents()
    agent.index_documents(sample_docs, "demo_index")
    print("Indexed sample documents")
    
    # Process a query
    query = "What is artificial intelligence and how does it work?"
    print(f"Processing query: {query}")
    result = agent.research_query(query)
    
    print(f"Query type: {result['query_type']}")
    print(f"Reasoning quality: {result['reasoning_quality']}")
    print(f"Final answer: {result['final_answer'][:200]}...")
    
    # Export report
    report_path = agent.save_pdf_report(session_id, "demo_report")
    print(f"Report saved to: {report_path}")


def run_interactive_demo(agent):
    """Run interactive demonstration"""
    print("=== Interactive Demo ===")
    print("Type 'quit' to exit, 'help' for commands")
    
    session_id = agent.start_research_session("interactive_demo")
    print(f"Started session: {session_id}")
    
    # Index sample documents
    sample_docs = DocumentProcessor.create_sample_documents()
    agent.index_documents(sample_docs, "interactive_index")
    print("Indexed sample documents")
    
    while True:
        try:
            query = input("\nEnter your research query: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'help':
                print("Commands:")
                print("  quit - Exit the demo")
                print("  help - Show this help")
                print("  summary - Show session summary")
                print("  export [format] - Export current session")
                continue
            elif query.lower() == 'summary':
                summary = agent.get_research_summary(session_id)
                print(json.dumps(summary, indent=2))
                continue
            elif query.lower().startswith('export'):
                parts = query.split()
                format_type = parts[1] if len(parts) > 1 else 'markdown'
                report_path = agent.save_pdf_report(session_id, f"interactive_report") if format_type == 'pdf' else None
                if not report_path:
                    content = agent.export_research_report(session_id, format_type)
                    print(content)
                else:
                    print(f"Report saved to: {report_path}")
                continue
            
            if not query:
                continue
            
            print(f"Processing: {query}")
            result = agent.research_query(query)
            
            print(f"\nAnswer: {result['final_answer']}")
            
            if result['follow_up_questions']:
                print("\nSuggested follow-up questions:")
                for i, follow_up in enumerate(result['follow_up_questions'], 1):
                    print(f"{i}. {follow_up}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nDemo ended. Goodbye!")


def save_result(result, output_path, format_type):
    """Save result to file"""
    if format_type == 'json':
        content = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        content = result.get('final_answer', 'No answer generated')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Result saved to: {output_path}")


def display_result(result, format_type):
    """Display result to console"""
    if format_type == 'json':
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Query Type: {result.get('query_type', 'unknown')}")
        print(f"Reasoning Quality: {result.get('reasoning_quality', 'unknown')}")
        print(f"Final Answer: {result.get('final_answer', 'No answer generated')}")
        
        if result.get('follow_up_questions'):
            print("\nSuggested Follow-up Questions:")
            for i, follow_up in enumerate(result['follow_up_questions'], 1):
                print(f"{i}. {follow_up}")


if __name__ == "__main__":
    main()
