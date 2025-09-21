"""
PDF export functionality for research reports
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from typing import List, Dict, Any
from datetime import datetime
import os


class PDFExporter:
    """Handles PDF export of research reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Title style
        if 'CustomTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            ))
        
        # Section header style
        if 'SectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                spaceBefore=20,
                textColor=colors.darkblue
            ))
        
        # Subsection style
        if 'Subsection' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='Subsection',
                parent=self.styles['Heading3'],
                fontSize=12,
                spaceAfter=8,
                spaceBefore=12,
                textColor=colors.darkgreen
            ))
        
        # Body text style
        if 'BodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                alignment=TA_JUSTIFY
            ))
        
        # Quote style
        if 'Quote' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='Quote',
                parent=self.styles['Normal'],
                fontSize=9,
                leftIndent=20,
                rightIndent=20,
                spaceAfter=6,
                textColor=colors.grey,
                fontStyle='italic'
            ))
    
    def export_research_report(self, research_data: Dict[str, Any], 
                             output_path: str) -> str:
        """Export research data to PDF format"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Add title
            title = Paragraph("Research Report", self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Add metadata
            metadata = self._create_metadata_table(research_data)
            story.append(metadata)
            story.append(Spacer(1, 20))
            
            # Add executive summary
            if 'summary' in research_data:
                story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
                story.append(Paragraph(research_data['summary'], self.styles['BodyText']))
                story.append(Spacer(1, 20))
            
            # Add research queries and results
            queries = research_data.get('queries', [])
            for i, query_data in enumerate(queries, 1):
                story = self._add_query_section(story, query_data, i)
                story.append(Spacer(1, 20))
            
            # Add conclusions
            if 'conclusions' in research_data:
                story.append(Paragraph("Conclusions", self.styles['SectionHeader']))
                story.append(Paragraph(research_data['conclusions'], self.styles['BodyText']))
                story.append(Spacer(1, 20))
            
            # Add sources
            sources = self._extract_sources(queries)
            if sources:
                story.append(Paragraph("Sources", self.styles['SectionHeader']))
                story = self._add_sources_section(story, sources)
            
            # Build PDF
            doc.build(story)
            return output_path
            
        except Exception as e:
            raise Exception(f"Error creating PDF: {str(e)}")
    
    def _create_metadata_table(self, research_data: Dict[str, Any]) -> Table:
        """Create metadata table for the report"""
        data = [
            ['Session ID', research_data.get('session_id', 'N/A')],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Queries', str(len(research_data.get('queries', [])))],
            ['Query Types', ', '.join(set(q.get('result', {}).get('query_type', 'unknown') 
                                        for q in research_data.get('queries', [])))],
            ['Total Sources', str(research_data.get('total_sources_accessed', 0))]
        ]
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _add_query_section(self, story: List, query_data: Dict[str, Any], 
                          query_num: int) -> List:
        """Add a query section to the story"""
        result = query_data.get('result', {})
        
        # Query header
        query_text = query_data.get('query', 'No query')
        story.append(Paragraph(f"Query {query_num}: {query_text}", self.styles['SectionHeader']))
        
        # Query metadata
        metadata_data = [
            ['Type', result.get('query_type', 'unknown')],
            ['Quality', result.get('reasoning_quality', 'unknown')],
            ['Sources', str(result.get('total_sources', 0))]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[1*inch, 5*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 12))
        
        # Reasoning steps
        reasoning_steps = result.get('reasoning_steps', [])
        if reasoning_steps:
            story.append(Paragraph("Reasoning Process", self.styles['Subsection']))
            
            for i, step in enumerate(reasoning_steps, 1):
                story.append(Paragraph(f"Step {i}: {step.get('description', 'No description')}", 
                                     self.styles['Subsection']))
                story.append(Paragraph(step.get('answer', 'No answer'), self.styles['BodyText']))
                story.append(Paragraph(f"Confidence: {step.get('confidence', 0):.2f}", 
                                     self.styles['Quote']))
                story.append(Spacer(1, 8))
        
        # Final answer
        final_answer = result.get('final_answer', 'No final answer')
        story.append(Paragraph("Final Answer", self.styles['Subsection']))
        story.append(Paragraph(final_answer, self.styles['BodyText']))
        
        # Follow-up questions
        follow_ups = result.get('follow_up_questions', [])
        if follow_ups:
            story.append(Paragraph("Suggested Follow-up Questions", self.styles['Subsection']))
            for i, follow_up in enumerate(follow_ups, 1):
                story.append(Paragraph(f"{i}. {follow_up}", self.styles['BodyText']))
        
        return story
    
    def _extract_sources(self, queries: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sources from all queries"""
        sources = set()
        for query in queries:
            result = query.get('result', {})
            reasoning_steps = result.get('reasoning_steps', [])
            for step in reasoning_steps:
                step_sources = step.get('sources', [])
                sources.update(step_sources)
        return list(sources)
    
    def _add_sources_section(self, story: List, sources: List[str]) -> List:
        """Add sources section to the story"""
        if not sources:
            story.append(Paragraph("No sources available", self.styles['BodyText']))
            return story
        
        # Create sources table
        source_data = [[f"{i+1}.", source] for i, source in enumerate(sources)]
        
        sources_table = Table(source_data, colWidths=[0.5*inch, 5.5*inch])
        sources_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0)
        ]))
        
        story.append(sources_table)
        return story
    
    def export_simple_report(self, content: str, output_path: str, 
                           title: str = "Research Report") -> str:
        """Export simple text content to PDF"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Add title
            story.append(Paragraph(title, self.styles['CustomTitle']))
            story.append(Spacer(1, 20))
            
            # Add content
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), self.styles['BodyText']))
                    story.append(Spacer(1, 6))
            
            # Build PDF
            doc.build(story)
            return output_path
            
        except Exception as e:
            raise Exception(f"Error creating simple PDF: {str(e)}")
