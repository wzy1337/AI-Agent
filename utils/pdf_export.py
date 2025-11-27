# -*- coding: utf-8 -*-
"""
PDF export functionality for research reports.
"""

import re
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


def save_research_output_as_pdf(research_response, filename: str = None):
    """
    Save the research output as a formatted PDF.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_output_{timestamp}.pdf"
    
    # Ensure filename has .pdf extension
    if not filename.endswith('.pdf'):
        filename += '.pdf'
    
    # Create PDF
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for PDF elements
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c5f8d'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#2c5f8d'),
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        leading=14
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['BodyText'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=6,
        leading=14
    )
    
    # Helper function to clean text for PDF
    def clean_text(text):
        """Remove problematic characters and escape XML special chars"""
        if not text:
            return ""
        # Replace common problematic characters
        text = str(text)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        # Remove emoji and special unicode
        text = re.sub(r'[^\x00-\x7F\u00A0-\uFFFF]+', '', text)
        return text
    
    # Title Page
    story.append(Paragraph(clean_text(research_response.topic), title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Metadata
    metadata_data = [
        ['Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ['Total Events:', str(len(research_response.events))],
        ['Action Items:', str(len(research_response.action_items) if research_response.action_items else 0)],
    ]
    
    metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(metadata_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading1_style))
    story.append(Paragraph(clean_text(research_response.summary), body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Sources
    if research_response.source:
        story.append(Paragraph("Primary Sources", heading2_style))
        for idx, source in enumerate(research_response.source, 1):
            story.append(Paragraph(f"{idx}. {clean_text(source)}", bullet_style))
    
    story.append(PageBreak())
    
    # Events Section
    story.append(Paragraph("Detailed Analysis of Events", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    for idx, event in enumerate(research_response.events, 1):
        # Event Header
        story.append(Paragraph(f"Event {idx}: {clean_text(event.event)}", heading2_style))
        
        # Event Metadata Table
        event_metadata = [
            ['Category:', clean_text(event.category)],
            ['Location:', clean_text(event.location) if event.location else 'N/A'],
            ['Date Range:', clean_text(', '.join(event.date)) if event.date else 'N/A'],
            ['Relevance:', clean_text(event.relevance)],
            ['Confidence:', clean_text(event.confidence)],
            ['Signal Strength:', clean_text(event.signal_strength)],
        ]
        
        event_table = Table(event_metadata, colWidths=[1.5*inch, 4.5*inch])
        event_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1a5490')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        story.append(event_table)
        story.append(Spacer(1, 0.15*inch))
        
        # Description
        story.append(Paragraph("<b>Description:</b>", heading3_style))
        story.append(Paragraph(clean_text(event.description), body_style))
        
        # Impact
        story.append(Paragraph("<b>Impact Analysis:</b>", heading3_style))
        story.append(Paragraph(clean_text(event.impact), body_style))
        
        # Scenario
        story.append(Paragraph("<b>Scenarios:</b>", heading3_style))
        story.append(Paragraph(clean_text(event.scenario), body_style))
        
        # Actors
        story.append(Paragraph("<b>Key Actors:</b>", heading3_style))
        actors_text = ', '.join(event.actors) if event.actors else 'N/A'
        story.append(Paragraph(clean_text(actors_text), body_style))
        
        # Policy Intervention
        story.append(Paragraph("<b>Policy Intervention Recommendations:</b>", heading3_style))
        story.append(Paragraph(clean_text(event.policy_intervention), body_style))
        
        # Sources
        story.append(Paragraph("<b>Sources:</b>", heading3_style))
        if event.source:
            # Handle both list and string formats
            sources = event.source if isinstance(event.source, list) else event.source.split(', ')
            for source in sources:
                story.append(Paragraph(f"• {clean_text(source)}", bullet_style))
        
        # Informal Insights (if available)
        if hasattr(event, 'informal_insights') and event.informal_insights:
            story.append(Paragraph("<b>Informal Insights:</b>", heading3_style))
            story.append(Paragraph(clean_text(event.informal_insights), body_style))
        
        if idx < len(research_response.events):
            story.append(PageBreak())
    
    # Action Items
    if research_response.action_items:
        story.append(PageBreak())
        story.append(Paragraph("Recommended Action Items", heading1_style))
        story.append(Spacer(1, 0.1*inch))
        
        for idx, action in enumerate(research_response.action_items, 1):
            story.append(Paragraph(f"{idx}. {clean_text(action)}", bullet_style))
    
    # Repeated Events (if any)
    if hasattr(research_response, 'repeated_events') and research_response.repeated_events:
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Repeated Events", heading2_style))
        for event in research_response.repeated_events:
            story.append(Paragraph(f"• {clean_text(event)}", bullet_style))
    
    # Build PDF
    try:
        doc.build(story)
        file_size = Path(filename).stat().st_size / 1024
        print(f"\n✅ PDF saved: {filename}")
        print(f"📄 File size: {file_size:.2f} KB")
        return filename
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        return None
