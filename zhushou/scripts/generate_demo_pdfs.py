#!/usr/bin/env python3
"""
Generate demo policy PDF documents for the climate data MVP.
These are placeholder documents with realistic policy content structure.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
import os

def create_germany_pdf():
    """Generate Germany Renewable Energy Act 2021 demo PDF."""
    filename = "data_sample/policies/germany_renewable_energy_act_2021.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    doc = SimpleDocTemplate(filename, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='darkblue',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("Renewable Energy Act 2021 Amendment", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Federal Republic of Germany", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    
    content = """
    This Act serves the purpose of facilitating the sustainable development of energy supply,
    particularly for the climate and the environment, to reduce the economic costs of energy supply,
    also by incorporating long-term external effects, to conserve fossil energy resources and
    to promote the further development of technologies to generate electricity from renewable energy sources.
    """
    story.append(Paragraph(content, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Article 1: Expansion Targets", styles['Heading3']))
    content2 = """
    The share of renewable energy in gross electricity consumption is to be increased to at least
    65 percent by 2030. To achieve this target, the installed capacity of renewable energy installations
    is to be increased as follows: wind energy onshore to 71 gigawatts, wind energy offshore to
    20 gigawatts, solar energy to 100 gigawatts, and biomass to 8.4 gigawatts by 2030.
    """
    story.append(Paragraph(content2, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Article 2: Support Mechanisms", styles['Heading3']))
    content3 = """
    Financial support for renewable energy is provided through feed-in tariffs and market premiums.
    The support levels are determined through competitive bidding processes to ensure cost efficiency.
    Special provisions apply for small installations and community energy projects to ensure broad participation.
    Priority grid connection and dispatch are guaranteed for renewable energy sources.
    """
    story.append(Paragraph(content3, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Article 3: Investment Incentives", styles['Heading3']))
    content4 = """
    Tax incentives and subsidies are provided for renewable energy investments, particularly for
    energy storage systems and grid infrastructure modernization. Public buildings are required to
    install solar panels where technically feasible. Investment tax credits of up to 20% are available
    for qualifying renewable energy projects.
    """
    story.append(Paragraph(content4, styles['BodyText']))
    
    doc.build(story)
    print(f"✓ Generated: {filename}")

def create_india_pdf():
    """Generate India National Solar Mission 2022 demo PDF."""
    filename = "data_sample/policies/india_national_solar_mission_2022.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    doc = SimpleDocTemplate(filename, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='darkgreen',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("National Solar Mission 2022 Update", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Ministry of New and Renewable Energy, Government of India", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    
    content = """
    The Jawaharlal Nehru National Solar Mission (JNNSM) aims to establish India as a global leader
    in solar energy. The mission targets 100 GW of solar power capacity by 2022, with subsequent
    expansion to 280 GW by 2030 as part of India's commitment to clean energy transition and
    climate action goals.
    """
    story.append(Paragraph(content, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Section 1: Capacity Targets and Milestones", styles['Heading3']))
    content2 = """
    Grid-connected solar power plants: 40 GW. Rooftop and small solar power plants: 40 GW.
    Off-grid solar applications: 20 GW. State-level targets are allocated to ensure equitable
    distribution across regions with high solar potential. Special incentives for ultra-mega
    solar power parks exceeding 500 MW capacity.
    """
    story.append(Paragraph(content2, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Section 2: Financial Support and Subsidies", styles['Heading3']))
    content3 = """
    Central Financial Assistance (CFA) of 40% for residential rooftop installations up to 3 kW
    and 20% for installations between 3-10 kW. Concessional finance through Indian Renewable
    Energy Development Agency (IREDA) with interest rates 2% below market rates. Generation-based
    incentives for solar power producers. Viability Gap Funding (VGF) for solar parks development.
    """
    story.append(Paragraph(content3, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Section 3: Regulatory Standards and Mandates", styles['Heading3']))
    content4 = """
    Renewable Purchase Obligations (RPO) mandate that distribution companies purchase a minimum
    percentage of power from solar sources, increasing annually. Solar obligation for industrial
    and commercial consumers. Net metering regulations enable bidirectional flow and credit for
    excess generation. Standard technical specifications ensure quality and interoperability.
    """
    story.append(Paragraph(content4, styles['BodyText']))
    
    doc.build(story)
    print(f"✓ Generated: {filename}")

def create_brazil_pdf():
    """Generate Brazil Energy Expansion Plan 2023 demo PDF."""
    filename = "data_sample/policies/brazil_energy_expansion_plan_2023.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    doc = SimpleDocTemplate(filename, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='darkred',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("Ten-Year Energy Expansion Plan 2023-2032", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Ministry of Mines and Energy, Federative Republic of Brazil", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    
    content = """
    The Ten-Year Energy Expansion Plan establishes strategic guidelines for the expansion of
    Brazil's energy sector, emphasizing the development of renewable energy sources while
    maintaining energy security and competitive pricing. The plan leverages Brazil's natural
    advantages in hydropower, wind, and solar resources.
    """
    story.append(Paragraph(content, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Chapter 1: Renewable Energy Targets", styles['Heading3']))
    content2 = """
    By 2032, renewable sources should represent 90% of electricity generation capacity. Wind
    energy expansion: additional 30 GW onshore and 8 GW offshore. Solar photovoltaic: additional
    45 GW distributed and centralized. Hydropower: 8 GW through modernization and new plants.
    Biomass and biogas: 6 GW from sugarcane and agricultural residues.
    """
    story.append(Paragraph(content2, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Chapter 2: Investment Framework", styles['Heading3']))
    content3 = """
    Estimated total investment requirement: R$ 2.8 trillion over the planning period. Public-private
    partnerships encouraged through regulatory stability and long-term contracts. BNDES (National
    Development Bank) provides preferential financing lines for renewable projects with interest
    rates below market. Foreign investment welcome with guaranteed repatriation of profits and
    simplified licensing procedures.
    """
    story.append(Paragraph(content3, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Chapter 3: Transmission and Storage", styles['Heading3']))
    content4 = """
    Expansion of transmission network by 25,000 km to integrate renewable generation in remote areas.
    Mandatory installation of energy storage systems for intermittent renewable plants exceeding
    100 MW. Smart grid deployment to improve demand response and reduce curtailment. Tax exemptions
    for transmission equipment and battery storage systems to reduce capital costs.
    """
    story.append(Paragraph(content4, styles['BodyText']))
    
    doc.build(story)
    print(f"✓ Generated: {filename}")

if __name__ == "__main__":
    print("Generating demo policy PDF documents...")
    create_germany_pdf()
    create_india_pdf()
    create_brazil_pdf()
    print("\n✓ All demo PDFs generated successfully!")

