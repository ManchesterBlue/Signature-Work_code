#!/usr/bin/env python3
"""
ETL Step 1: Fetch and Load Data
Inputs: Raw data files from data_sample/ directory
Outputs: Loaded DataFrames and file paths
Invariants: All specified files must exist and be readable
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import yaml
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/01_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_schema(schema_path: str = 'etl/schema.yaml') -> Dict:
    """Load and parse the data schema configuration."""
    try:
        with open(schema_path, 'r') as f:
            schema = yaml.safe_load(f)
        logger.info(f"Loaded schema from {schema_path}")
        return schema
    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
        raise


def fetch_indicators(data_dir: str, schema: Dict) -> pd.DataFrame:
    """
    Fetch climate indicators CSV file.
    Validates that file exists and can be loaded.
    """
    file_path = os.path.join(data_dir, schema['indicators']['file'])
    logger.info(f"Fetching indicators from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Indicators file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} indicator records")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Date range: {df['year'].min()} - {df['year'].max()}")
        logger.info(f"Countries: {df['country'].unique().tolist()}")
        return df
    except Exception as e:
        logger.error(f"Failed to load indicators: {e}")
        raise


def fetch_policy_files(data_dir: str, schema: Dict) -> List[Dict]:
    """
    Fetch policy PDF files from the policies directory.
    Returns list of file metadata including paths.
    """
    policies_dir = os.path.join(data_dir, schema['policies']['directory'])
    pattern = schema['policies']['file_pattern']
    
    logger.info(f"Fetching policy files from {policies_dir}")
    
    if not os.path.exists(policies_dir):
        os.makedirs(policies_dir, exist_ok=True)
        logger.warning(f"Created policies directory: {policies_dir}")
    
    policy_files = []
    
    # For demo: if no PDFs exist, create simple text-based PDFs
    pdf_files = list(Path(policies_dir).glob(pattern))
    
    if not pdf_files:
        logger.warning("No policy PDFs found - generating demo PDFs")
        # Generate simple PDFs using basic approach
        demo_pdfs = generate_demo_pdfs(policies_dir)
        pdf_files = [Path(p) for p in demo_pdfs]
    
    for pdf_path in pdf_files:
        # Extract metadata from filename (convention: country_name_year.pdf)
        filename = pdf_path.stem
        parts = filename.split('_')
        
        policy_info = {
            'filename': pdf_path.name,
            'path': str(pdf_path),
            'size_bytes': pdf_path.stat().st_size,
            'country': extract_country_from_filename(filename),
            'year': extract_year_from_filename(filename),
        }
        policy_files.append(policy_info)
        logger.info(f"Found policy: {policy_info['filename']} ({policy_info['size_bytes']} bytes)")
    
    logger.info(f"Total policy files found: {len(policy_files)}")
    return policy_files


def generate_demo_pdfs(policies_dir: str) -> List[str]:
    """Generate minimal demo PDF files if reportlab is available, else create text files."""
    demo_files = []
    
    policies = [
        {
            'name': 'germany_renewable_energy_act_2021.pdf',
            'content': '''Renewable Energy Act 2021 Amendment
Federal Republic of Germany

This Act serves the purpose of facilitating the sustainable development of energy supply.

Article 1: Expansion Targets
The share of renewable energy in gross electricity consumption is to be increased to at least
65 percent by 2030. Wind energy onshore: 71 GW, wind offshore: 20 GW, solar: 100 GW.

Article 2: Support Mechanisms
Financial support through feed-in tariffs and market premiums. Tax incentives for renewable
investments. Priority grid connection for renewable sources.

Article 3: Investment Incentives
Tax credits up to 20% for qualifying renewable energy projects. Subsidies for energy storage
and grid modernization. Public buildings must install solar panels where feasible.'''
        },
        {
            'name': 'india_national_solar_mission_2022.pdf',
            'content': '''National Solar Mission 2022 Update
Ministry of New and Renewable Energy, Government of India

The Jawaharlal Nehru National Solar Mission aims to establish India as a global leader
in solar energy. Target: 100 GW by 2022, expanding to 280 GW by 2030.

Section 1: Capacity Targets
Grid-connected solar: 40 GW. Rooftop solar: 40 GW. Off-grid solar: 20 GW.
Special incentives for ultra-mega solar parks exceeding 500 MW capacity.

Section 2: Financial Support and Subsidies
Central Financial Assistance of 40% for residential rooftop up to 3 kW.
Concessional finance through IREDA with interest rates 2% below market.
Generation-based incentives and Viability Gap Funding for solar parks.

Section 3: Regulatory Standards
Renewable Purchase Obligations mandate minimum solar power purchases.
Net metering regulations enable credit for excess generation.'''
        },
        {
            'name': 'brazil_energy_expansion_plan_2023.pdf',
            'content': '''Ten-Year Energy Expansion Plan 2023-2032
Ministry of Mines and Energy, Federative Republic of Brazil

Strategic guidelines for expansion of Brazil's energy sector with emphasis on renewables.

Chapter 1: Renewable Energy Targets
By 2032, renewable sources should represent 90% of electricity generation.
Wind: +30 GW onshore and +8 GW offshore. Solar: +45 GW. Hydropower: +8 GW.
Biomass and biogas: +6 GW from agricultural residues.

Chapter 2: Investment Framework
Total investment: R$ 2.8 trillion. Public-private partnerships encouraged.
BNDES provides preferential financing. Foreign investment welcome.

Chapter 3: Transmission and Storage
Transmission network expansion: 25,000 km. Mandatory energy storage for plants >100 MW.
Smart grid deployment. Tax exemptions for transmission equipment and batteries.'''
        }
    ]
    
    try:
        # Try using reportlab if available
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        
        for policy in policies:
            filepath = os.path.join(policies_dir, policy['name'])
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            for line in policy['content'].split('\n'):
                if line.strip():
                    story.append(Paragraph(line.strip(), styles['BodyText']))
                    story.append(Spacer(1, 12))
            
            doc.build(story)
            demo_files.append(filepath)
            logger.info(f"Generated demo PDF: {policy['name']}")
    except ImportError:
        logger.warning("reportlab not available - creating text-based demo files")
        # Fallback: create simple text files with .pdf extension for demo purposes
        for policy in policies:
            filepath = os.path.join(policies_dir, policy['name'])
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(policy['content'])
            demo_files.append(filepath)
            logger.info(f"Generated demo text file: {policy['name']}")
    
    return demo_files


def extract_country_from_filename(filename: str) -> str:
    """Extract country name from policy filename."""
    # Simple heuristic: first part before underscore
    parts = filename.lower().split('_')
    if parts:
        country = parts[0].capitalize()
        if country in ['Germany', 'India', 'Brazil']:
            return country
    return 'Unknown'


def extract_year_from_filename(filename: str) -> int:
    """Extract year from policy filename."""
    import re
    # Look for 4-digit year pattern
    match = re.search(r'20\d{2}', filename)
    if match:
        return int(match.group())
    return 0


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("ETL Step 1: Fetch and Load Data")
    logger.info("="*60)
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Load schema
    schema = load_schema()
    
    # Data directory
    data_dir = os.getenv('DATA_DIR', 'data_sample')
    logger.info(f"Data directory: {data_dir}")
    
    # Fetch indicators
    indicators_df = fetch_indicators(data_dir, schema)
    
    # Fetch policy files
    policy_files = fetch_policy_files(data_dir, schema)
    
    # Save intermediate results
    output_dir = 'output'
    indicators_output = os.path.join(output_dir, 'fetched_indicators.csv')
    indicators_df.to_csv(indicators_output, index=False)
    logger.info(f"Saved fetched indicators to {indicators_output}")
    
    # Save policy file list
    policy_df = pd.DataFrame(policy_files)
    policy_output = os.path.join(output_dir, 'fetched_policies.csv')
    policy_df.to_csv(policy_output, index=False)
    logger.info(f"Saved policy file list to {policy_output}")
    
    logger.info("="*60)
    logger.info("Step 1 Complete - Data fetched successfully")
    logger.info(f"Indicators: {len(indicators_df)} records")
    logger.info(f"Policy files: {len(policy_files)} files")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

