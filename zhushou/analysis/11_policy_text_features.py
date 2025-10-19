#!/usr/bin/env python3
"""
Analysis Step 2: Extract Policy Text Features
Inputs: Policy PDF files
Outputs: Policy intensity features per year/country
Method: Keyword extraction and simple topic modeling
"""

import os
import sys
import logging
from typing import Dict, List
import pandas as pd
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/11_policy_text_features.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from PDF file.
    Tries PyPDF2 first, falls back to reading as text for demo files.
    """
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        logger.debug(f"Extracted {len(text)} characters from {pdf_path}")
        return text
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed, trying text fallback: {e}")
        # Fallback for demo text files
        try:
            with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            logger.debug(f"Extracted {len(text)} characters (text mode) from {pdf_path}")
            return text
        except Exception as e2:
            logger.error(f"Failed to extract text from {pdf_path}: {e2}")
            return ""


def compute_keyword_features(text: str, keywords: List[str]) -> Dict[str, int]:
    """
    Compute frequency of policy-relevant keywords in text.
    Keywords indicate policy intensity and mechanisms.
    """
    text_lower = text.lower()
    
    features = {}
    for keyword in keywords:
        # Count occurrences (case-insensitive, word boundaries)
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        count = len(re.findall(pattern, text_lower))
        features[f'keyword_{keyword}'] = count
    
    return features


def compute_policy_intensity(text: str) -> float:
    """
    Compute overall policy intensity score based on action words.
    Higher score indicates more concrete policy commitments.
    """
    action_words = [
        'require', 'mandate', 'enforce', 'implement', 'establish',
        'invest', 'fund', 'allocate', 'provide', 'support',
        'target', 'goal', 'achieve', 'increase', 'reduce',
        'incentive', 'subsidy', 'tax', 'credit', 'regulation'
    ]
    
    text_lower = text.lower()
    total_words = len(text_lower.split())
    
    if total_words == 0:
        return 0.0
    
    action_count = sum(text_lower.count(word) for word in action_words)
    intensity = (action_count / total_words) * 100  # Percentage
    
    return intensity


def extract_policy_features(policy_files: pd.DataFrame) -> pd.DataFrame:
    """
    Extract text features from all policy documents.
    Returns dataframe with features per country/year.
    """
    logger.info(f"Extracting features from {len(policy_files)} policy documents")
    
    # Define policy-relevant keywords
    keywords = [
        'renewable', 'solar', 'wind', 'hydro', 'biomass',
        'subsidy', 'tax', 'incentive', 'credit', 'investment',
        'target', 'goal', 'capacity', 'generation', 'emission',
        'standard', 'regulation', 'mandate', 'policy', 'support'
    ]
    
    features_list = []
    
    for idx, row in policy_files.iterrows():
        pdf_path = row['path']
        
        if not os.path.exists(pdf_path):
            logger.warning(f"Policy file not found: {pdf_path}")
            continue
        
        logger.info(f"Processing: {row['filename']}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            logger.warning(f"No text extracted from {pdf_path}")
            continue
        
        # Compute features
        keyword_features = compute_keyword_features(text, keywords)
        intensity = compute_policy_intensity(text)
        
        # Extract metadata
        country = row.get('country', 'Unknown')
        year = row.get('year', 0)
        
        feature_row = {
            'country': country,
            'year': year,
            'filename': row['filename'],
            'text_length': len(text),
            'word_count': len(text.split()),
            'policy_intensity': intensity,
            **keyword_features
        }
        
        features_list.append(feature_row)
        logger.info(f"  Features: {len(text.split())} words, intensity={intensity:.2f}")
    
    features_df = pd.DataFrame(features_list)
    logger.info(f"Extracted features from {len(features_df)} documents")
    
    return features_df


def aggregate_features_by_year(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate policy features by country and year.
    Handles multiple policies per country/year.
    """
    logger.info("Aggregating features by country and year")
    
    # Group by country and year
    agg_funcs = {
        'text_length': 'sum',
        'word_count': 'sum',
        'policy_intensity': 'mean',
    }
    
    # Add aggregation for all keyword columns
    keyword_cols = [col for col in features_df.columns if col.startswith('keyword_')]
    for col in keyword_cols:
        agg_funcs[col] = 'sum'
    
    aggregated = features_df.groupby(['country', 'year']).agg(agg_funcs).reset_index()
    
    # Create composite features
    aggregated['total_policy_keywords'] = aggregated[keyword_cols].sum(axis=1)
    aggregated['keyword_density'] = (
        aggregated['total_policy_keywords'] / 
        aggregated['word_count'].replace(0, 1)
    ) * 100
    
    logger.info(f"Aggregated to {len(aggregated)} country-year observations")
    
    return aggregated


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Analysis Step 2: Extract Policy Text Features")
    logger.info("="*60)
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Load policy file list
    policy_files_path = 'output/fetched_policies.csv'
    if not os.path.exists(policy_files_path):
        logger.error(f"Policy files list not found: {policy_files_path}")
        logger.error("Please run ETL pipeline first")
        return 1
    
    policy_files = pd.read_csv(policy_files_path)
    logger.info(f"Loaded {len(policy_files)} policy files")
    
    # Extract features
    features_df = extract_policy_features(policy_files)
    
    if len(features_df) == 0:
        logger.error("No features extracted from policy documents")
        return 1
    
    # Aggregate by year
    aggregated_features = aggregate_features_by_year(features_df)
    
    # Save results
    output_path = 'output/policy_features.csv'
    aggregated_features.to_csv(output_path, index=False)
    logger.info(f"Saved policy features to {output_path}")
    
    logger.info("="*60)
    logger.info("Step 2 Complete - Policy features extracted")
    logger.info(f"Documents processed: {len(features_df)}")
    logger.info(f"Country-year observations: {len(aggregated_features)}")
    logger.info("="*60)
    
    # Display summary statistics
    logger.info("\nPolicy Intensity Summary:")
    logger.info(aggregated_features[['country', 'year', 'policy_intensity', 
                                     'total_policy_keywords', 'keyword_density']].to_string())
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

