#!/usr/bin/env python3
"""
ETL Step 2: Clean Data and Compute Hashes
Inputs: Fetched data from step 1
Outputs: Cleaned data with SHA-256 hashes, artifacts index
Invariants: Hash must be deterministic and verifiable
"""

import os
import sys
import logging
import hashlib
from pathlib import Path
from typing import Dict, List
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/02_clean_and_hash.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compute_file_hash(filepath: str) -> str:
    """
    Compute SHA-256 hash of a file.
    Uses chunked reading for memory efficiency with large files.
    """
    sha256_hash = hashlib.sha256()
    
    try:
        with open(filepath, 'rb') as f:
            # Read in 64kb chunks
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        
        hash_hex = sha256_hash.hexdigest()
        logger.debug(f"Computed hash for {filepath}: {hash_hex[:16]}...")
        return hash_hex
    except Exception as e:
        logger.error(f"Failed to hash file {filepath}: {e}")
        raise


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Compute deterministic hash of a DataFrame.
    Converts to CSV bytes for consistent hashing.
    """
    # Convert to CSV bytes (deterministic)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    hash_hex = hashlib.sha256(csv_bytes).hexdigest()
    logger.debug(f"Computed DataFrame hash: {hash_hex[:16]}...")
    return hash_hex


def clean_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize indicator data.
    - Remove duplicates
    - Sort by country and year
    - Validate ranges
    - Handle missing values
    """
    logger.info(f"Cleaning indicators data ({len(df)} rows)")
    
    original_count = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['country', 'year'])
    if len(df) < original_count:
        logger.warning(f"Removed {original_count - len(df)} duplicate rows")
    
    # Sort for determinism
    df = df.sort_values(['country', 'year']).reset_index(drop=True)
    
    # Validate ranges
    assert df['year'].between(2000, 2030).all(), "Year out of valid range"
    assert df['renewable_share'].between(0, 100).all(), "Renewable share out of range"
    assert df['co2_per_capita'].ge(0).all(), "CO2 per capita cannot be negative"
    assert df['installed_capacity'].ge(0).all(), "Installed capacity cannot be negative"
    
    # Round to consistent precision
    df['renewable_share'] = df['renewable_share'].round(1)
    df['co2_per_capita'] = df['co2_per_capita'].round(2)
    df['installed_capacity'] = df['installed_capacity'].round(0)
    
    logger.info(f"Cleaned indicators: {len(df)} rows")
    return df


def create_artifacts_index(indicators_file: str, policy_files: List[str], 
                          output_dir: str) -> pd.DataFrame:
    """
    Create comprehensive artifacts index with hashes and metadata.
    This index will be used for blockchain registration.
    """
    logger.info("Creating artifacts index with hashes")
    
    artifacts = []
    
    # Add indicators file
    if os.path.exists(indicators_file):
        file_hash = compute_file_hash(indicators_file)
        artifacts.append({
            'filename': os.path.basename(indicators_file),
            'filetype': 'csv',
            'year': None,  # Covers multiple years
            'country': 'Multi',  # Covers multiple countries
            'sha256_hex': file_hash,
            'source_url': 'demo://data_sample/indicators.csv',
            'license': 'CC0-1.0',
            'local_path': indicators_file,
            'size_bytes': os.path.getsize(indicators_file)
        })
        logger.info(f"Indexed indicators: {file_hash[:16]}...")
    
    # Add policy files
    for policy_path in policy_files:
        if os.path.exists(policy_path):
            file_hash = compute_file_hash(policy_path)
            
            # Extract metadata from filename
            filename = Path(policy_path).stem
            country = extract_country_from_filename(filename)
            year = extract_year_from_filename(filename)
            
            artifacts.append({
                'filename': os.path.basename(policy_path),
                'filetype': 'pdf',
                'year': year,
                'country': country,
                'sha256_hex': file_hash,
                'source_url': f'demo://data_sample/policies/{os.path.basename(policy_path)}',
                'license': 'Demo-Public-Domain',
                'local_path': policy_path,
                'size_bytes': os.path.getsize(policy_path)
            })
            logger.info(f"Indexed policy {country} {year}: {file_hash[:16]}...")
    
    artifacts_df = pd.DataFrame(artifacts)
    
    # Save artifacts index
    index_path = os.path.join(output_dir, 'artifacts_index.csv')
    artifacts_df.to_csv(index_path, index=False)
    logger.info(f"Saved artifacts index to {index_path}")
    
    return artifacts_df


def extract_country_from_filename(filename: str) -> str:
    """Extract country name from filename."""
    parts = filename.lower().split('_')
    if parts:
        country = parts[0].capitalize()
        return country
    return 'Unknown'


def extract_year_from_filename(filename: str) -> int:
    """Extract year from filename."""
    import re
    match = re.search(r'20\d{2}', filename)
    if match:
        return int(match.group())
    return 0


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("ETL Step 2: Clean Data and Compute Hashes")
    logger.info("="*60)
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load fetched data
    indicators_df = pd.read_csv('output/fetched_indicators.csv')
    policy_files_df = pd.read_csv('output/fetched_policies.csv')
    
    # Clean indicators
    cleaned_indicators = clean_indicators(indicators_df)
    
    # Save cleaned indicators
    cleaned_path = os.path.join(output_dir, 'cleaned_indicators.csv')
    cleaned_indicators.to_csv(cleaned_path, index=False)
    logger.info(f"Saved cleaned indicators to {cleaned_path}")
    
    # Get policy file paths
    policy_paths = policy_files_df['path'].tolist()
    
    # Create artifacts index with hashes
    artifacts_df = create_artifacts_index(cleaned_path, policy_paths, output_dir)
    
    logger.info("="*60)
    logger.info("Step 2 Complete - Data cleaned and hashed")
    logger.info(f"Total artifacts indexed: {len(artifacts_df)}")
    logger.info(f"Hash algorithm: SHA-256")
    logger.info("="*60)
    
    # Display sample hashes
    logger.info("\nSample hashes:")
    for idx, row in artifacts_df.iterrows():
        logger.info(f"  {row['filename']}: {row['sha256_hex'][:32]}...")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

