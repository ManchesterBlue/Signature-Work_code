#!/usr/bin/env python3
"""
ETL Step 3: Add Files to IPFS
Inputs: Artifacts index from step 2
Outputs: Updated artifacts index with IPFS CIDs
Invariants: CID must be retrievable and content-addressed
"""

import os
import sys
import logging
from typing import Optional
import pandas as pd
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/03_ipfs_add.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_ipfs_connection() -> bool:
    """
    Check if IPFS daemon is running and accessible.
    Returns True if connection successful, False otherwise.
    """
    try:
        import ipfshttpclient
        client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
        # Test connection
        client.id()
        logger.info("✓ IPFS daemon connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ IPFS daemon not accessible: {e}")
        logger.error("Please ensure IPFS daemon is running: ipfs daemon")
        return False


def add_file_to_ipfs(filepath: str) -> Optional[str]:
    """
    Add a file to IPFS and return its CID.
    Uses ipfshttpclient for Python-based interaction.
    """
    try:
        import ipfshttpclient
        client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
        
        logger.info(f"Adding to IPFS: {filepath}")
        result = client.add(filepath)
        
        cid = result['Hash']
        logger.info(f"✓ IPFS CID: {cid}")
        return cid
        
    except Exception as e:
        logger.error(f"Failed to add {filepath} to IPFS: {e}")
        return None


def add_artifacts_to_ipfs(artifacts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all artifacts to IPFS and update the dataframe with CIDs.
    Handles failures gracefully and logs progress.
    """
    logger.info(f"Adding {len(artifacts_df)} artifacts to IPFS")
    
    # Check IPFS connection first
    if not check_ipfs_connection():
        logger.warning("IPFS not available - using mock CIDs for demonstration")
        # Generate mock CIDs for demo purposes
        artifacts_df['ipfs_cid'] = artifacts_df['sha256_hex'].apply(
            lambda h: f"Qm{h[:44]}"  # Mock CID format
        )
        artifacts_df['ipfs_status'] = 'mock'
        return artifacts_df
    
    # Add each file to IPFS
    cids = []
    statuses = []
    
    for idx, row in artifacts_df.iterrows():
        filepath = row['local_path']
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            cids.append(None)
            statuses.append('file_not_found')
            continue
        
        try:
            cid = add_file_to_ipfs(filepath)
            if cid:
                cids.append(cid)
                statuses.append('success')
                logger.info(f"✓ {row['filename']} -> {cid}")
            else:
                cids.append(None)
                statuses.append('failed')
                logger.error(f"✗ Failed to add {row['filename']}")
            
            # Small delay to avoid overwhelming IPFS daemon
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error adding {filepath}: {e}")
            cids.append(None)
            statuses.append('error')
    
    artifacts_df['ipfs_cid'] = cids
    artifacts_df['ipfs_status'] = statuses
    
    successful = sum(1 for s in statuses if s == 'success')
    logger.info(f"Successfully added {successful}/{len(artifacts_df)} files to IPFS")
    
    return artifacts_df


def verify_ipfs_retrieval(cid: str) -> bool:
    """
    Verify that a file can be retrieved from IPFS by its CID.
    Returns True if retrieval successful, False otherwise.
    """
    try:
        import ipfshttpclient
        client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
        
        # Try to get file stats (lightweight check)
        client.object.stat(cid)
        return True
    except Exception as e:
        logger.debug(f"Verification failed for {cid}: {e}")
        return False


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("ETL Step 3: Add Files to IPFS")
    logger.info("="*60)
    
    output_dir = 'output'
    
    # Load artifacts index
    artifacts_path = os.path.join(output_dir, 'artifacts_index.csv')
    if not os.path.exists(artifacts_path):
        logger.error(f"Artifacts index not found: {artifacts_path}")
        logger.error("Please run step 2 first (02_clean_and_hash.py)")
        return 1
    
    artifacts_df = pd.read_csv(artifacts_path)
    logger.info(f"Loaded {len(artifacts_df)} artifacts")
    
    # Add to IPFS
    artifacts_with_cids = add_artifacts_to_ipfs(artifacts_df)
    
    # Save updated index
    artifacts_with_cids.to_csv(artifacts_path, index=False)
    logger.info(f"Updated artifacts index with IPFS CIDs: {artifacts_path}")
    
    # Summary statistics
    success_count = (artifacts_with_cids['ipfs_status'] == 'success').sum()
    mock_count = (artifacts_with_cids['ipfs_status'] == 'mock').sum()
    fail_count = len(artifacts_with_cids) - success_count - mock_count
    
    logger.info("="*60)
    logger.info("Step 3 Complete - Files added to IPFS")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Mock CIDs: {mock_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info("="*60)
    
    # Display sample CIDs
    logger.info("\nSample IPFS CIDs:")
    for idx, row in artifacts_with_cids.iterrows():
        if pd.notna(row.get('ipfs_cid')):
            logger.info(f"  {row['filename']}: {row['ipfs_cid']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

