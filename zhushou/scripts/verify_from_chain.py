#!/usr/bin/env python3
"""
Verification CLI: Download files from IPFS using CIDs from blockchain,
recompute SHA-256 hashes, and verify they match on-chain hashes.

This script demonstrates end-to-end data provenance verification.
"""

import os
import sys
import argparse
import logging
import json
import hashlib
import tempfile
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
from web3 import Web3
from tabulate import tabulate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_contract_info(contract_file: str) -> Dict:
    """Load contract address and deployment info."""
    with open(contract_file, 'r') as f:
        return json.load(f)


def connect_to_blockchain(rpc_url: str, contract_address: str) -> tuple:
    """Connect to blockchain and load contract."""
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to blockchain at {rpc_url}")
    
    logger.info(f"✓ Connected to blockchain (Chain ID: {w3.eth.chain_id})")
    
    # Load contract ABI
    abi_path = 'artifacts/contracts/ClimateDataRegistry.sol/ClimateDataRegistry.json'
    with open(abi_path, 'r') as f:
        contract_data = json.load(f)
        abi = contract_data['abi']
    
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=abi
    )
    
    logger.info(f"✓ Contract loaded at {contract_address}")
    
    return w3, contract


def get_all_records(contract) -> List[Dict]:
    """Retrieve all records from the blockchain."""
    try:
        record_count = contract.functions.getRecordCount().call()
        logger.info(f"Found {record_count} records on chain")
        
        records = []
        for i in range(record_count):
            record = contract.functions.getRecord(i).call()
            records.append({
                'record_id': i,
                'data_hash': record[0].hex(),
                'source_url': record[1],
                'license': record[2],
                'timestamp': record[3],
                'ipfs_cid': record[4],
                'uploader': record[5]
            })
        
        return records
        
    except Exception as e:
        logger.error(f"Error retrieving records: {e}")
        return []


def download_from_ipfs(cid: str, output_path: str) -> bool:
    """
    Download file from IPFS by CID.
    Returns True if successful, False otherwise.
    """
    try:
        import ipfshttpclient
        client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
        
        logger.debug(f"Downloading CID: {cid}")
        client.get(cid, target=output_path)
        
        # ipfshttpclient.get creates a subdirectory with the CID name
        # Move the actual file to the desired location
        downloaded_file = os.path.join(output_path, cid)
        if os.path.isfile(downloaded_file):
            final_path = output_path + '_file'
            os.rename(downloaded_file, final_path)
            os.rmdir(output_path)
            os.rename(final_path, output_path)
        
        return True
        
    except Exception as e:
        logger.debug(f"IPFS download failed: {e}")
        return False


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    
    with open(filepath, 'rb') as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def verify_record(record: Dict, temp_dir: str, use_ipfs: bool = True) -> Dict:
    """
    Verify a single record by downloading from IPFS and checking hash.
    Returns verification result dictionary.
    """
    record_id = record['record_id']
    expected_hash = record['data_hash']
    ipfs_cid = record['ipfs_cid']
    
    logger.info(f"\nVerifying Record #{record_id}")
    logger.info(f"  CID: {ipfs_cid}")
    logger.info(f"  Expected hash: {expected_hash[:32]}...")
    
    result = {
        'record_id': record_id,
        'ipfs_cid': ipfs_cid,
        'expected_hash': expected_hash,
        'actual_hash': None,
        'verified': False,
        'error': None
    }
    
    if not use_ipfs:
        # For mock CIDs, try to find file locally
        artifacts_df = pd.read_csv('output/artifacts_index.csv')
        matching = artifacts_df[artifacts_df['sha256_hex'] == expected_hash]
        
        if len(matching) > 0:
            local_path = matching.iloc[0]['local_path']
            if os.path.exists(local_path):
                actual_hash = compute_file_hash(local_path)
                result['actual_hash'] = actual_hash
                result['verified'] = (actual_hash == expected_hash)
                logger.info(f"  Local verification: {'✓ PASS' if result['verified'] else '✗ FAIL'}")
                return result
            else:
                result['error'] = 'Local file not found'
                logger.warning(f"  Local file not found: {local_path}")
                return result
        else:
            result['error'] = 'No matching artifact in index'
            return result
    
    # Download from IPFS
    temp_file = os.path.join(temp_dir, f'record_{record_id}')
    
    if download_from_ipfs(ipfs_cid, temp_file):
        if os.path.exists(temp_file):
            # Compute hash
            actual_hash = compute_file_hash(temp_file)
            result['actual_hash'] = actual_hash
            
            # Verify
            result['verified'] = (actual_hash == expected_hash)
            
            if result['verified']:
                logger.info(f"  ✓ VERIFIED: Hashes match!")
            else:
                logger.error(f"  ✗ FAILED: Hash mismatch!")
                logger.error(f"    Expected: {expected_hash}")
                logger.error(f"    Actual:   {actual_hash}")
            
            # Cleanup
            os.remove(temp_file)
        else:
            result['error'] = 'Downloaded file not found'
            logger.error(f"  ✗ Downloaded file not found")
    else:
        result['error'] = 'IPFS download failed'
        logger.warning(f"  ⚠ IPFS download failed - trying local verification")
        # Fallback to local verification
        return verify_record(record, temp_dir, use_ipfs=False)
    
    return result


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(
        description='Verify data provenance from blockchain and IPFS'
    )
    parser.add_argument(
        '--sample', 
        type=int, 
        default=None,
        help='Number of random records to verify (default: all)'
    )
    parser.add_argument(
        '--contract', 
        default='contracts/contract-address.json',
        help='Path to contract address JSON file'
    )
    parser.add_argument(
        '--network', 
        default='localhost',
        help='Network name (default: localhost)'
    )
    parser.add_argument(
        '--rpc-url',
        default='http://127.0.0.1:8545',
        help='RPC URL (default: http://127.0.0.1:8545)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print(" DATA PROVENANCE VERIFICATION")
    print("="*70)
    
    # Load contract info
    try:
        contract_info = load_contract_info(args.contract)
        contract_address = contract_info['address']
    except Exception as e:
        logger.error(f"Failed to load contract info: {e}")
        return 1
    
    # Connect to blockchain
    try:
        w3, contract = connect_to_blockchain(args.rpc_url, contract_address)
    except Exception as e:
        logger.error(f"Failed to connect to blockchain: {e}")
        logger.error("Ensure Hardhat node is running: npm run node")
        return 1
    
    # Get all records
    records = get_all_records(contract)
    
    if not records:
        logger.error("No records found on chain")
        return 1
    
    # Sample records if requested
    if args.sample and args.sample < len(records):
        import random
        records = random.sample(records, args.sample)
        logger.info(f"Sampling {args.sample} random records for verification")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temp directory: {temp_dir}")
        
        # Verify each record
        results = []
        for record in records:
            result = verify_record(record, temp_dir)
            results.append(result)
        
        # Summary
        print("\n" + "="*70)
        print(" VERIFICATION SUMMARY")
        print("="*70)
        
        verified_count = sum(1 for r in results if r['verified'])
        failed_count = sum(1 for r in results if not r['verified'] and r['error'] is None)
        error_count = sum(1 for r in results if r['error'] is not None)
        
        print(f"\nTotal records checked: {len(results)}")
        print(f"✓ Verified:            {verified_count}")
        print(f"✗ Failed verification: {failed_count}")
        print(f"⚠ Errors:              {error_count}")
        
        # Detailed table
        print("\n" + "="*70)
        print(" DETAILED RESULTS")
        print("="*70 + "\n")
        
        table_data = []
        for r in results:
            status = "✓ PASS" if r['verified'] else ("✗ FAIL" if r['error'] is None else f"⚠ {r['error']}")
            table_data.append([
                r['record_id'],
                r['ipfs_cid'][:20] + "..." if len(r['ipfs_cid']) > 20 else r['ipfs_cid'],
                r['expected_hash'][:16] + "...",
                status
            ])
        
        print(tabulate(table_data, 
                      headers=['Record ID', 'IPFS CID', 'Hash', 'Status'],
                      tablefmt='grid'))
        
        print("\n" + "="*70)
        
        # Exit code
        if verified_count == len(results):
            print("✓ All records verified successfully!")
            print("="*70)
            return 0
        else:
            print("⚠ Some records failed verification")
            print("="*70)
            return 1


if __name__ == '__main__':
    sys.exit(main())

