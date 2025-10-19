#!/usr/bin/env python3
"""
ETL Step 4: Register Artifacts on Blockchain
Inputs: Artifacts index with IPFS CIDs from step 3
Outputs: Transaction log with on-chain registration details
Invariants: Each artifact must have unique hash and valid CID
"""

import os
import sys
import logging
import json
from typing import Dict, Optional
import pandas as pd
from web3 import Web3
from eth_utils import to_bytes
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/04_register_onchain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BlockchainRegistry:
    """
    Wrapper for interacting with ClimateDataRegistry smart contract.
    Handles connection, transaction submission, and receipt validation.
    """
    
    def __init__(self, rpc_url: str, contract_address: str, abi_path: str):
        """Initialize connection to blockchain and contract."""
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to blockchain at {rpc_url}")
        
        logger.info(f"✓ Connected to blockchain (Chain ID: {self.w3.eth.chain_id})")
        
        # Load contract ABI
        with open(abi_path, 'r') as f:
            contract_data = json.load(f)
            # Extract ABI from Hardhat artifacts format
            if 'abi' in contract_data:
                abi = contract_data['abi']
            else:
                abi = contract_data
        
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi
        )
        logger.info(f"✓ Contract loaded at {contract_address}")
        
        # Get default account
        self.account = self.w3.eth.accounts[0]
        balance = self.w3.eth.get_balance(self.account)
        logger.info(f"✓ Using account {self.account} (Balance: {self.w3.from_wei(balance, 'ether')} ETH)")
    
    def register_record(self, data_hash: str, source_url: str, 
                       license_: str, ipfs_cid: str) -> Optional[Dict]:
        """
        Register a data record on the blockchain.
        Returns transaction receipt if successful, None otherwise.
        """
        try:
            # Convert hash string to bytes32
            hash_bytes = to_bytes(hexstr=data_hash)
            
            # Build transaction
            tx = self.contract.functions.addRecord(
                hash_bytes,
                source_url,
                license_,
                ipfs_cid
            ).build_transaction({
                'from': self.account,
                'nonce': self.w3.eth.get_transaction_count(self.account),
                'gas': 500000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=None)
            tx_hash = self.w3.eth.send_transaction(tx)
            
            logger.info(f"Transaction sent: {tx_hash.hex()}")
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                logger.info(f"✓ Transaction confirmed in block {receipt['blockNumber']}")
                
                # Extract record ID from event logs
                record_id = self.extract_record_id_from_receipt(receipt)
                
                return {
                    'tx_hash': receipt['transactionHash'].hex(),
                    'block_number': receipt['blockNumber'],
                    'gas_used': receipt['gasUsed'],
                    'record_id': record_id,
                    'status': 'success'
                }
            else:
                logger.error(f"✗ Transaction failed: {tx_hash.hex()}")
                return {
                    'tx_hash': receipt['transactionHash'].hex(),
                    'block_number': receipt['blockNumber'],
                    'gas_used': receipt['gasUsed'],
                    'record_id': None,
                    'status': 'failed'
                }
                
        except Exception as e:
            logger.error(f"Error registering record: {e}")
            return None
    
    def extract_record_id_from_receipt(self, receipt: Dict) -> Optional[int]:
        """Extract record ID from RecordAdded event in transaction receipt."""
        try:
            # Get RecordAdded event
            event_abi = [e for e in self.contract.abi if e.get('type') == 'event' 
                        and e.get('name') == 'RecordAdded'][0]
            
            # Process logs
            for log in receipt['logs']:
                try:
                    event = self.contract.events.RecordAdded().process_log(log)
                    return event['args']['recordId']
                except:
                    continue
            
            # Fallback: use contract method to get last record ID
            count = self.contract.functions.getRecordCount().call()
            return count - 1 if count > 0 else None
            
        except Exception as e:
            logger.debug(f"Could not extract record ID: {e}")
            return None
    
    def verify_record(self, record_id: int, expected_hash: str) -> bool:
        """Verify that a record exists on-chain with the expected hash."""
        try:
            record = self.contract.functions.getRecord(record_id).call()
            actual_hash = record[0].hex()  # dataHash is first field
            
            if actual_hash == expected_hash:
                logger.info(f"✓ Record {record_id} verified: hash matches")
                return True
            else:
                logger.error(f"✗ Record {record_id} hash mismatch!")
                logger.error(f"  Expected: {expected_hash}")
                logger.error(f"  Actual: {actual_hash}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying record {record_id}: {e}")
            return False


def load_contract_info() -> Dict:
    """Load contract address and network info."""
    address_file = 'contracts/contract-address.json'
    
    if not os.path.exists(address_file):
        raise FileNotFoundError(
            f"Contract address file not found: {address_file}\n"
            f"Please deploy contract first: npm run deploy"
        )
    
    with open(address_file, 'r') as f:
        return json.load(f)


def register_artifacts(artifacts_df: pd.DataFrame, registry: BlockchainRegistry) -> pd.DataFrame:
    """
    Register all artifacts on the blockchain.
    Updates dataframe with transaction information.
    """
    logger.info(f"Registering {len(artifacts_df)} artifacts on blockchain")
    
    tx_records = []
    
    for idx, row in artifacts_df.iterrows():
        logger.info(f"\nRegistering artifact {idx+1}/{len(artifacts_df)}: {row['filename']}")
        
        # Check if CID is available
        if pd.isna(row.get('ipfs_cid')) or not row.get('ipfs_cid'):
            logger.warning(f"Skipping {row['filename']}: No IPFS CID")
            tx_records.append({
                'artifact_filename': row['filename'],
                'sha256_hex': row['sha256_hex'],
                'ipfs_cid': None,
                'tx_hash': None,
                'block_number': None,
                'record_id': None,
                'status': 'skipped_no_cid'
            })
            continue
        
        # Register on blockchain
        result = registry.register_record(
            data_hash=row['sha256_hex'],
            source_url=row['source_url'],
            license_=row['license'],
            ipfs_cid=row['ipfs_cid']
        )
        
        if result:
            tx_records.append({
                'artifact_filename': row['filename'],
                'sha256_hex': row['sha256_hex'],
                'ipfs_cid': row['ipfs_cid'],
                'tx_hash': result['tx_hash'],
                'block_number': result['block_number'],
                'record_id': result['record_id'],
                'gas_used': result['gas_used'],
                'status': result['status']
            })
            logger.info(f"✓ Registered: Tx {result['tx_hash'][:16]}... Record ID: {result['record_id']}")
        else:
            tx_records.append({
                'artifact_filename': row['filename'],
                'sha256_hex': row['sha256_hex'],
                'ipfs_cid': row['ipfs_cid'],
                'tx_hash': None,
                'block_number': None,
                'record_id': None,
                'status': 'failed'
            })
            logger.error(f"✗ Failed to register {row['filename']}")
        
        # Small delay between transactions
        time.sleep(0.5)
    
    return pd.DataFrame(tx_records)


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("ETL Step 4: Register Artifacts on Blockchain")
    logger.info("="*60)
    
    output_dir = 'output'
    
    # Load artifacts index
    artifacts_path = os.path.join(output_dir, 'artifacts_index.csv')
    if not os.path.exists(artifacts_path):
        logger.error(f"Artifacts index not found: {artifacts_path}")
        return 1
    
    artifacts_df = pd.read_csv(artifacts_path)
    logger.info(f"Loaded {len(artifacts_df)} artifacts")
    
    # Load contract info
    try:
        contract_info = load_contract_info()
        logger.info(f"Contract address: {contract_info['address']}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # Initialize blockchain registry
    try:
        rpc_url = os.getenv('HARDHAT_RPC_URL', 'http://127.0.0.1:8545')
        abi_path = 'artifacts/contracts/ClimateDataRegistry.sol/ClimateDataRegistry.json'
        
        registry = BlockchainRegistry(
            rpc_url=rpc_url,
            contract_address=contract_info['address'],
            abi_path=abi_path
        )
    except Exception as e:
        logger.error(f"Failed to initialize blockchain connection: {e}")
        logger.error("Ensure Hardhat node is running: npm run node")
        return 1
    
    # Register artifacts
    tx_log_df = register_artifacts(artifacts_df, registry)
    
    # Save transaction log
    tx_log_path = os.path.join(output_dir, 'tx_log.csv')
    tx_log_df.to_csv(tx_log_path, index=False)
    logger.info(f"\nSaved transaction log to {tx_log_path}")
    
    # Summary statistics
    success_count = (tx_log_df['status'] == 'success').sum()
    failed_count = (tx_log_df['status'] == 'failed').sum()
    skipped_count = (tx_log_df['status'].str.contains('skipped')).sum()
    
    logger.info("="*60)
    logger.info("Step 4 Complete - Artifacts registered on blockchain")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Skipped: {skipped_count}")
    logger.info("="*60)
    
    # Display sample transactions
    logger.info("\nSample transactions:")
    for idx, row in tx_log_df.iterrows():
        if row['status'] == 'success' and pd.notna(row.get('tx_hash')):
            logger.info(f"  {row['artifact_filename']}")
            logger.info(f"    Tx: {row['tx_hash']}")
            logger.info(f"    Record ID: {row['record_id']}")
            logger.info(f"    Block: {row['block_number']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

