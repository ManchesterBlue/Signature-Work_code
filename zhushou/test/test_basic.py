"""
Basic smoke tests for core functionality
"""

import pytest
import hashlib
import pandas as pd
from pathlib import Path


def test_hash_function():
    """Test that hash function is deterministic."""
    data = b"test data"
    hash1 = hashlib.sha256(data).hexdigest()
    hash2 = hashlib.sha256(data).hexdigest()
    assert hash1 == hash2


def test_indicators_file_exists():
    """Test that sample indicators file exists."""
    assert Path('data_sample/indicators.csv').exists()


def test_indicators_loads():
    """Test that indicators can be loaded as DataFrame."""
    df = pd.read_csv('data_sample/indicators.csv')
    assert len(df) > 0
    assert 'country' in df.columns
    assert 'year' in df.columns
    assert 'renewable_share' in df.columns


def test_contract_file_structure():
    """Test that contract file exists and has expected structure."""
    contract_path = Path('contracts/ClimateDataRegistry.sol')
    assert contract_path.exists()
    
    content = contract_path.read_text()
    assert 'contract ClimateDataRegistry' in content
    assert 'function addRecord' in content
    assert 'function getRecord' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

