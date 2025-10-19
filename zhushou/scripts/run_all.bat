@echo off
REM One-click setup and run script for Windows
REM Simplified version - runs core pipeline

echo ============================================================
echo Green Energy Blockchain MVP - Setup and Run
echo ============================================================
echo.

REM Create directories
if not exist "logs" mkdir logs
if not exist "output" mkdir output
if not exist "docs\figures" mkdir docs\figures

echo [1/8] Installing dependencies...
call npm install
pip install -r requirements.txt

echo.
echo [2/8] Running ETL Pipeline...
python etl/01_fetch.py
python etl/02_clean_and_hash.py

echo.
echo [3/8] Running Analysis...
python analysis/10_baseline_models.py
python analysis/11_policy_text_features.py
python analysis/12_merge_and_plots.py

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo To run the full blockchain pipeline:
echo   1. Start Hardhat: npm run node
echo   2. Deploy contract: npm run deploy
echo   3. Run: python etl/03_ipfs_add.py
echo   4. Run: python etl/04_register_onchain.py
echo   5. Verify: python scripts/verify_from_chain.py
echo.
echo Report available at: docs/report.html
echo ============================================================
pause

