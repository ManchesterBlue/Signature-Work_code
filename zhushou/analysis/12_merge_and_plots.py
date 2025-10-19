#!/usr/bin/env python3
"""
Analysis Step 3: Merge Data and Generate Visualizations
Inputs: Cleaned indicators, policy features, model results
Outputs: Plots (trends, feature importance, lag analysis) and HTML report
Note: Results are correlation-based, not causal relationships
"""

import os
import sys
import logging
import json
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/12_merge_and_plots.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_data() -> Dict[str, pd.DataFrame]:
    """Load all required data files."""
    logger.info("Loading data files")
    
    data = {}
    
    # Indicators
    indicators_path = 'output/cleaned_indicators.csv'
    if os.path.exists(indicators_path):
        data['indicators'] = pd.read_csv(indicators_path)
        logger.info(f"Loaded indicators: {len(data['indicators'])} records")
    
    # Policy features
    policy_path = 'output/policy_features.csv'
    if os.path.exists(policy_path):
        data['policy_features'] = pd.read_csv(policy_path)
        logger.info(f"Loaded policy features: {len(data['policy_features'])} records")
    
    # Model results
    results_path = 'output/model_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            data['model_results'] = json.load(f)
        logger.info("Loaded model results")
    
    return data


def merge_indicators_and_policy(indicators: pd.DataFrame, 
                                policy_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge indicator data with policy features.
    Left join to retain all indicator observations.
    """
    logger.info("Merging indicators with policy features")
    
    merged = pd.merge(
        indicators,
        policy_features[['country', 'year', 'policy_intensity', 
                        'total_policy_keywords', 'keyword_density']],
        on=['country', 'year'],
        how='left'
    )
    
    # Fill missing policy features with 0 (no policy document for that year)
    policy_cols = ['policy_intensity', 'total_policy_keywords', 'keyword_density']
    merged[policy_cols] = merged[policy_cols].fillna(0)
    
    logger.info(f"Merged dataset: {len(merged)} records")
    return merged


def plot_indicator_trends(df: pd.DataFrame, output_dir: str):
    """Plot time trends of key climate indicators by country."""
    logger.info("Generating indicator trend plots")
    
    indicators = ['renewable_share', 'co2_per_capita', 'installed_capacity']
    titles = ['Renewable Energy Share (%)', 'CO₂ per Capita (metric tons)', 
              'Installed Capacity (MW)']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (indicator, title) in enumerate(zip(indicators, titles)):
        ax = axes[idx]
        
        for country in df['country'].unique():
            country_data = df[df['country'] == country]
            ax.plot(country_data['year'], country_data[indicator], 
                   marker='o', label=country, linewidth=2)
        
        ax.set_xlabel('Year')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'indicator_trends.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {filepath}")
    plt.close()


def plot_feature_importance(model_results: Dict, output_dir: str):
    """Plot feature importance from RandomForest model."""
    logger.info("Generating feature importance plot")
    
    # Extract feature importance for renewable_share target
    target = 'renewable_share'
    if target not in model_results:
        logger.warning(f"No model results for {target}")
        return
    
    rf_importance = model_results[target]['random_forest']['feature_importance']
    
    # Convert to sorted list
    features = list(rf_importance.keys())
    importances = list(rf_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1][:10]  # Top 10
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(sorted_features)), sorted_importances, color='steelblue')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 10 Feature Importances (RandomForest - Renewable Share Prediction)')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (feat, imp) in enumerate(zip(sorted_features, sorted_importances)):
        ax.text(imp, i, f' {imp:.3f}', va='center')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {filepath}")
    plt.close()


def plot_policy_lag_analysis(df: pd.DataFrame, output_dir: str):
    """
    Plot lag analysis: policy intensity vs. next year renewable share change.
    Note: This shows correlation, not causation.
    """
    logger.info("Generating policy lag analysis plot")
    
    # Create lag features
    df_sorted = df.sort_values(['country', 'year']).copy()
    df_sorted['renewable_next_year'] = df_sorted.groupby('country')['renewable_share'].shift(-1)
    df_sorted['renewable_change'] = df_sorted['renewable_next_year'] - df_sorted['renewable_share']
    
    # Filter valid observations
    valid_data = df_sorted[
        (df_sorted['policy_intensity'] > 0) & 
        (df_sorted['renewable_change'].notna())
    ]
    
    if len(valid_data) == 0:
        logger.warning("No valid data for lag analysis")
        return
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot: Policy intensity vs renewable share change
    ax1 = axes[0]
    for country in valid_data['country'].unique():
        country_data = valid_data[valid_data['country'] == country]
        ax1.scatter(country_data['policy_intensity'], 
                   country_data['renewable_change'],
                   label=country, s=100, alpha=0.7)
    
    # Add trend line
    if len(valid_data) > 2:
        z = np.polyfit(valid_data['policy_intensity'], 
                      valid_data['renewable_change'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_data['policy_intensity'].min(), 
                             valid_data['policy_intensity'].max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend (slope={z[0]:.3f})')
    
    ax1.set_xlabel('Policy Intensity (current year)')
    ax1.set_ylabel('Renewable Share Change (next year, %)')
    ax1.set_title('Policy Intensity vs. Next-Year Renewable Growth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot by country
    ax2 = axes[1]
    country_list = valid_data['country'].unique()
    data_for_box = [valid_data[valid_data['country'] == c]['policy_intensity'].values 
                    for c in country_list]
    ax2.boxplot(data_for_box, labels=country_list)
    ax2.set_ylabel('Policy Intensity')
    ax2.set_title('Policy Intensity Distribution by Country')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'policy_lag_analysis.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {filepath}")
    plt.close()


def generate_html_report(data: Dict, output_dir: str):
    """Generate comprehensive HTML report with plots and results."""
    logger.info("Generating HTML report")
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Climate Data Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .header p {
            margin: 5px 0;
            opacity: 0.9;
        }
        .section {
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .metric-card h3 {
            margin: 0 0 5px 0;
            font-size: 14px;
            color: #666;
        }
        .metric-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .plot {
            width: 100%;
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 5px;
        }
        .disclaimer {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .disclaimer strong {
            color: #856404;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #667eea;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌍 Climate Data Analysis Report</h1>
        <p>Blockchain-Verified Green Energy Data Analytics</p>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="disclaimer">
        <strong>⚠️ Important Note:</strong> This analysis presents preliminary findings based on 
        correlation-based statistical models. The relationships shown do not imply causation. 
        Policy effects on renewable energy adoption are influenced by numerous confounding factors 
        including economic conditions, technological progress, and international commitments. 
        These results should be considered exploratory and require further validation with 
        comprehensive data and causal inference methods.
    </div>
    
    <div class="section">
        <h2>📊 Dataset Overview</h2>
        <div class="metrics">
            <div class="metric-card">
                <h3>Countries</h3>
                <div class="value">{n_countries}</div>
            </div>
            <div class="metric-card">
                <h3>Years Covered</h3>
                <div class="value">{year_range}</div>
            </div>
            <div class="metric-card">
                <h3>Data Points</h3>
                <div class="value">{n_records}</div>
            </div>
            <div class="metric-card">
                <h3>Policy Documents</h3>
                <div class="value">{n_policies}</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Indicator Trends</h2>
        <p>Time series of key climate and energy indicators across countries.</p>
        <img src="indicator_trends.png" alt="Indicator Trends" class="plot">
    </div>
    
    <div class="section">
        <h2>🤖 Predictive Model Performance</h2>
        <p>Machine learning models trained to predict renewable energy share based on historical data and features.</p>
        <table>
            <tr>
                <th>Model</th>
                <th>Test R²</th>
                <th>Test RMSE</th>
                <th>CV RMSE Mean</th>
            </tr>
            {model_performance_rows}
        </table>
        <img src="feature_importance.png" alt="Feature Importance" class="plot">
    </div>
    
    <div class="section">
        <h2>📜 Policy Analysis</h2>
        <p>Analysis of policy text intensity and correlation with renewable energy growth. 
        Note: Correlations shown do not establish causal relationships.</p>
        <img src="policy_lag_analysis.png" alt="Policy Lag Analysis" class="plot">
    </div>
    
    <div class="section">
        <h2>🔐 Data Provenance</h2>
        <p>All data files are cryptographically hashed (SHA-256), stored on IPFS, and registered 
        on a local Ethereum blockchain for transparency and reproducibility.</p>
        <ul>
            <li><strong>Blockchain:</strong> Local Hardhat PoA Network</li>
            <li><strong>Storage:</strong> IPFS (Content-Addressed)</li>
            <li><strong>Verification:</strong> Run <code>python scripts/verify_from_chain.py</code></li>
        </ul>
    </div>
    
    <div class="footer">
        <p>Green Energy Blockchain MVP | MIT License | 2025</p>
        <p>For verification, see <code>docs/README_onepager.md</code></p>
    </div>
</body>
</html>
"""
    
    # Fill in template values
    indicators = data.get('indicators', pd.DataFrame())
    policy_features = data.get('policy_features', pd.DataFrame())
    model_results = data.get('model_results', {})
    
    # Build model performance rows
    model_rows = ""
    if 'renewable_share' in model_results:
        for model_type in ['elasticnet', 'random_forest']:
            if model_type in model_results['renewable_share']:
                metrics = model_results['renewable_share'][model_type]['metrics']
                model_name = metrics.get('model_type', model_type)
                test_r2 = metrics.get('test_r2', 0)
                test_rmse = metrics.get('test_rmse', 0)
                cv_rmse = metrics.get('cv_rmse_mean', 0)
                
                model_rows += f"""
            <tr>
                <td>{model_name}</td>
                <td>{test_r2:.3f}</td>
                <td>{test_rmse:.3f}</td>
                <td>{cv_rmse:.3f}</td>
            </tr>
"""
    
    html_content = html_content.format(
        timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        n_countries=indicators['country'].nunique() if len(indicators) > 0 else 0,
        year_range=f"{indicators['year'].min()}-{indicators['year'].max()}" if len(indicators) > 0 else "N/A",
        n_records=len(indicators),
        n_policies=len(policy_features),
        model_performance_rows=model_rows
    )
    
    # Save HTML
    html_path = os.path.join(output_dir, 'report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Saved HTML report: {html_path}")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Analysis Step 3: Merge Data and Generate Plots")
    logger.info("="*60)
    
    os.makedirs('logs', exist_ok=True)
    figures_dir = 'docs/figures'
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    
    # Load all data
    data = load_data()
    
    if 'indicators' not in data:
        logger.error("Indicators data not found")
        return 1
    
    # Merge indicators with policy features
    if 'policy_features' in data:
        merged_df = merge_indicators_and_policy(
            data['indicators'], 
            data['policy_features']
        )
    else:
        logger.warning("Policy features not found, using indicators only")
        merged_df = data['indicators']
    
    # Generate plots
    plot_indicator_trends(merged_df, figures_dir)
    
    if 'model_results' in data:
        plot_feature_importance(data['model_results'], figures_dir)
    
    if 'policy_features' in data:
        plot_policy_lag_analysis(merged_df, figures_dir)
    
    # Generate HTML report
    generate_html_report(data, 'docs')
    
    logger.info("="*60)
    logger.info("Step 3 Complete - Plots and report generated")
    logger.info(f"Figures saved to: {figures_dir}/")
    logger.info(f"HTML report: docs/report.html")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

