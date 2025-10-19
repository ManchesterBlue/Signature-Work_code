#!/usr/bin/env python3
"""
Analysis Step 1: Baseline Predictive Models
Inputs: Cleaned climate indicators
Outputs: Model metrics, feature importances, predictions
Models: ElasticNet and RandomForest with cross-validation
"""

import os
import sys
import logging
import json
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/10_baseline_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def prepare_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for modeling.
    Creates lag features and temporal features.
    """
    logger.info(f"Preparing features for target: {target_col}")
    
    # Sort by country and year
    df = df.sort_values(['country', 'year']).reset_index(drop=True)
    
    # Create lag features (previous year values)
    feature_cols = ['renewable_share', 'co2_per_capita', 'installed_capacity']
    lag_features = {}
    
    for col in feature_cols:
        if col != target_col:
            # Lag 1 year
            df[f'{col}_lag1'] = df.groupby('country')[col].shift(1)
            # Year-over-year change
            df[f'{col}_change'] = df.groupby('country')[col].diff()
    
    # Temporal features
    df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    
    # One-hot encode country
    country_dummies = pd.get_dummies(df['country'], prefix='country')
    df = pd.concat([df, country_dummies], axis=1)
    
    # Select feature columns
    feature_columns = [col for col in df.columns if col not in 
                      ['country', 'year', target_col] and 
                      not col.startswith('Unnamed')]
    
    # Remove rows with NaN (from lagging)
    df_clean = df.dropna()
    
    X = df_clean[feature_columns]
    y = df_clean[target_col]
    
    logger.info(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Feature columns: {feature_columns}")
    
    return X, y, feature_columns


def train_elasticnet(X: pd.DataFrame, y: pd.Series, cv_folds: int = 3) -> Dict:
    """
    Train ElasticNet regression model with cross-validation.
    ElasticNet combines L1 and L2 regularization for feature selection and stability.
    """
    logger.info("Training ElasticNet model")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=RANDOM_SEED
    )
    
    # Train model
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=10000)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    metrics = {
        'model_type': 'ElasticNet',
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }
    
    # Cross-validation
    if cv_folds > 0 and len(X) > cv_folds:
        cv_scores = cross_val_score(
            model, X_scaled, y, cv=cv_folds, 
            scoring='neg_mean_squared_error'
        )
        metrics['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
        metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
    
    # Feature importances (coefficients)
    feature_importance = dict(zip(X.columns, model.coef_))
    
    logger.info(f"ElasticNet - Test R²: {metrics['test_r2']:.3f}, Test RMSE: {metrics['test_rmse']:.3f}")
    
    return {
        'metrics': metrics,
        'feature_importance': feature_importance,
        'model': model,
        'scaler': scaler
    }


def train_random_forest(X: pd.DataFrame, y: pd.Series, cv_folds: int = 3) -> Dict:
    """
    Train RandomForest regression model with cross-validation.
    RandomForest captures non-linear relationships and interactions.
    """
    logger.info("Training RandomForest model")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=5, 
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    metrics = {
        'model_type': 'RandomForest',
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }
    
    # Cross-validation
    if cv_folds > 0 and len(X) > cv_folds:
        cv_scores = cross_val_score(
            model, X, y, cv=cv_folds, 
            scoring='neg_mean_squared_error'
        )
        metrics['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
        metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
    
    # Feature importances
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    
    logger.info(f"RandomForest - Test R²: {metrics['test_r2']:.3f}, Test RMSE: {metrics['test_rmse']:.3f}")
    
    return {
        'metrics': metrics,
        'feature_importance': feature_importance,
        'model': model
    }


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Analysis Step 1: Baseline Predictive Models")
    logger.info("="*60)
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Load cleaned indicators
    data_path = 'output/cleaned_indicators.csv'
    if not os.path.exists(data_path):
        logger.error(f"Cleaned data not found: {data_path}")
        return 1
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} indicator records")
    
    # Store all results
    all_results = {}
    
    # Model renewable_share as target
    target = 'renewable_share'
    logger.info(f"\n{'='*60}")
    logger.info(f"Modeling target: {target}")
    logger.info(f"{'='*60}")
    
    X, y, feature_cols = prepare_features(df, target)
    
    # Train models
    elasticnet_results = train_elasticnet(X, y, cv_folds=3)
    rf_results = train_random_forest(X, y, cv_folds=3)
    
    all_results[target] = {
        'elasticnet': {
            'metrics': elasticnet_results['metrics'],
            'feature_importance': {k: float(v) for k, v in 
                                 sorted(elasticnet_results['feature_importance'].items(), 
                                       key=lambda x: abs(x[1]), reverse=True)[:10]}
        },
        'random_forest': {
            'metrics': rf_results['metrics'],
            'feature_importance': {k: float(v) for k, v in 
                                 sorted(rf_results['feature_importance'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]}
        }
    }
    
    # Save results
    results_path = 'output/model_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved model results to {results_path}")
    
    logger.info("="*60)
    logger.info("Step 1 Complete - Baseline models trained")
    logger.info(f"Target variable: {target}")
    logger.info(f"ElasticNet Test R²: {elasticnet_results['metrics']['test_r2']:.3f}")
    logger.info(f"RandomForest Test R²: {rf_results['metrics']['test_r2']:.3f}")
    logger.info("="*60)
    
    # Display top features
    logger.info("\nTop 5 features (RandomForest):")
    for feat, imp in list(rf_results['feature_importance'].items())[:5]:
        logger.info(f"  {feat}: {imp:.4f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

