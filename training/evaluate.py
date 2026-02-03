"""
TFT Model Complete Analysis
===========================
Comprehensive evaluation with heatmaps, feature importance, and diagnostic plots.
Supports CLI arguments for model/data selection.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Enable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import argparse
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

sns.set_palette("husl")
plt.style.use('seaborn-v0_8-whitegrid')
torch.set_float32_matmul_precision('medium')

# =============================================================================
# DEFAULTS - use relative paths from script location
# =============================================================================
BASE_DIR = Path(__file__).parent.parent  # tftproj root
DEFAULT_MODEL = BASE_DIR / 'experiments/main/model/tft-main.ckpt'
DEFAULT_DATA = BASE_DIR / 'experiments/main/data/tft_training_data-main.csv'
DEFAULT_OUTPUT = BASE_DIR / 'experiments/main/analysis'

ENCODER_LENGTH = 96
PREDICTION_LENGTH = 24
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10


def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive TFT Model Evaluation')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL),
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default=str(DEFAULT_DATA),
                        help='Path to training data CSV')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT),
                        help='Output directory for results')
    parser.add_argument('--skip-importance', action='store_true',
                        help='Skip model-based feature importance (use if GPU crashes)')
    return parser.parse_args()


def get_file_stamp(model_path: Path) -> str:
    """Extract version and MAE from model filename for output naming."""
    ver_match = re.search(r'tft-(v?\d+|main)', model_path.name)
    mae_match = re.search(r'val_MAE=(\d+)', model_path.name)
    version = ver_match.group(1) if ver_match else 'model'
    mae = f"_MAE{mae_match.group(1)}" if mae_match else ''
    return f"{version}{mae}"


def load_model(model_path: Path):
    """Load TFT model from checkpoint."""
    print(f"Loading model: {model_path.name}")
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def get_features(model) -> list:
    """Extract feature names from model checkpoint."""
    exclude = {'encoder_length', 'target_demand_center', 'target_demand_scale', 'relative_time_idx'}
    return [f for f in model.hparams.x_reals if f not in exclude]


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare training data."""
    print(f"Loading data: {data_path.name}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['time_idx'] = range(len(df))
    df['group'] = 'Germany'
    df['season'] = df['season'].astype(str)
    return df


def create_dataset(df: pd.DataFrame, features: list) -> TimeSeriesDataSet:
    """Create TimeSeriesDataSet for training reference."""
    train_end = int(len(df) * TRAIN_RATIO)
    train_df = df[:train_end].copy()
    
    return TimeSeriesDataSet(
        train_df,
        time_idx='time_idx',
        target='target_demand',
        group_ids=['group'],
        min_encoder_length=ENCODER_LENGTH,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=PREDICTION_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=['group'],
        time_varying_known_categoricals=['season'],
        time_varying_known_reals=features,
        time_varying_unknown_reals=[],
        target_normalizer=GroupNormalizer(groups=['group'], transformation='softplus', center=True),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def generate_predictions(model, dataset, df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions on test set."""
    val_end = int(len(df) * (TRAIN_RATIO + VAL_RATIO))
    
    test_dataset = TimeSeriesDataSet.from_dataset(dataset, df, predict=False, stop_randomization=True)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=256, num_workers=0)
    
    print(f"Dataset: {len(test_dataset):,} sequences")
    
    print("Generating predictions...")
    with torch.no_grad():
        predictions = model.predict(test_loader, mode="prediction", return_x=True)
    
    pred_values = predictions.output.cpu().numpy()
    print(f"Generated: {len(pred_values):,} sequences")
    
    results = []
    pred_idx = 0
    
    for x, y in test_loader:
        batch_size = y[0].shape[0]
        decoder_idx = x['decoder_time_idx'].cpu().numpy()
        actuals = y[0].cpu().numpy()
        
        for i in range(batch_size):
            start_idx = int(decoder_idx[i, 0])
            
            if start_idx < val_end or start_idx >= len(df):
                pred_idx += 1
                continue
            
            ts_start = df.loc[start_idx, 'timestamp']
            
            for h in range(PREDICTION_LENGTH):
                ts = ts_start + pd.Timedelta(hours=h)
                results.append({
                    'timestamp': ts,
                    'hour': ts.hour,
                    'dow': ts.dayofweek,
                    'month': ts.month,
                    'week': ts.isocalendar()[1],
                    'horizon': h + 1,
                    'predicted': pred_values[pred_idx, h],
                    'actual': actuals[i, h],
                    'error': pred_values[pred_idx, h] - actuals[i, h],
                    'abs_error': abs(pred_values[pred_idx, h] - actuals[i, h]),
                })
            pred_idx += 1
    
    return pd.DataFrame(results)


def print_metrics(results_df: pd.DataFrame) -> tuple:
    """Print and return evaluation metrics."""
    mae = results_df['abs_error'].mean()
    rmse = np.sqrt((results_df['error']**2).mean())
    bias = results_df['error'].mean()
    
    print(f"\nTest Performance:")
    print(f"  MAE:  {mae:,.0f} MW")
    print(f"  RMSE: {rmse:,.0f} MW")
    print(f"  Bias: {bias:+,.0f} MW")
    
    return mae, rmse, bias


def create_time_aggregations(results_df: pd.DataFrame) -> dict:
    """Create time-based aggregations."""
    hourly = results_df.groupby('hour').agg({'abs_error': 'mean', 'error': 'mean'}).round(0)
    hourly.columns = ['mae', 'bias']
    hourly = hourly.reset_index()
    
    daily = results_df.groupby('dow').agg({'abs_error': 'mean', 'error': 'mean'}).round(0)
    daily.columns = ['mae', 'bias']
    daily = daily.reset_index()
    all_days = pd.DataFrame({'dow': range(7)})
    daily = all_days.merge(daily, on='dow', how='left')
    
    monthly = results_df.groupby('month').agg({'abs_error': 'mean', 'error': 'mean'}).round(0)
    monthly.columns = ['mae', 'bias']
    monthly = monthly.reset_index()
    
    weekly = results_df.groupby('week').agg({'abs_error': 'mean', 'error': 'mean'}).round(0)
    weekly.columns = ['mae', 'bias']
    weekly = weekly.reset_index()
    
    horizon = results_df.groupby('horizon').agg({'abs_error': ['mean', 'std'], 'error': 'mean'}).round(0)
    horizon.columns = ['mae', 'std', 'bias']
    horizon = horizon.reset_index()
    
    return {
        'hourly': hourly,
        'daily': daily,
        'monthly': monthly,
        'weekly': weekly,
        'horizon': horizon
    }


def plot_error_breakdown(aggs: dict, mae: float, bias: float, output_dir: Path, file_stamp: str):
    """Create comprehensive error breakdown plots."""
    print("Creating error breakdown plots...")
    
    hourly = aggs['hourly']
    daily = aggs['daily']
    monthly = aggs['monthly']
    horizon = aggs['horizon']
    
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Hourly MAE
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(hourly['hour'], hourly['mae'], marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.axhline(mae, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Overall: {mae:.0f}')
    ax.set_xlabel('Hour', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE (MW)', fontsize=11, fontweight='bold')
    ax.set_title('MAE by Hour', fontsize=13, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Daily MAE
    ax = fig.add_subplot(gs[0, 1])
    colors = ['steelblue']*5 + ['coral']*2
    daily_mae = daily['mae'].fillna(0)
    ax.bar(range(7), daily_mae, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(mae, color='red', linestyle='--', linewidth=2, alpha=0.7)
    for i, val in enumerate(daily['mae']):
        if pd.notna(val) and val > 0:
            ax.text(i, val, f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE (MW)', fontsize=11, fontweight='bold')
    ax.set_title('MAE by Day', fontsize=13, fontweight='bold')
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.grid(axis='y', alpha=0.3)
    
    # Hourly Bias
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(hourly['hour'], hourly['bias'], marker='s', linewidth=2, markersize=8, color='coral')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axhline(bias, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Hour', fontsize=11, fontweight='bold')
    ax.set_ylabel('Bias (MW)', fontsize=11, fontweight='bold')
    ax.set_title('Bias by Hour', fontsize=13, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    
    # Daily Bias
    ax = fig.add_subplot(gs[1, 1])
    daily_bias = daily['bias'].fillna(0)
    colors_bias = ['green' if b > 0 else 'red' if b < 0 else 'gray' for b in daily_bias]
    ax.bar(range(7), daily_bias, color=colors_bias, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    for i, val in enumerate(daily['bias']):
        if pd.notna(val) and abs(val) > 0:
            ax.text(i, val, f'{int(val):+d}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold', fontsize=9)
    ax.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
    ax.set_ylabel('Bias (MW)', fontsize=11, fontweight='bold')
    ax.set_title('Bias by Day', fontsize=13, fontweight='bold')
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.grid(axis='y', alpha=0.3)
    
    # Monthly MAE
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(monthly['month'], monthly['mae'], color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(mae, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE (MW)', fontsize=11, fontweight='bold')
    ax.set_title('MAE by Month', fontsize=13, fontweight='bold')
    ax.set_xticks(monthly['month'])
    ax.set_xticklabels([month_names[int(m)] for m in monthly['month']])
    ax.grid(axis='y', alpha=0.3)
    
    # Horizon MAE
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(horizon['horizon'], horizon['mae'], marker='o', linewidth=2, markersize=8, color='green')
    ax.fill_between(horizon['horizon'], horizon['mae'] - horizon['std'], horizon['mae'] + horizon['std'], 
                    color='green', alpha=0.2)
    ax.axhline(mae, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE (MW)', fontsize=11, fontweight='bold')
    ax.set_title('MAE by Horizon', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.suptitle('Error Breakdown by Time Dimensions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'error_breakdown_{file_stamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: error_breakdown_{file_stamp}.png")


def plot_heatmaps(results_df: pd.DataFrame, output_dir: Path, file_stamp: str):
    """Create MAE and Bias heatmaps by hour and day of week."""
    print("Creating heatmaps...")
    
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # MAE Heatmap
    mae_pivot = results_df.pivot_table(values='abs_error', index='hour', columns='dow', aggfunc='mean')
    ax = axes[0]
    sns.heatmap(mae_pivot, cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'MAE (MW)'}, ax=ax)
    ax.set_title('MAE by Hour and Day of Week', fontsize=14, fontweight='bold')
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Hour', fontsize=12)
    ax.set_xticklabels(dow_names, rotation=45)
    
    # Bias Heatmap
    bias_pivot = results_df.pivot_table(values='error', index='hour', columns='dow', aggfunc='mean')
    ax = axes[1]
    max_abs = max(abs(bias_pivot.min().min()), abs(bias_pivot.max().max()))
    sns.heatmap(bias_pivot, cmap='RdBu_r', center=0, annot=True, fmt='.0f', 
                cbar_kws={'label': 'Bias (MW)'}, ax=ax, vmin=-max_abs, vmax=max_abs)
    ax.set_title('Bias by Hour and Day of Week', fontsize=14, fontweight='bold')
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Hour', fontsize=12)
    ax.set_xticklabels(dow_names, rotation=45)
    
    plt.suptitle('Error Heatmaps - MAE and Bias by Hour/Day', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'heatmaps_{file_stamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: heatmaps_{file_stamp}.png")


def plot_weekly_breakdown(aggs: dict, mae: float, bias: float, output_dir: Path, file_stamp: str):
    """Create weekly performance breakdown."""
    weekly = aggs['weekly']
    
    if len(weekly) <= 1:
        return
    
    print("Creating weekly breakdown...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    ax = axes[0]
    ax.bar(weekly['week'], weekly['mae'], color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(mae, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Week', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE (MW)', fontsize=11, fontweight='bold')
    ax.set_title('MAE by Week', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1]
    colors_week = ['green' if b > 0 else 'red' for b in weekly['bias']]
    ax.bar(weekly['week'], weekly['bias'], color=colors_week, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axhline(bias, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Week', fontsize=11, fontweight='bold')
    ax.set_ylabel('Bias (MW)', fontsize=11, fontweight='bold')
    ax.set_title('Bias by Week', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Weekly Performance - Test Set', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'weekly_{file_stamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: weekly_{file_stamp}.png")


def analyze_feature_importance(model, dataset, df: pd.DataFrame, features: list, 
                                output_dir: Path, file_stamp: str,
                                skip_model_interpretation: bool = False):
    """Extract and plot feature importance and correlations."""
    print("\nExtracting feature correlations...")
    
    val_end = int(len(df) * (TRAIN_RATIO + VAL_RATIO))
    
    try:
        # Calculate correlations (doesn't require GPU)
        test_data = df[val_end:].copy()
        correlations = []
        for feat in features:
            if feat in test_data.columns:
                corr = test_data[feat].corr(test_data['target_demand'])
                correlations.append({
                    'feature': feat,
                    'correlation': abs(corr),
                    'correlation_raw': corr
                })
        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        
        # Save correlation analysis
        corr_df.to_csv(output_dir / f'feature_correlations_{file_stamp}.csv', index=False)
        print(f"✓ Saved: feature_correlations_{file_stamp}.csv")
        
        # Plot correlation only
        fig, ax = plt.subplots(figsize=(12, max(10, len(corr_df)*0.3)))
        all_corr = corr_df.sort_values('correlation_raw', key=abs, ascending=False).iloc[::-1]
        colors_corr = ['green' if c > 0 else 'red' for c in all_corr['correlation_raw']]
        ax.barh(range(len(all_corr)), all_corr['correlation_raw'], color=colors_corr, alpha=0.8)
        ax.set_yticks(range(len(all_corr)))
        ax.set_yticklabels(all_corr['feature'], fontsize=8)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('Correlation with Target', fontsize=11, fontweight='bold')
        ax.set_title('Feature Correlations with Target Demand', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_correlations_{file_stamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: feature_correlations_{file_stamp}.png")
        
        if skip_model_interpretation:
            print("Model interpretation skipped (GPU stability)")
            return
        
        # Model-based importance (may fail on some GPUs)
        print("Extracting model-based feature importance...")
        max_rows = 50 * (ENCODER_LENGTH + PREDICTION_LENGTH)
        sample_start = val_end
        sample_end = min(val_end + max_rows, len(df))
        sample_df = df.iloc[sample_start:sample_end].copy().reset_index(drop=True)
        
        if len(sample_df) < ENCODER_LENGTH + PREDICTION_LENGTH:
            print("Not enough data for feature importance analysis")
            return
        
        sample_df['time_idx'] = range(len(sample_df))
        
        sample_dataset = TimeSeriesDataSet.from_dataset(
            dataset, sample_df, predict=False, stop_randomization=True
        )
        
        n_seq = len(sample_dataset)
        print(f"Using {n_seq} sequences for importance analysis...")
        
        if n_seq == 0:
            print("No valid sequences for importance analysis")
            return
        
        # Use small batch size for GPU stability
        batch_size = min(16, n_seq)
        sample_loader = sample_dataset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0, shuffle=False
        )
        
        with torch.no_grad():
            raw_predictions = model.predict(sample_loader, mode="raw", return_x=False)
        
        interpretation = model.interpret_output(raw_predictions, reduction='sum')
        
        importance = interpretation['encoder_variables']
        if torch.is_tensor(importance):
            importance = importance.cpu().numpy()
        
        var_names = list(model.hparams.x_reals)
        
        if importance.ndim > 1:
            importance = importance.mean(axis=0)
        
        n_vars = min(len(var_names), len(importance))
        importance_df = pd.DataFrame({
            'feature': var_names[:n_vars],
            'importance': importance[:n_vars]
        }).sort_values('importance', ascending=False)
        
        # Calculate correlations
        test_data = df[val_end:].copy()
        correlations = []
        for feat in features:
            if feat in test_data.columns:
                corr = test_data[feat].corr(test_data['target_demand'])
                correlations.append({
                    'feature': feat,
                    'correlation': abs(corr),
                    'correlation_raw': corr
                })
        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        
        # Merge and save
        feature_analysis = importance_df.merge(
            corr_df[['feature', 'correlation', 'correlation_raw']], 
            on='feature', how='left'
        )
        feature_analysis.to_csv(output_dir / f'feature_analysis_{file_stamp}.csv', index=False)
        print(f"✓ Saved: feature_analysis_{file_stamp}.csv")
        
        # Plot feature importance and correlation
        fig, axes = plt.subplots(1, 2, figsize=(20, max(12, len(importance_df)*0.3)))
        
        ax = axes[0]
        all_imp = importance_df.iloc[::-1]
        colors_imp = ['steelblue' if 'lag' in f else 'coral' if 'temp' in f or 'heat' in f else 'green' 
                      for f in all_imp['feature']]
        ax.barh(range(len(all_imp)), all_imp['importance'], color=colors_imp, alpha=0.8)
        ax.set_yticks(range(len(all_imp)))
        ax.set_yticklabels(all_imp['feature'], fontsize=8)
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_title('All Feature Importance', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        ax = axes[1]
        all_corr = corr_df.sort_values('correlation_raw', key=abs, ascending=False).iloc[::-1]
        colors_corr = ['green' if c > 0 else 'red' for c in all_corr['correlation_raw']]
        ax.barh(range(len(all_corr)), all_corr['correlation_raw'], color=colors_corr, alpha=0.8)
        ax.set_yticks(range(len(all_corr)))
        ax.set_yticklabels(all_corr['feature'], fontsize=8)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('Correlation with Target', fontsize=11, fontweight='bold')
        ax.set_title('All Feature Correlations', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Feature Analysis - Importance & Correlation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_importance_{file_stamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: feature_importance_{file_stamp}.png")
        
        # Feature categories
        def categorize_feature(name):
            if 'lag' in name or 'rolling' in name:
                return 'Demand Lags'
            elif 'temp' in name or 'heat' in name or 'cold' in name:
                return 'Weather'
            elif any(x in name for x in ['hour', 'dow', 'month', 'day_', 'weekend', 'weekday', 'holiday']):
                return 'Temporal'
            elif 'regime' in name:
                return 'Regime'
            elif 'interaction' in name or 'product' in name:
                return 'Interactions'
            else:
                return 'Other'
        
        importance_df['category'] = importance_df['feature'].apply(categorize_feature)
        
        category_stats = importance_df.groupby('category').agg({
            'importance': ['sum', 'mean', 'count']
        }).round(4)
        category_stats.columns = ['total_importance', 'avg_importance', 'count']
        category_stats = category_stats.sort_values('total_importance', ascending=False).reset_index()
        
        print("\nFeature Categories:")
        print(category_stats.to_string(index=False))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors_cat = ['steelblue', 'coral', 'green', 'purple', 'orange', 'brown']
        
        ax = axes[0]
        ax.barh(range(len(category_stats)), category_stats['total_importance'], 
                color=colors_cat[:len(category_stats)], alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(category_stats)))
        ax.set_yticklabels(category_stats['category'], fontsize=10, fontweight='bold')
        ax.set_xlabel('Total Importance', fontsize=11, fontweight='bold')
        ax.set_title('Feature Importance by Category', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        ax = axes[1]
        ax.barh(range(len(category_stats)), category_stats['count'], 
                color=colors_cat[:len(category_stats)], alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(category_stats)))
        ax.set_yticklabels(category_stats['category'], fontsize=10, fontweight='bold')
        ax.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
        ax.set_title('Feature Count by Category', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Feature Category Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_categories_{file_stamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: feature_categories_{file_stamp}.png")
        
    except Exception as e:
        print(f"Feature importance failed: {e}")
        traceback.print_exc()


def save_report(results_df: pd.DataFrame, aggs: dict, mae: float, rmse: float, bias: float,
                output_dir: Path, file_stamp: str):
    """Save text report with metrics and insights."""
    report_path = output_dir / f'report_{file_stamp}.txt'
    
    hourly = aggs['hourly']
    daily = aggs['daily']
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TFT MODEL ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"MAE:  {mae:10,.0f} MW\n")
        f.write(f"RMSE: {rmse:10,.0f} MW\n")
        f.write(f"Bias: {bias:+10,.0f} MW\n\n")
        
        f.write("BY HOUR\n")
        f.write("-"*80 + "\n")
        for _, row in hourly.iterrows():
            f.write(f"  {int(row['hour']):02d}:00 - MAE: {row['mae']:6.0f}, Bias: {row['bias']:+7.0f}\n")
        f.write("\n")
        
        f.write("BY DAY\n")
        f.write("-"*80 + "\n")
        for _, row in daily.iterrows():
            if pd.notna(row['mae']):
                f.write(f"  {dow_names[int(row['dow'])]:3s} - MAE: {row['mae']:6.0f}, Bias: {row['bias']:+7.0f}\n")
        f.write("\n")
        
        f.write("KEY INSIGHTS\n")
        f.write("-"*80 + "\n")
        worst_hour = hourly.loc[hourly['mae'].idxmax()]
        best_hour = hourly.loc[hourly['mae'].idxmin()]
        f.write(f"Worst hour: {int(worst_hour['hour']):02d}:00 (MAE: {worst_hour['mae']:.0f} MW)\n")
        f.write(f"Best hour:  {int(best_hour['hour']):02d}:00 (MAE: {best_hour['mae']:.0f} MW)\n")
    
    print(f"✓ Saved: report_{file_stamp}.txt")


def main():
    args = parse_args()
    
    model_path = Path(args.model)
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    file_stamp = get_file_stamp(model_path)
    
    print("="*80)
    print("COMPREHENSIVE TFT ANALYSIS")
    print("="*80)
    print(f"Model:  {model_path.name}")
    print(f"Data:   {data_path.name}")
    print(f"Output: {output_dir}")
    print(f"Stamp:  {file_stamp}")
    print("="*80 + "\n")
    
    # Load model and data
    model = load_model(model_path)
    features = get_features(model)
    print(f"Features from checkpoint: {len(features)}")
    
    df = load_data(data_path)
    print(f"Records: {len(df):,}")
    
    # Validate features
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"\nMissing features: {missing[:5]}...")
        features = [f for f in features if f in df.columns]
        print(f"Using {len(features)} available features")
    
    # Create dataset and generate predictions
    dataset = create_dataset(df, features)
    results_df = generate_predictions(model, dataset, df)
    
    if len(results_df) == 0:
        print("No test predictions generated!")
        return
    
    print(f"\nResults: {len(results_df):,} predictions")
    print(f"Period: {results_df['timestamp'].min().date()} to {results_df['timestamp'].max().date()}")
    
    # Merge with features for analysis
    results_df = results_df.merge(df[['timestamp'] + features], on='timestamp', how='left')
    
    # Metrics
    mae, rmse, bias = print_metrics(results_df)
    
    # Aggregations
    aggs = create_time_aggregations(results_df)
    
    # Plots
    plot_error_breakdown(aggs, mae, bias, output_dir, file_stamp)
    plot_heatmaps(results_df, output_dir, file_stamp)
    plot_weekly_breakdown(aggs, mae, bias, output_dir, file_stamp)
    
    # Feature importance (may fail on some GPU setups)
    try:
        analyze_feature_importance(model, dataset, df, features, output_dir, file_stamp,
                                   skip_model_interpretation=args.skip_importance)
    except Exception as e:
        print(f"Feature importance skipped: {e}")
    
    # Save report and predictions
    save_report(results_df, aggs, mae, rmse, bias, output_dir, file_stamp)
    results_df.to_csv(output_dir / f'predictions_{file_stamp}.csv', index=False)
    print(f"✓ Saved: predictions_{file_stamp}.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.glob(f'*{file_stamp}*')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
