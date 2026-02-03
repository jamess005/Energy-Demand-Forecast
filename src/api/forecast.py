"""
Energy Demand Forecasting Pipeline
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from prepare_data import get_db_engine, load_historical_data, generate_features, generate_future_features, ensure_model_features

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from bias_correction import apply_bias_correction

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / 'experiments/main/model/tft-main.ckpt'
PREDICTION_LENGTH = 24
USE_BIAS_CORRECTION = True


def extract_model_features(model) -> list:
    """Extract required feature names from model checkpoint."""
    exclude = {'encoder_length', 'target_demand_center', 'target_demand_scale', 'relative_time_idx'}
    return [f for f in model.hparams.x_reals if f not in exclude]


def load_model(model_path: Path):
    """Load TFT model from checkpoint."""
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def forecast(hours: int = 24, forecast_from_date: str = None, apply_correction: bool = True) -> pd.DataFrame:
    """Generate demand forecast."""
    model = load_model(MODEL_PATH)
    encoder_length = model.hparams.max_encoder_length
    model_features = extract_model_features(model)
    
    engine = get_db_engine()
    historical_df = load_historical_data(engine, forecast_from_date, limit=encoder_length + 200)
    historical_df = generate_features(historical_df)
    
    future_df = generate_future_features(historical_df, hours)
    
    keep_rows = min(len(historical_df), encoder_length + 200)
    recent_hist = historical_df.tail(keep_rows).copy()
    combined = pd.concat([recent_hist, future_df], ignore_index=True)
    combined['time_idx'] = range(len(combined))
    combined['group'] = 'Germany'
    
    combined = ensure_model_features(combined, model_features)
    
    if 'season' in combined.columns:
        combined['season'] = combined['season'].astype(str)
    
    min_idx = len(combined) - hours - encoder_length
    
    pred_dataset = TimeSeriesDataSet(
        combined[combined['time_idx'] >= min_idx],
        time_idx='time_idx',
        target='target_demand',
        group_ids=['group'],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=PREDICTION_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=['group'],
        time_varying_known_categoricals=['season'] if 'season' in combined.columns else [],
        time_varying_known_reals=model_features,
        time_varying_unknown_reals=[],
        target_normalizer=GroupNormalizer(groups=['group'], transformation='softplus', center=True),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    pred_loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    
    with torch.no_grad():
        predictions = model.predict(pred_loader, mode='prediction')
    
    preds = predictions.cpu().numpy()
    last_preds = preds[-1] if len(preds.shape) > 1 else preds
    last_preds = last_preds[:hours]
    
    results = future_df[['timestamp']].copy()
    results['predicted_demand'] = last_preds
    
    if apply_correction and USE_BIAS_CORRECTION:
        results = apply_bias_correction(results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate energy demand forecast')
    parser.add_argument('--hours', type=int, default=24, help='Hours to forecast')
    parser.add_argument('--date', type=str, help='Forecast from date (DD/MM/YYYY)')
    parser.add_argument('--no-correction', action='store_true', help='Disable bias correction')
    args = parser.parse_args()
    
    results = forecast(
        hours=args.hours, 
        forecast_from_date=args.date, 
        apply_correction=not args.no_correction
    )
    
    print(f"\n{'='*60}")
    print(f"FORECAST: {args.hours} hours from {args.date or 'latest data'}")
    print(f"Model: {MODEL_PATH.stem}")
    print(f"{'='*60}\n")
    
    if 'corrected_demand' in results.columns:
        display = results[['timestamp', 'predicted_demand', 'bias_correction', 'corrected_demand']].copy()
        display.columns = ['Timestamp', 'Raw (MW)', 'Correction', 'Final (MW)']
        print(f"Raw range: {results['predicted_demand'].min():.0f} - {results['predicted_demand'].max():.0f} MW")
        print(f"Corrected range: {results['corrected_demand'].min():.0f} - {results['corrected_demand'].max():.0f} MW")
    else:
        display = results[['timestamp', 'predicted_demand']].copy()
        display.columns = ['Timestamp', 'Prediction (MW)']
        print(f"Range: {results['predicted_demand'].min():.0f} - {results['predicted_demand'].max():.0f} MW")
    
    print(f"\n{display.to_string(index=False)}\n")
    return results


if __name__ == '__main__':
    main()
