"""
Energy Demand Forecasting Pipeline — LSTM Seq2Seq v2

Loads the LSTM model, feature scaler, and normalisation stats from the
training/models directory.  Generates 24h-ahead forecasts with optional
bias correction.
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')

BASE_DIR   = Path(__file__).parent.parent.parent
MODEL_DIR  = BASE_DIR / 'training' / 'models'
TRAIN_DIR  = BASE_DIR / 'training'

# Add training/ and src/api/ to path so we can import model & API classes
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from train import (                        # noqa: E402
    LSTMSeq2Seq, HParams, FeatureScaler,
    ALL_FEATURES, DECODER_FEATURES,
    ENCODER_FEATURE_DIM, DECODER_FEATURE_DIM,
    DEFAULT_HP,
)
from prepare_data import (                 # noqa: E402
    get_db_engine, load_historical_data,
    generate_features, generate_future_features,
    ensure_model_features,
)
from bias_correction import apply_bias_correction  # noqa: E402

PREDICTION_LENGTH  = 24
USE_BIAS_CORRECTION = True


# ── Model discovery ──────────────────────────────────────────────────────────

def _find_best_checkpoint() -> Path:
    """Find the LSTM checkpoint with the lowest val_MAE in its filename."""
    ckpts = sorted(MODEL_DIR.glob('lstm-*.ckpt'))
    if not ckpts:
        raise FileNotFoundError(
            f"No LSTM checkpoints in {MODEL_DIR}. Run training/train.py first."
        )
    best, best_mae = ckpts[0], float('inf')
    for p in ckpts:
        m = re.search(r'val_MAE=(\d+)', p.stem)
        if m:
            mae = int(m.group(1))
            if mae < best_mae:
                best, best_mae = p, mae
    return best


# ── Model & artefact loading ─────────────────────────────────────────────────

def _load_artefacts():
    """Load norm stats, feature scaler, and the LSTM model."""
    # Norm stats (demand mean/std)
    with open(MODEL_DIR / 'lstm_norm_stats.json') as f:
        ns = json.load(f)
    demand_mean = ns['demand_mean']
    demand_std  = ns['demand_std']

    # Feature scaler
    with open(MODEL_DIR / 'lstm_feature_scaler.json') as f:
        sc = json.load(f)
    scaler = FeatureScaler()
    scaler.means = sc['means']
    scaler.stds  = sc['stds']

    # Hyperparams & model
    hp = DEFAULT_HP
    ckpt_path = _find_best_checkpoint()
    model = LSTMSeq2Seq.load_from_checkpoint(
        str(ckpt_path), hp=hp, demand_std=demand_std,
    )
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model, scaler, demand_mean, demand_std, hp, device, ckpt_path


# ── Forecast generation ──────────────────────────────────────────────────────

def forecast(
    hours: int = 24,
    forecast_from_date: Optional[str] = None,
    apply_correction: bool = True,
) -> pd.DataFrame:
    """Generate demand forecast.

    1. Load historical data from the database.
    2. Build encoder context (96 h of known features + demand).
    3. Build decoder context (24 h of known-future features).
    4. Run the LSTM Seq2Seq model to get normalised predictions.
    5. De-normalise to MW and optionally apply bias correction.
    """
    model, scaler, demand_mean, demand_std, hp, device, ckpt_path = (
        _load_artefacts()
    )
    encoder_length = hp.encoder_length

    # ── Pull data ────────────────────────────────────────────────────────────
    engine = get_db_engine()
    historical_df = load_historical_data(
        engine, forecast_from_date, limit=encoder_length + 200,
    )
    historical_df = generate_features(historical_df)
    future_df     = generate_future_features(historical_df, hours)

    # We need encoder_length rows of history + hours rows of future
    recent_hist = historical_df.tail(encoder_length).copy()
    combined    = pd.concat([recent_hist, future_df], ignore_index=True)

    # ── Ensure all required features exist ───────────────────────────────────
    combined = ensure_model_features(combined, ALL_FEATURES)

    # ── Normalise features ───────────────────────────────────────────────────
    enc_feats = scaler.transform(combined, ALL_FEATURES)        # (N, 22)
    dec_feats = scaler.transform(combined, DECODER_FEATURES)    # (N, 20)

    # Normalised demand for encoder only (history window)
    demand_vals = np.asarray(combined['target_demand'].values, dtype=np.float32)
    demand_norm = ((demand_vals - demand_mean) / demand_std).reshape(-1, 1)

    # Encoder input = [features, demand_norm]
    enc_input = np.concatenate([enc_feats, demand_norm], axis=1)  # (N, 23)

    # ── Build windows ────────────────────────────────────────────────────────
    # Encoder window: first encoder_length rows (historical)
    enc_x = torch.from_numpy(
        enc_input[:encoder_length][np.newaxis]              # (1, enc_len, 23)
    ).to(device)
    # Decoder window: next hours rows (future)
    dec_x = torch.from_numpy(
        dec_feats[encoder_length : encoder_length + hours][np.newaxis]  # (1, hours, 20)
    ).to(device)

    # ── Inference ────────────────────────────────────────────────────────────
    with torch.no_grad():
        preds_norm = model(enc_x, dec_x)            # (1, hours)

    preds_mw = preds_norm.cpu().numpy().flatten() * demand_std + demand_mean

    # ── Build result DataFrame ───────────────────────────────────────────────
    results = future_df[['timestamp']].head(hours).copy()
    results['predicted_demand'] = preds_mw[:hours]

    if apply_correction and USE_BIAS_CORRECTION:
        results = apply_bias_correction(results)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate energy demand forecast')
    parser.add_argument('--hours', type=int, default=24, help='Hours to forecast')
    parser.add_argument('--date', type=str, help='Forecast from date (DD/MM/YYYY)')
    parser.add_argument('--no-correction', action='store_true',
                        help='Disable bias correction')
    args = parser.parse_args()

    results = forecast(
        hours=args.hours,
        forecast_from_date=args.date,
        apply_correction=not args.no_correction,
    )

    ckpt = _find_best_checkpoint()
    print(f"\n{'=' * 60}")
    print(f"FORECAST: {args.hours} hours from {args.date or 'latest data'}")
    print(f"Model: {ckpt.stem}")
    print(f"{'=' * 60}\n")

    if 'corrected_demand' in results.columns:
        display = results[['timestamp', 'predicted_demand',
                           'bias_correction', 'corrected_demand']].copy()
        display.columns = ['Timestamp', 'Raw (MW)', 'Correction', 'Final (MW)']
        print(f"Raw range      : {results['predicted_demand'].min():.0f}"
              f" – {results['predicted_demand'].max():.0f} MW")
        print(f"Corrected range: {results['corrected_demand'].min():.0f}"
              f" – {results['corrected_demand'].max():.0f} MW")
    else:
        display = results[['timestamp', 'predicted_demand']].copy()
        display.columns = ['Timestamp', 'Prediction (MW)']
        print(f"Range: {results['predicted_demand'].min():.0f}"
              f" – {results['predicted_demand'].max():.0f} MW")

    print(f"\n{display.to_string(index=False)}\n")
    return results


if __name__ == '__main__':
    main()
