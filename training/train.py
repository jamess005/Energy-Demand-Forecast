"""
TFT Training Script
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import matplotlib
matplotlib.use('Agg')

import argparse
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
import warnings
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_float32_matmul_precision('medium')
pl.seed_everything(SEED, workers=True)

# Configuration - use relative paths from script location
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'tft_training_data-main.csv'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DATASET_PATH = MODEL_DIR / 'training_dataset.pkl'

# Architecture
HIDDEN_SIZE = 32
HIDDEN_CONTINUOUS_SIZE = 16
ATTENTION_HEADS = 4
DROPOUT = 0.3
ENCODER_LENGTH = 48
PREDICTION_LENGTH = 24

# Training
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
GRADIENT_CLIP = 0.1
LR_PATIENCE = 3
LR_FACTOR = 2.0
LR_MIN = 1e-6
EARLY_STOP_PATIENCE = 7

# Split
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10


def load_data():
    """Load training data."""
    if not DATA_PATH.exists():
        available = sorted(DATA_PATH.parent.glob('tft_training_data-v*.csv'), reverse=True)
        if not available:
            raise FileNotFoundError("No training data found. Run complete_data.py first.")
        actual_path = available[0]
    else:
        actual_path = DATA_PATH
    
    df = pd.read_csv(actual_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['time_idx'] = range(len(df))
    df['group'] = 'Germany'
    df['season'] = df['season'].astype(str)
    return df


def create_datasets(df: pd.DataFrame):
    """Create train/val datasets."""
    train_end = int(len(df) * TRAIN_RATIO)
    val_end = int(len(df) * (TRAIN_RATIO + VAL_RATIO))
    
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end].copy()
    
    print(f"Data: {len(df):,} | Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(df) - val_end:,}")
    
    features = [
        'hour_sin', 'hour_cos', 'dow_sin', 'month_cos',
        'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6',
        'is_weekend', 'is_weekday', 'is_public_holiday',
        'is_monday_after_weekend', 'is_friday_before_weekend',
        'is_early_morning', 'is_morning_ramp', 'is_night',
        'day_transition_type', 'is_peak_hour', 'is_valley_hour', 'hour_squared',
        'daylight_savings_winter', 'daylight_savings_summer',
        'temperature', 'heating_demand', 'heating_demand_sq',
        'temp_severity', 'heating_demand_log', 'monday_cold_multiplier', 'is_cold',
        'regime_0', 'regime_1', 'regime_2', 'regime_3',
        'demand_lag_24h_norm', 'demand_lag_168h_norm', 'demand_delta_24h',
        'demand_lag_ratio', 'lag_reliability', 'demand_lag_adjusted',
        'transition_adjustment', 'demand_rolling_std_7d',
        'lag_24h_was_weekend', 'lag_168h_was_weekend',
        'demand_lag_24h_sq', 'demand_lag_168h_sq', 'demand_lag_24h_log', 'temp_lag_24h',
        'peak_lag_interaction', 'peak_heating_interaction', 'weekend_temp_interaction',
        'heating_hour_cos_product', 'temp_lag_ratio_interaction',
        'night_temp_interaction', 'weekend_transition_temp', 'dow_sin_temp',
    ]
    
    training = TimeSeriesDataSet(
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
        allow_missing_timesteps=False,
    )
    
    with open(TRAINING_DATASET_PATH, 'wb') as f:
        pickle.dump(training, f)
    
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False)
    return training, validation


def create_model(training: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Create TFT model."""
    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEADS,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        loss=MAE(),
        logging_metrics=[MAE()],
        optimizer="adamw",
        weight_decay=WEIGHT_DECAY,
        reduce_on_plateau_patience=LR_PATIENCE,
        reduce_on_plateau_reduction=LR_FACTOR,
        reduce_on_plateau_min_lr=LR_MIN,
        log_interval=-1,
        log_val_interval=1,
    )


class TrainingMonitor(Callback):
    """Training progress monitor."""
    
    def __init__(self):
        super().__init__()
        self.epoch_start = None
        self.best_mae = float('inf')
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()
        
    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        
        elapsed = time.time() - self.epoch_start
        logs = trainer.callback_metrics
        
        train_loss = float(logs.get('train_loss_epoch', logs.get('train_loss', 0)))
        val_loss = float(logs.get('val_loss', 0))
        val_mae = float(logs.get('val_MAE', 0))
        lr = trainer.optimizers[0].param_groups[0]['lr']
        
        is_best = val_mae < self.best_mae
        if is_best:
            self.best_mae = val_mae
        
        star = "★" if is_best else ""
        print(f"  Epoch {trainer.current_epoch:3d}  |  Train: {train_loss:7.0f}  |  "
              f"Val: {val_loss:7.0f}  |  MAE: {val_mae:6.0f}  |  {elapsed:5.1f}s  |  LR: {lr:.1e}  {star}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    epochs = 10 if args.quick else args.epochs
    
    print("="*70)
    print("TFT TRAINING")
    print("="*70)
    
    df = load_data()
    training, validation = create_datasets(df)
    
    train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    
    model = create_model(training)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    monitor = TrainingMonitor()
    callbacks = [
        monitor,
        ModelCheckpoint(
            dirpath=MODEL_DIR,
            filename='tft-v1-{epoch:02d}-{val_MAE:.0f}',
            monitor='val_MAE',
            mode='min',
            save_top_k=3,
        ),
        EarlyStopping(monitor='val_MAE', patience=EARLY_STOP_PATIENCE, mode='min'),
    ]
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        gradient_clip_val=GRADIENT_CLIP,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=50,
    )
    
    print("\n" + "-"*70)
    print(f"   Epoch  |  Train Loss  |  Val Loss  |    MAE  |   Time  |      LR")
    print("-"*70)
    
    trainer.fit(model, train_loader, val_loader)
    
    print("-"*70)
    print(f"Best MAE: {monitor.best_mae:,.0f}")
    print("="*70)


if __name__ == '__main__':
    main()
