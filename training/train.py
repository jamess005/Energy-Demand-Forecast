"""
LSTM Seq2Seq Training Script — v2 (attention + Optuna)
Architecture: Encoder-Decoder LSTM with Bahdanau attention
Feature set: 22 features (same as TFT v2 — shared via complete_data.py)

v1 diagnosis (mean-prediction collapse, val MAE ~ 7,870 MW):
  The vanilla seq2seq passed encoder state to the decoder only as the initial
  (h, c) hidden state. The decoder LSTM overwrote the demand-history signal
  within ~5 decode steps, collapsing to unconditional-mean prediction.
  Val MAE of ~7,870 MW = E[|X|]*sigma = 0.798 * 9,856 — confirming constant output.

v2 fixes:
  1. Bahdanau attention: decoder attends over ALL encoder hidden states at each
     step, preventing demand-history signal washout.
  2. Demand shortcut: Linear(enc_len, 24) from encoder demand → output.
     Provides instant persistence baseline; attention path learns corrections.
  3. MSE training loss: gradient proportional to error magnitude, giving
     stronger signal to escape mean-prediction equilibrium vs L1.
  4. Pre-computed encoder projection: W_enc(enc_outputs) computed once per
     forward pass instead of 24 times — saves 23 redundant matmuls.
  5. Optuna hyperparameter tuning: --optuna flag runs automated sweep over
     encoder_length, hidden_size, layers, dropout, lr, batch_size, etc.

Usage:
  python train.py                                   # 50-epoch default training
  python train.py --quick                           # 10-epoch smoke test
  python train.py --optuna                          # 50-trial Optuna sweep
  python train.py --optuna --n-trials 100 --tune-epochs 20
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
import warnings
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torch.utils.data import Dataset, DataLoader

try:
    import optuna
    import optuna.trial
    import optuna.pruners
    HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    HAS_OPTUNA = False

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_float32_matmul_precision('medium')
pl.seed_everything(SEED, workers=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / 'tft_training_data-main.csv'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature sets ──────────────────────────────────────────────────────────────
ALL_FEATURES = [
    'hour_sin', 'hour_cos',
    'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6',
    'month_sin', 'month_cos',
    'is_public_holiday', 'is_weekend',
    'heating_demand', 'temp_lag_24h',
    'humidity', 'rain', 'snowfall',
    'demand_lag_168h_norm', 'demand_rolling_std_7d',
    'heating_hour_cos_product', 'weekend_temp_interaction',
]

# Known-future only — excludes historical demand lag features
DECODER_FEATURES = [
    'hour_sin', 'hour_cos',
    'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6',
    'month_sin', 'month_cos',
    'is_public_holiday', 'is_weekend',
    'heating_demand', 'temp_lag_24h',
    'humidity', 'rain', 'snowfall',
    'heating_hour_cos_product', 'weekend_temp_interaction',
]

ENCODER_FEATURE_DIM = len(ALL_FEATURES) + 1   # +1 for normalised target_demand
DECODER_FEATURE_DIM = len(DECODER_FEATURES)

# ── Split ─────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10


# ── Hyperparameters ───────────────────────────────────────────────────────────

@dataclass
class HParams:
    """All tuneable hyperparameters in one place."""
    encoder_length:      int   = 96    # sweep shows enc=96 consistently wins
    decoder_length:      int   = 24
    hidden_size:         int   = 128
    num_layers:          int   = 1     # sweep shows L=1 always beats L=2/3
    dropout:             float = 0.3
    batch_size:          int   = 256
    learning_rate:       float = 1e-3
    weight_decay:        float = 1e-3
    gradient_clip:       float = 5.0
    lr_patience:         int   = 6     # was 3 — 3 was causing premature LR decay
    lr_factor:           float = 0.5
    lr_min:              float = 1e-6
    early_stop_patience: int   = 10

DEFAULT_HP = HParams(
    # Best from 100-trial epoch-1 cold-start sweep (trial 78, 1,392 MW)
    encoder_length  = 96,
    num_layers      = 1,
    hidden_size     = 288,
    dropout         = 0.15,
    batch_size      = 128,
    learning_rate   = 1.875e-3,
    weight_decay    = 3.565e-4,
    gradient_clip   = 1.28,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature normalisation
# ─────────────────────────────────────────────────────────────────────────────

class FeatureScaler:
    """Z-score normalisation fitted on training data only."""

    def __init__(self):
        self.means: dict[str, float] = {}
        self.stds:  dict[str, float] = {}

    def fit(self, df: pd.DataFrame, features: list[str]) -> 'FeatureScaler':
        for f in features:
            self.means[f] = float(df[f].mean())
            self.stds[f]  = float(df[f].std())
            if self.stds[f] < 1e-8:
                self.stds[f] = 1.0   # constant feature — avoid div/0
        return self

    def transform(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        out = np.zeros((len(df), len(features)), dtype=np.float32)
        for i, f in enumerate(features):
            vals = np.asarray(df[f].values)
            out[:, i] = (vals - self.means[f]) / self.stds[f]
        return out

    def save(self, path: Path):
        with open(path, 'w') as fp:
            json.dump({'means': self.means, 'stds': self.stds}, fp, indent=2)
        print(f"Feature scaler saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DemandDataset(Dataset):
    """
    Sliding window dataset with configurable window lengths.
      enc_x  : (encoder_length, ENCODER_FEATURE_DIM)
      dec_x  : (decoder_length, DECODER_FEATURE_DIM)
      target : (decoder_length,)
    """

    def __init__(self, df: pd.DataFrame, scaler: FeatureScaler,
                 demand_mean: float, demand_std: float, hp: HParams):
        demand_norm = (
            (np.asarray(df['target_demand'].values) - demand_mean) / demand_std
        ).astype(np.float32)

        all_feat = scaler.transform(df, ALL_FEATURES)      # (N, 22)
        dec_feat = scaler.transform(df, DECODER_FEATURES)  # (N, 20)

        enc_demand     = demand_norm.reshape(-1, 1)
        self.enc_array = np.concatenate([all_feat, enc_demand], axis=1).astype(np.float32)
        self.dec_array = dec_feat.astype(np.float32)
        self.demand    = demand_norm

        self.encoder_length = hp.encoder_length
        self.decoder_length = hp.decoder_length
        self.n_samples = len(df) - hp.encoder_length - hp.decoder_length + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        enc_end = idx + self.encoder_length
        dec_end = enc_end + self.decoder_length
        return (
            torch.from_numpy(self.enc_array[idx:enc_end]),
            torch.from_numpy(self.dec_array[enc_end:dec_end]),
            torch.from_numpy(self.demand[enc_end:dec_end]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention with pre-computable encoder projection.
    Pre-computing W_enc(enc_outputs) once saves 23 redundant matmuls
    across the 24-step decode loop.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_enc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_dec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v     = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_proj: torch.Tensor,
                enc_outputs: torch.Tensor):
        """
        Args:
            dec_hidden:   (B, H)  decoder's last-layer hidden state
            enc_proj:     (B, T_enc, H)  pre-computed W_enc(enc_outputs)
            enc_outputs:  (B, T_enc, H)  raw encoder outputs for weighted sum
        Returns:
            context:  (B, H)
            weights:  (B, T_enc)
        """
        score = self.v(torch.tanh(
            enc_proj + self.W_dec(dec_hidden).unsqueeze(1)
        )).squeeze(-1)                          # (B, T_enc)
        weights = torch.softmax(score, dim=-1)  # (B, T_enc)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)  # (B, H)
        return context, weights


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class LSTMSeq2Seq(pl.LightningModule):
    """
    Encoder-Decoder LSTM with Bahdanau attention.

    Without attention the decoder overwrites the encoder's demand-history signal
    within ~5 steps, collapsing to mean-prediction (val MAE ~ 7,870 MW).

    Fix: at each decode step, attend over ALL encoder hidden states to maintain
    fresh access to demand history. A demand shortcut (Linear enc→24) provides
    an instant persistence baseline; the attention path learns corrections.
    """

    def __init__(self, hp: HParams = DEFAULT_HP, demand_std: float = 1.0):
        super().__init__()
        self.hp = hp
        self.demand_std = demand_std  # for logging MW in checkpoint filenames

        self.encoder = nn.LSTM(
            input_size=ENCODER_FEATURE_DIM,
            hidden_size=hp.hidden_size,
            num_layers=hp.num_layers,
            dropout=hp.dropout if hp.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.attention = BahdanauAttention(hp.hidden_size)

        # Explicit dropout — nn.LSTM dropout=0 when num_layers=1,
        # so these are the ONLY dropout applied during training.
        self.enc_dropout = nn.Dropout(hp.dropout)

        # Decoder input: known-future features + attention context
        self.decoder = nn.LSTM(
            input_size=DECODER_FEATURE_DIM + hp.hidden_size,
            hidden_size=hp.hidden_size,
            num_layers=hp.num_layers,
            dropout=hp.dropout if hp.num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output projection: decoder hidden + attention context -> scalar
        self.output_proj = nn.Sequential(
            nn.Linear(hp.hidden_size * 2, hp.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.hidden_size // 2, 1),
        )

        # Direct demand history -> output shortcut (persistence baseline)
        self.demand_shortcut = nn.Linear(hp.encoder_length, hp.decoder_length)

        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_weights(self):
        """Orthogonal init for recurrent weights, forget-gate bias = 1."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name and ('encoder' in name or 'decoder' in name):
                nn.init.zeros_(param)
                # Forget-gate bias = 1  (gate order: input, forget, cell, output)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, enc_x, dec_x):
        # Training noise: mild perturbation on z-scored inputs prevents
        # memorisation of exact training sequences (disabled at eval)
        if self.training:
            enc_x = enc_x + torch.randn_like(enc_x) * 0.05

        # Encode full history — keep all hidden states for attention
        enc_out, (h, c) = self.encoder(enc_x)   # enc_out: (B, T_enc, H)
        enc_out = self.enc_dropout(enc_out)      # explicit dropout (LSTM gives 0 at L=1)

        # Persistence shortcut: direct linear from demand history -> output
        enc_demand = enc_x[:, :, -1]                    # (B, T_enc)
        shortcut = self.demand_shortcut(enc_demand)      # (B, T_dec)

        # Pre-compute encoder projection (saves 23 redundant matmuls)
        enc_proj = self.attention.W_enc(enc_out)         # (B, T_enc, H)

        # Decode step-by-step with attention
        dec_h, dec_c = h, c
        outputs = []

        for t in range(self.hp.decoder_length):
            # Attend over encoder outputs using decoder's last-layer hidden
            context, _ = self.attention(dec_h[-1], enc_proj, enc_out)  # (B, H)

            # Decoder input = known-future features + attention context
            dec_input_t = torch.cat([
                dec_x[:, t : t + 1, :],    # (B, 1, D_dec)
                context.unsqueeze(1),       # (B, 1, H)
            ], dim=-1)                      # (B, 1, D_dec + H)

            dec_out_t, (dec_h, dec_c) = self.decoder(dec_input_t, (dec_h, dec_c))

            # Output projection gets decoder hidden + attention context
            combined = torch.cat([dec_out_t.squeeze(1), context], dim=-1)  # (B, 2H)
            outputs.append(self.output_proj(combined))                     # (B, 1)

        return torch.cat(outputs, dim=-1) + shortcut   # (B, T_dec) residual

    def training_step(self, batch, batch_idx):
        enc_x, dec_x, target = batch
        preds = self(enc_x, dec_x)
        loss = self.loss_fn(preds, target)
        with torch.no_grad():
            mae = nn.functional.l1_loss(preds, target)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        enc_x, dec_x, target = batch
        preds = self(enc_x, dec_x)
        loss = self.loss_fn(preds, target)
        with torch.no_grad():
            mae = nn.functional.l1_loss(preds, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        self.log('val_MAE', mae * self.demand_std, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hp.learning_rate,
            weight_decay=self.hp.weight_decay,
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', patience=self.hp.lr_patience,
            factor=self.hp.lr_factor, min_lr=self.hp.lr_min,
        )
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'monitor': 'val_mae'}}


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class TrainingMonitor(Callback):
    """Epoch-level MAE logger.
    verbose=False : silent
    compact=True  : one compact line per epoch (used during Optuna trials)
    verbose=True  : full banner output (standalone training)
    """

    def __init__(self, demand_std: float, verbose: bool = True, compact: bool = False):
        super().__init__()
        self.demand_std  = demand_std
        self.verbose     = verbose
        self.compact     = compact
        self.epoch_start: float | None = None
        self.best_mae    = float('inf')
        self._train_mw: float | None = None  # captured in on_train_epoch_end

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """Store train MAE immediately after training epoch — always available here."""
        train_mae = trainer.callback_metrics.get('train_mae')
        if train_mae is not None:
            self._train_mw = float(train_mae) * self.demand_std

    def on_validation_epoch_end(self, trainer, pl_module):
        """Print epoch summary once both train + val metrics are available."""
        if trainer.sanity_checking:
            return

        val_mae = trainer.callback_metrics.get('val_mae')
        if val_mae is None:
            return

        elapsed  = time.time() - (self.epoch_start or time.time())
        val_mw   = float(val_mae) * self.demand_std
        train_mw = self._train_mw
        lr       = trainer.optimizers[0].param_groups[0]['lr']

        is_best = val_mw < self.best_mae
        if is_best:
            self.best_mae = val_mw

        star = "\u2605" if is_best else " "

        if self.compact:
            gap_str = f"  gap={val_mw - train_mw:+.0f}" if train_mw is not None else ""
            print(
                f"    e{trainer.current_epoch:02d}  {val_mw:6,.0f} MW {star}"
                f"{gap_str}  [{elapsed:.0f}s]  lr={lr:.1e}",
                flush=True,
            )
        elif self.verbose:
            gap = (val_mw - train_mw) if train_mw is not None else float('nan')
            print(
                f"  Epoch {trainer.current_epoch:3d}  |  "
                f"Train: {train_mw or 0:6.0f} MW  |  "
                f"Val: {val_mw:6.0f} MW  |  "
                f"Gap: {gap:+5.0f} MW  |  "
                f"{elapsed:5.1f}s  |  LR: {lr:.1e}  {star}"
            )


class OptunaPruningCallback(Callback):
    """Reports val_mae to Optuna and prunes underperforming trials."""

    def __init__(self, trial):  # type: ignore[no-untyped-def]
        super().__init__()
        self.trial = trial

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val_mae = trainer.callback_metrics.get('val_mae')
        if val_mae is not None:
            self.trial.report(float(val_mae), step=trainer.current_epoch)
            if self.trial.should_prune():
                assert optuna is not None
                raise optuna.TrialPruned()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        available = sorted(DATA_PATH.parent.glob('tft_training_data-v*.csv'), reverse=True)
        if not available:
            raise FileNotFoundError("No training data found. Run complete_data.py first.")
        actual_path = available[0]
        print(f"  Using: {actual_path.name}")
    else:
        actual_path = DATA_PATH

    df = pd.read_csv(actual_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


def create_dataloaders(df: pd.DataFrame, hp: HParams, verbose: bool = True):
    train_end = int(len(df) * TRAIN_RATIO)
    val_end   = int(len(df) * (TRAIN_RATIO + VAL_RATIO))

    train_df = df[:train_end].copy()
    val_df   = df[train_end:val_end].copy()

    # Fit on training data ONLY
    scaler      = FeatureScaler().fit(train_df, ALL_FEATURES)
    demand_mean = float(train_df['target_demand'].mean())
    demand_std  = float(train_df['target_demand'].std())

    if verbose:
        print(f"Data    : {len(df):,} rows  |  Train: {len(train_df):,}  |  "
              f"Val: {len(val_df):,}")
        print(f"Demand  : mean={demand_mean:,.0f} MW  std={demand_std:,.0f} MW")
        print(f"Features: encoder={ENCODER_FEATURE_DIM} dims  "
              f"decoder={DECODER_FEATURE_DIM}+{hp.hidden_size}(attn) dims")

        print("\nScale check (normalised stds — should all be ~1.0):")
        for f in ['humidity', 'heating_demand', 'temp_lag_24h',
                  'demand_lag_168h_norm', 'demand_rolling_std_7d', 'rain']:
            raw    = train_df[f]
            normed = (raw - scaler.means[f]) / scaler.stds[f]
            print(f"  {f:35s}  raw std={raw.std():8.3f}  →  normed std={normed.std():.3f}")

    train_ds = DemandDataset(train_df, scaler, demand_mean, demand_std, hp)
    val_ds   = DemandDataset(val_df,   scaler, demand_mean, demand_std, hp)

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=hp.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    return train_loader, val_loader, scaler, demand_mean, demand_std


# ─────────────────────────────────────────────────────────────────────────────
# Single training run
# ─────────────────────────────────────────────────────────────────────────────

def train_single(hp: HParams, df: pd.DataFrame, epochs: int,
                 trial=None, verbose: bool = True,
                 compact: bool = False) -> float:  # type: ignore[no-untyped-def]
    """
    Run one training session. Returns best val MAE in MW.
    Pass an Optuna trial for pruning support.
    compact=True: one line per epoch (used by Optuna trials).
    """
    # In compact (Optuna) mode suppress scale-check: it's identical every trial
    train_loader, val_loader, scaler, demand_mean, demand_std = \
        create_dataloaders(df, hp, verbose=verbose and not compact)

    model   = LSTMSeq2Seq(hp, demand_std=demand_std)
    n_param = sum(p.numel() for p in model.parameters())
    if verbose and not compact:
        print(f"\nParameters: {n_param:,}")

    monitor: TrainingMonitor = TrainingMonitor(demand_std, verbose=verbose, compact=compact)
    callbacks: list[Callback] = [monitor]

    if trial is not None and HAS_OPTUNA:
        callbacks.append(OptunaPruningCallback(trial))

    # Only save checkpoints for real training runs, not Optuna trials
    if trial is None:
        callbacks.append(ModelCheckpoint(
            dirpath=MODEL_DIR,
            filename='lstm-v2-{epoch:02d}-{val_MAE:.0f}',
            monitor='val_MAE',
            mode='min',
            save_top_k=3,
        ))

    callbacks.append(EarlyStopping(
        monitor='val_mae',
        patience=hp.early_stop_patience,
        mode='min',
    ))

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        gradient_clip_val=hp.gradient_clip,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=50,
        logger=False,
    )

    if verbose and not compact:
        print("\n" + "-" * 75)
        print("   Epoch  |  Train MAE   |   Val MAE   |    Gap     |   Time  |      LR")
        print("-" * 75)

    trainer.fit(model, train_loader, val_loader)

    best_mae_mw = monitor.best_mae

    if verbose:
        print("-" * 75)
        print(f"Best Val MAE : {best_mae_mw:,.0f} MW")
        print(f"TFT v1 ref   : 1,454 MW  (30f h=32)")
        print(f"TFT v2 ref   : 1,766 MW  (22f h=64 — overfit)")
        print("=" * 75)

        scaler.save(MODEL_DIR / 'lstm_feature_scaler.json')
        with open(MODEL_DIR / 'lstm_norm_stats.json', 'w') as f:
            json.dump({'demand_mean': demand_mean, 'demand_std': demand_std}, f, indent=2)
        print(f"Demand stats saved: {MODEL_DIR / 'lstm_norm_stats.json'}")

    return best_mae_mw


# ─────────────────────────────────────────────────────────────────────────────
# Optuna hyperparameter sweep
# ─────────────────────────────────────────────────────────────────────────────

def objective(trial, df: pd.DataFrame, tune_epochs: int) -> float:  # type: ignore[no-untyped-def]
    """Optuna objective — returns val MAE in MW (lower is better)."""
    # encoder_length=96 and num_layers=1 are fixed — sweep data (10 trials) showed
    # enc=96 wins every good trial; L=2/3 universally stalls at ~3,000 MW.
    hp = HParams(
        encoder_length  = 96,
        num_layers      = 1,
        hidden_size     = trial.suggest_int('hidden_size', 96, 288, step=32),
        dropout         = trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
        batch_size      = trial.suggest_categorical('batch_size', [128, 256]),
        learning_rate   = trial.suggest_float('learning_rate', 3e-4, 2e-3, log=True),
        weight_decay    = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        gradient_clip   = trial.suggest_float('gradient_clip', 1.0, 10.0),
    )

    print(f"\n{'─' * 75}")
    print(f"Trial {trial.number:3d}  |  h={hp.hidden_size}  drop={hp.dropout:.2f}  "
          f"lr={hp.learning_rate:.1e}  bs={hp.batch_size}  "
          f"wd={hp.weight_decay:.1e}  clip={hp.gradient_clip:.1f}  "
          f"[enc=96 L=1 fixed]")

    trial_start = time.time()
    result = train_single(hp, df, epochs=tune_epochs, trial=trial,
                          verbose=True, compact=True)
    trial_elapsed = time.time() - trial_start
    mins, secs = divmod(int(trial_elapsed), 60)
    best_so_far = getattr(trial.study, '_cached_best', float('inf'))
    if result < best_so_far:
        trial.study._cached_best = result  # type: ignore[attr-defined]
    best_so_far = min(result, best_so_far)
    print(f"  \u2192 Trial {trial.number}: {result:,.0f} MW  [{mins}m {secs:02d}s]  "
          f"best={best_so_far:,.0f} MW")
    return result


def run_optuna(df: pd.DataFrame, n_trials: int, tune_epochs: int):
    """Run Optuna hyperparameter sweep."""
    if not HAS_OPTUNA or optuna is None:
        print("ERROR: optuna not installed.  pip install optuna")
        return

    # Suppress Lightning spam during sweep
    logging.getLogger('lightning.pytorch').setLevel(logging.WARNING)

    print("=" * 75)
    print("OPTUNA HYPERPARAMETER SWEEP")
    print(f"  trials={n_trials}  epochs_per_trial={tune_epochs}")
    if tune_epochs == 1:
        print(f"  mode=epoch-1 cold-start search  (no pruning)")
    else:
        print(f"  pruner=MedianPruner(startup=5, warmup=5)")
    print("=" * 75)

    sweep_start = time.time()
    completed_times: list[float] = []

    def _timed_objective(trial) -> float:  # type: ignore[no-untyped-def]
        t0  = time.time()
        val = objective(trial, df, tune_epochs)
        elapsed = time.time() - t0
        completed_times.append(elapsed)
        n_done = len(completed_times)
        avg    = sum(completed_times) / n_done
        eta_s  = avg * (n_trials - n_done)
        eta_m, eta_s2 = divmod(int(eta_s), 60)
        eta_h, eta_m  = divmod(eta_m, 60)
        total_elapsed = time.time() - sweep_start
        el_m, el_s = divmod(int(total_elapsed), 60)
        el_h, el_m = divmod(el_m, 60)
        if eta_h:
            eta_str = f"{eta_h}h {eta_m}m"
        else:
            eta_str = f"{eta_m}m {eta_s2:02d}s"
        elapsed_str = f"{el_h}h {el_m}m {el_s:02d}s" if el_h else f"{el_m}m {el_s:02d}s"
        print(
            f"  [{n_done}/{n_trials}]  elapsed={elapsed_str}  ETA=~{eta_str}",
            flush=True,
        )
        return val

    # With tune_epochs=1 MedianPruner is useless; use NopPruner for a clean
    # cold-start sweep. MedianPruner still helps for longer trial budgets.
    pruner = (
        optuna.pruners.NopPruner()
        if tune_epochs <= 2
        else optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study = optuna.create_study(
        direction='minimize',
        study_name='lstm-seq2seq-v2',
        pruner=pruner,
    )

    study.optimize(
        _timed_objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    # ── Results ──
    print("\n" + "=" * 75)
    print("OPTUNA RESULTS")
    print("=" * 75)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED]

    print(f"Trials: {len(completed)} completed, {len(pruned)} pruned, "
          f"{len(study.trials) - len(completed) - len(pruned)} failed")
    print(f"\nBest val MAE: {study.best_value:,.0f} MW  (trial {study.best_trial.number})")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v}")

    # Top 5 trials
    sorted_trials = sorted(
        completed,
        key=lambda t: t.value if t.value is not None else float('inf'),
    )[:5]
    print("\nTop 5 trials:")
    for t in sorted_trials:
        print(f"  Trial {t.number:3d}  |  {t.value:,.0f} MW  |  "
              f"h={t.params['hidden_size']}  drop={t.params['dropout']:.2f}  "
              f"lr={t.params['learning_rate']:.1e}  wd={t.params['weight_decay']:.1e}  "
              f"clip={t.params['gradient_clip']:.1f}")

    print("=" * 75)

    # Save results
    results_path = MODEL_DIR / 'optuna_results.json'
    results = {
        'best_value_mw': study.best_value,
        'best_params': study.best_params,
        'n_completed': len(completed),
        'n_pruned': len(pruned),
        'trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
            }
            for t in study.trials
        ],
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    # Print update instructions
    bp = study.best_params
    print(f"\nTo retrain with best hyperparameters, update DEFAULT_HP in train.py:")
    print(f"  DEFAULT_HP = HParams(")
    for k, v in bp.items():
        print(f"      {k}={v!r},")
    print(f"  )")
    print(f"  Then run: python training/train.py --epochs 50")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LSTM Seq2Seq training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--quick', action='store_true', help='10-epoch smoke test')
    parser.add_argument('--optuna', action='store_true',
                        help='Run Optuna hyperparameter sweep')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of Optuna trials (default: 100)')
    parser.add_argument('--tune-epochs', type=int, default=1,
                        help='Epochs per Optuna trial (default: 1 — cold-start search)')
    args = parser.parse_args()

    df = load_data()

    if args.optuna:
        run_optuna(df, args.n_trials, args.tune_epochs)
        return

    hp     = DEFAULT_HP
    epochs = 10 if args.quick else args.epochs

    print("=" * 75)
    print("LSTM SEQ2SEQ TRAINING — v2 (attention)")
    print(f"  hidden={hp.hidden_size}  layers={hp.num_layers}  dropout={hp.dropout}")
    print(f"  encoder={hp.encoder_length}h  decoder={hp.decoder_length}h")
    print(f"  lr={hp.learning_rate}  weight_decay={hp.weight_decay}  "
          f"grad_clip={hp.gradient_clip}")
    print(f"  early_stop_patience={hp.early_stop_patience}")
    print("=" * 75)

    train_single(hp, df, epochs=epochs, verbose=True)


if __name__ == '__main__':
    main()
