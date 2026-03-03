# Energy Demand Forecasting — LSTM Seq2Seq

Hourly electricity demand forecasting for Germany using deep learning time-series models. Demonstrates end-to-end ML engineering: data pipeline automation, feature engineering, systematic bias correction, and model deployment.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)

## Project Overview

This system forecasts hourly electricity demand 24 hours ahead for Germany's power grid. It combines real-time API data collection, advanced feature engineering, and an LSTM Seq2Seq model with Bahdanau attention and empirical bias correction.

### What It Does

- **Data Collection**: Automated pipeline fetching demand (ENTSO-E), weather (Open-Meteo), and Nager.Date holiday data
- **Feature Engineering**: 22 temporal, weather, and lag features with rolling statistics
- **24-Hour Forecasts**: Produces hourly predictions with ~1,100 MW MAE (seasonal bias correction) on the held-out test set
- **Bias Correction**: Two-stage adjustment (Hour×DoW and seasonal) targeting identified error patterns

### Key Components

- **LSTM Seq2Seq with Bahdanau Attention**: Encoder-decoder architecture — decoder attends over all encoder hidden states at each decode step, trained end-to-end with standard backprop
- **PostgreSQL Data Pipeline**: Normalised schema with automated duplicate detection and DST handling
- **Feature Engineering**: Temporal encodings (sine/cosine), 168 h weekly lag, rolling statistics, weather interactions
- **Empirical Bias Correction**: Two-stage adjustment for hour-of-day×day-of-week and seasonal error patterns based on residual analysis
- **API Feeds**: Continuous data updates from ENTSO-E, Open-Meteo, and Nager.Date

## Why These Methods?

The model initially showed systematic bias patterns despite good overall metrics:

- **Nocturnal and midday over-prediction**: +200 to +500 MW error during hours 0–4 and 10–13
- **Weekday variation**: Fridays and Sundays most over-predicted; Wednesdays systematically under-predicted (up to −1,089 MW bias at H06)
- **Seasonal drift**: Winter MAE (~1,375 MW) substantially higher than summer (~934 MW) due to heating demand volatility

Engineering solutions compensate for these challenges:

- **Empirical bias correction**: Statistical adjustments from hour×day residual patterns reduce mean bias from +209 MW to near zero
- **Seasonal stratification**: Separate correction tables per (hour, day-of-week, season) reduce test MAE by ~93 MW vs raw LSTM
- **96-hour encoder window**: Four-day historical context lets the model see structurally similar days in the lookback
- **Weekly lag feature**: `demand_lag_168h_norm` gives explicit access to the same hour one week earlier — the strongest single predictor

With more diverse training data or a dedicated holiday-demand module, seasonal variance could be reduced further.

## Technical Stack

```
ENTSO-E + Open-Meteo + Nager.Date → PostgreSQL → Feature Engineering → LSTM Seq2Seq → Bias Correction → 24h Forecast
```

**Model**: LSTM Seq2Seq with Bahdanau attention (96 h encoder, 24 h prediction)  
**Database**: PostgreSQL with normalised time-series schema  
**Framework**: PyTorch Lightning with gradient clipping and early stopping  
**Data Pipeline**: Automated API feeds with retry logic and duplicate detection  
**Deployment**: Docker Compose with FastAPI service (optional)

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Val MAE** | 1,364 MW | Validation set (Sep 2024 – Jun 2025), epoch 8 checkpoint |
| **Test MAE (raw)** | ~1,200 MW | Held-out test set (Jun 2025 – Mar 2026), 100-window sample |
| **Test MAE (seasonal corrected)** | ~1,100 MW | Same test set, seasonal bias correction applied |
| **Encoder Window** | 96 hours | Historical context for predictions |
| **Forecast Horizon** | 24 hours | Hourly predictions |
| **Training Data** | 2019–2025 | ~63,000 hourly observations |
| **Best month** | ~873 MW | July 2025 (summer, stable demand) |
| **Worst month** | ~1,568 MW | January 2026 (post-holiday demand spikes) |

**Bias Correction Impact**: Seasonal correction reduces mean bias from +209 MW to +12 MW and lowers test MAE by ~93 MW. Hour×DoW correction alone gives −46 MW; seasonal adds a further −47 MW on top.

## Installation & Usage

### Quick Start with Docker

```bash
# Clone repository
git clone https://github.com/jamess005/Energy-Demand-Forecast.git
cd Energy-Demand-Forecast

# Set up environment
cp .env.example .env
# Edit .env with your ENTSOE_API_KEY

# Start PostgreSQL
docker-compose up -d postgres

# Install dependencies
pip install -r requirements.txt

# Collect initial data
python src/data/feeds/entsoe_feed.py
python src/data/feeds/weather_feed.py
python src/data/feeds/holidays_feed.py

# Generate forecast
python src/api/forecast.py --hours 24
```

### Manual Installation

```bash
# Install PostgreSQL 14+
# Create database: createdb energy_forecast

# Install dependencies
pip install -r requirements.txt

# Initialize database schema
psql -U postgres -d energy_forecast -f scripts/db_init.sql

# Configure environment
cp .env.example .env
# Add ENTSOE_API_KEY and database credentials

# Collect data and run forecast
python src/data/feeds/entsoe_feed.py
python src/api/forecast.py --hours 24
```

### Usage

```bash
# 24-hour forecast from latest data
python src/api/forecast.py --hours 24

# Forecast from specific date
python src/api/forecast.py --hours 24 --date "25/01/2025"

# Disable bias correction
python src/api/forecast.py --hours 24 --no-correction
```

Output includes predicted demand (MW), bias corrections, timestamp range, and forecast statistics.

## Training Details

**Model**: LSTM Seq2Seq with Bahdanau attention  
**Hardware**: AMD Radeon RX 7800 XT  
**Convergence**: Epoch 8 (early stopping, patience 10)  
**Hyperparameter search**: 100-trial Optuna cold-start sweep on epoch-1 validation MAE

**Best-run Hyperparameters** (Optuna trial 78):

| Parameter | Value |
|-----------|-------|
| Encoder length | 96 h |
| Decoder / forecast length | 24 h |
| Hidden size | 288 |
| LSTM layers | 1 |
| Dropout | 0.15 |
| Batch size | 128 |
| Learning rate | 1.875 × 10⁻³ |
| Weight decay | 3.565 × 10⁻⁴ |
| Gradient clip | 1.28 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 6) |

**Dataset splits**:
- Training: 80% (2019-01 – 2024-09)
- Validation: 10% (2024-09 – 2025-06)
- Test: 10% (2025-06 – 2026-03)

## Feature Engineering

The model uses **22 engineered features** across four categories:

**Temporal encodings** (smooth circular representation):

| Feature | Description |
|---------|-------------|
| `hour_sin`, `hour_cos` | Hour of day as sine/cosine pair |
| `month_sin`, `month_cos` | Month of year as sine/cosine pair |
| `dow_0` … `dow_6` | One-hot day-of-week (Mon=0, Sun=6) |

**Calendar flags**:

| Feature | Description |
|---------|-------------|
| `is_public_holiday` | German national holiday (Nager.Date) |
| `is_weekend` | Saturday or Sunday indicator |

**Weather & demand drivers**:

| Feature | Description |
|---------|-------------|
| `heating_demand` | Degrees below 15 °C threshold (HDD proxy) |
| `temp_lag_24h` | Air temperature 24 hours prior |
| `humidity` | Relative humidity (%) |
| `rain` | Precipitation (mm/h) |
| `snowfall` | Snowfall (cm/h) |

**Lag & rolling statistics**:

| Feature | Description |
|---------|-------------|
| `demand_lag_168h_norm` | Normalised demand exactly one week prior — strongest single predictor |
| `demand_rolling_std_7d` | 7-day rolling demand std dev (regime / volatility signal) |

**Cross-feature interactions**:

| Feature | Description |
|---------|-------------|
| `heating_hour_cos_product` | `heating_demand × hour_cos` — cold-morning peak interaction |
| `weekend_temp_interaction` | Weekend flag × temperature — weekend demand is more temp-sensitive |

## Data Sources

| Source | Data Type | Update Frequency | API |
|--------|-----------|------------------|-----|
| ENTSO-E | Actual demand | Hourly | [Transparency Platform](https://transparency.entsoe.eu/) |
| Open-Meteo | Temperature, humidity, precipitation | Hourly | [Archive API](https://open-meteo.com/) |
| Nager.Date | German public holidays | Annual | [Holiday API](https://date.nager.at/) |

## Model Limitations

The model struggles with:

- **Holiday demand patterns**: The binary `is_public_holiday` flag is insufficient for post-holiday demand rebounds. Jan 7, 2026 (day after New Year, demand ~65 GW) had MAE of ~5,346 MW; Dec 30, 2025 (holiday week, ~50 GW) had ~4,028 MW
- **Wednesday anomaly**: Raw predictions consistently under-forecast mid-week demand (bias up to −1,089 MW at H06×Wed), suggesting the model misses a mid-week demand recovery pattern
- **Winter volatility**: Cold snaps cause demand spikes the 96-hour lookback doesn't always capture; January test MAE ~1,568 MW
- **Fixed feature set**: Model does not incorporate energy prices, industrial shutdowns, or probabilistic weather forecasts

A production system would benefit from:
- A dedicated holiday-proximity feature (e.g. `days_since_holiday`, `days_to_holiday`)
- Extended training data (10+ years) to encounter more rare events
- Ensemble combining LSTM with gradient boosted trees for holiday cells
- Continuous online recalibration of bias tables

## What This Demonstrates

This portfolio piece shows practical ML engineering skills:

1. **End-to-end ML pipeline**: Data collection → feature engineering → model training → inference
2. **Error analysis & correction**: Identified systematic bias through residual analysis, implemented two-stage empirical fixes
3. **Hyperparameter optimisation**: 100-trial Optuna sweep with cold-start proxy metric, converging to hidden_size=288, lr=1.875e-3
4. **Production-ready code**: Automated data feeds, error handling, duplicate detection, logging
5. **Database design**: Normalised PostgreSQL schema for time-series data with proper indexing
6. **API integration**: Resilient data fetching from ENTSO-E, Open-Meteo, and Nager.Date APIs

This represents a real-world ML engineering workflow: time-series forecasting with imperfect data, systematic debugging, and practical engineering to produce reliable outputs.

## Future Improvements

### If Deploying to Production

- **Continuous retraining**: Automated model updates as new data accumulates
- **Ensemble forecasting**: Combine LSTM with XGBoost/LightGBM for robust predictions
- **Multi-region expansion**: Extend beyond Germany to other European power markets
- **Real-time API**: FastAPI service for on-demand forecasts with authentication
- **Monitoring dashboard**: Track prediction accuracy, data pipeline health, model drift

### Architecture Enhancements

Current approach: Single LSTM Seq2Seq model with post-hoc bias correction.

Production approach would incorporate:
1. **Multi-model ensemble**: LSTM + gradient boosted trees + ARIMA for complementary strengths
2. **Online learning**: Incremental model updates without full retraining
3. **Uncertainty quantification**: Prediction intervals using quantile regression
4. **Explainability tools**: SHAP values for feature importance tracking
5. **Alert system**: Notifications for unusual predictions or data pipeline failures

## Project Structure

```
Energy-Demand-Forecaster/
├── src/
│   ├── api/                    # Forecasting API
│   │   ├── main.py            # FastAPI application
│   │   ├── forecast.py        # Main forecasting script
│   │   ├── prepare_data.py    # Data loading & feature generation
│   │   └── bias_correction.py # Systematic bias correction
│   └── data/
│       ├── feeds/             # Live data collection
│       │   ├── entsoe_feed.py    # ENTSO-E demand data
│       │   ├── weather_feed.py   # Open-Meteo weather
│       │   ├── holidays_feed.py  # German holidays
│       │   └── run_all_feeds.py  # Pipeline orchestrator
│       ├── processing/        # Feature engineering modules
│       └── validation/        # Data quality checks
├── training/
│   ├── train.py              # LSTM Seq2Seq model & training script
│   └── models/               # Saved checkpoints, scalers, bias tables
├── notebooks/
│   ├── 02_feature_engineering.ipynb  # Feature exploration & validation
│   └── 03_model_evaluation.ipynb     # Full error analysis & bias correction
├── config/                   # YAML configuration files
├── scripts/                  # Database initialisation scripts
├── docker-compose.yml        # PostgreSQL + optional API service
├── Dockerfile                # Container definition
├── .dockerignore             # Docker build exclusions
├── .env.example              # Environment variable template
├── requirements.txt          # Production dependencies
└── README.md
```

## Configuration Files

**Environment Variables** (`.env`):
```env
# Database
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=energy_forecast

# APIs
ENTSOE_API_KEY=your_key_here
```

**YAML Configs** (`config/`):
- `model_config.yaml`: Model hyperparameters and training settings
- `data_config.yaml`: Feature definitions and database connection
- `api_config.yaml`: API server configuration and CORS settings

**Note**: The YAML configs contain default localhost settings and are safe to commit. Sensitive credentials (API keys, passwords) should only be in `.env` (which is gitignored).

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

- **ENTSO-E**: Transparency Platform for electricity demand data
- **Open-Meteo**: Weather archive API
- **Nager.Date**: German holiday calendar
- **PyTorch Lightning**: Training framework and checkpointing utilities

## Contact

**James Scott** — Machine Learning Engineer  
[GitHub](https://github.com/jamess005) | [LinkedIn](https://www.linkedin.com/in/jamesscott005)

---

*This project demonstrates end-to-end ML engineering: from automated data pipelines to systematic bias correction, showing practical skills for production time-series forecasting systems.*
