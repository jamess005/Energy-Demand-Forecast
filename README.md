# Energy Demand Forecasting with Temporal Fusion Transformer

Hourly electricity demand forecasting for Germany using deep learning time-series models. Demonstrates end-to-end ML engineering: data pipeline automation, feature engineering, systematic bias correction, and model deployment.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)

## Project Overview

This system forecasts hourly electricity demand 24 hours ahead for Germany's power grid. It combines real-time API data collection, advanced feature engineering, and a Temporal Fusion Transformer model with empirical bias correction.

### What It Does

- **Data Collection**: Automated pipeline fetching demand (ENTSO-E), weather (Open-Meteo), and Nager.date data
- **Feature Engineering**: 30+ temporal, weather, and lag features with rolling statistics
- **24-Hour Forecasts**: Produces hourly predictions with ~1,270 MW MAE on validation data
- **Bias Correction**: Systematic adjustment for identified weekday/weekend and evening patterns

### Key Components

- **Temporal Fusion Transformer**: Multi-head attention with variable selection for time-series forecasting
- **PostgreSQL Data Pipeline**: Normalized schema with automated duplicate detection and DST handling
- **Advanced Feature Engineering**: Temporal encodings, lag features (24h and 168h), rolling statistics
- **Empirical Bias Correction**: Hour-of-day and day-of-week adjustments based on error pattern analysis
- **API Feeds**: Continuous data updates from ENTSO-E, Open-Meteo, and Nager.Date

## Why These Methods?

The model initially showed systematic bias patterns despite good overall metrics:

- **Evening underprediction**: Consistent -200 to -300 MW error during peak hours
- **Weekday/weekend split**: Different error patterns between workdays and weekends
- **96-hour encoder overfitting**: Full weekly context led to overfitting on weekly patterns

Engineering solutions compensate for these challenges:

- **Empirical bias correction**: Statistical adjustments based on hour/day error patterns reduce systematic bias
- **48-hour encoder window**: Shorter context (vs. 96h or 168h) improved generalization and reduced overfitting
- **Evening-specific lag features**: Targeted feature engineering captures peak hour dynamics
- **Removed day-ahead forecast**: Eliminated data leakage from features unavailable at inference time

With more diverse training data or longer training horizons, some corrections could be relaxed. The current approach ensures reliable predictions despite model limitations.

## Technical Stack

```
ENTSO-E + Open-Meteo + Nager.Date → PostgreSQL → Feature Engineering → TFT (PyTorch) → Bias Correction → 24h Forecast
```

**Model**: Temporal Fusion Transformer (48h encoder, 24h prediction)  
**Database**: PostgreSQL with normalized time-series schema  
**Framework**: PyTorch Lightning with gradient clipping and early stopping  
**Data Pipeline**: Automated API feeds with retry logic and duplicate detection  
**Deployment**: Docker Compose with FastAPI service (optional)

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **MAE** | 1,270 MW | Mean absolute error on validation set |
| **Encoder Window** | 48 hours | Historical context for predictions |
| **Forecast Horizon** | 24 hours | Hourly predictions |
| **Training Data** | 2019-2025 | ~6 years of historical data |

**Bias Correction Impact**: Reduces systematic evening underprediction by 200-300 MW and corrects weekday/weekend error patterns.

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

**Model**: Temporal Fusion Transformer  
**Hardware**: AMD Radeon RX 7800 XT  
**Training Time**: ~20-30 epochs to convergence  

**Hyperparameters**:
- Hidden size: 32
- Attention heads: 4
- Dropout: 0.3
- Learning rate: 0.0001 (with cosine decay)
- Batch size: 512
- Encoder length: 48 hours
- Prediction length: 24 hours

**Dataset**:
- Training split: 80%
- Validation split: 10%
- Test split: 10%
- Total records: ~52,000 hourly observations (2019-2025)

## Feature Engineering

The model uses 30+ engineered features across multiple categories:

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
        'night_temp_interaction', 'weekend_transition_temp', 'dow_sin_temp'

## Data Sources

| Source | Data Type | Update Frequency | API |
|--------|-----------|------------------|-----|
| ENTSO-E | Actual demand, day-ahead forecasts | Hourly | [Transparency Platform](https://transparency.entsoe.eu/) |
| Open-Meteo | Temperature, humidity, precipitation | Hourly | [Archive API](https://open-meteo.com/) |
| Nager.Date | German public holidays | Annual | [Holiday API](https://date.nager.at/) |

## Model Limitations

The model struggles with:

- **Novel patterns**: Performance degrades for unprecedented demand spikes (e.g., extreme weather events)
- **Multi-week forecasts**: Accuracy drops significantly beyond 48-hour horizon
- **Rapid industrial shifts**: Slow to adapt to sudden changes in baseload consumption

These limitations stem from:
- **Training data constraints**: 6 years of history may not capture rare events or long-term trends
- **Static feature set**: Model doesn't incorporate supplementary factors such as energy price
- **Fixed encoder window**: 48-hour context may miss longer-term seasonal patterns
- **Abnormally warm temperatures**: Record breaking temerature with especially warm winters causing bias

A production system would benefit from:
- Extended training data (10+ years)
- Incorporation of wider factors to explain sepcific patterns
- Ensemble methods combining multiple forecast horizons

## What This Demonstrates

This portfolio piece shows practical ML engineering skills:

1. **End-to-end ML pipeline**: Data collection → feature engineering → model training → inference
2. **Error analysis & correction**: Identified systematic bias through residual analysis, implemented empirical fixes
3. **Production-ready code**: Automated data feeds, error handling, duplicate detection, logging
4. **Database design**: Normalized PostgreSQL schema for time-series data with proper indexing
5. **API integration**: Resilient data fetching from ENTSO-E, Open-Meteo, and Nager.Date APIs
6. **Hyperparameter tuning**: Iterative refinement from 96h encoder (overfitting) to 48h (optimal)

This represents a real-world ML engineering workflow: time-series forecasting with imperfect data, systematic debugging, and practical engineering to produce reliable outputs.

## Future Improvements

### If Deploying to Production

- **Continuous retraining**: Automated model updates as new data accumulates
- **Ensemble forecasting**: Combine TFT with XGBoost/LightGBM for robust predictions
- **Multi-region expansion**: Extend beyond Germany to other European power markets
- **Real-time API**: FastAPI service for on-demand forecasts with authentication
- **Monitoring dashboard**: Track prediction accuracy, data pipeline health, model drift

### Architecture Enhancements

Current approach: Single TFT model with post-hoc bias correction.

Production approach would incorporate:
1. **Multi-model ensemble**: TFT + gradient boosted trees + ARIMA for complementary strengths
2. **Online learning**: Incremental model updates without full retraining
3. **Uncertainty quantification**: Prediction intervals using quantile regression
4. **Explainability tools**: SHAP values for feature importance tracking
5. **Alert system**: Notifications for unusual predictions or data pipeline failures

## Project Structure

```
Energy-Demand-Forecaster/
├── src/
│   ├── api/                    # Forecasting API
│   │   ├── forecast.py        # Main forecasting script
│   │   ├── prepare_data.py    # Data loading & feature generation
│   │   └── bias_correction.py # Systematic bias correction
│   └── data/
│       ├── feeds/             # Live data collection
│       │   ├── entsoe_feed.py    # ENTSO-E demand data
│       │   ├── weather_feed.py   # Open-Meteo weather
│       │   └── holidays_feed.py  # German holidays
│       ├── processing/        # Feature engineering modules
│       └── validation/        # Data quality checks
├── training/
│   ├── train.py              # Model training script
│   └── evaluate.py           # Model evaluation
├── config/                   # YAML configuration files
├── docker-compose.yml        # PostgreSQL + optional API service
├── Dockerfile                # Container definition
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
- `model_config.yaml`: TFT hyperparameters and training settings
- `data_config.yaml`: Feature definitions and database connection
- `api_config.yaml`: API server configuration and CORS settings

**Note**: The YAML configs contain default localhost settings and are safe to commit. Sensitive credentials (API keys, passwords) should only be in `.env` (which is gitignored).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgements

- **ENTSO-E**: Transparency Platform for electricity demand data
- **Open-Meteo**: Weather archive API
- **Nager.Date**: German holiday calendar
- **PyTorch Forecasting**: TFT implementation and utilities

## Contact

**James Scott** - Machine Learning Engineer  
[GitHub](https://github.com/jamess005) | [LinkedIn](https://www.linkedin.com/in/jamesscott005)

---

*This project demonstrates end-to-end ML engineering: from automated data pipelines to systematic bias correction, showing practical skills for production time-series forecasting systems.*
