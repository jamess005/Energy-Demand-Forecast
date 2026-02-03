"""
Energy Demand Forecast API
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from forecast import forecast, MODEL_PATH


class ForecastRequest(BaseModel):
    hours: int = 24
    forecast_from_date: Optional[str] = None
    apply_bias_correction: bool = True


class ForecastPoint(BaseModel):
    timestamp: datetime
    predicted_demand: float
    corrected_demand: Optional[float] = None
    bias_correction: Optional[float] = None


class ForecastResponse(BaseModel):
    forecasts: List[ForecastPoint]
    generated_at: datetime
    model_version: str
    bias_correction_applied: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool


app = FastAPI(
    title="Energy Demand Forecast API",
    description="24-hour ahead energy demand forecasting using Temporal Fusion Transformer",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    model_loaded = MODEL_PATH.exists()
    db_connected = False
    
    try:
        from prepare_data import get_db_engine
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        db_connected = True
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if model_loaded and db_connected else "degraded",
        model_loaded=model_loaded,
        database_connected=db_connected,
    )


@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate demand forecast."""
    if not 1 <= request.hours <= 168:
        raise HTTPException(400, "Hours must be between 1 and 168")
    
    if request.forecast_from_date:
        valid = False
        for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%d/%m/%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S']:
            try:
                pd.to_datetime(request.forecast_from_date, format=fmt)
                valid = True
                break
            except ValueError:
                continue
        if not valid:
            raise HTTPException(400, "Date must be in UK format: DD/MM/YYYY")
    
    try:
        results = forecast(
            hours=request.hours,
            forecast_from_date=request.forecast_from_date,
            apply_correction=request.apply_bias_correction
        )
        
        forecasts = []
        has_correction = 'corrected_demand' in results.columns
        
        for _, row in results.iterrows():
            point = ForecastPoint(
                timestamp=row['timestamp'],
                predicted_demand=float(row['predicted_demand']),
                corrected_demand=float(row['corrected_demand']) if has_correction else None,
                bias_correction=float(row['bias_correction']) if has_correction else None,
            )
            forecasts.append(point)
        
        return ForecastResponse(
            forecasts=forecasts,
            generated_at=datetime.utcnow(),
            model_version=MODEL_PATH.stem.split('-')[1],
            bias_correction_applied=has_correction,
        )
        
    except Exception as e:
        raise HTTPException(500, f"Forecast generation failed: {str(e)}")


@app.get("/forecast/simple")
async def simple_forecast(hours: int = 24, date: Optional[str] = None, bias_correction: bool = True):
    """Simple GET endpoint for forecasting."""
    return await generate_forecast(ForecastRequest(
        hours=hours,
        forecast_from_date=date,
        apply_bias_correction=bias_correction
    ))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
