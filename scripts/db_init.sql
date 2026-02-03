-- Database initialization script for Energy Demand Forecasting

-- Create weather table
CREATE TABLE IF NOT EXISTS weather (
    id SERIAL PRIMARY KEY,
    date_time TIMESTAMP NOT NULL,
    "temperature_2m(°C)" FLOAT,
    "relative_humidity_2m(%)" FLOAT,
    "rain(mm)" FLOAT,
    "snow_depth(m)" FLOAT,
    "snowfall(cm)" FLOAT,
    daylight_savings_winter BOOLEAN,
    daylight_savings_summer BOOLEAN,
    UNIQUE(date_time)
);

-- Create energy_demand table
CREATE TABLE IF NOT EXISTS energy_demand (
    id SERIAL PRIMARY KEY,
    date_time TIMESTAMP NOT NULL,
    "actual_demand(MW)" FLOAT,
    "demand_forecast(MW)" FLOAT,
    daylight_savings_winter BOOLEAN,
    daylight_savings_summer BOOLEAN,
    UNIQUE(date_time)
);

-- Create energy_unavailability table
CREATE TABLE IF NOT EXISTS energy_unavailability (
    id SERIAL PRIMARY KEY,
    date_time TIMESTAMP NOT NULL,
    "planned_unavailability(MW)" FLOAT DEFAULT 0,
    "actual_unavailability(MW)" FLOAT DEFAULT 0,
    "total_unavailability(MW)" FLOAT DEFAULT 0,
    daylight_savings_winter BOOLEAN,
    daylight_savings_summer BOOLEAN,
    UNIQUE(date_time)
);

-- Create holidays table
CREATE TABLE IF NOT EXISTS holidays (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    is_public_holiday BOOLEAN DEFAULT FALSE,
    is_national BOOLEAN DEFAULT FALSE,
    holiday_type VARCHAR(50),
    counties_affected INTEGER DEFAULT 0,
    daylight_savings_winter BOOLEAN,
    daylight_savings_summer BOOLEAN,
    UNIQUE(date)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_weather_date_time ON weather(date_time);
CREATE INDEX IF NOT EXISTS idx_energy_demand_date_time ON energy_demand(date_time);
CREATE INDEX IF NOT EXISTS idx_energy_unavailability_date_time ON energy_unavailability(date_time);
CREATE INDEX IF NOT EXISTS idx_holidays_date ON holidays(date);

-- Create a view for joined data
CREATE OR REPLACE VIEW forecasting_data AS
SELECT 
    ed.date_time,
    ed."actual_demand(MW)",
    ed."demand_forecast(MW)",
    w."temperature_2m(°C)",
    w."relative_humidity_2m(%)",
    w."rain(mm)",
    w."snow_depth(m)",
    h.is_public_holiday,
    h.is_national,
    h.holiday_type,
    ed.daylight_savings_winter,
    ed.daylight_savings_summer
FROM energy_demand ed
LEFT JOIN weather w ON DATE_TRUNC('hour', ed.date_time) = DATE_TRUNC('hour', w.date_time)
LEFT JOIN holidays h ON DATE(ed.date_time) = h.date
ORDER BY ed.date_time;
