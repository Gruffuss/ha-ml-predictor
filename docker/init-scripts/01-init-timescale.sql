-- TimescaleDB initialization script for Home Assistant ML Predictor
-- This script sets up the database with TimescaleDB extension and basic tables

-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create application database if it doesn't exist
-- (This is handled by POSTGRES_DB environment variable)

-- Connect to the application database
\c occupancy_prediction;

-- Enable TimescaleDB extension on the application database
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create basic table structure (will be managed by SQLAlchemy migrations)
-- This is a minimal setup to ensure TimescaleDB is properly initialized

-- Grant necessary permissions to the application user
GRANT USAGE ON SCHEMA public TO occupancy_user;
GRANT CREATE ON SCHEMA public TO occupancy_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO occupancy_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO occupancy_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO occupancy_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO occupancy_user;

-- Configure TimescaleDB settings for optimal performance
-- Set chunk time interval to 1 day for sensor events
SELECT set_config('timescaledb.max_background_workers', '4', false);

-- Create a sample hypertable to verify TimescaleDB is working
-- (This will be replaced by proper migrations)
CREATE TABLE IF NOT EXISTS _timescale_test (
    time TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION
);

-- Convert to hypertable
SELECT create_hypertable('_timescale_test', 'time', if_not_exists => TRUE);

-- Clean up test table
DROP TABLE IF EXISTS _timescale_test;

-- Log successful initialization
\echo 'TimescaleDB extension initialized successfully for occupancy_prediction database'