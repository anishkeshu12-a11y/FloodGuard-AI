import pandas as pd
import numpy as np
import os

def generate_spatial_data(grid_size=30, days=15, seed=42):
    """
    Generates synthetic spatial-temporal flood risk data.
    Features: latitude, longitude, elevation, slope, rainfall, rainfall_lag_1, rainfall_lag_2, humidity, temperature, day
    Target: flood_risk (0: Low, 1: Medium, 2: High)
    """
    np.random.seed(seed)
    
    # center roughly on latitude 28.61, longitude 77.20 (New Delhi region)
    base_lat = 28.61
    base_lon = 77.20
    
    lats = np.linspace(base_lat - 0.1, base_lat + 0.1, grid_size)
    lons = np.linspace(base_lon - 0.1, base_lon + 0.1, grid_size)
    
    records = []
    
    # Generate static features for the grid (elevation and slope)
    elevation_grid = np.random.uniform(200, 300, (grid_size, grid_size))
    # Make elevation have a gradient (river valley in the center)
    for i in range(grid_size):
        for j in range(grid_size):
            dist_to_center = abs(i - grid_size/2) + abs(j - grid_size/2)
            elevation_grid[i, j] += dist_to_center * 2.5
            
    slope_grid = np.random.uniform(0, 15, (grid_size, grid_size))
    
    print(f"Generating data for a {grid_size}x{grid_size} grid over {days} days...")
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            elev = elevation_grid[i, j]
            slope = slope_grid[i, j]
            for day in range(days):
                # Weather factors
                # Rainfall (exponential to simulate dry days and occasional heavy downpours)
                rainfall = np.random.exponential(scale=12.0)
                rainfall_lag_1 = np.random.exponential(scale=10.0)
                rainfall_lag_2 = np.random.exponential(scale=8.0)
                
                # Temperature (uniform, 20°C to 40°C in monsoonal/tropical regions)
                temperature = np.random.uniform(20.0, 40.0)
                
                # Humidity (uniform, 50% to 100% - highly correlated with rainfall/risk)
                humidity = np.random.uniform(50.0, 100.0)
                
                # Risk formula: higher rainfall, higher humidity, lower elevation, lower slope -> higher risk
                # Soil saturation factor is modeled by humidity and lag rain
                soil_saturation = (humidity / 100.0) * 1.2
                rain_impact = (rainfall * soil_saturation) + (rainfall_lag_1 * 0.7) + (rainfall_lag_2 * 0.4)
                
                # Lower elevation and lower slope increase flood susceptibility
                elev_susceptibility = 300.0 / (elev + 1e-5)
                slope_susceptibility = 1.0 / (slope + 1.0)
                
                risk_score = rain_impact * elev_susceptibility * (1.0 + slope_susceptibility * 0.5)
                
                # Classify based on score thresholds
                if risk_score > 25.0:
                    risk_label = 2  # High Risk
                elif risk_score > 12.0:
                    risk_label = 1  # Medium Risk
                else:
                    risk_label = 0  # Low Risk
                
                records.append({
                    "day": day,
                    "latitude": lat,
                    "longitude": lon,
                    "elevation": elev,
                    "slope": slope,
                    "rainfall": rainfall,
                    "rainfall_lag_1": rainfall_lag_1,
                    "rainfall_lag_2": rainfall_lag_2,
                    "temperature": temperature,
                    "humidity": humidity,
                    "flood_risk": risk_label
                })
                
    df = pd.DataFrame(records)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/flood_data.csv', index=False)
    print(f"Successfully generated {len(df)} records in data/flood_data.csv")
    print(df['flood_risk'].value_counts())

if __name__ == "__main__":
    generate_spatial_data(grid_size=30, days=15) # 900 grid cells * 15 days = 13,500 records
