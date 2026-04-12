import pandas as pd
import numpy as np
import os

def generate_spatial_data(grid_size=50, days=30):
    # center roughly on a generic city (e.g., latitude 28.61, longitude 77.20 - New Delhi)
    base_lat = 28.61
    base_lon = 77.20
    
    lats = np.linspace(base_lat - 0.1, base_lat + 0.1, grid_size)
    lons = np.linspace(base_lon - 0.1, base_lon + 0.1, grid_size)
    
    records = []
    
    # Generate static features for the grid
    elevation_grid = np.random.uniform(200, 300, (grid_size, grid_size))
    # Make elevation have a gradient (river valley)
    for i in range(grid_size):
        for j in range(grid_size):
            dist_to_center = abs(i - grid_size/2) + abs(j - grid_size/2)
            elevation_grid[i, j] += dist_to_center * 2
            
    slope_grid = np.random.uniform(0, 15, (grid_size, grid_size))
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            elev = elevation_grid[i, j]
            slope = slope_grid[i, j]
            for day in range(days):
                # Daily weather
                rainfall = np.random.exponential(scale=10) # 0 to ~50 mm
                rainfall_lag_1 = np.random.exponential(scale=12)
                rainfall_lag_2 = np.random.exponential(scale=15)
                
                # Risk formula (higher rainfall + lower elevation + lower slope -> higher risk)
                risk_score = (rainfall + rainfall_lag_1 * 0.8 + rainfall_lag_2 * 0.5) / (elev/100) - slope * 0.5
                
                if risk_score > 30:
                    risk_label = 2 # High
                elif risk_score > 15:
                    risk_label = 1 # Medium
                else:
                    risk_label = 0 # Low
                
                records.append({
                    "latitude": lat,
                    "longitude": lon,
                    "elevation": elev,
                    "slope": slope,
                    "rainfall": rainfall,
                    "rainfall_lag_1": rainfall_lag_1,
                    "rainfall_lag_2": rainfall_lag_2,
                    "flood_risk": risk_label
                })
                
    df = pd.DataFrame(records)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/flood_data.csv', index=False)
    print(f"Generated {len(df)} records in data/flood_data.csv")

if __name__ == "__main__":
    generate_spatial_data(grid_size=30, days=10) # 30x30 = 900 points * 10 days = 9000 rows
