import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import os

def train():
    df = pd.read_csv('data/flood_data.csv')
    
    features = ['elevation', 'slope', 'rainfall', 'rainfall_lag_1', 'rainfall_lag_2']
    X = df[features]
    y = df['flood_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=3
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Weighted F1 Score: {f1:.4f}")
    
    os.makedirs('models', exist_ok=True)
    model.save_model('models/xgboost_flood_model.json')
    print("Model saved to models/xgboost_flood_model.json")
    
    # Save a typical grid coordinates for streamlit app without the weather
    grid_df = df[['latitude', 'longitude', 'elevation', 'slope']].drop_duplicates()
    grid_df.to_csv('data/grid_data.csv', index=False)

if __name__ == "__main__":
    train()
