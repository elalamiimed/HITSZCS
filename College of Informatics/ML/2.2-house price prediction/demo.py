"""
Quick demonstration of house price prediction
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def quick_demo():
    """Quick demonstration of house price prediction"""
    print("🏠 California House Price Prediction Demo")
    print("=" * 50)
    
    # Load data
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target * 100000  # Convert to actual dollars
    
    print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest (best performing model)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 3 Most Important Features:")
    for idx, row in feature_importance.head(3).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Example prediction
    example_house = {
        'MedInc': 4.0,
        'HouseAge': 20,
        'AveRooms': 6,
        'AveBedrms': 1,
        'Population': 1000,
        'AveOccup': 3,
        'Latitude': 34.0,
        'Longitude': -118.0
    }
    
    example_scaled = scaler.transform(pd.DataFrame([example_house]))
    predicted_price = model.predict(example_scaled)[0]
    
    print(f"\n💡 Example Prediction:")
    print(f"House with ${example_house['MedInc']*100}k median income")
    print(f"Predicted price: ${predicted_price:,.2f}")
    
    return model, scaler

if __name__ == "__main__":
    quick_demo()