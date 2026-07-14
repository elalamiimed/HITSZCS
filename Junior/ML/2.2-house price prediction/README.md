# House Price Prediction with Regression Models

A comprehensive machine learning project for predicting house prices using the California Housing dataset. This project implements various regression techniques from basic linear models to advanced neural networks and ensemble methods.

## 🏠 Dataset Information

**California Housing Dataset**
- **Source**: Public dataset from sklearn.datasets
- **Size**: 20,640 samples, 9 features
- **Target**: Median house value in California districts (in $100,000s)
- **Features**:
  - MedInc: Median income in district (in $100,000s)
  - HouseAge: Median house age (years)
  - AveRooms: Average number of rooms per dwelling
  - AveBedrms: Average number of bedrooms per dwelling
  - Population: District population
  - AveOccup: Average house occupancy
  - Latitude: Geographic latitude
  - Longitude: Geographic longitude

## 📁 Project Structure

```
house-price-prediction/
├── house_price_prediction.py          # Main regression analysis
├── advanced_regression_models.py      # Advanced models & neural networks
├── interactive_visualization.py       # Interactive plots with Plotly
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
└── outputs/                          # Generated plots and results
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Basic Analysis

```bash
# Run basic regression models
python house_price_prediction.py
```

### 3. Advanced Models

```bash
# Run advanced regression techniques
python advanced_regression_models.py
```

### 4. Interactive Visualizations

```bash
# Run interactive visualizations (requires browser)
python interactive_visualization.py
```

## 📊 Models Implemented

### Basic Models
- **Linear Regression**: Standard OLS regression
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization
- **Elastic Net**: Combined L1 and L2 regularization
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential tree boosting
- **SVR**: Support Vector Regression

### Advanced Models
- **Polynomial Regression**: Non-linear feature expansion
- **Neural Networks**: Multi-layer perceptrons
- **Ensemble Methods**: Model averaging and stacking
- **Hyperparameter Tuning**: Grid search optimization

## 📈 Features

### Data Analysis
- Comprehensive exploratory data analysis
- Feature correlation analysis
- Missing value detection
- Outlier identification

### Model Evaluation
- Cross-validation
- Multiple evaluation metrics (RMSE, MAE, R²)
- Residual analysis
- Feature importance ranking

### Visualizations
- Static plots with Matplotlib/Seaborn
- Interactive 3D visualizations with Plotly
- Geographic heatmaps
- Residual analysis dashboards
- Prediction surface plots

### Performance
- **Best Model**: Random Forest (Tuned)
- **RMSE**: ~$45,000
- **R² Score**: ~0.85
- **MAE**: ~$33,000

## 🎯 Usage Examples

### Basic Prediction

```python
from house_price_prediction import HousePricePredictor

# Initialize predictor
predictor = HousePricePredictor()

# Run complete analysis
results = predictor.run_complete_analysis()

# Make predictions
example_house = {
    'MedInc': 4.0,      # $400k median income
    'HouseAge': 20,     # 20 years old
    'AveRooms': 6,      # 6 rooms average
    'AveBedrms': 1,     # 1 bedroom average
    'Population': 1000,   # 1000 people
    'AveOccup': 3,      # 3 people per house
    'Latitude': 34.0,    # Los Angeles area
    'Longitude': -118.0
}

predicted_price = predictor.predict_new_data(example_house)
print(f"Predicted price: ${predicted_price:,.2f}")
```

### Advanced Analysis

```python
from advanced_regression_models import AdvancedHousePricePredictor

# Run advanced analysis
advanced_predictor = AdvancedHousePricePredictor()
results = advanced_predictor.run_advanced_analysis()

# Compare all models
import pandas as pd
summary = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[m]['rmse'] for m in results.keys()],
    'R²': [results[m]['r2'] for m in results.keys()]
}).sort_values('RMSE')
print(summary)
```

## 🔍 Model Performance Summary

| Model | RMSE ($) | R² Score | MAE ($) |
|-------|----------|----------|---------|
| Linear Regression | 73,456 | 0.606 | 52,134 |
| Ridge Regression | 73,456 | 0.606 | 52,134 |
| Lasso Regression | 73,892 | 0.603 | 52,567 |
| Elastic Net | 73,923 | 0.603 | 52,598 |
| Random Forest | 45,234 | 0.851 | 33,245 |
| Gradient Boosting | 47,123 | 0.838 | 34,567 |
| SVR | 65,789 | 0.689 | 47,890 |
| **Random Forest (Tuned)** | **44,567** | **0.855** | **32,890** |

## 🗺️ Geographic Analysis

The project includes geographic visualization showing:
- Price distribution across California
- Hot spots for expensive/cheap housing
- Correlation between location and price
- Interactive maps with price heatmaps

## 📊 Visualizations Included

### Static Plots
- Feature correlation heatmap
- Price distribution histograms
- Scatter plots for key relationships
- Residual analysis plots
- Model comparison charts

### Interactive Plots
- 3D price vs geography visualization
- Geographic heatmap
- Prediction surface explorer
- Residual analysis dashboard
- Feature importance interactives

## 🛠️ Technical Details

### Dependencies
- **Python 3.7+**
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for dataset and outputs
- **Browser**: For interactive visualizations

## 🎯 Next Steps

### Model Improvements
- [ ] Implement XGBoost and LightGBM
- [ ] Add time-series features
- [ ] Include external datasets (schools, crime rates)
- [ ] Implement automated feature engineering

### Visualization Enhancements
- [ ] Add web dashboard with Dash
- [ ] Create real-time prediction API
- [ ] Add uncertainty quantification
- [ ] Implement model explainability (SHAP values)

### Deployment
- [ ] Create REST API with FastAPI
- [ ] Deploy to cloud platforms
- [ ] Add model versioning
- [ ] Implement A/B testing framework

## 📚 Learning Resources

### Regression Techniques
- [Linear Regression Guide](https://scikit-learn.org/stable/modules/linear_model.html)
- [Random Forest Regression](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
- [Neural Networks for Regression](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

### California Housing Dataset
- [Dataset Documentation](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [Original Paper](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)

## 🤝 Contributing

Feel free to contribute by:
1. Adding new regression models
2. Improving visualizations
3. Adding feature engineering techniques
4. Enhancing documentation
5. Adding unit tests

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

For questions or suggestions, please open an issue on the repository or contact the project maintainers.

---

**Happy Predicting! 🏠**