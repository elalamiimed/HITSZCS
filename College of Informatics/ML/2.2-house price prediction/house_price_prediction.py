"""
House Price Prediction using California Housing Dataset
This script implements various regression models to predict house prices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HousePricePredictor:
    """
    A comprehensive house price prediction system using various regression models
    """
    
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load California housing dataset"""
        print("Loading California Housing Dataset...")
        housing = fetch_california_housing()
        self.data = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.data['Price'] = housing.target
        self.X = self.data.drop('Price', axis=1)
        self.y = self.data['Price'] * 100000  # Convert to actual dollars
        print(f"Dataset loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        # Check for missing values
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        # Create visualizations
        self.create_visualizations()
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('California Housing Dataset - Exploratory Analysis', fontsize=16)
        
        # Price distribution
        axes[0, 0].hist(self.y, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('House Price Distribution')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Median income vs Price
        axes[0, 1].scatter(self.data['MedInc'], self.y, alpha=0.5, s=10)
        axes[0, 1].set_title('Median Income vs House Price')
        axes[0, 1].set_xlabel('Median Income ($100k)')
        axes[0, 1].set_ylabel('House Price ($)')
        
        # House age vs Price
        axes[0, 2].scatter(self.data['HouseAge'], self.y, alpha=0.5, s=10)
        axes[0, 2].set_title('House Age vs House Price')
        axes[0, 2].set_xlabel('House Age (years)')
        axes[0, 2].set_ylabel('House Price ($)')
        
        # Average rooms vs Price
        axes[1, 0].scatter(self.data['AveRooms'], self.y, alpha=0.5, s=10)
        axes[1, 0].set_title('Average Rooms vs House Price')
        axes[1, 0].set_xlabel('Average Rooms')
        axes[1, 0].set_ylabel('House Price ($)')
        
        # Population vs Price
        axes[1, 1].scatter(self.data['Population'], self.y, alpha=0.5, s=10)
        axes[1, 1].set_title('Population vs House Price')
        axes[1, 1].set_xlabel('Population')
        axes[1, 1].set_ylabel('House Price ($)')
        
        # Correlation heatmap
        corr_matrix = self.data.corr()
        im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 2].set_title('Feature Correlation Matrix')
        axes[1, 2].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 2].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 2].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        axes[1, 2].set_yticklabels(corr_matrix.columns)
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split and scale the data"""
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
    def train_models(self):
        """Train multiple regression models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use scaled data for SVR, original for others
            if name == 'SVR':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            self.models[name] = model
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            self.results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  MAE: ${mae:,.2f}")
            print(f"  R²: {r2:.4f}")
            print()
            
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for best models"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # Random Forest tuning
        print("Tuning Random Forest...")
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        rf_grid.fit(self.X_train, self.y_train)
        print(f"Best Random Forest params: {rf_grid.best_params_}")
        
        # Update model with best parameters
        self.models['Random Forest (Tuned)'] = rf_grid.best_estimator_
        y_pred = rf_grid.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        self.results['Random Forest (Tuned)'] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
        
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'RMSE': [self.results[m]['rmse'] for m in self.results.keys()],
            'MAE': [self.results[m]['mae'] for m in self.results.keys()],
            'R²': [self.results[m]['r2'] for m in self.results.keys()]
        })
        
        comparison_df = comparison_df.sort_values('RMSE')
        print("\nModel Performance Comparison:")
        print(comparison_df.round(2))
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE comparison
        bars1 = ax1.barh(comparison_df['Model'], comparison_df['RMSE'])
        ax1.set_xlabel('RMSE ($)')
        ax1.set_title('Model RMSE Comparison')
        ax1.invert_yaxis()
        
        # R² comparison
        bars2 = ax2.barh(comparison_df['Model'], comparison_df['R²'])
        ax2.set_xlabel('R² Score')
        ax2.set_title('Model R² Score Comparison')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
        
    def feature_importance_analysis(self):
        """Analyze feature importance for tree-based models"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Get feature importance from Random Forest
        rf_model = self.models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (Random Forest):")
        print(feature_importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance - Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def residual_analysis(self):
        """Perform residual analysis for the best model"""
        print("\n" + "="*50)
        print("RESIDUAL ANALYSIS")
        print("="*50)
        
        # Use Random Forest as the best model
        best_model_name = 'Random Forest'
        y_pred = self.results[best_model_name]['predictions']
        
        residuals = self.y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residual plot
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Price ($)')
        axes[0, 0].set_ylabel('Residuals ($)')
        axes[0, 0].set_title('Residual Plot')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        
        # Actual vs Predicted
        axes[1, 1].scatter(self.y_test, y_pred, alpha=0.5)
        axes[1, 1].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Price ($)')
        axes[1, 1].set_ylabel('Predicted Price ($)')
        axes[1, 1].set_title('Actual vs Predicted Prices')
        
        plt.tight_layout()
        plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def predict_new_data(self, new_data_dict):
        """Make predictions for new data"""
        new_data = pd.DataFrame([new_data_dict])
        prediction = self.models['Random Forest'].predict(new_data)[0]
        return prediction
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting House Price Prediction Analysis")
        print("=" * 60)
        
        # Load and explore data
        self.load_data()
        self.explore_data()
        
        # Prepare data
        self.prepare_data()
        
        # Train models
        self.train_models()
        
        # Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Compare models
        comparison = self.compare_models()
        
        # Feature importance
        self.feature_importance_analysis()
        
        # Residual analysis
        self.residual_analysis()
        
        print("\n" + "="*60)
        print("🏁 ANALYSIS COMPLETE!")
        print("="*60)
        
        # Best model summary
        best_model = comparison.iloc[0]
        print(f"\nBest Model: {best_model['Model']}")
        print(f"RMSE: ${best_model['RMSE']:,.2f}")
        print(f"MAE: ${best_model['MAE']:,.2f}")
        print(f"R² Score: {best_model['R²']:.4f}")
        
        return comparison

# Example usage
if __name__ == "__main__":
    predictor = HousePricePredictor()
    results = predictor.run_complete_analysis()
    
    # Example prediction
    example_house = {
        'MedInc': 4.0,      # Median income in $100k
        'HouseAge': 20,     # House age in years
        'AveRooms': 6,      # Average number of rooms
        'AveBedrms': 1,     # Average number of bedrooms
        'Population': 1000, # Population
        'AveOccup': 3,      # Average occupancy
        'Latitude': 34.0,   # Latitude
        'Longitude': -118.0 # Longitude
    }
    
    predicted_price = predictor.predict_new_data(example_house)
    print(f"\n💡 Example Prediction:")
    print(f"House with median income ${example_house['MedInc']*100}k")
    print(f"Predicted price: ${predicted_price:,.2f}")