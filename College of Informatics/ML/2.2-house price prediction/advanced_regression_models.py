"""
Advanced Regression Models for House Price Prediction
Including polynomial features, stacking, and neural networks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedHousePricePredictor:
    """
    Advanced house price prediction with polynomial features and neural networks
    """
    
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_data(self):
        """Load and prepare California housing dataset"""
        print("Loading California Housing Dataset...")
        housing = fetch_california_housing()
        self.data = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.data['Price'] = housing.target * 100000  # Convert to actual dollars
        self.X = self.data.drop('Price', axis=1)
        self.y = self.data['Price']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"Dataset: {len(self.X)} samples, {len(self.X.columns)} features")
        
    def polynomial_regression_analysis(self):
        """Test polynomial features with different degrees"""
        print("\n" + "="*50)
        print("POLYNOMIAL REGRESSION ANALYSIS")
        print("="*50)
        
        degrees = [1, 2, 3]
        polynomial_results = {}
        
        for degree in degrees:
            print(f"\nTesting polynomial degree {degree}...")
            
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(self.X_train)
            X_test_poly = poly.transform(self.X_test)
            
            # Scale features
            scaler_poly = StandardScaler()
            X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly)
            X_test_poly_scaled = scaler_poly.transform(X_test_poly)
            
            # Train Ridge regression to prevent overfitting
            model = Ridge(alpha=1.0)
            model.fit(X_train_poly_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_poly_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            polynomial_results[f'Polynomial (degree {degree})'] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred,
                'features': X_train_poly.shape[1]
            }
            
            print(f"  Features: {X_train_poly.shape[1]}")
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  R²: {r2:.4f}")
            
        self.results.update(polynomial_results)
        return polynomial_results
        
    def neural_network_analysis(self):
        """Test neural network models with different architectures"""
        print("\n" + "="*50)
        print("NEURAL NETWORK ANALYSIS")
        print("="*50)
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        architectures = [
            (100,),                    # Single hidden layer
            (100, 50),                # Two hidden layers
            (100, 50, 25),           # Three hidden layers
            (200, 100),              # Wider layers
        ]
        
        nn_results = {}
        
        for i, hidden_layer_sizes in enumerate(architectures):
            print(f"\nTesting architecture {hidden_layer_sizes}...")
            
            # Create and train neural network
            nn = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            nn.fit(X_train_scaled, self.y_train)
            y_pred = nn.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            arch_name = f'NN {str(hidden_layer_sizes)}'
            nn_results[arch_name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred,
                'iterations': nn.n_iter_
            }
            
            print(f"  Iterations: {nn.n_iter_}")
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  R²: {r2:.4f}")
            
        self.results.update(nn_results)
        return nn_results
        
    def ensemble_methods(self):
        """Test ensemble methods including stacking"""
        print("\n" + "="*50)
        print("ENSEMBLE METHODS")
        print("="*50)
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Base models
        base_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge(alpha=1.0)),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]
        
        ensemble_results = {}
        
        # Individual models
        for name, model in base_models:
            if name in ['lr', 'ridge']:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            else:
                model.fit(X_train_scaled, self.y_train)
                y_pred = model.predict(X_test_scaled)
                
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            
            ensemble_results[name.upper()] = {
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
        # Simple averaging ensemble
        predictions_list = [ensemble_results[name]['predictions'] for name in ensemble_results.keys()]
        avg_predictions = np.mean(predictions_list, axis=0)
        
        rmse = np.sqrt(mean_squared_error(self.y_test, avg_predictions))
        r2 = r2_score(self.y_test, avg_predictions)
        
        ensemble_results['Ensemble (Avg)'] = {
            'rmse': rmse,
            'r2': r2,
            'predictions': avg_predictions
        }
        
        self.results.update(ensemble_results)
        return ensemble_results
        
    def plot_advanced_results(self):
        """Create comprehensive visualizations for advanced models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect all results
        model_names = list(self.results.keys())
        rmses = [self.results[m]['rmse'] for m in model_names]
        r2_scores = [self.results[m]['r2'] for m in model_names]
        
        # RMSE comparison
        axes[0, 0].barh(model_names, rmses)
        axes[0, 0].set_xlabel('RMSE ($)')
        axes[0, 0].set_title('RMSE Comparison - Advanced Models')
        axes[0, 0].invert_yaxis()
        
        # R² comparison
        axes[0, 1].barh(model_names, r2_scores)
        axes[0, 1].set_xlabel('R² Score')
        axes[0, 1].set_title('R² Score Comparison - Advanced Models')
        axes[0, 1].invert_yaxis()
        
        # Best model residual plot
        best_model_name = min(model_names, key=lambda x: self.results[x]['rmse'])
        best_predictions = self.results[best_model_name]['predictions']
        residuals = self.y_test - best_predictions
        
        axes[1, 0].scatter(best_predictions, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Price ($)')
        axes[1, 0].set_ylabel('Residuals ($)')
        axes[1, 0].set_title(f'Residual Plot - {best_model_name}')
        
        # Actual vs Predicted for best model
        axes[1, 1].scatter(self.y_test, best_predictions, alpha=0.5)
        axes[1, 1].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Price ($)')
        axes[1, 1].set_ylabel('Predicted Price ($)')
        axes[1, 1].set_title(f'Actual vs Predicted - {best_model_name}')
        
        plt.tight_layout()
        plt.savefig('advanced_models_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_advanced_analysis(self):
        """Run complete advanced analysis"""
        print("🚀 Starting Advanced House Price Prediction Analysis")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Run all analyses
        poly_results = self.polynomial_regression_analysis()
        nn_results = self.neural_network_analysis()
        ensemble_results = self.ensemble_methods()
        
        # Create visualizations
        self.plot_advanced_results()
        
        # Summary
        print("\n" + "="*60)
        print("🏁 ADVANCED ANALYSIS COMPLETE!")
        print("="*60)
        
        # Find best model
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        best_rmse = self.results[best_model]['rmse']
        best_r2 = self.results[best_model]['r2']
        
        print(f"\nBest Advanced Model: {best_model}")
        print(f"RMSE: ${best_rmse:,.2f}")
        print(f"R² Score: {best_r2:.4f}")
        
        return self.results

# Usage example
if __name__ == "__main__":
    advanced_predictor = AdvancedHousePricePredictor()
    results = advanced_predictor.run_advanced_analysis()
    
    # Print summary table
    summary_df = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE': [results[m]['rmse'] for m in results.keys()],
        'R²': [results[m]['r2'] for m in results.keys()]
    }).sort_values('RMSE')
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(summary_df.round(2))