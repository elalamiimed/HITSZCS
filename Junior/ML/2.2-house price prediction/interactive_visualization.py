"""
Interactive Visualizations for House Price Prediction
Using Plotly for interactive charts and dashboards
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class InteractiveHousePriceVisualizer:
    """
    Interactive visualizations for house price prediction analysis
    """
    
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        """Load and prepare data with predictions"""
        print("Loading data for interactive visualizations...")
        
        # Load data
        housing = fetch_california_housing()
        self.data = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.data['Price'] = housing.target * 100000
        
        # Split and train model
        X = self.data.drop('Price', axis=1)
        y = self.data['Price']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Add predictions to data
        self.data['Predicted_Price'] = self.model.predict(X)
        self.data['Residual'] = self.data['Price'] - self.data['Predicted_Price']
        
        # Add geographic coordinates for mapping
        self.data['Latitude'] = self.data['Latitude']
        self.data['Longitude'] = self.data['Longitude']
        
        print("Data prepared for visualization!")
        
    def create_interactive_scatter_matrix(self):
        """Create interactive scatter matrix"""
        features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'Price']
        
        fig = px.scatter_matrix(
            self.data[features],
            dimensions=features,
            title="Interactive Scatter Matrix - California Housing Features",
            height=800
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.show()
        
    def create_3d_price_geography(self):
        """Create 3D visualization of price vs geography"""
        fig = px.scatter_3d(
            self.data.sample(n=5000, random_state=42),  # Sample for performance
            x='Longitude',
            y='Latitude',
            z='Price',
            color='MedInc',
            size='Population',
            hover_data=['HouseAge', 'AveRooms'],
            title="3D Visualization: House Prices vs Geography",
            labels={'Price': 'House Price ($)', 'MedInc': 'Median Income ($100k)'}
        )
        
        fig.update_layout(height=800)
        fig.show()
        
    def create_price_distribution_dashboard(self):
        """Create interactive price distribution dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Distribution', 'Log Price Distribution',
                           'Price by Median Income', 'Price by House Age'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Price distribution
        fig.add_trace(
            go.Histogram(x=self.data['Price'], nbinsx=50, name='Price'),
            row=1, col=1
        )
        
        # Log price distribution
        fig.add_trace(
            go.Histogram(x=np.log1p(self.data['Price']), nbinsx=50, name='Log Price'),
            row=1, col=2
        )
        
        # Price vs Median Income
        sample_data = self.data.sample(n=5000, random_state=42)
        fig.add_trace(
            go.Scatter(
                x=sample_data['MedInc'],
                y=sample_data['Price'],
                mode='markers',
                marker=dict(color=sample_data['HouseAge'], colorscale='Viridis', showscale=True),
                name='Income vs Price'
            ),
            row=2, col=1
        )
        
        # Price vs House Age
        fig.add_trace(
            go.Scatter(
                x=sample_data['HouseAge'],
                y=sample_data['Price'],
                mode='markers',
                marker=dict(color=sample_data['MedInc'], colorscale='Plasma', showscale=True),
                name='Age vs Price'
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Log(Price)", row=1, col=2)
        fig.update_xaxes(title_text="Median Income ($100k)", row=2, col=1)
        fig.update_xaxes(title_text="House Age (years)", row=2, col=2)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, title_text="Price Distribution Dashboard")
        fig.show()
        
    def create_geographic_heatmap(self):
        """Create geographic heatmap of house prices"""
        # Sample data for performance
        sample_data = self.data.sample(n=10000, random_state=42)
        
        fig = px.density_mapbox(
            sample_data,
            lat='Latitude',
            lon='Longitude',
            z='Price',
            radius=10,
            center=dict(lat=36.0, lon=-119.5),
            zoom=4.5,
            mapbox_style="open-street-map",
            title="Geographic Heatmap of House Prices - California",
            color_continuous_scale="Viridis",
            range_color=[0, 500000]
        )
        
        fig.update_layout(height=600)
        fig.show()
        
    def create_feature_correlation_heatmap(self):
        """Create interactive correlation heatmap"""
        corr_matrix = self.data.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Interactive Feature Correlation Heatmap"
        )
        
        fig.update_layout(height=600)
        fig.show()
        
    def create_residual_analysis_dashboard(self):
        """Create interactive residual analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Actual vs Predicted', 'Residual Distribution',
                           'Residuals vs Predicted', 'Feature Importance'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Actual vs Predicted
        sample_data = self.data.sample(n=5000, random_state=42)
        fig.add_trace(
            go.Scatter(
                x=sample_data['Price'],
                y=sample_data['Predicted_Price'],
                mode='markers',
                marker=dict(size=3, opacity=0.6),
                name='Actual vs Predicted'
            ),
            row=1, col=1
        )
        
        # Add perfect prediction line
        min_price = min(sample_data['Price'].min(), sample_data['Predicted_Price'].min())
        max_price = max(sample_data['Price'].max(), sample_data['Predicted_Price'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_price, max_price],
                y=[min_price, max_price],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction'
            ),
            row=1, col=1
        )
        
        # Residual distribution
        fig.add_trace(
            go.Histogram(x=sample_data['Residual'], nbinsx=50, name='Residuals'),
            row=1, col=2
        )
        
        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(
                x=sample_data['Predicted_Price'],
                y=sample_data['Residual'],
                mode='markers',
                marker=dict(size=3, opacity=0.6),
                name='Residuals vs Predicted'
            ),
            row=2, col=1
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.data.drop(['Price', 'Predicted_Price', 'Residual'], axis=1).columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=feature_importance['importance'],
                y=feature_importance['feature'],
                orientation='h',
                name='Feature Importance'
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Actual Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Residual ($)", row=1, col=2)
        fig.update_xaxes(title_text="Predicted Price ($)", row=2, col=1)
        fig.update_xaxes(title_text="Importance", row=2, col=2)
        
        fig.update_yaxes(title_text="Predicted Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Residual ($)", row=2, col=1)
        fig.update_yaxes(title_text="Feature", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, title_text="Residual Analysis Dashboard")
        fig.show()
        
    def create_prediction_explorer(self):
        """Create interactive prediction explorer"""
        # Create a grid of median income and house age values
        medinc_range = np.linspace(self.data['MedInc'].min(), self.data['MedInc'].max(), 20)
        houseage_range = np.linspace(self.data['HouseAge'].min(), self.data['HouseAge'].max(), 20)
        
        medinc_grid, houseage_grid = np.meshgrid(medinc_range, houseage_range)
        
        # Create prediction grid
        prediction_grid = []
        for mi, ha in zip(medinc_grid.ravel(), houseage_grid.ravel()):
            # Use median values for other features
            sample_data = {
                'MedInc': mi,
                'HouseAge': ha,
                'AveRooms': self.data['AveRooms'].median(),
                'AveBedrms': self.data['AveBedrms'].median(),
                'Population': self.data['Population'].median(),
                'AveOccup': self.data['AveOccup'].median(),
                'Latitude': self.data['Latitude'].median(),
                'Longitude': self.data['Longitude'].median()
            }
            
            pred = self.model.predict(pd.DataFrame([sample_data]))[0]
            prediction_grid.append(pred)
        
        prediction_grid = np.array(prediction_grid).reshape(medinc_grid.shape)
        
        fig = go.Figure(data=[
            go.Surface(
                z=prediction_grid,
                x=medinc_range,
                y=houseage_range,
                colorscale='Viridis',
                name='Predicted Price'
            )
        ])
        
        fig.update_layout(
            title='Interactive Price Prediction Surface',
            scene=dict(
                xaxis_title='Median Income ($100k)',
                yaxis_title='House Age (years)',
                zaxis_title='Predicted Price ($)'
            ),
            height=600
        )
        
        fig.show()
        
    def run_all_visualizations(self):
        """Run all interactive visualizations"""
        print("🎨 Starting Interactive Visualization Suite")
        print("=" * 50)
        
        # Load data
        self.load_and_prepare_data()
        
        print("\n📊 Creating interactive visualizations...")
        
        # Create all visualizations
        print("1. Creating scatter matrix...")
        self.create_interactive_scatter_matrix()
        
        print("2. Creating 3D geography visualization...")
        self.create_3d_price_geography()
        
        print("3. Creating price distribution dashboard...")
        self.create_price_distribution_dashboard()
        
        print("4. Creating geographic heatmap...")
        self.create_geographic_heatmap()
        
        print("5. Creating correlation heatmap...")
        self.create_feature_correlation_heatmap()
        
        print("6. Creating residual analysis dashboard...")
        self.create_residual_analysis_dashboard()
        
        print("7. Creating prediction explorer...")
        self.create_prediction_explorer()
        
        print("\n✅ All interactive visualizations created!")

# Usage
if __name__ == "__main__":
    visualizer = InteractiveHousePriceVisualizer()
    visualizer.run_all_visualizations()