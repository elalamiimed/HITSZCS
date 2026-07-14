"""
Data exploration and visualization for California Housing Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

def explore_data():
    """Comprehensive data exploration"""
    print("🔍 California Housing Dataset Exploration")
    print("=" * 50)
    
    # Load data
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target * 100000  # Convert to actual dollars
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Basic info
    print("\n📊 Dataset Info:")
    print(df.info())
    
    print("\n📈 Basic Statistics:")
    print(df.describe().round(2))
    
    # Missing values
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if any(missing > 0) else "No missing values")
    
    # Price statistics
    print(f"\n💰 Price Statistics:")
    print(f"Mean: ${df['Price'].mean():,.2f}")
    print(f"Median: ${df['Price'].median():,.2f}")
    print(f"Std Dev: ${df['Price'].std():,.2f}")
    print(f"Min: ${df['Price'].min():,.2f}")
    print(f"Max: ${df['Price'].max():,.2f}")
    
    # Create visualizations
    create_visualizations(df)
    
    return df

def create_visualizations(df):
    """Create comprehensive visualizations"""
    
    # 1. Price distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.hist(df['Price'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.title('House Price Distribution')
    plt.axvline(df['Price'].mean(), color='red', linestyle='--', label=f'Mean: ${df["Price"].mean():,.0f}')
    plt.axvline(df['Price'].median(), color='green', linestyle='--', label=f'Median: ${df["Price"].median():,.0f}')
    plt.legend()
    
    # 2. Log price distribution
    plt.subplot(2, 3, 2)
    plt.hist(np.log1p(df['Price']), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Log(Price)')
    plt.ylabel('Frequency')
    plt.title('Log House Price Distribution')
    
    # 3. Price by geographic location
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(df['Longitude'], df['Latitude'], c=df['Price'], 
                         alpha=0.5, s=10, cmap='viridis')
    plt.colorbar(scatter, label='Price ($)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('House Prices by Geographic Location')
    
    # 4. Feature correlation heatmap
    plt.subplot(2, 3, 4)
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    # 5. Price vs Median Income
    plt.subplot(2, 3, 5)
    plt.scatter(df['MedInc'], df['Price'], alpha=0.5, s=10)
    plt.xlabel('Median Income ($100k)')
    plt.ylabel('Price ($)')
    plt.title('Price vs Median Income')
    
    # 6. Price vs House Age
    plt.subplot(2, 3, 6)
    plt.scatter(df['HouseAge'], df['Price'], alpha=0.5, s=10)
    plt.xlabel('House Age (years)')
    plt.ylabel('Price ($)')
    plt.title('Price vs House Age')
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional detailed plots
    create_detailed_plots(df)

def create_detailed_plots(df):
    """Create more detailed visualizations"""
    
    # Create pairplot for key features
    key_features = ['MedInc', 'HouseAge', 'AveRooms', 'Price']
    
    plt.figure(figsize=(12, 10))
    sns.pairplot(df[key_features], diag_kind='kde', height=2.5)
    plt.suptitle('Pairplot of Key Features', y=1.02)
    plt.tight_layout()
    plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Box plots for price ranges
    plt.figure(figsize=(15, 6))
    
    # Create price categories
    df['Price_Category'] = pd.cut(df['Price'], bins=4, 
                                  labels=['Low', 'Medium', 'High', 'Very High'])
    
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x='Price_Category', y='MedInc')
    plt.title('Median Income by Price Category')
    plt.ylabel('Median Income ($100k)')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x='Price_Category', y='HouseAge')
    plt.title('House Age by Price Category')
    plt.ylabel('House Age (years)')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(data=df, x='Price_Category', y='AveRooms')
    plt.title('Average Rooms by Price Category')
    plt.ylabel('Average Rooms')
    
    plt.tight_layout()
    plt.savefig('price_categories.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Geographic heatmap
    create_geographic_heatmap(df)

def create_geographic_heatmap(df):
    """Create geographic heatmap of prices"""
    
    # Sample data for performance
    sample_df = df.sample(n=5000, random_state=42)
    
    # Create 2D histogram
    plt.figure(figsize=(12, 8))
    
    # Create hexbin plot
    hb = plt.hexbin(sample_df['Longitude'], sample_df['Latitude'], 
                    C=sample_df['Price'], gridsize=50, cmap='viridis')
    
    plt.colorbar(hb, label='Average Price ($)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution of House Prices')
    
    # Add California outline approximation
    plt.xlim(-124, -114)
    plt.ylim(32, 42)
    
    plt.tight_layout()
    plt.savefig('geographic_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_outliers(df):
    """Analyze outliers in the dataset"""
    print("\n🚨 Outlier Analysis")
    print("=" * 30)
    
    # Price outliers
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]
    print(f"Price outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    # Feature outliers
    for col in df.columns:
        if col != 'Price_Category':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                print(f"{col} outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

def generate_summary_report(df):
    """Generate a summary report"""
    print("\n📋 Summary Report")
    print("=" * 50)
    
    # Key insights
    print("\n🔑 Key Insights:")
    print(f"1. Price range: ${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}")
    print(f"2. Most expensive areas: ${df.groupby(['Latitude', 'Longitude'])['Price'].mean().nlargest(3)}")
    print(f"3. Strongest price predictor: MedInc (correlation: {df['MedInc'].corr(df['Price']):.3f})")
    print(f"4. Average house age: {df['HouseAge'].mean():.1f} years")
    print(f"5. Average rooms: {df['AveRooms'].mean():.1f}")
    
    # Geographic insights
    print(f"\n🗺️ Geographic Insights:")
    print(f"- Northern California (higher latitudes) generally more expensive")
    print(f"- Coastal areas (lower longitudes) show higher prices")
    print(f"- Central valley areas are more affordable")

if __name__ == "__main__":
    df = explore_data()
    analyze_outliers(df)
    generate_summary_report(df)