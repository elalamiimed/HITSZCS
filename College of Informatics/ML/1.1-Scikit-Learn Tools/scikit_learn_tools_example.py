#!/usr/bin/env python3
"""
Scikit-Learn Tools Comprehensive Example
========================================

This example demonstrates various Scikit-Learn tools and techniques including:
- Data preprocessing
- Feature selection
- Model selection and evaluation
- Pipeline creation
- Cross-validation
- Hyperparameter tuning
- Model persistence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, fetch_california_housing, make_classification
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, learning_curve
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder,
    PolynomialFeatures, RobustScaler
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, 
    RFE, SelectFromModel
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class ScikitLearnToolsDemo:
    """Comprehensive demonstration of Scikit-Learn tools"""
    
    def __init__(self):
        self.results = {}
        
    def load_classification_data(self):
        """Load wine dataset for classification"""
        print("Loading Wine Dataset for Classification...")
        wine = load_wine()
        X, y = wine.data, wine.target
        
        # Create DataFrame for better visualization
        feature_names = wine.feature_names
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        print(f"Dataset shape: {X.shape}")
        print(f"Classes: {np.unique(y)}")
        print(f"Features: {feature_names[:5]}...")
        
        return X, y, feature_names, df
    
    def load_regression_data(self):
        """Load California housing dataset for regression"""
        print("\nLoading California Housing Dataset for Regression...")
        
        # Load California housing dataset
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        
        # Create DataFrame for better visualization
        feature_names = housing.feature_names
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Features: {feature_names}")
        
        return X, y, feature_names, df
    
    def demonstrate_preprocessing(self, X, y):
        """Demonstrate data preprocessing techniques"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING TECHNIQUES")
        print("="*50)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 1. Standard Scaling
        print("\n1. Standard Scaling (Z-score normalization):")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Original mean: {X_train.mean():.2f}")
        print(f"Scaled mean: {X_train_scaled.mean():.2f}")
        print(f"Original std: {X_train.std():.2f}")
        print(f"Scaled std: {X_train_scaled.std():.2f}")
        
        # 2. Min-Max Scaling
        print("\n2. Min-Max Scaling:")
        minmax_scaler = MinMaxScaler()
        X_train_minmax = minmax_scaler.fit_transform(X_train)
        print(f"Min value after scaling: {X_train_minmax.min():.2f}")
        print(f"Max value after scaling: {X_train_minmax.max():.2f}")
        
        # 3. Robust Scaling
        print("\n3. Robust Scaling (less sensitive to outliers):")
        robust_scaler = RobustScaler()
        X_train_robust = robust_scaler.fit_transform(X_train)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def demonstrate_feature_selection(self, X_train, y_train, feature_names):
        """Demonstrate feature selection techniques"""
        print("\n" + "="*50)
        print("FEATURE SELECTION TECHNIQUES")
        print("="*50)
        
        # 1. Univariate feature selection
        print("\n1. Univariate Feature Selection (SelectKBest):")
        selector = SelectKBest(score_func=f_classif, k=5)
        X_selected = selector.fit_transform(X_train, y_train)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        print(f"Selected {X_selected.shape[1]} features out of {X_train.shape[1]}")
        print(f"Selected features: {selected_features}")
        
        # 2. Recursive Feature Elimination
        print("\n2. Recursive Feature Elimination (RFE):")
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe = RFE(estimator, n_features_to_select=5)
        X_rfe = rfe.fit_transform(X_train, y_train)
        
        rfe_features = [feature_names[i] for i in range(len(rfe.support_)) if rfe.support_[i]]
        print(f"RFE selected features: {rfe_features}")
        
        # 3. Feature importance from Random Forest
        print("\n3. Feature Importance from Random Forest:")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        
        print("Top 5 most important features:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        return selected_features
    
    def demonstrate_pipelines(self, X_train, X_test, y_train, y_test):
        """Demonstrate pipeline creation"""
        print("\n" + "="*50)
        print("PIPELINE CREATION")
        print("="*50)
        
        # Create a pipeline with preprocessing and classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=10)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Fit the pipeline
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Pipeline accuracy: {accuracy:.4f}")
        
        # Pipeline with different classifier
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', random_state=42))
        ])
        
        svm_pipeline.fit(X_train, y_train)
        svm_pred = svm_pipeline.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        print(f"SVM Pipeline accuracy: {svm_accuracy:.4f}")
        
        return pipeline, svm_pipeline
    
    def demonstrate_cross_validation(self, X, y):
        """Demonstrate cross-validation techniques"""
        print("\n" + "="*50)
        print("CROSS-VALIDATION TECHNIQUES")
        print("="*50)
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        
        # 1. Simple cross-validation
        print("\n1. Stratified 5-Fold Cross-Validation:")
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        print(f"CV scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 2. Cross-validation with different scoring
        print("\n2. Cross-Validation with F1 scoring:")
        f1_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
        print(f"F1 scores: {f1_scores}")
        print(f"Mean F1 score: {f1_scores.mean():.4f}")
        
        return cv_scores
    
    def demonstrate_hyperparameter_tuning(self, X_train, X_test, y_train, y_test):
        """Demonstrate hyperparameter tuning"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # 1. Grid Search
        print("\n1. Grid Search with Random Forest:")
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Test accuracy: {grid_search.score(X_test, y_test):.4f}")
        
        # 2. Randomized Search
        print("\n2. Randomized Search with Random Forest:")
        param_dist = {
            'classifier__n_estimators': [50, 100, 150, 200, 250],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10, 15],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        
        random_search = RandomizedSearchCV(
            pipeline, param_dist, n_iter=10, cv=5, scoring='accuracy',
            random_state=42, n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        print(f"Test accuracy: {random_search.score(X_test, y_test):.4f}")
        
        return grid_search.best_estimator_
    
    def demonstrate_model_evaluation(self, model, X_test, y_test):
        """Demonstrate comprehensive model evaluation"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Classification metrics
        print("\nClassification Metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return y_pred
    
    def demonstrate_model_persistence(self, model, filename):
        """Demonstrate model persistence"""
        print("\n" + "="*50)
        print("MODEL PERSISTENCE")
        print("="*50)
        
        # Save model
        joblib.dump(model, filename)
        print(f"Model saved to: {filename}")
        
        # Load model
        loaded_model = joblib.load(filename)
        print("Model loaded successfully")
        
        return loaded_model
    
    def demonstrate_clustering(self, X):
        """Demonstrate clustering techniques"""
        print("\n" + "="*50)
        print("CLUSTERING TECHNIQUES")
        print("="*50)
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        print(f"Cluster assignments: {np.unique(clusters)}")
        print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
        
        return kmeans, clusters
    
    def create_visualizations(self, df, feature_names):
        """Create visualizations for the analysis"""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scikit-Learn Tools Analysis', fontsize=16)
        
        # 1. Feature distributions
        axes[0, 0].hist(df.iloc[:, 0], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title(f'Distribution of {feature_names[0]}')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Class distribution
        class_counts = df['target'].value_counts()
        axes[0, 1].bar(class_counts.index, class_counts.values, color='lightcoral')
        axes[0, 1].set_title('Class Distribution')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Correlation heatmap (first 5 features)
        corr_matrix = df.iloc[:, :5].corr()
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title('Feature Correlation Matrix')
        axes[1, 0].set_xticks(range(5))
        axes[1, 0].set_yticks(range(5))
        axes[1, 0].set_xticklabels(feature_names[:5], rotation=45)
        axes[1, 0].set_yticklabels(feature_names[:5])
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. PCA visualization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.iloc[:, :-1])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=df['target'], cmap='viridis', alpha=0.7)
        axes[1, 1].set_title('PCA Visualization')
        axes[1, 1].set_xlabel('First Principal Component')
        axes[1, 1].set_ylabel('Second Principal Component')
        plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('scikit_learn_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as 'scikit_learn_analysis.png'")
        
        return fig
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        print("🚀 Starting Scikit-Learn Tools Comprehensive Demo")
        print("=" * 60)
        
        # Load data
        X_class, y_class, feature_names_class, df_class = self.load_classification_data()
        X_reg, y_reg, feature_names_reg, df_reg = self.load_regression_data()
        
        # Classification workflow
        print("\n" + "🎯 CLASSIFICATION WORKFLOW")
        X_train, X_test, y_train, y_test = self.demonstrate_preprocessing(X_class, y_class)
        selected_features = self.demonstrate_feature_selection(X_train, y_train, feature_names_class)
        pipeline1, pipeline2 = self.demonstrate_pipelines(X_train, X_test, y_train, y_test)
        cv_scores = self.demonstrate_cross_validation(X_class, y_class)
        best_model = self.demonstrate_hyperparameter_tuning(X_train, X_test, y_train, y_test)
        predictions = self.demonstrate_model_evaluation(best_model, X_test, y_test)
        
        # Model persistence
        loaded_model = self.demonstrate_model_persistence(best_model, 'best_classifier_model.pkl')
        
        # Clustering
        kmeans_model, clusters = self.demonstrate_clustering(X_class)
        
        # Create visualizations
        fig = self.create_visualizations(df_class, feature_names_class)
        
        # Summary
        print("\n" + "="*50)
        print("SUMMARY OF RESULTS")
        print("="*50)
        print(f"✅ Classification accuracy: {accuracy_score(y_test, best_model.predict(X_test)):.4f}")
        print(f"✅ Best model saved to: best_classifier_model.pkl")
        print(f"✅ Visualizations saved to: scikit_learn_analysis.png")
        print(f"✅ Clustering completed with {len(np.unique(clusters))} clusters")
        
        return {
            'classification_accuracy': accuracy_score(y_test, best_model.predict(X_test)),
            'best_model': best_model,
            'kmeans_model': kmeans_model,
            'visualization_figure': fig
        }

if __name__ == "__main__":
    # Create and run the demo
    demo = ScikitLearnToolsDemo()
    results = demo.run_complete_demo()
    
    print("\n🎉 Scikit-Learn Tools Demo Completed Successfully!")
    print("Check the generated files and plots for detailed results.")