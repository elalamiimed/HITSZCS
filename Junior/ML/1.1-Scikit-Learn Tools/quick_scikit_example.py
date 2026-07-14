#!/usr/bin/env python3
"""
Quick Scikit-Learn Tools Example
===============================

A concise example showing essential Scikit-Learn tools in action.
Perfect for getting started quickly!
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def quick_demo():
    """Quick demonstration of essential Scikit-Learn tools"""
    
    print("🚀 Quick Scikit-Learn Demo")
    print("-" * 30)
    
    # 1. Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    print(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 4. Train model
    pipeline.fit(X_train, y_train)
    print("✅ Model trained successfully!")
    
    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # 6. Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # 7. Detailed classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # 8. Save model
    joblib.dump(pipeline, 'quick_iris_model.pkl')
    print("💾 Model saved as 'quick_iris_model.pkl'")
    
    # 9. Load and test saved model
    loaded_model = joblib.load('quick_iris_model.pkl')
    test_accuracy = loaded_model.score(X_test, y_test)
    print(f"🔍 Loaded model accuracy: {test_accuracy:.4f}")
    
    print("\n🎉 Demo completed! Check the saved model file.")
    
    return pipeline

if __name__ == "__main__":
    model = quick_demo()