"""
线性回归 (Linear Regression)
实现简单线性回归和多元线性回归
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

class LinearRegressionLab:
    def __init__(self):
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_sample_data(self, n_samples=100, n_features=1, noise=0.1):
        """生成样本数据"""
        np.random.seed(42)
        if n_features == 1:
            # 简单线性回归数据
            X = np.random.randn(n_samples, 1) * 3
            y = 2 * X.ravel() + 3 + np.random.randn(n_samples) * noise
            return X, y
        else:
            # 多元线性回归数据
            X = np.random.randn(n_samples, n_features)
            true_coef = np.array([1.5, -2.0, 0.5, 3.0, -1.0][:n_features])
            y = X @ true_coef + np.random.randn(n_samples) * noise
            return X, y
    
    def fit_model(self, X, y, test_size=0.2):
        """训练线性回归模型"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
    
    def evaluate(self):
        """评估模型性能"""
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        print("=== 线性回归模型评估 ===")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"R² 分数: {r2:.4f}")
        print(f"截距: {self.model.intercept_:.4f}")
        if len(self.model.coef_) == 1:
            print(f"系数: {self.model.coef_[0]:.4f}")
        else:
            print(f"系数: {self.model.coef_}")
            
        return {"mse": mse, "rmse": rmse, "r2": r2}
    
    def visualize_simple_regression(self):
        """可视化简单线性回归"""
        if self.X_train.shape[1] != 1:
            print("仅适用于单变量线性回归")
            return
            
        plt.figure(figsize=(12, 4))
        
        # 训练集
        plt.subplot(1, 2, 1)
        plt.scatter(self.X_train, self.y_train, alpha=0.7, label='Training Data')
        x_line = np.linspace(self.X_train.min(), self.X_train.max(), 100).reshape(-1, 1)
        y_line = self.model.predict(x_line)
        plt.plot(x_line, y_line, 'r-', label='Regression Line')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Training Set')
        plt.legend()
        
        # 测试集
        plt.subplot(1, 2, 2)
        plt.scatter(self.X_test, self.y_test, alpha=0.7, label='Test Data')
        plt.plot(x_line, y_line, 'r-', label='Regression Line')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Test Set')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def residual_analysis(self):
        """残差分析"""
        y_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_pred
        
        plt.figure(figsize=(12, 4))
        
        # 残差图
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Q-Q图
        plt.subplot(1, 2, 2)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        
        plt.tight_layout()
        plt.show()

def demo_linear_regression():
    """演示线性回归"""
    print("=== 简单线性回归演示 ===")
    lab = LinearRegressionLab()
    
    # 简单线性回归
    X, y = lab.generate_sample_data(n_samples=100, n_features=1)
    lab.fit_model(X, y)
    lab.evaluate()
    lab.visualize_simple_regression()
    lab.residual_analysis()
    
    print("\n=== 多元线性回归演示 ===")
    # 多元线性回归
    X_multi, y_multi = lab.generate_sample_data(n_samples=100, n_features=3)
    lab_multi = LinearRegressionLab()
    lab_multi.fit_model(X_multi, y_multi)
    lab_multi.evaluate()

if __name__ == "__main__":
    demo_linear_regression()