"""
逻辑回归 (Logistic Regression)
实现二分类和多分类逻辑回归
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class LogisticRegressionLab:
    def __init__(self, multi_class='auto'):
        self.model = LogisticRegression(multi_class=multi_class, random_state=42)
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_binary_data(self, n_samples=300):
        """生成二分类数据"""
        np.random.seed(42)
        X, y = make_classification(n_samples=n_samples, n_features=2, 
                                 n_redundant=0, n_informative=2, 
                                 n_clusters_per_class=1, random_state=42)
        return X, y
    
    def generate_multiclass_data(self, n_samples=300, n_classes=3):
        """生成多分类数据"""
        np.random.seed(42)
        X, y = make_blobs(n_samples=n_samples, centers=n_classes, 
                         n_features=2, random_state=42, cluster_std=1.5)
        return X, y
    
    def fit_model(self, X, y, test_size=0.2, scale=True):
        """训练逻辑回归模型"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if scale:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, X):
        """预测"""
        if hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)
    
    def evaluate(self):
        """评估模型性能"""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print("=== 逻辑回归模型评估 ===")
        print(f"准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(self.y_test, y_pred))
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return {"accuracy": accuracy}
    
    def visualize_binary_classification(self):
        """可视化二分类结果"""
        if len(np.unique(self.y_train)) != 2:
            print("仅适用于二分类问题")
            return
            
        plt.figure(figsize=(15, 5))
        
        # 决策边界
        plt.subplot(1, 3, 1)
        h = 0.02
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        Z = self.model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, 
                   cmap='RdYlBu', edgecolors='black')
        plt.title('决策边界')
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        
        # 训练集结果
        plt.subplot(1, 3, 2)
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, 
                   cmap='RdYlBu', edgecolors='black')
        plt.title('训练集')
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        
        # 测试集结果
        plt.subplot(1, 3, 3)
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, 
                   cmap='RdYlBu', edgecolors='black')
        plt.title('True Labels')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_multiclass_classification(self):
        """可视化多分类结果"""
        n_classes = len(np.unique(self.y_train))
        if n_classes <= 2:
            print("仅适用于多分类问题")
            return
            
        plt.figure(figsize=(15, 5))
        
        # 决策边界
        plt.subplot(1, 3, 1)
        h = 0.02
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        scatter = plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, 
                            cmap='viridis', edgecolors='black')
        plt.title('决策边界')
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.colorbar(scatter)
        
        # 训练集
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, 
                            cmap='viridis', edgecolors='black')
        plt.title('训练集')
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.colorbar(scatter)
        
        # 测试集
        plt.subplot(1, 3, 3)
        scatter = plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, 
                            cmap='viridis', edgecolors='black')
        plt.title('True Labels')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
    
    def sigmoid_function(self, z):
        """sigmoid函数"""
        return 1 / (1 + np.exp(-z))
    
    def plot_sigmoid(self):
        """绘制sigmoid函数"""
        z = np.linspace(-10, 10, 100)
        sigmoid = self.sigmoid_function(z)
        
        plt.figure(figsize=(8, 6))
        plt.plot(z, sigmoid, 'b-', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.title('Sigmoid函数')
        plt.xlabel('z')
        plt.ylabel('σ(z)')
        plt.ylim(-0.1, 1.1)
        plt.show()

def demo_logistic_regression():
    """演示逻辑回归"""
    print("=== 二分类逻辑回归演示 ===")
    lab_binary = LogisticRegressionLab()
    
    # 二分类
    X_binary, y_binary = lab_binary.generate_binary_data(n_samples=300)
    lab_binary.fit_model(X_binary, y_binary)
    lab_binary.evaluate()
    lab_binary.visualize_binary_classification()
    lab_binary.plot_sigmoid()
    
    print("\n=== 多分类逻辑回归演示 ===")
    lab_multi = LogisticRegressionLab(multi_class='multinomial')
    
    # 多分类
    X_multi, y_multi = lab_multi.generate_multiclass_data(n_samples=300, n_classes=4)
    lab_multi.fit_model(X_multi, y_multi)
    lab_multi.evaluate()
    lab_multi.visualize_multiclass_classification()

if __name__ == "__main__":
    demo_logistic_regression()