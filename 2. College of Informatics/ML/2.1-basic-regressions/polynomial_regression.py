"""
多项式回归 (Polynomial Regression)
实现多项式回归和非线性拟合
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import seaborn as sns

class PolynomialRegressionLab:
    def __init__(self):
        self.models = {}
        self.degrees = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_nonlinear_data(self, n_samples=100, noise=0.1, func_type='quadratic'):
        """生成非线性数据"""
        np.random.seed(42)
        X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
        
        if func_type == 'quadratic':
            y = 0.5 * X.ravel()**2 + X.ravel() + 2 + np.random.randn(n_samples) * noise
        elif func_type == 'cubic':
            y = 0.1 * X.ravel()**3 - 0.5 * X.ravel()**2 - X.ravel() + 1 + np.random.randn(n_samples) * noise
        elif func_type == 'sinusoidal':
            y = np.sin(X.ravel()) + 0.5 * X.ravel() + np.random.randn(n_samples) * noise
        elif func_type == 'complex':
            y = 0.1 * X.ravel()**4 - 0.5 * X.ravel()**3 + X.ravel()**2 + np.random.randn(n_samples) * noise
        
        return X, y
    
    def fit_polynomial_models(self, X, y, degrees=[1, 2, 3, 4, 5], test_size=0.2):
        """训练不同阶数的多项式回归模型"""
        self.degrees = degrees
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        for degree in degrees:
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('linear', LinearRegression())
            ])
            model.fit(self.X_train, self.y_train)
            self.models[degree] = model
    
    def evaluate_models(self):
        """评估不同阶数的模型"""
        results = {}
        
        print("=== 多项式回归模型评估 ===")
        print("阶数\t训练R²\t测试R²\t训练RMSE\t测试RMSE")
        print("-" * 60)
        
        for degree in self.degrees:
            model = self.models[degree]
            
            # 训练集预测
            y_train_pred = model.predict(self.X_train)
            train_r2 = r2_score(self.y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            
            # 测试集预测
            y_test_pred = model.predict(self.X_test)
            test_r2 = r2_score(self.y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            
            results[degree] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            }
            
            print(f"{degree}\t{train_r2:.4f}\t{test_r2:.4f}\t{train_rmse:.4f}\t{test_rmse:.4f}")
        
        return results
    
    def visualize_polynomial_fits(self):
        """可视化多项式拟合"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # 绘制不同阶数的拟合曲线
        X_plot = np.linspace(self.X_train.min(), self.X_train.max(), 300).reshape(-1, 1)
        
        for i, degree in enumerate(self.degrees):
            if i < len(axes):
                model = self.models[degree]
                
                # 训练数据
                axes[i].scatter(self.X_train, self.y_train, alpha=0.6, label='训练数据')
                axes[i].scatter(self.X_test, self.y_test, alpha=0.6, color='red', label='测试数据')
                
                # 拟合曲线
                y_plot = model.predict(X_plot)
                axes[i].plot(X_plot, y_plot, 'g-', linewidth=2, label=f'阶数 {degree}')
                
                axes[i].set_xlabel('X')
                axes[i].set_ylabel('y')
                axes[i].set_title(f'多项式回归 (阶数 {degree})')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(self.degrees), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self):
        """绘制学习曲线"""
        plt.figure(figsize=(12, 5))
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = {}
        val_scores = {}
        
        for degree in self.degrees:
            train_scores[degree] = []
            val_scores[degree] = []
            
            for train_size in train_sizes:
                n_samples = int(len(self.X_train) * train_size)
                if n_samples > 0:
                    X_subset = self.X_train[:n_samples]
                    y_subset = self.y_train[:n_samples]
                    
                    model = Pipeline([
                        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                        ('linear', LinearRegression())
                    ])
                    model.fit(X_subset, y_subset)
                    
                    # 训练集得分
                    train_pred = model.predict(X_subset)
                    train_score = r2_score(y_subset, train_pred)
                    train_scores[degree].append(train_score)
                    
                    # 验证集得分
                    val_pred = model.predict(self.X_test)
                    val_score = r2_score(self.y_test, val_pred)
                    val_scores[degree].append(val_score)
        
        # 绘制学习曲线
        plt.subplot(1, 2, 1)
        for degree in self.degrees:
            plt.plot(train_sizes * 100, train_scores[degree], 'o-', label=f'Training Set (Degree {degree})')
        plt.xlabel('Training Samples (%)')
        plt.ylabel('R² Score')
        plt.title('Training Set Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for degree in self.degrees:
            plt.plot(train_sizes * 100, val_scores[degree], 'o-', label=f'Validation Set (Degree {degree})')
        plt.xlabel('Training Samples (%)')
        plt.ylabel('R² Score')
        plt.title('Validation Set Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_bias_variance_tradeoff(self):
        """分析偏差-方差权衡"""
        degrees_extended = range(1, 11)
        train_r2_scores = []
        test_r2_scores = []
        
        for degree in degrees_extended:
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('linear', LinearRegression())
            ])
            model.fit(self.X_train, self.y_train)
            
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            
            train_r2_scores.append(r2_score(self.y_train, train_pred))
            test_r2_scores.append(r2_score(self.y_test, test_pred))
        
        plt.figure(figsize=(10, 6))
        plt.plot(degrees_extended, train_r2_scores, 'o-', label='Training R²', linewidth=2)
        plt.plot(degrees_extended, test_r2_scores, 's-', label='Testing R²', linewidth=2)
        plt.xlabel('Polynomial Degree')
        plt.ylabel('R² Score')
        plt.title('Bias-Variance Tradeoff Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(degrees_extended)
        plt.show()
    
    def find_best_degree(self, max_degree=10):
        """找到最佳多项式阶数"""
        degrees_extended = range(1, max_degree + 1)
        test_scores = []
        
        for degree in degrees_extended:
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('linear', LinearRegression())
            ])
            model.fit(self.X_train, self.y_train)
            
            test_pred = model.predict(self.X_test)
            test_score = r2_score(self.y_test, test_pred)
            test_scores.append(test_score)
        
        best_degree = degrees_extended[np.argmax(test_scores)]
        best_score = max(test_scores)
        
        print(f"最佳多项式阶数: {best_degree}")
        print(f"测试集 R² 分数: {best_score:.4f}")
        
        return best_degree, best_score

def demo_polynomial_regression():
    """演示多项式回归"""
    print("=== 多项式回归演示 ===")
    
    # 二次函数
    lab = PolynomialRegressionLab()
    X, y = lab.generate_nonlinear_data(n_samples=100, func_type='quadratic')
    lab.fit_polynomial_models(X, y, degrees=[1, 2, 3, 4, 5])
    lab.evaluate_models()
    lab.visualize_polynomial_fits()
    lab.plot_learning_curves()
    lab.analyze_bias_variance_tradeoff()
    lab.find_best_degree()
    
    print("\n=== 复杂函数演示 ===")
    # 复杂函数
    lab_complex = PolynomialRegressionLab()
    X_complex, y_complex = lab_complex.generate_nonlinear_data(n_samples=150, func_type='complex')
    lab_complex.fit_polynomial_models(X_complex, y_complex, degrees=[1, 3, 5, 7, 9])
    lab_complex.evaluate_models()
    lab_complex.visualize_polynomial_fits()

if __name__ == "__main__":
    demo_polynomial_regression()