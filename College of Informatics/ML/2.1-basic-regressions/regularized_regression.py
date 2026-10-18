"""
正则化回归 (Regularized Regression)
实现Ridge、Lasso和Elastic Net回归
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, validation_curve
import seaborn as sns

class RegularizedRegressionLab:
    def __init__(self):
        self.ridge_model = None
        self.lasso_model = None
        self.elastic_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_high_dimensional_data(self, n_samples=100, n_features=20, noise=0.1, 
                                     redundant_features=True):
        """生成高维数据用于演示正则化"""
        np.random.seed(42)
        
        if redundant_features:
            # 创建具有冗余特征的数据
            X = np.random.randn(n_samples, n_features)
            
            # 只有少数特征真正相关
            true_coef = np.zeros(n_features)
            true_coef[:5] = [3, -2, 1.5, -1, 0.5]  # 前5个特征有效
            
            y = X @ true_coef + np.random.randn(n_samples) * noise
            
            # 添加高度相关的特征
            X = np.hstack([X, X[:, :3] + np.random.randn(n_samples, 3) * 0.01])
            
            return X, y
        else:
            # 标准线性数据
            X = np.random.randn(n_samples, n_features)
            true_coef = np.random.randn(n_features) * 0.5
            y = X @ true_coef + np.random.randn(n_samples) * noise
            return X, y
    
    def generate_polynomial_data(self, n_samples=100, noise=0.1):
        """生成多项式数据用于过拟合演示"""
        np.random.seed(42)
        X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
        y = 0.5 * X.ravel()**2 + X.ravel() + 2 + np.random.randn(n_samples) * noise
        
        # 创建高阶多项式特征
        poly = PolynomialFeatures(degree=15, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        return X_poly, y, X
    
    def fit_regularized_models(self, X, y, test_size=0.2, alphas=None):
        """训练正则化模型"""
        if alphas is None:
            alphas = [0.001, 0.01, 0.1, 1, 10, 100]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 标准化数据
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        self.ridge_results = {}
        self.lasso_results = {}
        self.elastic_results = {}
        
        for alpha in alphas:
            # Ridge回归
            ridge = Ridge(alpha=alpha)
            ridge.fit(self.X_train, self.y_train)
            self.ridge_results[alpha] = {
                'model': ridge,
                'train_score': ridge.score(self.X_train, self.y_train),
                'test_score': ridge.score(self.X_test, self.y_test),
                'coef': ridge.coef_
            }
            
            # Lasso回归
            lasso = Lasso(alpha=alpha)
            lasso.fit(self.X_train, self.y_train)
            self.lasso_results[alpha] = {
                'model': lasso,
                'train_score': lasso.score(self.X_train, self.y_train),
                'test_score': lasso.score(self.X_test, self.y_test),
                'coef': lasso.coef_,
                'n_zero_coef': np.sum(lasso.coef_ == 0)
            }
            
            # Elastic Net回归
            elastic = ElasticNet(alpha=alpha, l1_ratio=0.5)  # 平衡L1和L2
            elastic.fit(self.X_train, self.y_train)
            self.elastic_results[alpha] = {
                'model': elastic,
                'train_score': elastic.score(self.X_train, self.y_train),
                'test_score': elastic.score(self.X_test, self.y_test),
                'coef': elastic.coef_,
                'n_zero_coef': np.sum(elastic.coef_ == 0)
            }
    
    def compare_regularization_methods(self):
        """比较不同正则化方法"""
        alphas = list(self.ridge_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 模型性能比较
        ax1 = axes[0, 0]
        ridge_scores = [self.ridge_results[a]['test_score'] for a in alphas]
        lasso_scores = [self.lasso_results[a]['test_score'] for a in alphas]
        elastic_scores = [self.elastic_results[a]['test_score'] for a in alphas]
        
        ax1.plot(alphas, ridge_scores, 'o-', label='Ridge', linewidth=2)
        ax1.plot(alphas, lasso_scores, 's-', label='Lasso', linewidth=2)
        ax1.plot(alphas, elastic_scores, '^-', label='Elastic Net', linewidth=2)
        ax1.set_xlabel('正则化参数 α')
        ax1.set_ylabel('测试集 R² 分数')
        ax1.set_title('不同正则化方法的性能比较')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 系数路径 - Ridge
        ax2 = axes[0, 1]
        for i in range(min(10, len(self.ridge_results[alphas[0]]['coef']))):
            coef_path = [self.ridge_results[a]['coef'][i] for a in alphas]
            ax2.plot(alphas, coef_path, label=f'Feature {i+1}')
        ax2.set_xlabel('Regularization Parameter α')
        ax2.set_ylabel('Coefficient Value')
        ax2.set_title('Ridge Coefficient Path')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 系数路径 - Lasso
        ax3 = axes[1, 0]
        for i in range(min(10, len(self.lasso_results[alphas[0]]['coef']))):
            coef_path = [self.lasso_results[a]['coef'][i] for a in alphas]
            ax3.plot(alphas, coef_path, label=f'Feature {i+1}')
        ax3.set_xlabel('Regularization Parameter α')
        ax3.set_ylabel('Coefficient Value')
        ax3.set_title('Lasso Coefficient Path')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 零系数数量
        ax4 = axes[1, 1]
        lasso_zeros = [self.lasso_results[a]['n_zero_coef'] for a in alphas]
        elastic_zeros = [self.elastic_results[a]['n_zero_coef'] for a in alphas]
        
        ax4.plot(alphas, lasso_zeros, 'o-', label='Lasso', linewidth=2)
        ax4.plot(alphas, elastic_zeros, 's-', label='Elastic Net', linewidth=2)
        ax4.set_xlabel('正则化参数 α')
        ax4.set_ylabel('零系数数量')
        ax4.set_title('特征选择效果')
        ax4.set_xscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_coefficient_shrinkage(self):
        """可视化系数收缩"""
        from sklearn.linear_model import LinearRegression
        # 获取无正则化的系数
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        original_coef = lr.coef_

        # 获取不同正则化方法的系数
        alpha = 1.0
        ridge_coef = self.ridge_results[alpha]['coef']
        lasso_coef = self.lasso_results[alpha]['coef']
        elastic_coef = self.elastic_results[alpha]['coef']

        plt.figure(figsize=(15, 8))

        # 系数比较
        feature_indices = range(len(original_coef))

        plt.subplot(2, 1, 1)
        plt.bar([i-0.3 for i in feature_indices], original_coef, width=0.2, 
                label='No Regularization', alpha=0.8)
        plt.bar([i-0.1 for i in feature_indices], ridge_coef, width=0.2, 
                label='Ridge', alpha=0.8)
        plt.bar([i+0.1 for i in feature_indices], lasso_coef, width=0.2, 
                label='Lasso', alpha=0.8)
        plt.bar([i+0.3 for i in feature_indices], elastic_coef, width=0.2, 
                label='Elastic Net', alpha=0.8)

        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.title('Coefficient Comparison Across Regularization Methods (α=1.0)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 系数绝对值
        plt.subplot(2, 1, 2)
        plt.bar([i-0.3 for i in feature_indices], np.abs(original_coef), width=0.2, 
                label='No Regularization', alpha=0.8)
        plt.bar([i-0.1 for i in feature_indices], np.abs(ridge_coef), width=0.2, 
                label='Ridge', alpha=0.8)
        plt.bar([i+0.1 for i in feature_indices], np.abs(lasso_coef), width=0.2, 
                label='Lasso', alpha=0.8)
        plt.bar([i+0.3 for i in feature_indices], np.abs(elastic_coef), width=0.2, 
                label='Elastic Net', alpha=0.8)

        plt.xlabel('Feature Index')
        plt.ylabel('Absolute Coefficient Value')
        plt.title('Absolute Coefficient Comparison Across Regularization Methods (α=1.0)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    def plot_validation_curves(self):
        """绘制验证曲线"""
        param_range = [0.001, 0.01, 0.1, 1, 10, 100]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Ridge验证曲线
        train_scores, val_scores = validation_curve(
            Ridge(), self.X_train, self.y_train,
            param_name='alpha', param_range=param_range,
            cv=5, scoring='r2'
        )
        
        axes[0].plot(param_range, np.mean(train_scores, axis=1), 'o-', label='训练得分')
        axes[0].plot(param_range, np.mean(val_scores, axis=1), 's-', label='验证得分')
        axes[0].set_xlabel('正则化参数 α')
        axes[0].set_ylabel('R² 分数')
        axes[0].set_title('Ridge回归验证曲线')
        axes[0].set_xscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Lasso验证曲线
        train_scores, val_scores = validation_curve(
            Lasso(max_iter=10000), self.X_train, self.y_train,
            param_name='alpha', param_range=param_range,
            cv=5, scoring='r2'
        )
        
        axes[1].plot(param_range, np.mean(train_scores, axis=1), 'o-', label='训练得分')
        axes[1].plot(param_range, np.mean(val_scores, axis=1), 's-', label='验证得分')
        axes[1].set_xlabel('正则化参数 α')
        axes[1].set_ylabel('R² 分数')
        axes[1].set_title('Lasso回归验证曲线')
        axes[1].set_xscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Elastic Net验证曲线
        train_scores, val_scores = validation_curve(
            ElasticNet(max_iter=10000), self.X_train, self.y_train,
            param_name='alpha', param_range=param_range,
            cv=5, scoring='r2'
        )
        
        axes[2].plot(param_range, np.mean(train_scores, axis=1), 'o-', label='训练得分')
        axes[2].plot(param_range, np.mean(val_scores, axis=1), 's-', label='验证得分')
        axes[2].set_xlabel('正则化参数 α')
        axes[2].set_ylabel('R² 分数')
        axes[2].set_title('Elastic Net验证曲线')
        axes[2].set_xscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def select_optimal_alpha(self, method='ridge'):
        """选择最优的正则化参数"""
        from sklearn.model_selection import GridSearchCV
        
        if method == 'ridge':
            model = Ridge()
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
        elif method == 'lasso':
            model = Lasso(max_iter=10000)
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
        else:  # elastic net
            model = ElasticNet(max_iter=10000)
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1, 10],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"=== {method.upper()} 最优参数 ===")
        print(f"最优参数: {grid_search.best_params_}")
        print(f"最优得分: {grid_search.best_score_:.4f}")
        print(f"测试集得分: {grid_search.score(self.X_test, self.y_test):.4f}")
        
        return grid_search.best_params_, grid_search.best_score_

def demo_regularized_regression():
    """演示正则化回归"""
    print("=== 正则化回归演示 ===")
    
    # 高维数据演示
    lab = RegularizedRegressionLab()
    X, y = lab.generate_high_dimensional_data(n_samples=100, n_features=20)
    lab.fit_regularized_models(X, y)
    lab.compare_regularization_methods()
    lab.visualize_coefficient_shrinkage()
    lab.plot_validation_curves()
    
    # 选择最优参数
    lab.select_optimal_alpha('ridge')
    lab.select_optimal_alpha('lasso')
    lab.select_optimal_alpha('elastic')
    
    print("\n=== 多项式过拟合演示 ===")
    # 多项式过拟合演示
    lab_poly = RegularizedRegressionLab()
    X_poly, y_poly, _ = lab_poly.generate_polynomial_data(n_samples=100)
    lab_poly.fit_regularized_models(X_poly, y_poly, alphas=[0.001, 0.01, 0.1, 1, 10])
    lab_poly.compare_regularization_methods()

if __name__ == "__main__":
    demo_regularized_regression()