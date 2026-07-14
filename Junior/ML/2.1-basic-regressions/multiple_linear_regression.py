"""
多元线性回归 (Multiple Linear Regression)
实现多个自变量预测一个因变量的回归分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class MultipleLinearRegressionLab:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def generate_housing_data(self, n_samples=1000, n_features=8):
        """生成房价预测数据集"""
        np.random.seed(42)
        
        # 模拟房价数据
        size = np.random.normal(1500, 500, n_samples)  # 房屋面积
        bedrooms = np.random.randint(1, 6, n_samples)  # 卧室数量
        bathrooms = np.random.randint(1, 4, n_samples)  # 浴室数量
        age = np.random.randint(0, 50, n_samples)  # 房龄
        garage = np.random.randint(0, 3, n_samples)  # 车库数量
        distance = np.random.normal(10, 5, n_samples)  # 到市中心距离
        school_rating = np.random.randint(1, 11, n_samples)  # 学校评分
        crime_rate = np.random.uniform(0, 0.1, n_samples)  # 犯罪率
        
        # 生成房价（基于多个特征的复杂关系）
        price = (
            50 * size +
            10000 * bedrooms +
            8000 * bathrooms -
            500 * age +
            5000 * garage -
            2000 * distance +
            3000 * school_rating -
            50000 * crime_rate +
            np.random.normal(0, 10000, n_samples)
        )
        
        X = np.column_stack([size, bedrooms, bathrooms, age, garage, 
                          distance, school_rating, crime_rate])
        
        self.feature_names = [
            '房屋面积', '卧室数量', '浴室数量', '房龄', 
            '车库数量', '到市中心距离', '学校评分', '犯罪率'
        ]
        
        return X, price
    
    def generate_economic_data(self, n_samples=500):
        """生成经济数据集"""
        np.random.seed(42)
        
        # 经济指标
        gdp_growth = np.random.normal(2.5, 1, n_samples)  # GDP增长率
        inflation = np.random.normal(2, 0.5, n_samples)  # 通胀率
        unemployment = np.random.normal(5, 1, n_samples)  # 失业率
        interest_rate = np.random.normal(3, 1, n_samples)  # 利率
        consumer_confidence = np.random.normal(100, 10, n_samples)  # 消费者信心指数
        
        # 生成股票收益
        stock_return = (
            0.5 * gdp_growth -
            0.3 * inflation -
            0.4 * unemployment +
            0.2 * interest_rate +
            0.001 * consumer_confidence +
            np.random.normal(0, 2, n_samples)
        )
        
        X = np.column_stack([gdp_growth, inflation, unemployment, 
                          interest_rate, consumer_confidence])
        
        self.feature_names = [
            'GDP增长率', '通胀率', '失业率', '利率', '消费者信心指数'
        ]
        
        return X, stock_return
    
    def load_sample_dataset(self):
        """加载示例数据集"""
        # 创建合成数据集
        np.random.seed(42)
        n_samples = 200
        
        # 特征
        x1 = np.random.uniform(0, 10, n_samples)  # 学习时长
        x2 = np.random.uniform(0, 100, n_samples)  # 平时成绩
        x3 = np.random.uniform(1, 5, n_samples)  # 出勤率
        x4 = np.random.uniform(0, 10, n_samples)  # 作业完成度
        
        # 考试成绩
        y = 2 * x1 + 0.5 * x2 + 3 * x3 + 1.5 * x4 + np.random.normal(0, 5, n_samples)
        
        X = np.column_stack([x1, x2, x3, x4])
        self.feature_names = ['学习时长', '平时成绩', '出勤率', '作业完成度']
        
        return X, y
    
    def fit_model(self, X, y, test_size=0.2, scale=True):
        """训练多元线性回归模型"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if scale:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            self.model.fit(self.X_train, self.y_train)
        else:
            self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        """评估模型性能"""
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # 计算各种指标
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_mse = mean_squared_error(self.y_train, y_pred_train)
        test_mse = mean_squared_error(self.y_test, y_pred_test)
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        print("=== 多元线性回归模型评估 ===")
        print(f"训练集 R²: {train_r2:.4f}")
        print(f"测试集 R²: {test_r2:.4f}")
        print(f"训练集 MSE: {train_mse:.4f}")
        print(f"测试集 MSE: {test_mse:.4f}")
        print(f"训练集 RMSE: {train_rmse:.4f}")
        print(f"测试集 RMSE: {test_rmse:.4f}")
        print(f"训练集 MAE: {train_mae:.4f}")
        print(f"测试集 MAE: {test_mae:.4f}")
        
        # 交叉验证
        if hasattr(self, 'X_train'):
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                      cv=5, scoring='r2')
            print(f"交叉验证 R² (均值±标准差): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae
        }
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        if self.feature_names is None:
            self.feature_names = [f'特征{i+1}' for i in range(len(self.model.coef_))]
        
        # 获取系数
        coefficients = self.model.coef_
        
        # 创建DataFrame便于分析
        coef_df = pd.DataFrame({
            '特征': self.feature_names,
            '系数': coefficients,
            '绝对系数': np.abs(coefficients)
        }).sort_values('绝对系数', ascending=False)
        
        print("\n=== 特征重要性分析 ===")
        print(coef_df)
        
        # 可视化
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        colors = ['green' if c > 0 else 'red' for c in coef_df['系数']]
        plt.barh(coef_df['特征'], coef_df['系数'], color=colors, alpha=0.7)
        plt.xlabel('回归系数')
        plt.title('特征系数值')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.barh(coef_df['特征'], coef_df['绝对系数'], color='skyblue', alpha=0.7)
        plt.xlabel('系数绝对值')
        plt.title('特征重要性排序')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return coef_df
    
    def visualize_regression_results(self):
        """可视化回归结果"""
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测值 vs 实际值
        ax1 = axes[0, 0]
        ax1.scatter(self.y_train, y_pred_train, alpha=0.6, label='训练集', color='blue')
        ax1.scatter(self.y_test, y_pred_test, alpha=0.6, label='测试集', color='red')
        
        # 添加对角线
        min_val = min(ax1.get_xlim()[0], ax1.get_ylim()[0])
        max_val = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predicted vs Actual Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 残差图
        ax2 = axes[0, 1]
        train_residuals = self.y_train - y_pred_train
        test_residuals = self.y_test - y_pred_test
        
        ax2.scatter(y_pred_train, train_residuals, alpha=0.6, label='训练集', color='blue')
        ax2.scatter(y_pred_test, test_residuals, alpha=0.6, label='测试集', color='red')
        ax2.axhline(y=0, color='k', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差分布
        ax3 = axes[1, 0]
        ax3.hist(train_residuals, bins=30, alpha=0.7, label='训练集', color='blue')
        ax3.hist(test_residuals, bins=30, alpha=0.7, label='测试集', color='red')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residual Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q图
        ax4 = axes[1, 1]
        from scipy import stats
        stats.probplot(train_residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Training Residuals)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def detect_multicollinearity(self):
        """检测多重共线性"""
        if self.X_train is None:
            print("请先训练模型")
            return
        
        # 计算特征间的相关系数矩阵
        corr_matrix = np.corrcoef(self.X_train.T)
        
        plt.figure(figsize=(15, 6))
        
        # VIF值
        plt.subplot(1, 2, 1)
        vif_data = pd.DataFrame()
        vif_data["特征"] = self.feature_names
        vif_data["VIF"] = [variance_inflation_factor(self.X_train, i) 
                          for i in range(self.X_train.shape[1])]
        vif_data.plot(kind='bar', x='特征', y='VIF', ax=plt.gca(), legend=False)
        plt.axhline(y=10, color='r', linestyle='--', label='VIF=10')
        plt.xlabel('Features')
        plt.ylabel('VIF Value')
        plt.title('Variance Inflation Factor (VIF)')
        plt.legend()
        
        # 相关性矩阵
        plt.subplot(1, 2, 2)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                   xticklabels=self.feature_names,
                   yticklabels=self.feature_names,
                   cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
        
        print("\n=== 多重共线性检测 (VIF) ===")
        print(vif_data)

        print("\n=== Correlation Analysis ===")
        print("High Correlation Feature Pairs:")
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix[i,j]) > 0.8:
                    print(f"{self.feature_names[i]} - {self.feature_names[j]}: {corr_matrix[i,j]:.3f}")

        return vif_data
    
    def stepwise_selection(self, X, y, threshold_in=0.05, threshold_out=0.1):
        """逐步回归特征选择"""
        from sklearn.feature_selection import f_regression
        
        n_samples, n_features = X.shape
        included = []
        
        while True:
            changed = False
            
            # 前向选择
            excluded = list(set(range(n_features)) - set(included))
            new_pval = pd.Series(index=excluded)
            
            for new_column in excluded:
                model = LinearRegression()
                model.fit(X[:, included + [new_column]], y)
                
                _, pval = f_regression(X[:, included + [new_column]], y)
                new_pval[new_column] = pval[-1]
            
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                print(f"添加特征 {self.feature_names[best_feature]} (p-value: {best_pval:.4f})")
            
            # 后向消除
            if len(included) > 1:
                model = LinearRegression()
                model.fit(X[:, included], y)
                
                _, pvals = f_regression(X[:, included], y)
                worst_pval = pvals.max()
                
                if worst_pval > threshold_out:
                    worst_feature = included[np.argmax(pvals)]
                    included.remove(worst_feature)
                    changed = True
                    print(f"移除特征 {self.feature_names[worst_feature]} (p-value: {worst_pval:.4f})")
            
            if not changed:
                break
        
        return included
    
    def polynomial_multiple_regression(self, X, y, degree=2):
        """多项式多元回归"""
        # 创建多项式特征
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # 获取特征名称
        feature_names = poly.get_feature_names_out(self.feature_names)
        
        # 训练模型
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_poly, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        print(f"\n=== 多项式多元回归 (degree={degree}) ===")
        print(f"原始特征数: {X.shape[1]}")
        print(f"多项式特征数: {X_poly.shape[1]}")
        
        return self.evaluate_model()

def demo_multiple_linear_regression():
    """演示多元线性回归"""
    print("=== 多元线性回归演示 ===")
    
    lab = MultipleLinearRegressionLab()
    
    # 1. 示例数据集演示
    print("\n1. 学生成绩预测示例")
    X, y = lab.load_sample_dataset()
    lab.fit_model(X, y)
    lab.evaluate_model()
    lab.analyze_feature_importance()
    lab.visualize_regression_results()
    
    # 2. 房价预测演示
    print("\n2. 房价预测示例")
    X_housing, y_housing = lab.generate_housing_data()
    lab_housing = MultipleLinearRegressionLab()
    lab_housing.feature_names = [
        '房屋面积', '卧室数量', '浴室数量', '房龄', 
        '车库数量', '到市中心距离', '学校评分', '犯罪率'
    ]
    lab_housing.fit_model(X_housing, y_housing)
    lab_housing.evaluate_model()
    lab_housing.analyze_feature_importance()
    lab_housing.detect_multicollinearity()
    
    # 3. 经济数据演示
    print("\n3. 股票收益预测示例")
    X_econ, y_econ = lab.generate_economic_data()
    lab_econ = MultipleLinearRegressionLab()
    lab_econ.feature_names = ['GDP增长率', '通胀率', '失业率', '利率', '消费者信心指数']
    lab_econ.fit_model(X_econ, y_econ)
    lab_econ.evaluate_model()
    lab_econ.analyze_feature_importance()

if __name__ == "__main__":
    demo_multiple_linear_regression()