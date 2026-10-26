"""
主演示文件 - 综合所有回归方法
运行所有回归算法的演示
"""

import warnings
warnings.filterwarnings('ignore')

from linear_regression import LinearRegressionLab
from logistic_regression import LogisticRegressionLab
from polynomial_regression import PolynomialRegressionLab
from regularized_regression import RegularizedRegressionLab
from multiple_linear_regression import MultipleLinearRegressionLab

def print_header(title):
    """打印标题装饰"""
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

def main():
    """主函数 - 运行所有演示"""
    
    print_header("机器学习回归算法综合演示")
    
    # 1. 线性回归演示
    print_header("1. 线性回归 (Linear Regression)")
    try:
        from linear_regression import demo_linear_regression
        demo_linear_regression()
    except Exception as e:
        print(f"线性回归演示出错: {e}")
    
    input("\n按Enter键继续下一个演示...")
    
    # 2. 逻辑回归演示
    print_header("2. 逻辑回归 (Logistic Regression)")
    try:
        from logistic_regression import demo_logistic_regression
        demo_logistic_regression()
    except Exception as e:
        print(f"逻辑回归演示出错: {e}")
    
    input("\n按Enter键继续下一个演示...")
    
    # 3. 多项式回归演示
    print_header("3. 多项式回归 (Polynomial Regression)")
    try:
        from polynomial_regression import demo_polynomial_regression
        demo_polynomial_regression()
    except Exception as e:
        print(f"多项式回归演示出错: {e}")
    
    input("\n按Enter键继续下一个演示...")
    
    # 4. 正则化回归演示
    print_header("4. 正则化回归 (Ridge, Lasso, Elastic Net)")
    try:
        from regularized_regression import demo_regularized_regression
        demo_regularized_regression()
    except Exception as e:
        print(f"正则化回归演示出错: {e}")
    
    input("\n按Enter键继续下一个演示...")
    
    # 5. 多元线性回归演示
    print_header("5. 多元线性回归 (Multiple Linear Regression)")
    try:
        from multiple_linear_regression import demo_multiple_linear_regression
        demo_multiple_linear_regression()
    except Exception as e:
        print(f"多元线性回归演示出错: {e}")
    
    print_header("所有演示完成！")
    print("\n总结:")
    print("- 线性回归: 基础回归模型，适用于连续变量预测")
    print("- 逻辑回归: 二分类问题，使用sigmoid函数")
    print("- 多项式回归: 非线性关系建模，需注意过拟合")
    print("- 正则化回归: 防止过拟合，特征选择")
    print("- 多元线性回归: 多个自变量预测一个因变量")

if __name__ == "__main__":
    main()