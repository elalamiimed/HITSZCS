# 机器学习回归算法实验

本实验涵盖了机器学习中的主要回归算法，包括线性回归、逻辑回归、多项式回归、正则化回归和多类别回归。

## 📋 实验内容

### 1. 线性回归 (Linear Regression)
- **文件**: `linear_regression.py`
- **功能**: 
  - 简单线性回归（单变量）
  - 多元线性回归（多变量）
  - 模型评估指标（MSE, RMSE, R²）
  - 残差分析和可视化

### 2. 逻辑回归 (Logistic Regression)
- **文件**: `logistic_regression.py`
- **功能**:
  - 二分类逻辑回归
  - 多分类逻辑回归
  - Sigmoid函数可视化
  - 决策边界绘制

### 3. 多项式回归 (Polynomial Regression)
- **文件**: `polynomial_regression.py`
- **功能**:
  - 不同阶数的多项式拟合
  - 偏差-方差权衡分析
  - 学习曲线绘制
  - 过拟合检测

### 4. 正则化回归 (Regularized Regression)
- **文件**: `regularized_regression.py`
- **功能**:
  - Ridge回归 (L2正则化)
  - Lasso回归 (L1正则化)
  - Elastic Net回归 (L1+L2正则化)
  - 系数收缩可视化
  - 超参数调优

### 5. 多元线性回归 (Multiple Linear Regression)
- **文件**: `multiple_linear_regression.py`
- **功能**:
  - 多个自变量预测一个因变量
  - 特征重要性分析
  - 多重共线性检测
  - 逐步回归特征选择
  - 多种实际数据集示例

## 🚀 快速开始

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 运行演示

#### 方法1：逐个运行
```bash
# 运行单个算法演示
python linear_regression.py
python logistic_regression.py
python polynomial_regression.py
python regularized_regression.py
python multiple_linear_regression.py
```

#### 方法2：综合演示
```bash
# 运行所有算法演示
python main_demo.py
```

## 📊 算法对比

| 算法类型 | 适用场景 | 优点 | 缺点 |
|---------|---------|------|------|
| 线性回归 | 连续变量预测 | 简单、可解释性强 | 假设线性关系 |
| 逻辑回归 | 二分类问题 | 概率输出、计算高效 | 线性决策边界 |
| 多项式回归 | 非线性关系 | 拟合复杂模式 | 易过拟合 |
| Ridge回归 | 多重共线性 | 稳定系数估计 | 不能特征选择 |
| Lasso回归 | 特征选择 | 自动特征选择 | 可能过度稀疏 |
| Elastic Net | 高维数据 | 结合Ridge和Lasso优点 | 需要调两个参数 |

## 🔍 关键概念

### 1. 线性回归
- **目标**: 找到最佳拟合直线
- **损失函数**: 最小二乘法
- **评估指标**: R², MSE, RMSE

### 2. 逻辑回归
- **激活函数**: Sigmoid函数
- **损失函数**: 对数损失
- **输出**: 概率值 (0-1之间)

### 3. 正则化
- **Ridge (L2)**: 系数平方和惩罚
- **Lasso (L1)**: 系数绝对值和惩罚
- **Elastic Net**: L1和L2的组合

### 4. 多元线性回归
- **目标**: 多个自变量预测一个因变量
- **假设**: 线性关系
- **挑战**: 多重共线性、特征选择

## 📈 可视化输出

每个实验都会生成以下可视化：
- 数据分布图
- 模型拟合曲线
- 决策边界
- 学习曲线
- 混淆矩阵
- 系数路径图

## 🎯 学习目标

完成本实验后，你将掌握：
1. 各种回归算法的原理和实现
2. 如何选择合适的回归模型
3. 如何处理过拟合和欠拟合
4. 如何进行特征选择和超参数调优
5. 多类别分类的不同策略

## 📚 参考资料

- [scikit-learn官方文档](https://scikit-learn.org/)
- 《统计学习方法》
- 《机器学习》周志华
- 《Pattern Recognition and Machine Learning》

## 🐛 故障排除

如果遇到问题：
1. 确保所有依赖已正确安装
2. 检查Python版本 (推荐3.8+)
3. 查看错误信息，通常是数据维度不匹配
4. 尝试减少样本量或特征数

## 📝 扩展练习

1. 尝试不同的数据集
2. 调整超参数观察效果
3. 实现交叉验证
4. 添加特征工程步骤
5. 比较不同正则化参数的影响