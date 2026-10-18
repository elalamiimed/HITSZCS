"""
测试安装和环境配置
"""

try:
    import numpy as np
    print("✓ numpy 已安装")
except ImportError:
    print("✗ numpy 未安装")

try:
    import pandas as pd
    print("✓ pandas 已安装")
except ImportError:
    print("✗ pandas 未安装")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib 已安装")
except ImportError:
    print("✗ matplotlib 未安装")

try:
    import seaborn as sns
    print("✓ seaborn 已安装")
except ImportError:
    print("✗ seaborn 未安装")

try:
    import sklearn
    print("✓ scikit-learn 已安装")
    print(f"   版本: {sklearn.__version__}")
except ImportError:
    print("✗ scikit-learn 未安装")

try:
    import scipy
    print("✓ scipy 已安装")
except ImportError:
    print("✗ scipy 未安装")

print("\n环境测试完成！")