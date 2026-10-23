import numpy as np
import time
from typing import List, Tuple
import requests
import gzip
import os
from urllib.parse import urljoin
from pathlib import Path

class MLP:
    def __init__(self, layer_sizes: List[int], activation: str = 'sigmoid'):
        """
        初始化MLP模型
        
        参数:
            layer_sizes: 包含各层神经元数量的列表，例如[784, 128, 10]表示输入层784个神经元，隐藏层128个，输出层10个
            activation: 激活函数类型，支持'sigmoid'或'relu'
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation = activation
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            # 使用He初始化或Xavier初始化
            if activation == 'relu':
                weight_init = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i])
            else:  # sigmoid
                weight_init = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(1 / layer_sizes[i])
            self.weights.append(weight_init)
            self.biases.append(np.zeros((layer_sizes[i+1], 1)))
    
    def _sigmoid(self, z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, a):
        """Sigmoid函数的导数"""
        return a * (1 - a)
    
    def _relu(self, z):
        """ReLU激活函数"""
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        """ReLU函数的导数"""
        return np.where(z > 0, 1, 0)
    
    def forward(self, X):
        """
        前向传播计算
        
        参数:
            X: 输入数据，形状为(n_samples, n_features)
            
        返回:
            a_layers: 包含各层激活值的列表
            z_layers: 包含各层加权输入的列表
        """
        a = X.T  # 转换为(特征数, 样本数)
        a_layers = [a]
        z_layers = []
        
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            z_layers.append(z)
            
            if i == self.num_layers - 2:  # 输出层使用softmax激活
                a = self._softmax(z)
            elif self.activation == 'sigmoid':
                a = self._sigmoid(z)
            else:  # relu
                a = self._relu(z)
                
            a_layers.append(a)
        
        return a_layers, z_layers
    
    def _softmax(self, z):
        """Softmax激活函数，用于多分类问题"""
        # 为了数值稳定性，减去最大值
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def backward(self, X, y, a_layers, z_layers):
        """
        反向传播计算梯度
        
        参数:
            X: 输入数据，形状为(n_samples, n_features)
            y: 目标标签，形状为(n_samples, n_outputs)，应为one-hot编码
            a_layers: 前向传播得到的各层激活值
            z_layers: 前向传播得到的各层加权输入
            
        返回:
            weight_grads: 权重梯度列表
            bias_grads: 偏置梯度列表
        """
        n_samples = X.shape[0]
        weight_grads = [np.zeros_like(w) for w in self.weights]
        bias_grads = [np.zeros_like(b) for b in self.biases]
        
        # 计算输出层误差 (softmax + 交叉熵损失的导数)
        delta = a_layers[-1] - y.T  # 对于softmax和交叉熵损失，导数就是这样简单
        
        # 反向传播
        for i in reversed(range(self.num_layers - 1)):
            if i < self.num_layers - 2:  # 非输出层
                if self.activation == 'sigmoid':
                    delta = np.dot(self.weights[i+1].T, delta) * self._sigmoid_derivative(a_layers[i+1])
                else:  # relu
                    delta = np.dot(self.weights[i+1].T, delta) * self._relu_derivative(z_layers[i])
            
            weight_grads[i] = np.dot(delta, a_layers[i].T) / n_samples
            bias_grads[i] = np.sum(delta, axis=1, keepdims=True) / n_samples
        
        return weight_grads, bias_grads
    
    def update_parameters(self, weight_grads, bias_grads, learning_rate):
        """
        更新模型参数
        
        参数:
            weight_grads: 权重梯度列表
            bias_grads: 偏置梯度列表
            learning_rate: 学习率
        """
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * weight_grads[i]
            self.biases[i] -= learning_rate * bias_grads[i]
    
    def train(self, X, y, learning_rate=0.1, epochs=1000, print_every=100):
        """
        训练模型
        
        参数:
            X: 输入数据，形状为(n_samples, n_features)
            y: 目标标签，形状为(n_samples, n_outputs)，应为one-hot编码
            learning_rate: 学习率
            epochs: 训练轮数
            print_every: 每隔多少轮打印一次损失
        """
        for epoch in range(epochs):
            # 前向传播
            a_layers, z_layers = self.forward(X)
            
            # 计算损失 (交叉熵损失)
            loss = self._cross_entropy_loss(a_layers[-1], y)
            
            # 反向传播
            weight_grads, bias_grads = self.backward(X, y, a_layers, z_layers)
            
            # 更新参数
            self.update_parameters(weight_grads, bias_grads, learning_rate)
            
            # 打印损失
            if (epoch + 1) % print_every == 0:
                # 计算准确率
                predictions = np.argmax(a_layers[-1], axis=0)
                true_labels = np.argmax(y, axis=1)
                accuracy = np.mean(predictions == true_labels)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}, Accuracy: {accuracy:.4f}")
    
    def _cross_entropy_loss(self, y_pred, y_true):
        """计算交叉熵损失"""
        # 添加小的epsilon值防止对数计算溢出
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred.T), axis=0))
    
    def predict(self, X):
        """
        预测函数
        
        参数:
            X: 输入数据，形状为(n_samples, n_features)
            
        返回:
            预测结果，形状为(n_samples, n_outputs)
        """
        a_layers, _ = self.forward(X)
        return a_layers[-1].T  # 转换为(n_samples, n_outputs)

# MNIST数据加载和预处理
class MNISTLoader:
    """MNIST数据集加载器"""
    
    BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    FILES = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    def __init__(self, data_dir='./data/mnist'):
        """
        初始化MNIST加载器
        
        参数:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据缓存
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
    
    def _download_file(self, url, filename):
        """下载MNIST文件"""
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            print(f"下载文件: {filename}...")
            max_retries = 3
            retry_delay = 2  # seconds
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    break
                except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"重试 ({attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
            
            # 获取文件总大小
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # 过滤掉保持连接的chunk
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        # 显示进度
                        if total_size > 0:
                            percent = (downloaded_size / total_size) * 100
                            print(f"\r下载进度: {percent:.1f}% ({downloaded_size}/{total_size} bytes)", end='', flush=True)
            
            print(f"\n下载完成: {filename}")
        else:
            print(f"文件已存在: {filename}")
            
        return file_path
    
    def _read_images(self, filename):
        """读取图像数据"""
        with gzip.open(filename, 'rb') as f:
            # 读取文件头
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            
            # 读取图像数据
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_images, rows * cols)
            
            # 归一化处理
            return images.astype(np.float32) / 255.0
    
    def _read_labels(self, filename):
        """读取标签数据"""
        with gzip.open(filename, 'rb') as f:
            # 读取文件头
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            
            # 读取标签数据
            label_data = f.read()
            return np.frombuffer(label_data, dtype=np.uint8)
    
    def load_train_data(self):
        """加载训练数据"""
        if self.train_images is None or self.train_labels is None:
            # 下载训练图像
            images_url = urljoin(self.BASE_URL, self.FILES['train_images'])
            images_file = self._download_file(images_url, self.FILES['train_images'])
            
            # 下载训练标签
            labels_url = urljoin(self.BASE_URL, self.FILES['train_labels'])
            labels_file = self._download_file(labels_url, self.FILES['train_labels'])
            
            # 读取数据
            self.train_images = self._read_images(images_file)
            self.train_labels = self._read_labels(labels_file)
            
        return self.train_images, self.train_labels
    
    def load_test_data(self):
        """加载测试数据"""
        if self.test_images is None or self.test_labels is None:
            # 下载测试图像
            images_url = urljoin(self.BASE_URL, self.FILES['test_images'])
            images_file = self._download_file(images_url, self.FILES['test_images'])
            
            # 下载测试标签
            labels_url = urljoin(self.BASE_URL, self.FILES['test_labels'])
            labels_file = self._download_file(labels_url, self.FILES['test_labels'])
            
            # 读取数据
            self.test_images = self._read_images(images_file)
            self.test_labels = self._read_labels(labels_file)
            
        return self.test_images, self.test_labels
    
    def to_one_hot(self, labels, num_classes=10):
        """将标签转换为one-hot编码"""
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot

# 示例使用
def main():
    # 加载MNIST数据
    print("加载MNIST数据集...")
    loader = MNISTLoader()
    X_train, y_train = loader.load_train_data()
    X_test, y_test = loader.load_test_data()
    
    # 将标签转换为one-hot编码
    y_train_onehot = loader.to_one_hot(y_train)
    y_test_onehot = loader.to_one_hot(y_test)
    
    print(f"训练数据: {X_train.shape}, 标签: {y_train_onehot.shape}")
    print(f"测试数据: {X_test.shape}, 标签: {y_test_onehot.shape}")
    
    # 创建MLP模型
    model = MLP(layer_sizes=[784, 128, 64, 10], activation='relu')
    
    # 训练模型
    print("\n开始训练模型...")
    model.train(X_train, y_train_onehot, learning_rate=0.1, epochs=50, print_every=1)
    
    # 测试模型
    test_predictions = model.predict(X_test)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    
    test_accuracy = np.mean(test_pred_classes == y_test)
    print(f"\n测试准确率: {test_accuracy * 100:.2f}%")
    
    # 显示一些预测示例
    print("\n样本预测结果:")
    for i in range(10):
        print(f"样本 {i+1}: 真实标签 = {y_test[i]}, 预测标签 = {test_pred_classes[i]}")

if __name__ == "__main__":
    main()