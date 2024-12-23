import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
import os
from datetime import datetime
from tqdm import tqdm
import time

class KMeansModel:
    def __init__(self, data_dir='./data/cifar-10-batches-py', output_dir='./output/kmeans'):
        self.data_dir = data_dir
        # 配置K-means参数
        self.model = KMeans(
            n_clusters=10,    # CIFAR-10有10个类别
            init='k-means++', # 使用k-means++初始化
            n_init=10,        # 运行10次取最佳结果
            max_iter=300,     # 最大迭代次数
            tol=1e-4,        # 收敛容差
            random_state=42,
            verbose=0,        # 关闭详细输出
            algorithm='elkan' # 使用更快的Elkan算法
        )
        self.scaler = StandardScaler()
        self.history = {
            'inertia': [],      # 簇内平方和
            'silhouette': [],   # 轮廓系数
            'calinski': [],     # Calinski-Harabasz指数
            'davies': []        # Davies-Bouldin指数
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"run_{timestamp}")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        """加载CIFAR-10数据集"""
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        print("Loading training data...")
        # 加载训练数据
        X_train = []
        y_train = []
        for i in tqdm(range(1, 6), desc="Loading batches"):
            batch = unpickle(os.path.join(self.data_dir, f'data_batch_{i}'))
            X_train.append(batch[b'data'])
            y_train.extend(batch[b'labels'])
            time.sleep(0.1)  # 添加小延迟以显示进度

        X_train = np.vstack(X_train)
        y_train = np.array(y_train)

        print("Loading test data...")
        # 加载测试数据
        test_batch = unpickle(os.path.join(self.data_dir, 'test_batch'))
        X_test = test_batch[b'data']
        y_test = np.array(test_batch[b'labels'])

        print("Preprocessing data...")
        with tqdm(total=2, desc="Preprocessing") as pbar:
            # 预处理训练数据
            X_train = self.preprocess_data(X_train)
            pbar.update(1)
            time.sleep(0.1)
            
            # 预处理测试数据
            X_test = self.preprocess_data(X_test)
            pbar.update(1)
            time.sleep(0.1)

        return X_train, y_train, X_test, y_test

    def preprocess_data(self, data):
        """预处理数据"""
        # 将图像数据展平为一维向量
        flattened_data = data.reshape(data.shape[0], -1)
        
        # 如果是训练数据，拟合scaler；如果是测试数据，直接转换
        if not hasattr(self.scaler, 'mean_'):
            return self.scaler.fit_transform(flattened_data)
        return self.scaler.transform(flattened_data)

    def evaluate_clustering(self, X, labels):
        """评估聚类效果"""
        metrics = {}
        
        print("\nCalculating clustering metrics...")
        with tqdm(total=4, desc="Computing metrics") as pbar:
            # 计算轮廓系数
            metrics['silhouette'] = silhouette_score(X, labels)
            pbar.update(1)
            
            # 计算Calinski-Harabasz指数
            metrics['calinski'] = calinski_harabasz_score(X, labels)
            pbar.update(1)
            
            # 计算Davies-Bouldin指数
            metrics['davies'] = davies_bouldin_score(X, labels)
            pbar.update(1)
            
            # 计算惯性（簇内平方和）
            metrics['inertia'] = self.model.inertia_
            pbar.update(1)
        
        return metrics

    def plot_cluster_distribution(self, labels, true_labels):
        """绘制聚类分布图"""
        plt.figure(figsize=(12, 6))
        
        # 创建混淆矩阵
        confusion_matrix = np.zeros((10, 10))
        for i in range(len(labels)):
            confusion_matrix[true_labels[i]][labels[i]] += 1
            
        # 归一化混淆矩阵
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
        
        # 绘制热力图
        sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Cluster Distribution vs True Labels')
        plt.xlabel('Cluster Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(self.output_dir, 'cluster_distribution.png'))
        plt.close()

    def plot_cluster_sizes(self, labels):
        """绘制聚类大小分布图"""
        plt.figure(figsize=(10, 6))
        cluster_sizes = np.bincount(labels)
        plt.bar(range(len(cluster_sizes)), cluster_sizes)
        plt.title('Cluster Sizes')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Samples')
        for i, v in enumerate(cluster_sizes):
            plt.text(i, v, str(v), ha='center', va='bottom')
        plt.savefig(os.path.join(self.output_dir, 'cluster_sizes.png'))
        plt.close()

    def run(self):
        """运行完整的聚类和评估流程"""
        print("\n=== K-means Clustering on CIFAR-10 ===")
        X_train, y_train, X_test, y_test = self.load_data()
        
        # 保存测试数据供评估使用
        self.X_test = X_test
        self.y_test = y_test

        print(f"\nUsing full training set with {len(X_train)} samples...")
        
        print("\nTraining K-means model...")
        print("This may take a few minutes. Progress bar updates after each iteration.")
        
        # 训练模型
        start_time = time.time()
        self.model.fit(X_train)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # 获取聚类标签
        train_labels = self.model.labels_
        test_labels = self.model.predict(X_test)

        # 评估聚类效果
        train_metrics = self.evaluate_clustering(X_train, train_labels)
        test_metrics = self.evaluate_clustering(X_test, test_labels)

        # 更新历史记录
        for metric in train_metrics:
            self.history[metric].append(train_metrics[metric])

        # 打印评估结果
        print("\nTraining Set Metrics:")
        print(f"Silhouette Score: {train_metrics['silhouette']:.3f}")
        print(f"Calinski-Harabasz Score: {train_metrics['calinski']:.3f}")
        print(f"Davies-Bouldin Score: {train_metrics['davies']:.3f}")
        print(f"Inertia: {train_metrics['inertia']:.3f}")

        print("\nTest Set Metrics:")
        print(f"Silhouette Score: {test_metrics['silhouette']:.3f}")
        print(f"Calinski-Harabasz Score: {test_metrics['calinski']:.3f}")
        print(f"Davies-Bouldin Score: {test_metrics['davies']:.3f}")
        print(f"Inertia: {test_metrics['inertia']:.3f}")

        # 生成可视化
        print("\nGenerating visualizations...")
        self.plot_cluster_distribution(train_labels, y_train)
        self.plot_cluster_sizes(train_labels)
        
        # 保存结果摘要
        summary_path = os.path.join(self.output_dir, 'clustering_results.txt')
        with open(summary_path, 'w') as f:
            f.write("=== K-means Clustering Results ===\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of samples: {len(X_train)}\n")
            f.write(f"Number of clusters: {self.model.n_clusters}\n")
            f.write(f"Training time: {training_time:.2f} seconds\n\n")
            
            f.write("Training Set Metrics:\n")
            f.write(f"Silhouette Score: {train_metrics['silhouette']:.3f}\n")
            f.write(f"Calinski-Harabasz Score: {train_metrics['calinski']:.3f}\n")
            f.write(f"Davies-Bouldin Score: {train_metrics['davies']:.3f}\n")
            f.write(f"Inertia: {train_metrics['inertia']:.3f}\n\n")
            
            f.write("Test Set Metrics:\n")
            f.write(f"Silhouette Score: {test_metrics['silhouette']:.3f}\n")
            f.write(f"Calinski-Harabasz Score: {test_metrics['calinski']:.3f}\n")
            f.write(f"Davies-Bouldin Score: {test_metrics['davies']:.3f}\n")
            f.write(f"Inertia: {test_metrics['inertia']:.3f}\n")

        return {
            'model_name': 'K-means',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'training_time': training_time
        }
