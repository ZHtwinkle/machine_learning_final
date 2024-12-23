import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from datetime import datetime
from tqdm import tqdm

class KNNModel:
    def __init__(self, data_dir='./data/cifar-10-batches-py', output_dir='./output/knn'):
        self.data_dir = data_dir
        self.model = KNeighborsClassifier(
            n_neighbors=5,    # K值
            weights='distance',# 使用距离加权
            algorithm='auto', # 自动选择最优算法
            n_jobs=-1        # 使用所有CPU核心
        )
        self.scaler = StandardScaler()
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'distances': []  # 存储每个预测点到最近邻的距离
        }
        
        # 创建带时间戳的输出目录
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

        # 加载训练数据
        X_train, y_train = [], []
        for i in range(1, 6):
            batch = unpickle(f"{self.data_dir}/data_batch_{i}")
            X_train.append(batch[b'data'])
            y_train.extend(batch[b'labels'])
        X_train = np.vstack(X_train)
        y_train = np.array(y_train)

        # 加载测试数据
        test_batch = unpickle(f"{self.data_dir}/test_batch")
        X_test = test_batch[b'data']
        y_test = np.array(test_batch[b'labels'])

        # 数据预处理
        X_train = self.preprocess_data(X_train)
        X_test = self.preprocess_data(X_test)

        return X_train, y_train, X_test, y_test

    def preprocess_data(self, data):
        """预处理数据"""
        # 将图像数据展平为一维向量
        flattened_data = data.reshape(data.shape[0], -1)
        # 如果是训练数据，拟合scaler；如果是测试数据，直接转换
        if not hasattr(self.scaler, 'mean_'):
            return self.scaler.fit_transform(flattened_data)
        return self.scaler.transform(flattened_data)

    def run(self):
        """运行完整的训练和评估流程"""
        print("\n=== KNN on CIFAR-10 ===")
        print("Loading and preprocessing data...")
        X_train, y_train, X_test, y_test = self.load_data()
        
        # 保存测试数据供评估使用
        self.X_test = X_test
        self.y_test = y_test

        # 由于KNN计算量大，使用部分数据进行训练
        subset_size = 5000  # 使用5000个样本
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]

        # 划分训练集和验证集
        train_size = int(0.8 * len(X_train_subset))
        X_train_final = X_train_subset[:train_size]
        X_val = X_train_subset[train_size:]
        y_train_final = y_train_subset[:train_size]
        y_val = y_train_subset[train_size:]

        print("Training KNN model...")
        self.model.fit(X_train_final, y_train_final)

        # 计算训练和验证准确率
        train_acc = self.model.score(X_train_final, y_train_final)
        val_acc = self.model.score(X_val, y_val)

        # 计算到最近邻的距离
        distances, _ = self.model.kneighbors(X_val)
        mean_distances = distances.mean(axis=1)
        self.history['distances'] = mean_distances.tolist()

        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)

        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Validation Accuracy: {val_acc:.3f}")
        print(f"Average Distance to Nearest Neighbors: {np.mean(mean_distances):.3f}")

        # 在测试集上评估
        print("\nEvaluating on test set...")
        y_pred = self.model.predict(self.X_test)
        test_acc = self.model.score(self.X_test, self.y_test)
        
        # 打印分类报告
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # 生成可视化
        self._generate_visualizations(self.y_test, y_pred, mean_distances)
        
        # 保存结果摘要
        self._save_summary(test_acc, np.mean(mean_distances))
        
        return {
            'model_name': 'KNN',
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc
        }

    def _generate_visualizations(self, y_test, y_pred, distances):
        """生成所有可视化内容"""
        # 设置图片保存路径
        confusion_matrix_path = os.path.join(self.output_dir, 'knn_confusion_matrix.png')
        metrics_path = os.path.join(self.output_dir, 'knn_metrics.png')
        distances_path = os.path.join(self.output_dir, 'knn_distances_histogram.png')
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - KNN')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制性能指标图
        plt.figure(figsize=(10, 6))
        metrics = [self.history['train_acc'][-1], self.history['val_acc'][-1]]
        plt.bar(['Training', 'Validation'], metrics, color=['blue', 'orange'])
        plt.title('Model Accuracy - KNN')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for i, acc in enumerate(metrics):
            plt.text(i, acc, f'{acc:.3f}', ha='center', va='bottom')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制距离分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=50, edgecolor='black')
        plt.title('Distribution of Distances to Nearest Neighbors')
        plt.xlabel('Average Distance')
        plt.ylabel('Frequency')
        plt.savefig(distances_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization files have been saved to {self.output_dir}/")
        print(f"- {os.path.basename(confusion_matrix_path)}")
        print(f"- {os.path.basename(metrics_path)}")
        print(f"- {os.path.basename(distances_path)}")

    def _save_summary(self, test_acc, avg_distance):
        """保存实验结果摘要"""
        summary_path = os.path.join(self.output_dir, 'knn_results.txt')
        with open(summary_path, 'w') as f:
            f.write("=== KNN Model Results ===\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training Accuracy: {self.history['train_acc'][-1]:.3f}\n")
            f.write(f"Validation Accuracy: {self.history['val_acc'][-1]:.3f}\n")
            f.write(f"Test Accuracy: {test_acc:.3f}\n")
            f.write(f"Average Distance to Nearest Neighbors: {avg_distance:.3f}\n")
            f.write("\nModel Parameters:\n")
            f.write(f"- Number of Neighbors: {self.model.n_neighbors}\n")
            f.write(f"- Weight Function: {self.model.weights}\n")
            f.write(f"- Algorithm: {self.model.algorithm}\n")
        
        print(f"- {os.path.basename(summary_path)}")
