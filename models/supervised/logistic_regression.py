import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import pickle
import os
from datetime import datetime
from tqdm import tqdm

class LogisticRegressionCIFAR10:
    def __init__(self, data_dir='./data/cifar-10-batches-py', output_dir='./output/logistic_regression'):
        self.data_dir = data_dir
        # 使用SGDClassifier来实现在线学习，记录训练过程
        self.model = SGDClassifier(
            loss='log_loss',
            learning_rate='constant',  # 使用固定学习率
            eta0=0.001,  # 设置较小的学习率
            max_iter=100,
            random_state=42,
            n_jobs=-1  # 使用所有CPU核心
        )
        self.scaler = StandardScaler()
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
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

    def compute_probabilities(self, X):
        """安全地计算概率，避免数值问题"""
        scores = self.model.decision_function(X)
        # 使用softmax将分数转换为概率
        scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probas = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
        return np.clip(probas, 1e-10, 1.0)  # 裁剪概率值避免数值问题

    def run(self):
        """运行完整的训练和评估流程"""
        print("\n=== Logistic Regression on CIFAR-10 ===")
        print("Loading and preprocessing data...")
        X_train, y_train, X_test, y_test = self.load_data()
        
        # 保存测试数据供评估使用
        self.X_test = X_test
        self.y_test = y_test

        # 划分训练集和验证集
        train_size = int(0.8 * len(X_train))
        X_train_final, X_val = X_train[:train_size], X_train[train_size:]
        y_train_final, y_val = y_train[:train_size], y_train[train_size:]

        print("Training logistic regression model...")
        # 使用小批量训练来记录训练过程
        batch_size = 500
        n_batches = len(X_train_final) // batch_size
        n_epochs = 10

        for epoch in range(n_epochs):
            # 打乱数据
            indices = np.random.permutation(len(X_train_final))
            X_train_shuffled = X_train_final[indices]
            y_train_shuffled = y_train_final[indices]
            
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            # 批量训练
            for batch in tqdm(range(n_batches), desc="Training"):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                self.model.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
                
            # 记录训练集性能
            train_acc = self.model.score(X_train_final, y_train_final)
            train_pred_proba = self.compute_probabilities(X_train_final)
            train_loss = log_loss(y_train_final, train_pred_proba)
            
            # 记录验证集性能
            val_acc = self.model.score(X_val, y_val)
            val_pred_proba = self.compute_probabilities(X_val)
            val_loss = log_loss(y_val, val_pred_proba)
            
            # 保存历史记录
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
            print(f"Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")

        # 在测试集上评估
        y_pred = self.model.predict(self.X_test)
        test_acc = self.model.score(self.X_test, self.y_test)
        
        # 打印分类报告
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # 生成可视化
        self._generate_visualizations(self.y_test, y_pred)
        
        # 保存结果摘要
        self._save_summary(test_acc)
        
        return {
            'model_name': 'Logistic Regression',
            'train_acc': self.history['train_acc'][-1],
            'val_acc': self.history['val_acc'][-1],
            'test_acc': test_acc
        }

    def _generate_visualizations(self, y_test, y_pred):
        """生成所有可视化内容"""
        # 设置图片保存路径
        confusion_matrix_path = os.path.join(self.output_dir, 'logistic_regression_confusion_matrix.png')
        accuracy_plot_path = os.path.join(self.output_dir, 'logistic_regression_accuracy_curve.png')
        loss_plot_path = os.path.join(self.output_dir, 'logistic_regression_loss_curve.png')
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Logistic Regression')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制准确率变化曲线
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history['train_acc']) + 1)
        plt.plot(epochs, self.history['train_acc'], 'bo-', label='Training Accuracy')
        plt.plot(epochs, self.history['val_acc'], 'ro-', label='Validation Accuracy')
        plt.title('Model Accuracy over Epochs - Logistic Regression')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制损失变化曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_loss'], 'bo-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], 'ro-', label='Validation Loss')
        plt.title('Model Loss over Epochs - Logistic Regression')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization files have been saved to {self.output_dir}/")
        print(f"- {os.path.basename(confusion_matrix_path)}")
        print(f"- {os.path.basename(accuracy_plot_path)}")
        print(f"- {os.path.basename(loss_plot_path)}")

    def _save_summary(self, test_acc):
        """保存实验结果摘要"""
        summary_path = os.path.join(self.output_dir, 'logistic_regression_results.txt')
        with open(summary_path, 'w') as f:
            f.write("=== Logistic Regression Model Results ===\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Final Training Accuracy: {self.history['train_acc'][-1]:.3f}\n")
            f.write(f"Final Validation Accuracy: {self.history['val_acc'][-1]:.3f}\n")
            f.write(f"Final Test Accuracy: {test_acc:.3f}\n")
            f.write(f"Final Training Loss: {self.history['train_loss'][-1]:.3f}\n")
            f.write(f"Final Validation Loss: {self.history['val_loss'][-1]:.3f}\n")
            f.write("\nModel Parameters:\n")
            f.write(f"- Learning Rate: {self.model.learning_rate}\n")
            f.write(f"- Max Iterations: {self.model.max_iter}\n")
            f.write(f"- Loss Function: {self.model.loss}\n")
            f.write(f"- Random State: {self.model.random_state}\n")
        
        print(f"- {os.path.basename(summary_path)}")
