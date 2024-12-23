import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import pickle
import os
from datetime import datetime
from tqdm import tqdm
import time

class ImprovedSVMModel:
    def __init__(self, data_dir=None, output_dir=None):
        # 使用绝对路径
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(base_dir, 'data', 'cifar-10-batches-py')
        else:
            self.data_dir = data_dir

        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, 'output', 'improved_svm')

        self.n_components = 100  # PCA降维后的维度

        # 创建包含PCA和SVM的管道
        self.pipeline = Pipeline([
            ('pca', PCA(n_components=self.n_components, random_state=42)),
            ('svm', LinearSVC(
                dual=False,
                random_state=42,
                verbose=0,
                max_iter=1000
            ))
        ])

        # 包装成概率输出
        self.model = CalibratedClassifierCV(
            self.pipeline,
            cv=2,
            n_jobs=-1
        )

        self.scaler = StandardScaler()
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': [],
            'pca_explained_variance': None
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
            time.sleep(0.1)

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

    def perform_grid_search(self, X_train, y_train):
        """执行网格搜索找到最佳参数"""
        print("\nPerforming grid search for optimal parameters...")
        param_grid = {
            'svm__C': [0.1, 1.0],
            'svm__tol': [1e-3, 1e-2],
            'pca__n_components': [50, 100]
        }

        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=2,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print("\nBest parameters:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)

        # 使用最佳参数更新模型
        self.pipeline.set_params(**grid_search.best_params_)
        return grid_search.best_params_

    def plot_pca_variance(self, X_train):
        """绘制PCA方差解释率"""
        # 先单独拟合PCA来获取方差解释率
        pca = PCA(n_components=self.n_components, random_state=42)
        pca.fit(X_train)

        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(explained_variance_ratio) + 1),
                cumulative_variance_ratio, 'bo-')
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.grid(True)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')

        # 保存解释方差比
        self.history['pca_explained_variance'] = explained_variance_ratio

        plt.savefig(os.path.join(self.output_dir, 'pca_variance.png'))
        plt.close()

    def plot_learning_curves(self):
        """绘制学习曲线"""
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_acc'], label='Training')
        plt.plot(self.history['val_acc'], label='Validation')
        plt.title('Accuracy over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='Training')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title('Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'learning_curves.png'))
        plt.close()

    def run(self):
        """运行完整的训练和评估流程"""
        print("\n=== Improved SVM with PCA on CIFAR-10 ===")
        X_train, y_train, X_test, y_test = self.load_data()

        # 保存测试数据供评估使用
        self.X_test = X_test
        self.y_test = y_test

        print(f"\nUsing full training set with {len(X_train)} samples...")

        # 划分训练集和验证集
        train_size = int(0.8 * len(X_train))
        X_train_final = X_train[:train_size]
        X_val = X_train[train_size:]
        y_train_final = y_train[:train_size]
        y_val = y_train[train_size:]

        print(f"Training set size: {len(X_train_final)}")
        print(f"Validation set size: {len(X_val)}")

        # 执行网格搜索找到最佳参数
        best_params = self.perform_grid_search(X_train_final, y_train_final)

        print("\nTraining final model with best parameters...")
        print("This may take a few minutes. Progress bar updates after each iteration.")

        # 训练最终模型
        start_time = time.time()
        self.model.fit(X_train_final, y_train_final)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # 计算性能指标
        print("\nCalculating performance metrics...")
        with tqdm(total=6, desc="Computing metrics") as pbar:
            # 训练集性能
            train_acc = self.model.score(X_train_final, y_train_final)
            pbar.update(1)

            # 验证集性能
            val_acc = self.model.score(X_val, y_val)
            pbar.update(1)

            # 概率预测
            train_proba = self.model.predict_proba(X_train_final)
            pbar.update(1)
            val_proba = self.model.predict_proba(X_val)
            pbar.update(1)

            # 计算损失
            train_loss = -np.mean(np.log(np.clip(train_proba[np.arange(len(y_train_final)), y_train_final], 1e-10, 1)))
            pbar.update(1)
            val_loss = -np.mean(np.log(np.clip(val_proba[np.arange(len(y_val)), y_val], 1e-10, 1)))
            pbar.update(1)

        # 更新历史记录
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

        # 打印训练结果
        print(f"\nTraining Accuracy: {train_acc:.3f}")
        print(f"Validation Accuracy: {val_acc:.3f}")
        print(f"Training Loss: {train_loss:.3f}")
        print(f"Validation Loss: {val_loss:.3f}")

        # 在测试集上评估
        print("\nEvaluating on test set...")
        test_acc = self.model.score(self.X_test, self.y_test)
        y_pred = self.model.predict(self.X_test)

        # 打印分类报告
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # 生成可视化
        print("\nGenerating visualizations...")

        # 绘制PCA方差解释率
        self.plot_pca_variance(X_train_final)

        # 绘制学习曲线
        self.plot_learning_curves()

        # 绘制混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

        # 保存详细结果
        summary_path = os.path.join(self.output_dir, 'improved_svm_results.txt')
        with open(summary_path, 'w') as f:
            f.write("=== Improved SVM Model Results ===\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n\n")

            f.write("Model Configuration:\n")
            f.write("-" * 20 + "\n")
            f.write(f"PCA Components: {self.n_components}\n")
            f.write("Best Parameters:\n")
            for param, value in best_params.items():
                f.write(f"- {param}: {value}\n")

            f.write("\nPerformance Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Training Accuracy: {train_acc:.3f}\n")
            f.write(f"Validation Accuracy: {val_acc:.3f}\n")
            f.write(f"Test Accuracy: {test_acc:.3f}\n")
            f.write(f"Training Loss: {train_loss:.3f}\n")
            f.write(f"Validation Loss: {val_loss:.3f}\n")

            f.write("\nPCA Explained Variance:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total variance explained: {np.sum(self.history['pca_explained_variance']):.3f}\n")
            f.write(f"Number of components explaining 95% variance: {np.argmax(np.cumsum(self.history['pca_explained_variance']) >= 0.95) + 1}\n")

        print(f"\nResults have been saved to: {self.output_dir}")
        print("Generated files:")
        print("- improved_svm_results.txt")
        print("- pca_variance.png")
        print("- learning_curves.png")
        print("- confusion_matrix.png")

if __name__ == "__main__":
    model = ImprovedSVMModel()
    model.run()
