import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os
from datetime import datetime
from tqdm import tqdm
import time

class SVMModel:
    def __init__(self, data_dir='./data/cifar-10-batches-py', output_dir='./output/svm'):
        self.data_dir = data_dir
        # 使用更简单的线性SVM配置
        base_svm = LinearSVC(
            dual=False,      # 使用原始优化方法
            C=0.1,          # 降低正则化参数，使模型更简单
            max_iter=500,    # 减少最大迭代次数
            random_state=42,
            verbose=0,       # 关闭详细输出
            tol=1e-2        # 进一步提高收敛容差
        )
        # 简化概率校准过程
        self.model = CalibratedClassifierCV(
            base_svm,
            cv=2,           # 减少交叉验证折数
            n_jobs=-1       # 使用所有CPU
        )
        self.scaler = StandardScaler()
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
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

    def run(self):
        """运行完整的训练和评估流程"""
        print("\n=== SVM on CIFAR-10 ===")
        X_train, y_train, X_test, y_test = self.load_data()
        
        # 保存测试数据供评估使用
        self.X_test = X_test
        self.y_test = y_test

        # 使用完整训练集
        print(f"\nUsing full training set with {len(X_train)} samples...")
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(X_train))
        X_train_final = X_train[:train_size]
        X_val = X_train[train_size:]
        y_train_final = y_train[:train_size]
        y_val = y_train[train_size:]

        print(f"Training set size: {len(X_train_final)}")
        print(f"Validation set size: {len(X_val)}")

        print("\nTraining SVM model...")
        print("This may take a few minutes. Progress bar updates after each iteration.")
        
        # 使用自定义的训练进度显示
        start_time = time.time()
        self.model.fit(X_train_final, y_train_final)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        print("\nCalculating performance metrics...")
        with tqdm(total=6, desc="Computing metrics") as pbar:
            # 计算训练准确率
            train_acc = self.model.score(X_train_final, y_train_final)
            pbar.update(1)
            
            # 计算验证准确率
            val_acc = self.model.score(X_val, y_val)
            pbar.update(1)
            
            # 计算训练概率
            train_proba = self.model.predict_proba(X_train_final)
            pbar.update(1)
            
            # 计算验证概率
            val_proba = self.model.predict_proba(X_val)
            pbar.update(1)
            
            # 计算损失
            train_loss = -np.mean(np.log(np.clip(train_proba[np.arange(len(y_train_final)), y_train_final], 1e-10, 1)))
            pbar.update(1)
            
            val_loss = -np.mean(np.log(np.clip(val_proba[np.arange(len(y_val)), y_val], 1e-10, 1)))
            pbar.update(1)

        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

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

        # 生成混淆矩阵
        print("\nGenerating confusion matrix...")
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        
        return {
            'model_name': 'Linear SVM',
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc
        }
