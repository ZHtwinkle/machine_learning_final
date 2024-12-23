import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def load_cifar_batch(file_path):
    """加载CIFAR-10数据批次"""
    with open(file_path, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    data = batch[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch[b'labels']
    return data, labels

def plot_data_samples(data, labels, class_names, save_path):
    """展示每个类别的示例图片"""
    plt.figure(figsize=(8, 8))
    for i, class_name in enumerate(class_names):
        # 找到该类别的所有图片
        idx = np.where(labels == i)[0]
        # 随机选择9张图片
        sample_idx = np.random.choice(idx, 9, replace=False)
        
        for j, img_idx in enumerate(sample_idx):
            plt.subplot(10, 9, i*9 + j + 1)
            plt.imshow(data[img_idx])
            if j == 0:
                plt.ylabel(class_name)
            plt.xticks([])
            plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_data_distribution(train_labels, test_labels, class_names, save_path):
    """展示训练集和测试集的数据分布"""
    train_dist = [np.sum(train_labels == i) for i in range(len(class_names))]
    test_dist = [np.sum(test_labels == i) for i in range(len(class_names))]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_dist, width, label='Training Set')
    plt.bar(x + width/2, test_dist, width, label='Test Set')
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Data Distribution in CIFAR-10')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 创建保存图片的目录
    os.makedirs('plots', exist_ok=True)
    
    # 类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 数据路径
    data_dir = 'data/cifar-10-batches-py'  # 请确保这个路径正确
    
    # 加载训练数据
    X_train = []
    y_train = []
    for i in range(1, 6):
        data, labels = load_cifar_batch(f"{data_dir}/data_batch_{i}")
        X_train.append(data)
        y_train.extend(labels)
    
    X_train = np.concatenate(X_train)
    y_train = np.array(y_train)
    
    # 加载测试数据
    X_test, y_test = load_cifar_batch(f"{data_dir}/test_batch")
    
    # 生成数据样本图
    plot_data_samples(X_train, y_train, class_names, 'plots/data_samples.png')
    
    # 生成数据分布图
    plot_data_distribution(y_train, y_test, class_names, 'plots/data_distribution.png')

if __name__ == '__main__':
    main()
