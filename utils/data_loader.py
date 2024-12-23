import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # 将图像从(32, 32, 3)转换为(3, 32, 32)
        image = np.transpose(image.reshape(3, 32, 32), (1, 2, 0))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_cifar10(data_dir, batch_size=128):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载训练数据
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch = unpickle(f"{data_dir}/data_batch_{i}")
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    train_data = np.vstack(train_data)
    
    # 加载测试数据
    test_batch = unpickle(f"{data_dir}/test_batch")
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']
    
    # 创建数据集
    train_dataset = CIFAR10Dataset(train_data, train_labels, transform=transform)
    test_dataset = CIFAR10Dataset(test_data, test_labels, transform=transform)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader
