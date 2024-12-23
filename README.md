# CIFAR-10 图像分类

CIFAR-10数据集的图像分类，实现了多种机器学习模型，包括监督学习和无监督学习方法。

## 项目结构

```
MachineLearning/
├── data/                      # 数据目录
│   └── cifar-10-batches-py/  # CIFAR-10数据集
├── models/                    # 模型实现
│   ├── supervised/           # 监督学习模型
│   │   ├── logistic_regression.py
│   │   ├── svm.py
│   │   ├── improved_svm.py
│   │   └── knn.py
│   └── unsupervised/        # 无监督学习模型
│       └── kmeans.py
├── output/                   # 输出目录
├── utils/                    # 工具函数
├── main.py                   # 主程序
└── requirements.txt          # 依赖包列表
```

## 已实现的模型

### 监督学习模型

1. **逻辑回归 (Logistic Regression)**
   - 多类别分类
   - 使用One-vs-Rest策略
   - 包含正则化

2. **支持向量机 (SVM)**
   - 基础版本：使用线性核
   - 改进版本：
     - 包含PCA降维
     - 网格搜索优化参数
     - 详细的评估指标

3. **K近邻 (KNN)**
   - 使用欧氏距离
   - 自动选择最优K值

### 无监督学习模型

1. **K-means聚类**
   - 使用k-means++初始化
   - 包含多个评估指标：
     - 轮廓系数
     - Calinski-Harabasz指数
     - Davies-Bouldin指数


## 评估指标

所有模型都包含以下评估指标：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- 混淆矩阵

## 如何使用

1. **环境配置**
   ```bash
   pip install -r requirements.txt
   ```

2. **运行所有模型对比**
   ```bash
   python main.py
   ```

3. **单独运行特定模型**
   ```bash
   # 运行改进版SVM
   python models/supervised/improved_svm.py
   
   # 运行K-means聚类
   python models/unsupervised/kmeans.py
   ```

## 输出结果

所有模型的运行结果将保存在`output`目录下，包括：
- 训练日志
- 评估指标
- 可视化图表：
  - 学习曲线
  - 混淆矩阵热力图
  - ROC曲线
  - PCA方差解释率（适用于改进版SVM）
  - 聚类结果可视化（适用于K-means）

## 依赖包

主要依赖包包括：
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm


