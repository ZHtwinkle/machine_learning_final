from models.supervised.logistic_regression import LogisticRegressionCIFAR10
from models.supervised.svm import SVMModel
from models.supervised.knn import KNNModel
from models.unsupervised.kmeans import KMeansModel
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import seaborn as sns

def create_comparison_plots(results_list, output_dir):
    """创建模型比较的可视化图表"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 准确率对比图
    create_accuracy_comparison(results_list, output_dir)
    
    # 2. 性能指标对比图
    create_metrics_comparison(results_list, output_dir)
    
    # 3. 混淆矩阵对比图
    create_confusion_matrices(results_list, output_dir)
    
    # 4. 每个类别的性能对比图
    create_per_class_performance(results_list, output_dir)
    
    # 5. ROC曲线对比
    create_roc_curves(results_list, output_dir)

def create_accuracy_comparison(results_list, output_dir):
    """创建准确率对比图"""
    model_names = [r['model_name'] for r in results_list]
    train_accs = [r['train_acc'] for r in results_list]
    val_accs = [r['val_acc'] for r in results_list]
    test_accs = [r['test_acc'] for r in results_list]
    
    plt.figure(figsize=(12, 6))
    x = range(len(model_names))
    width = 0.25
    
    plt.bar([i - width for i in x], train_accs, width, label='Training', color='blue', alpha=0.7)
    plt.bar(x, val_accs, width, label='Validation', color='orange', alpha=0.7)
    plt.bar([i + width for i in x], test_accs, width, label='Test', color='green', alpha=0.7)
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i in x:
        plt.text(i - width, train_accs[i], f'{train_accs[i]:.3f}', ha='center', va='bottom')
        plt.text(i, val_accs[i], f'{val_accs[i]:.3f}', ha='center', va='bottom')
        plt.text(i + width, test_accs[i], f'{test_accs[i]:.3f}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'model_comparison_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_comparison(results_list, output_dir):
    """创建性能指标对比图"""
    plt.figure(figsize=(15, 6))
    
    metrics = ['precision_macro', 'recall_macro', 'f1_macro']
    metric_names = ['Precision', 'Recall', 'F1-Score']
    x = range(len(metric_names))
    width = 0.25
    
    for i, model in enumerate(results_list):
        values = [model[metric] for metric in metrics]
        plt.bar([j + i*width for j in x], values, width, 
                label=model['model_name'], alpha=0.7)
        
        # 添加数值标签
        for j, v in enumerate(values):
            plt.text(j + i*width, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics Comparison')
    plt.xticks([i + width for i in range(len(metric_names))], metric_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'model_comparison_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrices(results_list, output_dir):
    """创建混淆矩阵对比图"""
    fig, axes = plt.subplots(1, len(results_list), figsize=(20, 6))
    
    for i, results in enumerate(results_list):
        cm = confusion_matrix(results['y_true'], results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{results["model_name"]}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_per_class_performance(results_list, output_dir):
    """创建每个类别的性能对比图"""
    n_classes = 10  # CIFAR-10有10个类别
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    metrics = ['precision', 'recall', 'f1-score']
    titles = ['Precision per Class', 'Recall per Class', 'F1-Score per Class']
    
    for i, metric in enumerate(metrics):
        for results in results_list:
            # 从分类报告中提取每个类的指标
            report = classification_report(results['y_true'], results['y_pred'], 
                                        output_dict=True)
            class_scores = [report[str(j)][metric] for j in range(n_classes)]
            axes[i].plot(range(n_classes), class_scores, 
                        marker='o', label=results['model_name'])
        
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Score')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_curves(results_list, output_dir):
    """创建ROC曲线对比图"""
    plt.figure(figsize=(10, 8))
    
    for results in results_list:
        # 计算每个类别的ROC曲线和AUC
        y_true_bin = np.eye(10)[results['y_true']]  # 转换为one-hot编码
        y_score = results['y_proba']
        
        for i in range(10):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            if i == 0:  # 只在图例中显示一次模型名称
                plt.plot(fpr, tpr, alpha=0.3, 
                        label=f'{results["model_name"]} (avg AUC = {roc_auc:.3f})')
            else:
                plt.plot(fpr, tpr, alpha=0.3)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Classes and Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'roc_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_comparison_summary(results_list, output_dir):
    """保存模型比较的详细报告"""
    summary_path = os.path.join(output_dir, 'model_comparison_report.txt')
    
    with open(summary_path, 'w') as f:
        f.write("=== Model Comparison Report ===\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 写入每个模型的详细结果
        for results in results_list:
            f.write(f"\n{results['model_name']}:\n")
            f.write("=" * (len(results['model_name']) + 1) + "\n")
            
            # 基础指标
            f.write("\nBasic Metrics:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Training Accuracy: {results['train_acc']:.3f}\n")
            f.write(f"Validation Accuracy: {results['val_acc']:.3f}\n")
            f.write(f"Test Accuracy: {results['test_acc']:.3f}\n")
            
            # 详细性能指标
            f.write("\nDetailed Performance Metrics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Macro-average Precision: {results['precision_macro']:.3f}\n")
            f.write(f"Macro-average Recall: {results['recall_macro']:.3f}\n")
            f.write(f"Macro-average F1-Score: {results['f1_macro']:.3f}\n")
            
            # 分类报告
            f.write("\nDetailed Classification Report:\n")
            f.write("-" * 30 + "\n")
            f.write(results['classification_report'])
            f.write("\n")
        
        # 找出最佳模型（基于不同指标）
        f.write("\nBest Performing Models:\n")
        f.write("=" * 20 + "\n")
        
        metrics = {
            'Test Accuracy': lambda x: x['test_acc'],
            'Macro F1-Score': lambda x: x['f1_macro'],
            'Macro Precision': lambda x: x['precision_macro'],
            'Macro Recall': lambda x: x['recall_macro']
        }
        
        for metric_name, metric_func in metrics.items():
            best_model = max(results_list, key=metric_func)
            f.write(f"\nBest model by {metric_name}:\n")
            f.write(f"- Model: {best_model['model_name']}\n")
            f.write(f"- Score: {metric_func(best_model):.3f}\n")

def main():
    # 创建比较结果的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join('./output/model_comparison', f'run_{timestamp}')
    
    # 运行所有模型并收集详细指标
    results_list = []
    
    print("\n=== Running Logistic Regression ===")
    logistic_model = LogisticRegressionCIFAR10()
    results_lr = logistic_model.run()
    
    # 计算详细指标
    y_true = logistic_model.y_test
    y_pred = logistic_model.model.predict(logistic_model.X_test)
    y_proba = logistic_model.model.predict_proba(logistic_model.X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    results_lr.update({
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'classification_report': classification_report(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba
    })
    results_list.append(results_lr)
    
    print("\n=== Running SVM ===")
    svm_model = SVMModel()
    results_svm = svm_model.run()
    
    # 计算详细指标
    y_true = svm_model.y_test
    y_pred = svm_model.model.predict(svm_model.X_test)
    y_proba = svm_model.model.predict_proba(svm_model.X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    results_svm.update({
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'classification_report': classification_report(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba
    })
    results_list.append(results_svm)
    
    print("\n=== Running KNN ===")
    knn_model = KNNModel()
    results_knn = knn_model.run()
    
    # 计算详细指标
    y_true = knn_model.y_test
    y_pred = knn_model.model.predict(knn_model.X_test)
    y_proba = knn_model.model.predict_proba(knn_model.X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    results_knn.update({
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'classification_report': classification_report(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba
    })
    results_list.append(results_knn)

    # 运行K-means聚类
    print("\n=== Running K-means Clustering ===")
    kmeans_model = KMeansModel()
    results_kmeans = kmeans_model.run()
    
    # 生成监督模型的比较结果
    print("\n=== Generating Supervised Models Comparison ===")
    create_comparison_plots(results_list, comparison_dir)
    save_comparison_summary(results_list, comparison_dir)
    
    # 保存无监督学习结果
    print("\n=== Saving Unsupervised Learning Results ===")
    unsupervised_summary_path = os.path.join(comparison_dir, 'unsupervised_results.txt')
    with open(unsupervised_summary_path, 'w') as f:
        f.write("\n=== Unsupervised Learning Results ===\n")
        f.write("\nK-means Clustering Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training Time: {results_kmeans['training_time']:.2f} seconds\n\n")
        
        f.write("Training Set Metrics:\n")
        for metric, value in results_kmeans['train_metrics'].items():
            f.write(f"{metric.title()}: {value:.3f}\n")
        
        f.write("\nTest Set Metrics:\n")
        for metric, value in results_kmeans['test_metrics'].items():
            f.write(f"{metric.title()}: {value:.3f}\n")
    
    print(f"\nComparison results have been saved to: {comparison_dir}")
    print("- model_comparison_accuracy.png")
    print("- model_comparison_metrics.png")
    print("- confusion_matrices_comparison.png")
    print("- per_class_performance.png")
    print("- roc_curves_comparison.png")
    print("- model_comparison_report.txt")
    print("- unsupervised_results.txt")
if __name__ == "__main__":
    main()
