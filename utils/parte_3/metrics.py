import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    auc,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize

def plot_multiclass_roc(y_true, y_probs, class_names, result_save, modelo_nome="modelo"):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_probs = np.array(y_probs)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(8, 6))

    # ROC para cada classe
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    # ROC macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='navy', lw=2, linestyle='--',
             label=f'Média macro (AUC = {macro_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {modelo_nome}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(result_save + f'{modelo_nome}_roc_curve.png', dpi=300)
    plt.show()

def evaluate_imbalanced_dataset(model_name, all_labels, all_preds, all_probs, class_names, fine_tuning):
    """
    Avaliação específica para datasets desbalanceados
    """
    
    if fine_tuning:
        result_save = "./results/parte_3/ft/"
    else:
        result_save = "./results/parte_3/tl/"

    print(classification_report(all_labels, all_preds, target_names=class_names))
    # Retorne métricas para comparação
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Métricas específicas para desbalanceamento
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    print("=== MÉTRICAS POR CLASSE ===")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {precision[i]:.3f}")
        print(f"  Recall: {recall[i]:.3f}")
        print(f"  F1-Score: {f1[i]:.3f}")
        print(f"  Support: {support[i]}")
    
    # Métricas macro e micro
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    
    print(f"\nMacro F1-Score: {macro_f1:.3f}")
    print(f"Weighted F1-Score: {weighted_f1:.3f}")
    
    # === plot_roc_pr_curves ===
    #y_true, y_probs, class_names, result_save, modelo_nome="modelo"
    plot_multiclass_roc(all_labels, all_probs, class_names, modelo_nome=model_name, result_save=result_save)

    # === Matriz de confusão ===
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix - ' + model_name)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(result_save + f'{model_name}_matriz_confusao.png', bbox_inches='tight', dpi=300)
    plt.show()

    # === Gráficos de Precision, Recall, F1 ===
    x = np.arange(len(class_names))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')
    plt.xticks(x, class_names)
    plt.title("Métricas por Classe")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(result_save + f'/{model_name}_PRF1.png', bbox_inches='tight', dpi=300)
    plt.show()

    return report

def generating_graphs(num_epochs, model_name, train_losses, val_losses, train_accuracies, val_accuracies, fine_tuning):

    if fine_tuning:
        result_save = "./results/parte_3/ft/"
    else:
        result_save = "./results/parte_3/tl/"

    epochs = range(1, num_epochs + 1)

    # Plot da Loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Loss por Época')
    plt.legend()

    # Plot da Acurácia
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.title('Acurácia por Época')
    plt.legend()

    plt.tight_layout()
    plt.savefig(result_save + f"{model_name}", dpi=300)
    plt.show()


def save_metrics_to_csv(resnet_report, efficientnet_report, convnext_report, fine_tuning):
    if fine_tuning:
        result_save = "./results/parte_3/ft/"
    else:
        result_save = "./results/parte_3/tl/"

    df_resnet = pd.DataFrame(resnet_report).T
    df_efficientnet = pd.DataFrame(efficientnet_report).T
    df_convnext_t = pd.DataFrame(convnext_report).T
    
    df_resnet.to_csv(result_save + 'resnet50_results.csv')
    df_efficientnet.to_csv(result_save + 'efficientnetv2_s_results.csv')
    df_convnext_t.to_csv(result_save + 'convnext_t_results.csv')