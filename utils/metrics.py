import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def evaluate_imbalanced_dataset(model, testloader, class_names):
    """
    Avaliação específica para datasets desbalanceados
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in testloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
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
    
    return all_preds, all_labels

def save_metrics_to_csv(resnet_report, efficientnet_report):
    df_resnet = pd.DataFrame(resnet_report).T
    df_efficientnet = pd.DataFrame(efficientnet_report).T
    
    df_resnet.to_csv('./models/resnet_results.csv')
    df_efficientnet.to_csv('./models/efficientnet_results.csv')