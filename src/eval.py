import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import torch
from tqdm import tqdm

def evaluate_model(model, model_name, dataloader, class_names, device='cuda'):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - ' + model_name)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'./results/val_{model_name}_matriz_confusao.png', bbox_inches='tight', dpi=300)
    plt.show()
    # Retorne métricas para comparação
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return report