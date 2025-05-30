import torch
from tqdm import tqdm

from utils.parte_3.metrics import evaluate_imbalanced_dataset

def evaluate_model(model, model_name, dataloader, class_names, fine_tuning, device='cuda'):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = evaluate_imbalanced_dataset(model_name, all_labels, all_preds, all_probs, class_names, fine_tuning)
    
    return report