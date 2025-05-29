from matplotlib import pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import torch
import torch.nn as nn
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.Analyses import plot_distribution_comparison
from utils.BMWDataset import BMWDataset
from utils.class_weight import compute_class_weights

# Verifica se CUDA está disponível
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 1. PREPARAÇÃO DOS DADOS ==================

mat = scipy.io.loadmat('data/bmw10_release/bmw10_annos.mat')
annots = mat['annos'][0]

filepaths, labels = [], []
for entry in annots:
    img_path = entry[0][0]         
    label = int(entry[1][0][0])    
    filepaths.append(img_path)
    labels.append(label)

df = pd.DataFrame({'filepath': filepaths, 'label': labels})

# Filtrar classes 3, 4, 5
wanted = [3, 4, 5]
df['label_filtered'] = df['label'].apply(lambda x: x if x in wanted else 0)  # 0 = Outros

# Mapear para [0: Outros, 1: 3, 2: 4, 3: 5] (opcional, para facilitar a leitura da matriz)
mapping = {0:0, 3:1, 4:2, 5:3}
df['class_final'] = df['label_filtered'].map(mapping)

print("=== DISTRIBUIÇÃO ORIGINAL ===")
print(df['class_final'].value_counts().sort_index())
class_counts = df['class_final'].value_counts().sort_index()
print(f"Classe 0 (Outros): {class_counts[0]} amostras")
for i in range(1, 4):
    if i in class_counts:
        print(f"Classe {i}: {class_counts[i]} amostras")

train_df, test_df = train_test_split(df, test_size=0.25, stratify=df['class_final'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['class_final'], random_state=42)

img_dir = 'data/bmw10_release/bmw10_ims'

# Transform básico (sem augmentation) - para classe majoritária
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transform com augmentation - para classes minoritárias
augmented_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = BMWDataset(
    train_df, 
    img_dir, 
    basic_transform=basic_transform,
    augmented_transform=augmented_transform,
    minority_classes=[1, 2, 3],  # Classes 3, 4, 5 (mapeadas para 1, 2, 3)
    augment_factor=7  # Aumentar 4x as classes minoritárias
)

# Validação e Teste - sem augmentation
val_dataset = BMWDataset(
    val_df, 
    img_dir, 
    basic_transform=basic_transform,
    augmented_transform=None,  # Sem augmentation
    augment_factor=1  # Sem repetição
)

test_dataset = BMWDataset(
    test_df, 
    img_dir, 
    basic_transform=basic_transform,
    augmented_transform=None,  # Sem augmentation  
    augment_factor=1  # Sem repetição
)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

dataloaders = {
    x: [trainloader, valloader, testloader]
    for x in ['train', 'val', 'test']
}

plot_distribution_comparison(train_df, train_dataset, val_dataset, test_dataset)

# ================== 2. TRANSFER LEARNING ==================

def tl_resnet():
    """
    Configura o modelo ResNet50 para transfer learning.
    Converte a última camada para 4 classes (Outros, 3, 4, 5).
    Congele todas as camadas exceto a última.
    """
    # Verifica se CUDA está disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baixe modelo pré-treinado
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_classes = 4  # Outros, 3, 4, 5

    # Troque a última camada
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Congele todas as camadas, só deixa fc treinável
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)
    return model

def tl_efficientnetv2():
    """
    Configura o modelo EfficientNetV2 para transfer learning.
    Converte a última camada para 4 classes (Outros, 3, 4, 5).
    """

    # Baixe modelo pré-treinado
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    num_classes = 4  # Outros, 3, 4, 5

    # Troque a última camada
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Congele todas as camadas, só deixa classifier treinável
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)
    return model

# ================== 3. TREINAMENTO E AVALIAÇÃO ==================
def train_model(model, criterion, optimizer, dataloaders, num_epochs=5):
    model.to(DEVICE)
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return model

def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    # Retorne métricas para comparação
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    return report

if "__main__" == __name__:

    # Inicializa modelos
    resnet = tl_resnet()
    efficientnet = tl_efficientnetv2()

    # ================== 3. TREINAMENTO E AVALIAÇÃO ==================
    class_weights_tensor = compute_class_weights(train_df, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    print("Treinando ResNet50...")
    resnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-3)
    resnet = train_model(resnet, criterion, resnet_optimizer, dataloaders, num_epochs=40)
    torch.save(resnet.state_dict(), "./models/resnet50_tl.pth")
    print("Avaliando ResNet50...")
    resnet_metrics = evaluate_model(resnet, dataloaders['test'], [1,2,3,"Outros"])

    print("Treinando EfficientNetV2...") 
    efficientnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-3)   
    efficientnet = train_model(efficientnet, criterion, efficientnet_optimizer, dataloaders, num_epochs=40)
    torch.save(efficientnet.state_dict(), "./models/efficientnetv2s_tl.pth")
    print("Avaliando EfficientNetV2...")
    efficientnet_metrics = evaluate_model(efficientnet, dataloaders['test'], [1,2,3,"Outros"])