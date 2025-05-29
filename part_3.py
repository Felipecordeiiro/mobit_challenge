import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.analyses import plot_distribution_comparison
from src.eval import evaluate_model
from src.models import tl_efficientnetv2, tl_resnet
from src.train import train_model
from src.transforms import get_basic_transform, get_augmented_transform
from src.dataset import BMWDataset
from src.class_weight import compute_class_weights
from utils.data import get_data
from utils.metrics import save_metrics_to_csv

if "__main__" == __name__:
    # Verifica se CUDA está disponível
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"O ambiente está usando: {DEVICE})")

    # ================== 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ==================
    train_df, val_df, test_df = get_data()

    img_dir = 'data/bmw10_release/bmw10_ims'

    basic_transform = get_basic_transform()
    augmented_transform = get_augmented_transform()

    train_dataset = BMWDataset(
        train_df, 
        img_dir, 
        basic_transform=basic_transform,
        augmented_transform=augmented_transform,
        minority_classes=[1, 2, 3],
        augment_factor=7  # Aumentar 4x as classes minoritárias
    )

    # Validação e Teste - sem augmentation
    val_dataset = BMWDataset(
        val_df, 
        img_dir, 
        basic_transform=basic_transform,
        augmented_transform=None,
        augment_factor=1 
    )

    test_dataset = BMWDataset(
        test_df, 
        img_dir, 
        basic_transform=basic_transform,
        augmented_transform=None,  
        augment_factor=1 
    )

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    plot_distribution_comparison(train_df, train_dataset, val_dataset, test_dataset)
    # Inicializa modelos
    resnet = tl_resnet(device=DEVICE)
    efficientnet = tl_efficientnetv2(device=DEVICE)

    # ================== 2. TREINAMENTO E AVALIAÇÃO ==================
    class_weights_tensor = compute_class_weights(train_df, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    #print("Treinando ResNet50...")
    #resnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-3)
    #resnet = train_model(resnet, trainloader, valloader, criterion, resnet_optimizer, num_epochs=40)
    #torch.save(resnet.state_dict(), "./models/resnet50_tl.pth")
    resnet.load_state_dict(torch.load("./models/resnet50_tl.pth",  weights_only=True))
    print("Avaliando ResNet50...")
    resnet_metrics = evaluate_model(resnet, "ResNet50", testloader, ["Outros", "1", "2", "3"])

    #print("Treinando EfficientNetV2...") 
    #efficientnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-3)   
    #efficientnet = train_model(efficientnet, trainloader, valloader, criterion, resnet_optimizer, num_epochs=40)
    #torch.save(efficientnet.state_dict(), "./models/efficientnetv2s_tl.pth")
    #efficientnet.load_state_dict(torch.load("./models/efficientnetv2s_tl.pth",  weights_only=True))
    #print("Avaliando EfficientNetV2...")
    #efficientnet_metrics = evaluate_model(efficientnet, "EfficientNetV2", testloader, ["Outros", "1", "2", "3"])

    # Salvando resultados
    #save_metrics_to_csv(resnet_metrics, efficientnet_metrics)