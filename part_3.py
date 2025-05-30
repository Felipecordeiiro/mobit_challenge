import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.analyses import plot_distribution_comparison
from src.eval import evaluate_model
from src.models import tl_convnet_t, tl_efficientnetv2_s, tl_resnet50
from src.train import train_model
from utils.data import get_data
from utils.metrics import evaluate_imbalanced_dataset, save_metrics_to_csv

if "__main__" == __name__:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"O ambiente está usando: {DEVICE})")

    # ================== 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ==================
    train_df, val_df, test_df = get_data()

    img_dir = 'data/bmw10_release/bmw10_ims'

    basic_transform = get_basic_transform()
    augmented_transform = get_augmented_transform()

    print("Train dataset")
    train_dataset = BMWDataset(
        train_df, 
        img_dir, 
        basic_transform=basic_transform,
        augmented_transform=augmented_transform,
        minority_classes=[1, 2, 3],
        augment_factor=8  # Aumentar 7x as classes minoritárias
    )

    print("Val dataset")
    val_dataset = BMWDataset(
        val_df, 
        img_dir, 
        basic_transform=basic_transform,
        augmented_transform=None,
        augment_factor=1 
    )

    print("Test dataset")
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
    resnet = tl_resnet50(device=DEVICE, fine_tuning=True)
    efficientnet = tl_efficientnetv2_s(device=DEVICE, fine_tuning=True)
    convnext_t = tl_convnet_t(device=DEVICE, fine_tuning=True)

    # ================== 2. TREINAMENTO E AVALIAÇÃO ==================
    class_weights_tensor = compute_class_weights(train_df, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    class_name = ["Outros", "1", "2", "3"]

    os.makedirs("./models/", exist_ok=True)

    print("Treinando ResNet50...")
    resnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-4, weight_decay=1e-5)
    resnet = train_model(resnet, "ResNet50", trainloader, valloader, criterion, resnet_optimizer, num_epochs=15)
    torch.save(resnet.state_dict(), "./models/resnet50_tl.pth")
    #resnet.load_state_dict(torch.load("./models/resnet50_tl.pth",  weights_only=True))
    print("Avaliando ResNet50...")
    resnet_metrics = evaluate_model(resnet, "ResNet50", testloader, class_name)
    evaluate_imbalanced_dataset(resnet, "ResNet50", testloader, class_name)

    print("Treinando EfficientNetV2...") 
    efficientnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, efficientnet.parameters()), lr=1e-4, weight_decay=1e-5)   
    efficientnet = train_model(efficientnet, "EfficientNetV2s", trainloader, valloader, criterion, efficientnet_optimizer, num_epochs=15)
    torch.save(efficientnet.state_dict(), "./models/efficientnetv2s_tl.pth")
    #efficientnet.load_state_dict(torch.load("./models/efficientnetv2s_tl.pth",  weights_only=True))
    print("Avaliando EfficientNetV2...")
    efficientnet_metrics = evaluate_model(efficientnet, "EfficientNetV2", testloader, class_name)
    evaluate_imbalanced_dataset(efficientnet, "EfficientNetV2", testloader, class_name)

    print("Treinando ConvNeXt_Tiny...") 
    convnext_t_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, convnext_t.parameters()), lr=1e-4, weight_decay=1e-5)   
    convnext_t = train_model(convnext_t, "ConvNeXt_Tiny", trainloader, valloader, criterion, convnext_t_optimizer, num_epochs=15)
    torch.save(convnext_t.state_dict(), "./models/convnext_t.pth")
    #convnext_t.load_state_dict(torch.load("./models/convnext_t.pth",  weights_only=True))
    print("Avaliando ConvNext_Tiny...")
    convnext_t_metrics = evaluate_model(convnext_t, "ConvNeXt", testloader, class_name)
    evaluate_imbalanced_dataset(convnext_t, "ConvNeXt", testloader, class_name)

    # Salvando resultados
    save_metrics_to_csv(resnet_metrics, efficientnet_metrics, convnext_t_metrics)