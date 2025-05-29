import torchvision.models as models
import torch.nn as nn

def tl_resnet(device='cuda'):
    """
    Configura o modelo ResNet50 para transfer learning.
    Converte a última camada para 4 classes (Outros, 3, 4, 5).
    Congele todas as camadas exceto a última.
    """
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

def tl_efficientnetv2(device='cuda'):
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

    model = model.to(device)
    return model