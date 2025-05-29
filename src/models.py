import torchvision.models as models
import torch.nn as nn

NUM_CLASSES = 4 # Outros, 3, 4, 5

def tl_resnet50(device='cuda', fine_tuning=True):
    """
    Configura o modelo ResNet50 para transfer learning ou Fine-tuning.
    Converte a última camada para 4 classes (Outros, 3, 4, 5).
    Congele todas as camadas exceto a última.
    """
    # Baixe modelo pré-treinado
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Troque a última camada
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # Congele todas as camadas, só deixa fc treinável
    for param in model.parameters():
        param.requires_grad = False
    if fine_tuning:
        for param in model.layer4.parameters():
            param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)
    return model

def tl_efficientnetv2_s(device='cuda', fine_tuning=True):
    """
    Configura o modelo EfficientNetV2 para transfer learning ou Fine-tuning.
    Converte a última camada para 4 classes (Outros, 3, 4, 5).
    """

    # Baixe modelo pré-treinado
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    # Troque a última camada
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    # Congele todas as camadas, só deixa classifier treinável
    for param in model.parameters():
        param.requires_grad = False
    if fine_tuning:
        n_blocos = 2
        for param in model.features[-n_blocos].parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)
    return model

def tl_convnet_t(device='cuda', fine_tuning=True):
    """
    Configura o modelo ConvNeXt_Tiny para transfer learning ou Fine-tuning.
    Converte a última camada para 4 classes (Outros, 3, 4, 5).
    Congele todas as camadas exceto a última.
    """
    # Baixe modelo pré-treinado
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

    # Troque a última camada
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)

    # Congele todas as camadas, só deixa fc treinável
    for param in model.parameters():
        param.requires_grad = False
    if fine_tuning:
        for param in model.features[5].parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)
    return model