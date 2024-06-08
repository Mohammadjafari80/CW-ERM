import torch
from torchvision import models

# List of supported feature extractors
feature_extractors = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'wide_resnet101_2': models.wide_resnet101_2,
    'wide_resnet50_2': models.wide_resnet50_2,
    'vit_b_16': models.vit_b_16,
}

def get_pretrained_model(backbone='resnet18'):
    if backbone not in feature_extractors:
        raise ValueError(f'backbone {backbone} not supported')

    # Load the pretrained model
    model = feature_extractors[backbone](pretrained=True)
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Modify the model based on its type
    if 'vit' in backbone:
        model.heads = torch.nn.Identity()
    else:
        model = torch.nn.Sequential(*list(model.children())[:-1])
    
    return model
