# Standard Library

# Torch Library
import torch
import torch.nn as nn

# My Library
from .backbone import get_resnet, get_regnet, get_convnext


def get_model(backbone: str, modality: str) -> nn.Module:
    if backbone.lower().startswith("resnet"):
        model = get_resnet(backbone, modality)
    elif backbone.lower().startswith("regnety"):
        model = get_regnet(backbone, modality)
    elif backbone.lower().startswith("convnext"):
        model = get_convnext(backbone, modality)

    return model
