# Standard Library
from typing import Literal, Callable

# Third-Party Library
import timm

# Torch Library
import torch
import torchvision
import torch.nn as nn

# My Library
from .backbone import get_resnet, get_regnet, get_convnext
# from .temporal import VanillaGRUPrediction, UniTransGRUPrediction, BiTransGRUPrediction
from .shift import TemporalShift, GatedShift, get_shift_module_builder, insert_temporal_shift


class VideoLabelHierarchy(nn.Module):
    def __init__(
        self,
        backbone: torchvision.models.ResNet | timm.models.RegNet | timm.models.ConvNeXt,
        temporal
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.temporal = temporal

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # sourcery skip: inline-immediately-returned-variable

        # shape: [Batch, Temporal, Channel, Height, Width]
        batch, clip_len, channel, height, width = x.size()

        # transform the input to [Batch * Temporal, Channel, Height, Width] before go through backbone
        x_in = x.view(-1, channel, height, width)
        image_feature: torch.FloatTensor = self.backbone(x_in)

        # transform the input to [Batch, Temporal, Feature] before go through temporal
        image_feature_in = image_feature.view(batch, clip_len, -1)
        logits = self.temporal(image_feature_in)

        return logits


def get_model(
    backbone: Literal["resnet18", "resnet34", "resnet101", "resnet152",
                      "regnety_002", "regnety_008", "convnext_tiny", "convnext_large"],
    modality: str,
    clip_len: int,
    n_div: int,
    inplace_tsm: bool
) -> nn.Module:
    # get backbone
    if backbone.lower().startswith("resnet"):
        backbone = get_resnet(backbone, modality)
    elif backbone.lower().startswith("regnety"):
        backbone = get_regnet(backbone, modality)
    elif backbone.lower().startswith("convnext"):
        backbone = get_convnext(backbone, modality)

    # insert temporal shift module
    shift_module_builder = get_shift_module_builder(
        clip_len, n_div, inplace_tsm)
    backbone = insert_temporal_shift(backbone, shift_module_builder)

    # TODO: add temporal

    return backbone
