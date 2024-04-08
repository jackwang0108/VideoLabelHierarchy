# Standard Library
from typing import Literal

# Third-Party Library
import timm

# Torch Library
import torch
import torch.nn as nn
import torchvision.models


def get_inchannel(modality: str) -> int:
    return {"bw": 1, "flow": 2, "rgb": 3}[modality]


def get_resnet(
    backbone: Literal["resnet18", "resnet34", "resnet101", "resnet152"],
    modality: Literal["bw", "rgb", "flow"]
) -> torchvision.models.ResNet:
    assert backbone in (bn := ["resnet18", "resnet34", "resnet101",
                        "resnet152"]), f"Invalid resnet backbone, should be in {bn}"

    # get resnet model
    model: torchvision.models.ResNet = getattr(
        torchvision.models, backbone)(weights=f"{backbone}_Weights.DEFAULT" if modality == "rgb" else None)

    # set the name of the model
    model.name = backbone

    # discard the classification head
    model.feature_dim = model.fc.in_features
    model.fc = nn.Identity()

    # change the first conv kernel according to the modality
    # rgb has 3 in channel, flow has 2 in channel, black and white has 1 in channel
    in_channel = get_inchannel(modality)
    if modality != "rgb":
        origin_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            in_channel, origin_conv1.out_channels,
            kernel_size=origin_conv1.kernel_size, stride=origin_conv1.stride,
            padding=origin_conv1.padding, bias=origin_conv1.bias
        )

    return model


def get_regnet(
    backbone: Literal["regnety_002", "regnety_008"],
    modality: Literal["bw", "rgb", "flow"]
) -> timm.models.RegNet:
    assert backbone in (bn := ["regnety_002", "regnety_008"]
                        ), f"Invalid regnet backbone, should be in {bn}"

    # get regnet model
    model: timm.models.RegNet = timm.create_model(
        backbone, pretrained=modality == "rgb", in_chans=get_inchannel(modality))

    # set the name of the model
    model.name = backbone

    # discard the classification head
    model.feature_dim = model.head.fc.in_features
    model.head.fc = nn.Identity()

    return model


def get_convnext(
    backbone: Literal["convnext_tiny", "convnext_large"],
    modality: Literal["bw", "rgb", "flow"]
) -> timm.models.ConvNeXt:
    # get resnext model
    model: timm.models.ConvNeXt = timm.create_model(
        backbone, pretrained=modality == "rgb", in_chans=get_inchannel(modality))

    # set the name of the model
    model.name = backbone

    # discard the classification head
    model.feature_dim = model.head.fc.in_features
    model.head.fc = nn.Identity()

    return model


def get_backbone(
    backbone: Literal["resnet18", "resnet34", "resnet101", "resnet152",
                      "regnety_002", "regnety_008", "convnext_tiny", "convnext_large"],
    modality: Literal["bw", "rgb", "flow"]
) -> torchvision.models.ResNet | timm.models.RegNet | timm.models.ConvNeXt:
    if backbone.startswith("resnet"):
        return get_resnet(backbone, modality)
    elif backbone.startswith("regnet"):
        return get_regnet(backbone, modality)
    elif backbone.startswith("convnext"):
        return get_convnext(backbone, modality)
    else:
        raise NotImplementedError
