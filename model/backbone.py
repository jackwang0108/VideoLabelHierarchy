# Standard Library
from typing import Literal

# Third-Party Library
import timm

# Torch Library
import torch.nn as nn
import torchvision.models


def get_inchannel(modality: str) -> int:
    return {"bw": 1, "flow": 2, "rgb": 3}[modality]


def get_resnet(
    backbone: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    modality: Literal["bw", "rgb", "flow"]
) -> torchvision.models.ResNet:
    """
    get resnet backbone

    input shape: [Batch * Temporal, Channel, Height, Width]
    output shape: [Batch * Temporal, Feature], Feature=512 for resnet18/34, Feature=2048 for resnet50/101/152

    Args:
        backbone (Literal[&quot;resnet18&quot;, &quot;resnet34&quot;, &quot;resnet101&quot;, &quot;resnet152&quot;]): which resent to use
        modality (Literal[&quot;bw&quot;, &quot;rgb&quot;, &quot;flow&quot;]): modality of input, used to decided the input channel of first conv

    Returns:
        torchvision.models.ResNet: resnet model from torchvision
    """
    assert backbone in (bn := ["resnet18", "resnet34", "resnet50", "resnet101",
                        "resnet152"]), f"Invalid resnet backbone {backbone}, should be in {bn}"

    # get resnet model
    model: torchvision.models.ResNet = getattr(
        torchvision.models, backbone)(weights=f"ResNet{backbone[6:]}_Weights.DEFAULT" if modality == "rgb" else None)

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
    """
    get regnet backbone

    input shape: [Batch * Temporal, Channel, Height, Width]
    output shape: [Batch * Temporal, Feature], Feature=368 for regnety_002, Feature=768 for regnety_008

    Args:
        backbone (Literal[&quot;regnety_002&quot;, &quot;regnety_008&quot;]): which regnet to use
        modality (Literal[&quot;bw&quot;, &quot;rgb&quot;, &quot;flow&quot;]): modality of input, used to decided the input channel of first conv

    Returns:
        timm.models.RegNet: regnet model from timm
    """
    assert backbone in (bn := ["regnety_002", "regnety_008"]
                        ), f"Invalid regnet backbone {backbone}, should be in {bn}"

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
    """
    get convnext backbone

    input shape: [Batch * Temporal, Channel, Height, Width]
    output shape: [Batch * Temporal, Feature], Feature=768 for convnext_tiny, Feature=1536 for convnext_large

    Args:
        backbone (Literal[&quot;convnext_tiny&quot;, &quot;convnext_large&quot;]): which regnet to use
        modality (Literal[&quot;bw&quot;, &quot;rgb&quot;, &quot;flow&quot;]): modality of input, used to decided the input channel of first conv

    Returns:
        timm.models.ConvNext: convnext model from timm
    """
    assert backbone in (bn := ["convnext_tiny", "convnext_large"]
                        ), f"Invalid convnext backbone {backbone}, should be in {bn}"

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
