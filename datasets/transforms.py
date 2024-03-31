# Standard Library
import random
from typing import Optional

# Torch Library
import torch
import torch.nn as nn
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

# My Library
from utils.color import red, green


class SameRandomStateContext:
    def __enter__(self):
        self.random_state = random.getstate()

    def __exit__(self, exc_type, exc_value, traceback):
        random.setstate(self.random_state)


IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]


class ThreeCrop:
    """ Apply three crops to the input image along the width dimension. """

    def __init__(self, dim: int):
        self._dim = dim

    def __call__(self, img: torch.FloatTensor) -> torch.FloatTensor:
        c, h, w = img.shape[-3:]
        y = (h - self._dim) // 2
        dw = w - self._dim
        ret = [VF.crop(img, y, x, self._dim, self._dim)
               for x in (0, dw // 2, dw)]
        return torch.stack(ret)


class RandomHorizontalFlipFLow(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1)[0] < self.p:
            shape = img.shape
            img.view((-1,) + shape[-3:])[:, 1, :, :] *= -1
            return img.flip(-1)
        return img


class RandomOffsetFlow(nn.Module):

    def __init__(self, p=0.5, x=0.1, y=0.05):
        super().__init__()
        self.p = p
        self.x = x
        self.y = y

    def forward(self, img):
        if torch.rand(1)[0] < self.p:
            shape = img.shape
            view = img.view((-1,) + shape[-3:])
            view[:, 1, :, :] += (
                torch.rand(1, device=img.device)[0] * 2 - 1) * self.x
            view[:, 0, :, :] += (
                torch.rand(1, device=img.device)[0] * 2 - 1) * self.y
        return img


class RandomGaussianNoise(nn.Module):
    """ Apply random Gaussian noise to the input image with a given probability and standard deviation.  """

    def __init__(self, p: float = 0.5, s: float = 0.1):
        super().__init__()
        self.p = p
        self.std = s ** 0.5

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        v = torch.rand(1)[0]
        if v < self.p:
            img += torch.randn(img.shape, device=img.device) * self.std
        return img


class SeedableRandomSquareCrop:
    """ Apply a random square crop to the input image with a seedable behavior. The random crop status could be fixed. """

    def __init__(self, dim: int):
        self._dim = dim

    def __call__(self, img: torch.FloatTensor) -> torch.FloatTensor:
        c, h, w = img.shape[-3:]
        x, y = 0, 0
        if h > self._dim:
            y = random.randint(0, h - self._dim)
        if w > self._dim:
            x = random.randint(0, w - self._dim)
        return VF.crop(img, y, x, self._dim, self._dim)


def get_crop_transform(
    is_eval: bool,
    same_crop_transform: bool,
    multi_crop: bool = False,
    crop_dim: Optional[int] = None,
) -> ThreeCrop | SeedableRandomSquareCrop | VT.CenterCrop | VT.RandomCrop:
    """
    Returns the appropriate crop transform based on the input parameters.

    Args:
        is_eval: A boolean indicating if the transform is for evaluation.
        crop_dim: An integer specifying the dimensions of the crop.
        same_transform: A boolean indicating if the same random crop should be used for all frames in a clip.
        multi_crop: A boolean indicating if multiple crops are used (default is False).

    Returns:
        The selected crop transform based on the input parameters.
    """
    crop_transform = None
    if crop_dim is not None:
        if multi_crop:
            assert is_eval, "Using ThreeCrop is only allow in evaluation phase"
            crop_transform = ThreeCrop(crop_dim)
        elif is_eval:
            crop_transform = VT.CenterCrop(crop_dim)
        elif same_crop_transform:
            crop_transform = SeedableRandomSquareCrop(crop_dim)
        else:
            crop_transform = VT.RandomCrop(crop_dim)
    return crop_transform


def get_rgb_transform(
    is_eval: bool,
) -> torch.ScriptModule:
    """
    Returns a TorchScript transform on the RGB image.

    Args:
        is_eval: A boolean indicating if the transform is for evaluation.

    Returns:
        torch.ScriptModule: TorchScript module with RGB image transformations.
    """
    img_transforms = []

    # Transforms for training
    if not is_eval:
        img_transforms.extend([
            VT.RandomHorizontalFlip(),
            VT.RandomApply(nn.ModuleList([VT.ColorJitter(hue=0.2)]), p=0.25),
            VT.RandomApply(nn.ModuleList(
                [VT.ColorJitter(saturation=(0.7, 1.2))]), p=0.25),
            VT.RandomApply(nn.ModuleList(
                [VT.ColorJitter(brightness=(0.7, 1.2))]), p=0.25),
            VT.RandomApply(nn.ModuleList(
                [VT.ColorJitter(contrast=(0.7, 1.2))]), p=0.25),
            VT.RandomApply(nn.ModuleList(
                [VT.GaussianBlur(kernel_size=5)]), p=0.25),
        ])

    # Transforms for both training and evaluating
    img_transforms.append(VT.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return torch.jit.script(nn.Sequential(*img_transforms))


def get_bw_transform(
    is_eval: bool
) -> torch.ScriptModule:
    """
    Returns a TorchScript transform on the black and white image.

    Args:
        is_eval: A boolean indicating if the transform is for evaluation.

    Returns:
        torch.ScriptModule: TorchScript module with black and white image transformations.
    """
    img_transforms = []

    # Transforms for training
    if not is_eval:
        img_transforms.extend([
            VT.RandomHorizontalFlip(),
            VT.RandomApply(nn.ModuleList([VT.ColorJitter(hue=0.2)]), p=0.25),
        ])

    # Transforms for both training and evaluating
    img_transforms.append(VT.Grayscale())

    # Transforms for training
    if not is_eval:
        img_transforms.extend([
            VT.RandomApply(nn.ModuleList(
                [VT.ColorJitter(brightness=0.3)]), p=0.25),
            VT.RandomApply(nn.ModuleList(
                [VT.ColorJitter(contrast=0.3)]), p=0.25),
            VT.RandomApply(nn.ModuleList(
                [VT.GaussianBlur(kernel_size=5)]), p=0.25),
        ])

    # Transforms for both training and evaluating
    img_transforms.append(VT.Normalize(mean=[0.5], std=[0.5]))

    # Transforms for training
    if not is_eval:
        img_transforms.append(RandomGaussianNoise())

    return torch.jit.script(nn.Sequential(*img_transforms))


def get_flow_transform(
    is_eval: bool
) -> torch.ScriptModule:
    """
    Returns a TorchScript transform on the optical flow image.

    Args:
        is_eval: A boolean indicating if the transform is for evaluation.

    Returns:
        torch.ScriptModule: TorchScript module with optical flow image transformations.
    """
    img_transforms = [VT.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])]

    # Transforms for training
    if not is_eval:
        img_transforms.extend([
            RandomHorizontalFlipFLow(),
            RandomOffsetFlow(),
            RandomGaussianNoise()
        ])

    return torch.jit.script(nn.Sequential(*img_transforms))


def get_img_transform(
    is_eval: bool,
    modality: str,
) -> torch.ScriptModule:
    """
    Returns a TorchScript module with image transformation for images of modality.

    Args:
        is_eval: A boolean indicating if the transform is for evaluation.
        modality: A string specifying the type of input images (rgb, bw, flow).

    Returns:
        torch.ScriptModule: TorchScript module with image transformations based on the specified modality.
    """
    assert modality in (m := {"rgb", "bw", "flow"}), f"modality should be {
        green(m)} images, but got {red(modality)}"

    if modality == "rgb":
        return get_rgb_transform(is_eval)
    elif modality == "bw":
        return get_bw_transform(is_eval)
    else:
        return get_crop_transform(is_eval)


if __name__ == "__main__":
    get_img_transform(False, "gb")
