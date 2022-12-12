import torch
from torch import nn, Tensor
import torchvision.transforms.functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tmpl_image, obsv_image, target):
        for t in self.transforms:
            tmpl_image, obsv_image, target = t(tmpl_image, obsv_image, target)
        return tmpl_image, obsv_image, target

class PILToTensor(nn.Module):
    def forward(
        self, tmpl_image, obsv_image, target,
    ):
        tmpl_image = F.pil_to_tensor(tmpl_image)
        obsv_image = F.pil_to_tensor(obsv_image)
        return tmpl_image, obsv_image, target

class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, tmpl_image, obsv_image, target,
    ):
        tmpl_image = F.convert_image_dtype(tmpl_image, self.dtype)
        obsv_image = F.convert_image_dtype(obsv_image, self.dtype)
        return tmpl_image, obsv_image, target
