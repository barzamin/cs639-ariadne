import torch
import torch.nn as nn
import torchvision
import torchvision.models as tvm

class Ariadne(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        x = self.vgg.features(x)

        return x
