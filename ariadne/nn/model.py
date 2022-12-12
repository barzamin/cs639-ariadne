import torch
import torchvision
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers

from . import transform as T # pairwise transforms
from .rcnn import PairwiseFasterRCNN
from ..util.deeppcb import DefectType


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)

def ariadne_resnet50():
    num_classes = len(DefectType)

    trainable_backbone_layers = 3 # default from fasterrcnn_resnet50_fpn
    resnet_norm_layer = torchvision.ops.misc.FrozenBatchNorm2d

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, norm_layer=resnet_norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

    model = PairwiseFasterRCNN(
        backbone,
        num_classes=num_classes,
    )

    return model