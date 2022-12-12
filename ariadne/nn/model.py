import torch
import torch.nn as nn
import torchvision
from ..util.deeppcb import DefectType
# from torchvision.models import vgg16
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class ISAriadne(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # replace pretrained ROI heads
        num_classes = len(DefectType)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    def forward(self, impair, targets = None) -> torch.Tensor:
        print(impair)
        return self.model(imdiff, targets)
        # return x