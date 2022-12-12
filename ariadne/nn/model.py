import torch
import torch.nn as nn
import torchvision
# from torchvision.models import vgg16
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator

class ISAriadne(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # self.backbone = vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_FEATURES).features

        # self.model = FasterRCNN(self.backbone,
        #     num_classes=num_classes,
        #     rpn_anchor_generator=anchorgen,
        #     box_roi_pooler=roipooler)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")


    def forward(self, truth_images, , targets = None) -> torch.Tensor:
        return self.model(imdiff, targets)
        # return x