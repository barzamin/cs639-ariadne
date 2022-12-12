import torch
from torch import nn, Tensor
import torchvision.transforms as T
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

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, tmpl_image, obsv_image, target
    ):
        if torch.rand(1) < self.p:
            tmpl_image = F.hflip(tmpl_image)
            obsv_image = F.hflip(obsv_image)

            if target is not None:
                _, _, width = F.get_dimensions(tmpl_image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                # if "keypoints" in target:
                #     keypoints = target["keypoints"]
                #     keypoints = _flip_coco_person_keypoints(keypoints, width)
                #     target["keypoints"] = keypoints

        return tmpl_image, obsv_image, target

class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(
        self, tmpl_image, obsv_image, target
    ):
        if torch.rand(1) < self.p:
        # if True:
            tmpl_image = F.vflip(tmpl_image)
            obsv_image = F.vflip(obsv_image)

            if target is not None:
                _, height, _ = F.get_dimensions(tmpl_image)
                target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
                if "masks" in target:
                    raise NotImplementedError()
                #     target["masks"] = target["masks"].flip(-2)
                # if "keypoints" in target:
                #     keypoints = target["keypoints"]
                #     keypoints = _flip_coco_person_keypoints(keypoints, width)
                #     target["keypoints"] = keypoints


        return tmpl_image, obsv_image, target
