import torch
import torchvision
import torchvision.transforms.functional as tvtF
from ..util.deeppcb import DefectType

def show_results(img, prediction, score_thresh=0.8, **kwargs):
    LABELMAP = [v.name for v in DefectType]

    if 'scores' in prediction:
        mask = prediction['scores'] > score_thresh
    else:
        mask = torch.ones_like(prediction['labels'], dtype=torch.bool)

    return tvtF.to_pil_image(torchvision.utils.draw_bounding_boxes(
        tvtF.convert_image_dtype(img, torch.uint8),
        prediction['boxes'][mask],
        [LABELMAP[l] for l in prediction['labels'][mask]],
        **kwargs,
    ))
