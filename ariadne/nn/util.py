import torchvision.transforms.functional as tvtF

def show_results(img, prediction, score_thresh=0.8):
    LABELMAP = [v.name for v in DefectType]

    mask = prediction['scores'] > score_thresh

    return tvtF.to_pil_image(torchvision.utils.draw_bounding_boxes(
        tvtF.convert_image_dtype(img, torch.uint8),
        prediction['boxes'][mask],
        [LABELMAP[l] for l in prediction['labels'][mask]],
    ))
