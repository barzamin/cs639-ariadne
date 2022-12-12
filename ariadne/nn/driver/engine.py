import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from . import utils
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, logwriter=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    i = 0
    for tmpl_images, obsv_images, targets in metric_logger.log_every(data_loader, print_freq, header):
        tmpl_images = list(image.to(device) for image in tmpl_images)
        obsv_images = list(image.to(device) for image in obsv_images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(tmpl_images, obsv_images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if logwriter is not None:
            if i % 10 == 0:
                logwriter.add_scalar('train/lr', optimizer.param_groups[0]["lr"], i)
                logwriter.add_scalar('train/total_loss', losses_reduced.item(), i)
                logwriter.add_scalars('train/losses', {k: v.item() for k, v in loss_dict_reduced.items()}, i)

        i += 1

    return metric_logger


@torch.inference_mode()
def evaluate(model, data_loader, device):
    return