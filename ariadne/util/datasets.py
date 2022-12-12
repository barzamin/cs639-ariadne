import torch
from torch.utils.data import Dataset
from pathlib import Path
from .deeppcb import DeepPCBData
from PIL import Image

class DeepPCB(Dataset):
    def __init__(self, root: Path):
        self.ds = DeepPCBData(root)

    def __len__(self):
        return len(self.ds.pairs)

    def __getitem__(self, idx):
        pair = self.ds.pairs[idx]
        annot = DeepPCBData._read_annot(pair['annotpath'])
        img_obsv = Image.open(pair['obsvpath']).convert('1').convert("RGB")
        img_truth = Image.open(pair['truthpath']).convert('1').convert("RGB")


        n_objects = len(annot)
        boxes = torch.zeros((n_objects, 4), dtype=torch.float32)
        labels = torch.zeros((n_objects,), dtype=torch.int64)
        for i, defect in enumerate(annot):
            # boxes.append(defect.aslist())
            boxes[i, :] = defect.astensor()
            labels[i] = defect.ty

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return img_truth, img_obsv, {
            'boxes': boxes,
            'area': area,
            'labels': labels,
            'image_id': pair['pairid']
        }