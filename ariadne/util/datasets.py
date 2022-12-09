import torch
from torch.utils.data import Dataset
from pathlib import Path
from .deeppcb import DeepPCBData

class DeepPCB(Dataset):
    def __init__(self, root: Path):
        self.ds = DeepPCBData(root)

    def __len__(self):
        return len(self.ds.pairs)

    def __getitem__(self, idx):
        pair = self.ds.pairs[idx]
        annot = DeepPCBData._read_annot(pair['annotpath'])

        n_objects = len(annot)
        boxes = torch.zeros((n_objects, 4), dtype=torch.float32)
        labels = torch.zeros((n_objects,), dtype=torch.int64)
        for i, defect in enumerate(annot):
            # boxes.append(defect.aslist())
            boxes[i, :] = defect.astensor()
            labels[i] = defect.ty

        return {'boxes': boxes, 'labels': labels}