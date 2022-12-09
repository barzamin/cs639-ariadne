from pathlib import Path
import re
from pprint import pprint
from enum import IntEnum
import torch
from dataclasses import dataclass

class DefectType(IntEnum):
    BACKGROUND = 0
    OPEN       = 1
    SHORT      = 2
    MOUSEBITE  = 3
    SPUR       = 4
    COPPER     = 5
    PINHOLE    = 6

@dataclass
class Defect:
    x0: int
    y0: int
    x1: int
    y1: int
    ty: DefectType

    def aslist(self):
        return [self.x0, self.y0, self.x1, self.y1]

    def astensor(self):
        return torch.tensor(self.aslist())

    @property
    def upleft(self):
        return (self.x0, self.y0)

    @property
    def downright(self):
        return (self.x1, self.y1)

    @property
    def width(self):
        return self.y1 - self.y0

    @property
    def height(self):
        return self.y1 - self.y0

    def __repr__(self):
        return f'Defect({self.ty.name}: {self.upleft} to {self.downright})'

class DeepPCBData:
    def __init__(self, root):
        self.root = root
        self.pairs = []
        for annotpath in (self.root/'PCBData').glob('*/*_not/*.txt'):
            pair_id = annotpath.stem
            groupid = annotpath.parent.name.removesuffix('_not')

            obsvpath = annotpath.parent.parent/groupid/f'{pair_id}_test.jpg'
            truthpath = annotpath.parent.parent/groupid/f'{pair_id}_test.jpg'
            assert(obsvpath.exists() and truthpath.exists())

            self.pairs.append({
                'pairid': int(pair_id),
                'obsvpath': obsvpath,
                'truthpath': truthpath,
                'annotpath': annotpath,
            })

    def _read_annot(annotpath):
        annotations = []
        with open(annotpath) as f:
            for line in f:
                # (x0, y0) top-left, (x1, y1) bottom-right
                x0, y0, x1, y1, defect_ty = line.strip().split()
                x0, y0 = int(x0), int(y0)
                x1, y1 = int(x1), int(y1)
                defect_ty = DefectType(int(defect_ty))

                annotations.append(Defect(x0, y0, x1, y1, defect_ty))

        return annotations


if __name__ == '__main__':
    # ds = DeepPCBData(Path('/Users/moon/git/DeepPCB'))
    pprint(DeepPCBData._read_annot('/Users/moon/git/DeepPCB/PCBData/group13000/13000_not/13000031.txt'))
