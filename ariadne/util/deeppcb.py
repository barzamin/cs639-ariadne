from pathlib import Path
import re
from pprint import pprint
from enum import IntEnum
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

    @property
    def upleft(self):
        return (self.x0, self.y0)

    @property
    def downright(self):
        return (self.x1, self.y1)

    def __repr__(self):
        return f'Defect({self.ty.name}, {self.upleft} to {self.upleft})'
    
    

class DeepPCBData:
    def __init__(self, root):
        self.root = root
        pprint(self._getpaths())

    def _getpaths(self):
        groups = []
        for grouppath in (self.root/'PCBData').glob('group*'):
            if m := re.match(r'group(\d+)', grouppath.stem):
                gid, = m.groups()
                pairs = []
                for annotpath in (grouppath/f'{gid}_not').glob('*.txt'):
                    pairid = int(annotpath.stem)
                    testpath = grouppath/f'{gid}'/f'{pairid}_test.jpg'
                    templatepath = grouppath/f'{gid}'/f'{pairid}_temp.jpg'

                    pairs.append({
                        'pairid': pairid,
                        'testpath': testpath,
                        'templatepath': templatepath,
                        'annotpath': annotpath,
                    })

                groups.append((gid, pairs))

        return groups

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