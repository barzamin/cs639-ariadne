from ariadne.util.deeppcb import DeepPCBData
from pathlib import Path

if __name__ == '__main__':
    ds = DeepPCBData(Path('/Users/moon/git/DeepPCB'))
    # print(ds)
    for gid, group in ds.groups:
        for pair in group:
            print(pair['pairid'], DeepPCBData._read_annot(pair['annotpath']))