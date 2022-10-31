from torch.utils.data import Dataset
from pathlib import Path

class DeepPCB(Dataset):
	def __init__(self, root: Path):
