import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
import json

class MLDataset(Dataset):
    def __init__(self, img_path, label_path):
        self.path = img_path
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgs = np.load(f"{self.path}/{idx}.npy")
        seq_length = imgs.shape[0]
        label = self.labels[str(idx)]
        
        return torch.tensor(imgs, dtype=torch.float32), torch.tensor(label, dtype=torch.long), seq_length


def collate_fn(batch):
    sequences, targets, lengths = zip(*batch)
    
    padded_sequences = pad_sequence([seq for seq in sequences], batch_first=True)
    padded_targets = pad_sequence([tar for tar in targets], batch_first=True)
    
    return padded_sequences, padded_targets, torch.tensor(lengths, dtype=torch.float32)
