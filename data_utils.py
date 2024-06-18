import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

import random

import numpy as np
import json

class MLDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        self.path = img_path
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgs = np.load(f"{self.path}/{idx}.npy")

        seq_length = imgs.shape[0]
        label = self.labels[str(idx)]
        imgs = torch.tensor(imgs, dtype=torch.float32)
        
        # Apply data augmentation to each image if transform is provided
        if self.transform:

            #imgs = np.array([np.array(self.transform(Image.fromarray(img))) for img in imgs])
            imgs = np.array([self.transform(transforms.ToPILImage()(img.permute(2, 0, 1))).permute(1, 2, 0) for img in imgs])
            
        
        """
        # Padding images to length 10
        if seq_length < 10:
            padding = np.zeros((10 - seq_length, 28, 28, 3))
            imgs = np.vstack((imgs, padding))
        if len(label) < 10:
            label.extend([0] * (10 - len(label)))  # Assuming 0 is the padding index for labels
        """

        # Padding images and labels to length 10 at random positions
        new_seq_length = 0
        if seq_length < 10:
            num_padding = 10 - seq_length
            padding_positions = sorted(random.sample(range(10), num_padding))

            padded_imgs = np.zeros((10, 28, 28, 3), dtype=np.float32)
            padded_labels = np.zeros(10, dtype=np.int32)
            img_idx = 0
            label_idx = 0
            for i in range(10):
                if i not in padding_positions:
                    padded_imgs[i] = imgs[img_idx]
                    padded_labels[i] = label[label_idx]
                    img_idx += 1
                    label_idx += 1
                    if img_idx == seq_length:
                        new_seq_length = i + 1
            imgs = padded_imgs
            label = padded_labels

        return torch.tensor(imgs, dtype=torch.float32), torch.tensor(label, dtype=torch.long), new_seq_length

def collate_fn(batch, transform=None):
    sequences, targets, lengths = zip(*batch)
    if transform:
        augmented_sequences = []
        for seq in sequences:
            augmented_seq = np.array([np.array(transform(transforms.ToPILImage()(img.permute(2, 0, 1))).permute(1, 2, 0)) for img in seq])
            augmented_sequences.append(torch.tensor(augmented_seq, dtype=torch.float32))
        sequences = augmented_sequences
    #print(sequences.shape)
    padded_sequences = pad_sequence([seq for seq in sequences], batch_first=True)
    padded_targets = pad_sequence([tar for tar in targets], batch_first=True)
    
    return padded_sequences, padded_targets, torch.tensor(lengths, dtype=torch.float32)


augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(20),  # Rotate the image by up to 10 degrees
    transforms.RandomResizedCrop(28, scale=(1.0, 1.2)),  # Randomly crop and resize the image
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),  # Random color adjustments
    transforms.ToTensor()  # Convert the image to a tensor
])