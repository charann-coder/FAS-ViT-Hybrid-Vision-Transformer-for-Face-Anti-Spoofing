import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class LCCFASDRotationSSL(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, real_only=False):
        """
        Self-supervised dataset for rotation prediction.
        Args:
            csv_file (str): Path to CSV with image paths and labels.
            root_dir (str): Root directory containing the image files.
            transform (callable, optional): Optional transform to apply after rotation.
            real_only (bool): If True, only uses real images (label == 'real').
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        with open(csv_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue  # skip malformed lines
                img_path, label = parts[:2]
                if real_only and label.strip().lower() != 'real':
                    continue
                self.data.append((img_path.strip(), label.strip()))
                self.labels.append(int(label.strip()))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        img_rel_path, label = self.data[idx]
        img_path = os.path.join(self.root_dir, img_rel_path)
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
        image = Image.open(img_path).convert('RGB')

        # Use the label from the CSV to apply the correct rotation
        rotation_class = int(label)
        rotations = [0, 90, 180, 270]
        rotated_image = image.rotate(rotations[rotation_class])

        if self.transform:
            rotated_image = self.transform(rotated_image)

        return rotated_image, torch.tensor(rotation_class, dtype=torch.long)
