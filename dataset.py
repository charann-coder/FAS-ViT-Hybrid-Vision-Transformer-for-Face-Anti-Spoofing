import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class LCCFASDFromCSV(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, multitask=False):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.multitask = multitask
        
        self.label_map = {"real": 0, "spoof": 1}
        self.has_spoof_type = 'spoof_type' in self.data.columns
        if self.has_spoof_type:
            
            unique_spoof_types = self.data[self.data['spoof_type'].notna()]['spoof_type'].unique()
            self.spoof_type_map = {name: i for i, name in enumerate(sorted(unique_spoof_types))}
        else:
            self.spoof_type_map = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        label_str = row["label"].strip().lower()
        if label_str not in self.label_map:
            raise ValueError(f"Unknown label '{label_str}' at index {idx}")
        label = self.label_map[label_str]
        
        spoof_type = -1 # Default value
        if self.multitask and label == 1 and self.has_spoof_type:
            spoof_type_str = row.get("spoof_type", "").strip().lower()
            if spoof_type_str in self.spoof_type_map:
                spoof_type = self.spoof_type_map[spoof_type_str]

        if self.transform:
            image = self.transform(image)
        
        
        if self.multitask:
            targets = (label, spoof_type)
            return image, targets, row["image_path"]
        else:
            return image, label, row["image_path"]