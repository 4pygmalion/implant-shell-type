import os
import sys

ST_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(ST_DIR)
ROOT_DIR = os.path.dirname(EXP_DIR)
sys.path.insert(0, ROOT_DIR)
from core.transform import crop_ratio

import torch
import numpy as np
from PIL import Image


class STDataSet(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None, device: str = "cpu", **kwargs):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.device = device
        self.kwargs = kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert("RGB")
        crop_image_array, (_, _) = crop_ratio(np.array(image), is_st=True)
        crop_image_array = Image.fromarray(crop_image_array)

        if self.transform:
            return (
                self.transform(crop_image_array).to(self.device),
                torch.tensor([self.labels[idx]]).to(self.device).float(),
            )

        return (
            torch.tensor(np.array(crop_image_array)).to(self.device),
            torch.tensor([self.labels[idx]]).to(self.device).float(),
        )
