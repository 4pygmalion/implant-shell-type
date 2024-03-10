import os
import sys
import random
from typing import List

import torch
import numpy as np
from torchvision.transforms import ElasticTransform
from PIL import Image

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CORE_DIR)
sys.path.append(ROOT_DIR)

from core.transform import crop_usimage


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: List[str],
        size: tuple = (0, 0),
        transform=None,
        require_crop=True,
        label_hint: str = "rupture",
        device="cuda",
    ):
        """이미지 데이터 로더

        Args:
            image_paths (List[str]): 이미지 데이터 경로
        """
        self.image_paths = image_paths
        self.size = size
        self.transform = transform
        self.require_crop = require_crop
        self.label_hint = label_hint
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def preprocess(crop_image: np.ndarray, transform=None) -> torch.Tensor:
        crop_image = np.expand_dims(crop_image, axis=0)
        crop_image = np.repeat(crop_image, 3, axis=0)

        if transform is not None:
            image_wh3 = np.transpose(crop_image, (1, 2, 0))
            image = Image.fromarray(image_wh3)
            image: torch.Tensor = transform(image)

            return image

        return torch.from_numpy(crop_image)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert("L"))

        if self.require_crop:
            input_array, (x_min, y_min) = crop_usimage(image, size=self.size)

        else:
            input_array = image

        input_tensor = self.preprocess(input_array, transform=self.transform)

        if self.label_hint in image_path:
            label = np.array([1])
        else:
            label = np.array([0])

        label = label.astype(np.float32)
        label_tensor = torch.from_numpy(label)

        return (
            input_tensor.to(self.device).float(),
            label_tensor.to(self.device).float(),
        )


class ElasticDistDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: List[str],
        transform=None,
        label_hint: str = "rupture",
        crop: callable = None,
        distortion=ElasticTransform(alpha=100.0),
        device="cuda",
    ):
        """이미지 데이터 로더

        Args:
            image_paths (List[str]): 이미지 데이터 경로
        """
        self.image_paths = image_paths
        self.transform = transform
        self.label_hint = label_hint
        self.crop = crop
        self.device = device
        self.distortion = distortion

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def preprocess(crop_image: np.ndarray, transform=None) -> torch.Tensor:
        crop_image = np.expand_dims(crop_image, axis=0)
        crop_image = np.repeat(crop_image, 3, axis=0)

        if transform is not None:
            image_wh3 = np.transpose(crop_image, (1, 2, 0))
            image = Image.fromarray(image_wh3)
            image: torch.Tensor = transform(image)

            return image

        return torch.from_numpy(crop_image)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if self.label_hint in image_path:
            label = np.array([1])
        else:
            label = np.array([0])

        image = np.array(Image.open(image_path).convert("L"))

        if self.crop:
            input_array = self.crop(image)
            if isinstance(input_array, tuple):
                input_array, (_, _) = input_array
        else:
            input_array = image

        input_tensor = self.preprocess(input_array, transform=self.transform)

        if random.randint(0, 1) and self.distortion is not None:
            input_tensor = self.distortion(input_tensor)

        label = label.astype(np.float32)
        label_tensor = torch.from_numpy(label)

        return (
            input_tensor.to(self.device).float(),
            label_tensor.to(self.device).float(),
        )
