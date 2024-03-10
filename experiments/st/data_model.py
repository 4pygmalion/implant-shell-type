from __future__ import annotations

from typing import List, Iterable, Union
from dataclasses import dataclass, field


@dataclass
class STImage:
    image_path: str = None
    label: int = None

    def __post_init__(self):

        return


@dataclass
class STImages:
    s_image_paths: List[str] = field(default_factory=list)
    t_image_paths: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.image_paths and self.labels:
            return
        self.image_paths = self.s_image_paths + self.t_image_paths
        self.labels = [0 for _ in self.s_image_paths] + [1 for _ in self.t_image_paths]

    def __getitem__(self, index) -> Union[STImages, STImage]:
        if isinstance(index, slice):
            return STImages(self.s_image_paths[index], self.t_image_paths[index])
        if isinstance(index, Iterable):
            image_paths = [self.image_paths[i] for i in index]
            label = [self.labels[i] for i in index]
            return STImages(image_paths=image_paths, labels=label)

        return self.image_paths[index], self.labels[index]

    def __len__(self):
        return len(self.image_paths)

    @property
    def s_image_count(self):
        return len([label for label in self.labels if label == 0])

    @property
    def t_image_count(self):
        return len([label for label in self.labels if label == 1])

    @property
    def n_images(self):
        return len(self.image_paths)
