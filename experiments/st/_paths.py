from pathlib import Path
from dataclasses import dataclass


@dataclass
class Canon:
    data_root_dir: Path
    s: str = Path("st_paper_data/canon_s_no_rupture")
    t: str = Path("st_paper_data/canon_t_no_rupture")

    def __post_init__(self):
        self.s = self.data_root_dir / self.s
        self.t = self.data_root_dir / self.t


@dataclass
class Online:
    data_root_dir: Path
    s: str = Path("online/smooth")
    t: str = Path("online/texture")

    def __post_init__(self):
        self.s = self.data_root_dir / self.s
        self.t = self.data_root_dir / self.t


@dataclass
class CanonRupture:
    data_root_dir: Path
    s: str = Path("st_paper_data/canon_s_rupture")
    t: str = Path("st_paper_data/canon_t_rupture")

    def __post_init__(self):
        self.s = self.data_root_dir / self.s
        self.t = self.data_root_dir / self.t


@dataclass
class GE:
    data_root_dir: Path
    s: str = Path("st_paper_data/ge_s_no_rupture")
    t: str = Path("st_paper_data/ge_t_no_rupture")

    def __post_init__(self):
        self.s = self.data_root_dir / self.s
        self.t = self.data_root_dir / self.t


@dataclass
class DataPaths:
    data_root_dir: str

    def __post_init__(self):
        self.canon: Canon = Canon(self.data_root_dir)
        self.canon_rupture: CanonRupture = CanonRupture(self.data_root_dir)
        self.ge: GE = GE(self.data_root_dir)
        self.online: Online = Online(self.data_root_dir)
