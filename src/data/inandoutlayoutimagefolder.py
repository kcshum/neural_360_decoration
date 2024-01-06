from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, Dict

from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from data.nodata import NullIterableDataset
from data.utils import ImageFolderWithFilenames
from utils import print_once

_all__ = ['InandoutLayoutImageFolderDataModule']


@dataclass
class InandoutLayoutImageFolderDataModule(LightningDataModule):

    path: Union[str, Path]  # Root
    bgpath: Union[str, Path]
    edgepath: Union[str, Path]
    dataloader: Dict[str, Any]
    resolution: int = 256  # Image dimension

    def __post_init__(self):
        super().__init__()
        self.path = Path(self.path)
        self.bgpath = Path(self.bgpath)
        self.edgepath = Path(self.edgepath)

        self.stats = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
        self.transform = transforms.Compose([
            t for t in [
                transforms.Resize([self.resolution, self.resolution], InterpolationMode.LANCZOS),
                transforms.CenterCrop(self.resolution),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.stats['mean'], self.stats['std'], inplace=True),
            ]
        ])
        self.data = {}
        self.bgdata = {}
        self.edgedata = {}

    def setup(self, stage: Optional[str] = None):
        for split in ('train', 'validate', 'test'):
            path = self.path / split
            empty = True
            if path.exists():
                try:
                    print('trying to make dataset...')
                    self.data[split] = ImageFolderWithFilenames(path, transform=self.transform)
                    empty = False
                except FileNotFoundError:
                    print('FileNotFoundError')
                    pass
            if empty:
                print_once(
                    f'Warning: no images found in {path}. Using empty dataset for split {split}. '
                    f'Perhaps you set `dataset.path` incorrectly?')
                self.data[split] = NullIterableDataset(1)

        for split in ('train', 'validate', 'test'):
            bgpath = self.bgpath / split
            empty = True
            if bgpath.exists():
                try:
                    print('trying to make bg dataset...')
                    self.bgdata[split] = ImageFolderWithFilenames(bgpath, transform=self.transform)
                    empty = False
                except FileNotFoundError:
                    print('FileNotFoundError')
                    pass
            if empty:
                print_once(
                    f'Warning: no images found in {bgpath}. Using empty dataset for split {split}. '
                    f'Perhaps you set `dataset.path` incorrectly?')
                self.bgdata[split] = NullIterableDataset(1)

        for split in ('train', 'validate', 'test'):
            edgepath = self.edgepath / split
            empty = True
            if edgepath.exists():
                try:
                    print('trying to make bg dataset...')
                    self.edgedata[split] = ImageFolderWithFilenames(edgepath, transform=self.transform)
                    empty = False
                except FileNotFoundError:
                    print('FileNotFoundError')
                    pass
            if empty:
                print_once(
                    f'Warning: no images found in {edgepath}. Using empty dataset for split {split}. '
                    f'Perhaps you set `dataset.path` incorrectly?')
                self.edgedata[split] = NullIterableDataset(1)

    def train_dataloader(self):
        return self._get_dataloader('train')

    def val_dataloader(self):
        temp = self._get_dataloader('validate')
        loaders = {"bg": temp[0], "gt": temp[1], "edge": temp[2]}
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loaders

    def test_dataloader(self):
        temp = self._get_dataloader('test')
        loaders = {"bg": temp[0], "gt": temp[1], "edge": temp[2]}
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loaders

    def _get_dataloader(self, split: str):
        return [DataLoader(self.bgdata[split], **self.dataloader), DataLoader(self.data[split], **self.dataloader), DataLoader(self.edgedata[split], **self.dataloader)]
