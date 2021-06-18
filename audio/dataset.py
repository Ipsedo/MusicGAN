import torch as th
from torch.utils.data import Dataset

from typing import Tuple

from os import listdir
from os.path import isdir, isfile, join

from tqdm import tqdm


class AudioDataset(Dataset):

    def __init__(self, dataset_path: str) -> None:
        super().__init__()

        assert isdir(dataset_path)

        all_magn = [
            f for f in tqdm(listdir(dataset_path))
            if isfile(join(dataset_path, f)) and
               f.startswith("magn")
        ]

        all_phase = [
            f for f in tqdm(listdir(dataset_path))
            if isfile(join(dataset_path, f)) and
               f.startswith("phase")
        ]

        assert len(all_magn) == len(all_phase)

        self.__all_magn = sorted(all_magn)
        self.__all_phase = sorted(all_phase)

        self.__dataset_path = dataset_path

    def __getitem__(self, index: int):
        magn = th.load(join(
            self.__dataset_path,
            self.__all_magn[index]
        ))

        phase = th.load(join(
            self.__dataset_path,
            self.__all_phase[index]
        ))

        return th.stack([magn, phase], dim=0)

    def __len__(self):
        return len(self.__all_magn)
