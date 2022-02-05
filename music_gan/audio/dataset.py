import torch as th
from torch.utils.data import Dataset

from os import listdir
from os.path import isdir, isfile, join

from tqdm import tqdm


class AudioDataset(Dataset):

    def __init__(self, dataset_path: str) -> None:
        super().__init__()

        assert isdir(dataset_path)

        all_files = [
            f for f in tqdm(listdir(dataset_path))
            if isfile(join(dataset_path, f)) and
            f.startswith("magn_phase")
        ]

        self.__all_files = sorted(all_files)

        self.__dataset_path = dataset_path

    def __getitem__(self, index: int):
        magn_phase = th.load(join(
            self.__dataset_path,
            self.__all_files[index]
        ))

        return magn_phase

    def __len__(self):
        return len(self.__all_files)
