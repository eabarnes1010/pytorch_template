"""Data loader modules.

Classes
---------
CustomData(torch.utils.data.Dataset)

"""

from torch.utils.data import Dataset
import torch
import numpy as np
import pickle


class CustomData(torch.utils.data.Dataset):
    """
    Custom dataset for data in dictionaries.
    """

    def __init__(self, data_file):
        with open(data_file, "rb") as handle:
            dict_data = pickle.load(handle)

        self.input = np.moveaxis(dict_data["x"], -1, 1)
        self.input_unit = dict_data["emissions_left"]
        self.target = dict_data["y"]

        # self.input = np.moveaxis(dict_data["x"], -1, 1)[:100]
        # self.input_unit = dict_data["emissions_left"][:100]
        # self.target = dict_data["y"][:100]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        input = self.input[idx, ...]
        input_unit = self.input_unit[idx]

        target = self.target[idx]

        return (
            [
                torch.tensor(input, dtype=torch.float32),
                torch.tensor(input_unit, dtype=torch.float32),
            ],
            torch.tensor(target, dtype=torch.float32),
        )
