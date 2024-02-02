from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from abc import abstractmethod
import numpy as np


# class BaseDataset(Dataset):
#     """
#     Base class for all data loaders.
#     """

#     def __init__(self, data):
#         self.data = data[0]
#         self.target = data[1]

#     def __len__(self):
#         return len(self.target)

#     def __getitem__(self, idx):
#         input = self.data[idx, ...]
#         target = self.target[idx]

#         return input, target

# @abstractmethod
# def __getitem__(self, idx):
#     """
#     Get item in data.

#     """
#     raise NotImplementedError
