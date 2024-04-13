import torch
import numpy as np
from torch.utils.data import Dataset
from ..config import Config as cfg

class teleDataset(Dataset):

    def __init__(self, file):
        """
        Args:
            filepath (str): The file path of the binary data file.
        """
        # Load the data using memory mapping for efficient access
        self.data = np.memmap(file, dtype=np.uint16, mode='r')

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data) - cfg.data.block_size

    def __getitem__(self, index):
        """
        Returns a sample from the dataset at the specified index.
        Args:
            index (int): The index of the sample to return.
        """
        # Extract the input sequence x from the data
        x = torch.from_numpy((self.data[index:index+cfg.data.block_size]).astype(np.int64))
        # Extract the target sequence y from the data (shifted one step to the right)
        y = torch.from_numpy((self.data[index+1:index+1+cfg.data.block_size]).astype(np.int64))
        return x, y
