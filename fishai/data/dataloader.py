from torch.utils.data import Dataset
import torch
import os
from .window_function import f_window_gen


class FishDataset(Dataset):
    """
    Dataset class for the fish data.
    The dataset class is used to load the fish data and apply the window function to the data.

    ...

    Attributes
    ----------
    data : torch.Tensor
        The fish data tensor.
    input_width : int
        The width of the input window.
    output_width : int
        The width of the target window.
    offset_width : int
        The offset between the input and target windows.
    data_time : list
        List of input windows.
    target_time : list
        List of target windows.
    """

    def __init__(self, mode: str, input_width: int, output_width: int, offset_width: int) -> None:
        """
        Initializes the dataset class for the fish data.

        :param input_width: The width of the input window.
        :param output_width: The width of the target window.
        :param offset_width: The offset between the input and target windows.
        """
        super().__init__()
        assert mode in ['train', 'val', 'test'], 'Invalid mode'
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data', mode))

        cod_data = torch.load(os.path.join(data_dir, 'cod_data_' + mode + '_normalized.pt'))
        cod_data = torch.reshape(cod_data, (cod_data.shape[0], 1, cod_data.shape[1], cod_data.shape[2]))
        salinity_data = torch.load(os.path.join(data_dir, 'salinity_' + mode + '_normalized.pt'))
        salinity_data = torch.reshape(salinity_data,
                                      (salinity_data.shape[0], 1, salinity_data.shape[1], salinity_data.shape[2]))
        temperature_data = torch.load(os.path.join(data_dir, 'sst_' + mode + '_normalized.pt'))
        temperature_data = torch.reshape(temperature_data,
                                         (temperature_data.shape[0], 1, temperature_data.shape[1],
                                          temperature_data.shape[2]))
        data = torch.cat((cod_data, salinity_data, temperature_data), dim=1)

        del cod_data, salinity_data, temperature_data

        self.data = data
        self.input_width = input_width
        self.output_width = output_width
        self.offset_width = offset_width
        self.data_time, self.target_time = f_window_gen(data, input_width, output_width, offset_width)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        :return: The length of the dataset.
        """
        return len(self.data_time)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        Returns the input and target windows at the given index.

        :param idx: The index of the input and target windows.
        :return: The input and target windows at the given index.
        """
        return self.data_time[idx], self.target_time[idx]
