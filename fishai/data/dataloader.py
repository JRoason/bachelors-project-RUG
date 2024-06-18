from torch.utils.data import Dataset
import torch
import os
from data.window_function import f_window_gen


class FishDataset(Dataset):
    """
    Dataset class for the fish data.
    The dataset class is used to load the fish data and apply the window function to the data.

    ...

    Attributes
    ----------
    data : torch.Tensor
        The fish data tensor, containing the fish data, salinity and temperature data.
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

    def __init__(self, mode: str, input_width: int, output_width: int, offset_width: int,
                 matrix_structure: str) -> None:
        """
        Initializes the dataset class for the fish data.

        :param mode: The mode of the dataset.
        :param input_width: The width of the input window.
        :param output_width: The width of the target window.
        :param offset_width: The offset between the input and target windows.
        """
        super().__init__()
        assert mode in ['train', 'val', 'test'], 'Invalid mode'
        assert matrix_structure in ['diagonal', 'quadrant', 'regular'], 'Invalid matrix structure'
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data', mode))

        cod_data = torch.load(os.path.join(data_dir, 'cod_' + mode + '_normalized.pt'))
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

        self.matrix_structure = matrix_structure
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

        # return self.data_time[idx], self.target_time[idx]

        inputTensor = None
        targetTensor = None

        if self.matrix_structure == 'diagonal':
            cod_input = torch.block_diag(self.data_time[idx][0][0], self.data_time[idx][1][0],
                                         self.data_time[idx][2][0],
                                         self.data_time[idx][3][0])
            cod_input = torch.reshape(cod_input, (1, cod_input.shape[0], cod_input.shape[1]))
            salinity_input = torch.block_diag(self.data_time[idx][0][1], self.data_time[idx][1][1],
                                              self.data_time[idx][2][1], self.data_time[idx][3][1])
            salinity_input = torch.reshape(salinity_input, (1, salinity_input.shape[0], salinity_input.shape[1]))
            temperature_input = torch.block_diag(self.data_time[idx][0][2], self.data_time[idx][1][2],
                                                 self.data_time[idx][2][2], self.data_time[idx][3][2])
            temperature_input = torch.reshape(temperature_input,
                                              (1, temperature_input.shape[0], temperature_input.shape[1]))
            inputTensor = torch.cat((cod_input, salinity_input, temperature_input), dim=0)

            targetTensor = torch.block_diag(self.target_time[idx][0], self.target_time[idx][1],
                                            self.target_time[idx][2], self.target_time[idx][3])
            targetTensor = torch.reshape(targetTensor, (1, targetTensor.shape[0], targetTensor.shape[1]))
        elif self.matrix_structure == 'quadrant':
            inputTensor = torch.zeros((3, (self.input_width // 2) * self.data.shape[2], (self.input_width // 2) * self.data.shape[2]))
            targetTensor = torch.zeros(((self.output_width // 2) * self.data.shape[2], (self.output_width // 2) * self.data.shape[2]))
            for i in range(self.input_width):
                j = 0 if i < 2 else 1
                inputTensor[0, i % 2 * self.data.shape[2]:(i % 2 + 1) * self.data.shape[2],
                j * self.data.shape[2]:(j + 1) * self.data.shape[2]] = self.data_time[idx][i][0]
                inputTensor[1, i % 2 * self.data.shape[2]:(i % 2 + 1) * self.data.shape[2],
                j * self.data.shape[2]:(j + 1) * self.data.shape[2]] = self.data_time[idx][i][1]
                inputTensor[2, i % 2 * self.data.shape[2]:(i % 2 + 1) * self.data.shape[2],
                j * self.data.shape[2]:(j + 1) * self.data.shape[2]] = self.data_time[idx][i][2]
            for i in range(self.output_width):
                j = 0 if i == 0 or i == 3 else 1
                targetTensor[i % 2 * self.data.shape[2]:(i % 2 + 1) * self.data.shape[2],
                j * self.data.shape[2]:(j + 1) * self.data.shape[2]] = self.target_time[idx][i]
            targetTensor = torch.reshape(targetTensor, (1, targetTensor.shape[0], targetTensor.shape[1]))
        elif self.matrix_structure == 'regular':
            inputTensor = torch.Tensor(self.data_time[idx])
            targetTensor = torch.Tensor(self.target_time[idx])

        assert inputTensor is not None, 'Input tensor is None'
        assert targetTensor is not None, 'Target tensor is None'

        return inputTensor, targetTensor
