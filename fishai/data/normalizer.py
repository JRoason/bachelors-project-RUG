import torch
import pickle
import os
import sys


def normalize_train(x: torch.Tensor, name: str) -> torch.Tensor:
    # Create a mask of NaN values
    nan_mask = torch.isnan(x)

    # Replace NaNs with very low value (to find the max value)
    x_no_nan_max = x.clone()
    x_no_nan_max[nan_mask] = float('-inf')

    max_value = torch.max(x_no_nan_max)

    del x_no_nan_max

    x_no_nan_min = x.clone()
    x_no_nan_min[nan_mask] = float('inf')

    min_value = torch.min(x_no_nan_min)

    del x_no_nan_min

    # Save the minimum and maximum values, for later un-normalizing the outputs

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data', 'train'))

    file = open(path + '/minmax_values_' + name, 'wb')

    pickle.dump([min_value, max_value], file)

    file.close()

    # Normalize tensor

    normalized_tensor = (x - min_value) / (max_value - min_value)

    # Set NaNs to 0
    normalized_tensor[nan_mask] = 0.0

    return normalized_tensor


def normalize_val(x: torch.Tensor, name: str) -> torch.Tensor:
    if name not in ['sst', 'salinity', 'cod']:
        return

    nan_mask = torch.isnan(x)

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data', 'train'))

    file = open(path + '/minmax_values_' + name, 'rb')

    values = pickle.load(file)

    file.close()

    normalized_tensor = (x - values[0]) / (values[1] - values[0])

    normalized_tensor[nan_mask] = 0.0

    return normalized_tensor


# This function unnormalizes the data using the min and max values saved during the normalization process
def unnormalize(x: torch.Tensor, name: str) -> torch.Tensor:

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data', 'train'))

    file = open(path + '/minmax_values_' + name, 'rb')

    values = pickle.load(file)

    file.close()

    un_normalized_tensor = (x * (values[1] - values[0])) + values[0]

    return un_normalized_tensor


if __name__ == '__main__':
    var = sys.argv[1]
    data_name = sys.argv[2]
    assert var in ['train', 'val', 'test'], 'Invalid mode'
    assert data_name in ['sst', 'salinity', 'cod'], 'Invalid name'

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data', var))

    data = torch.load(os.path.join(data_dir, data_name + '_' + var + '_data' + '.pt'))

    if var == 'train':
        data = normalize_train(data, data_name)
        torch.save(data, os.path.join(data_dir, data_name + '_' + var + '_normalized.pt'))
    else:
        data = normalize_val(data, data_name)
        torch.save(data, os.path.join(data_dir, data_name + '_' + var + '_normalized.pt'))
