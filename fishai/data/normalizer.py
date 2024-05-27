import torch
import pickle

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

    file = open('minmax_values_' + name, 'wb')

    pickle.dump([min_value, max_value], file)

    file.close()

    # Normalize tensor

    normalized_tensor = (x - min_value) / (max_value - min_value)

    # Set NaNs to 0
    normalized_tensor[nan_mask] = 0.0

    return normalized_tensor

def normalize_val(x: torch.Tensor, name: str) -> torch.Tensor:

    if name not in ['sst', 'salinity', 'cod_data']:
        return
        
    nan_mask = torch.isnan(x)
        
    file = open('fishai/data/minmax_values_' + name, 'rb')

    values = pickle.load(file)

    file.close()

    normalized_tensor = (x - values[0]) / (values[1] - values[0])
    
    normalized_tensor[nan_mask] = 0.0
    
    return normalized_tensor
    
def unnormalize(x: torch.Tensor) -> torch.Tensor:

    file = open('minmax_values_cod_data', 'rb')

    values = pickle.load(file)

    file.close()
    
    un_normalized_tensor = (x * (values[1] - values[0])) + values[0]
    
    return un_normalized_tensor
    

      
    
