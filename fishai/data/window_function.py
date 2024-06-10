import torch


def f_window_gen(data: torch.Tensor, input_width: int, output_width: int, offset_width: int):
    """
    Generate input and target windows from the data tensor.

    :param data: The input data tensor.
    :param input_width: The width of the input window.
    :param output_width: The width of the target window.
    :param offset_width: The offset between the input and target windows.
    :return: The input and target windows.
    """
    data_time = []
    target_time = []
    for i in range(data.shape[0] - (input_width + output_width + offset_width)):
        data_time.append(data[i:i + input_width])
        target_time.append(data[i + input_width + offset_width:i + input_width + output_width + offset_width, 0])
    return data_time, target_time
