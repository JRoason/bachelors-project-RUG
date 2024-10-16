import torch
import torch.nn as nn

from encoder import Encoder


class ConvNet(nn.Module):
    """
    Regular convolutional neural network model for the prediction of fish data.

    ...

    Attributes
    ----------
    encoder : Encoder
        Encoder for the ConvNet model.
    """

    def __init__(self, in_channels: int, num_blocks: int, features: int, dropout: bool) -> None:
        """
        Initializes a ConvNet model.
        :param in_channels: The number of input channels of the input data.
        :param num_blocks: The number of encoder blocks in the model.
        :param features: Number of channels in the outputs/inputs of the encoder blocks.
        :param dropout: Whether to use dropout in the model.
        """
        super(ConvNet, self).__init__()
        self.encoder = Encoder(in_channels, num_blocks, features, dropout)
        self.final_conv = nn.Conv2d(features * 2 ** num_blocks, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvNet model.
        :param x: The input to the model, 10 days of fish data.
        :return: The output of the model, 5 subsequent days of fish data.
        """
        _, x = self.encoder(x)
        x = self.final_conv(x)
        return x


if __name__ == '__main__':
    # Example usage just to check the shapes of the output
    data = torch.rand(1, 3, 256, 256)
    # In this case, the data is a tensor of shape (1, 30, 134, 410), representing a batch of 1 image with 30 channels, and a resolution of 134x410 pixels.
    # The 30 channels refer to 3 x 10 channels, where the 3 channels refer to the fish map matrix, salinity and temperature data.
    # While the 10 refers to the 10 time steps of the data.
    print(data.shape)
    model = ConvNet(3, 1, 64, True)
    output = model(data)
    print(output.shape)