import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class UNet(nn.Module):
    """
    UNet model for the prediction of fish data.
    The UNet model consists of an encoder and a decoder.

    ...

    Attributes
    ----------
    encoder : Encoder
        Encoder for the UNet model.
    decoder : Decoder
        Decoder for the UNet model.
    """

    def __init__(self, in_channels: int, num_blocks: int, features: int) -> None:
        """
        Initializes a UNet model.
        :param in_channels: The number of input channels of the input data.
        :param num_blocks: The number of encoder/decoder blocks in the model.
        :param features: Number of channels in the outputs/inputs of the encoder/decoder blocks.
        """
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, num_blocks, features)
        self.decoder = Decoder(num_blocks, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet model.
        :param x: The input to the model, 10 days of fish data.
        :return: The output of the model, 5 subsequent days of fish data.
        """
        intermediates, x = self.encoder(x)
        return self.decoder(intermediates, x)


if __name__ == '__main__':
    # Example usage just to check the shapes of the output
    data = torch.rand(5, 3, 192, 512)
    # In this case, the data is a tensor of shape (1, 30, 134, 410), representing a batch of 1 image with 30 channels, and a resolution of 134x410 pixels.
    # The 30 channels refer to 3 x 10 channels, where the 3 channels refer to the fish map matrix, salinity and temperature data.
    # While the 10 refers to the 10 time steps of the data.
    print(data.shape)
    model = UNet(3, 3, 64)
    output = model(data)
    print(output.shape)
