import torch
import torch.nn as nn
from encoder_block import EncoderBlock


class Encoder(nn.Module):
    """
    Encoder for the UNet architecture.
    The encoder consists of two convolutional layers followed by a series of encoder blocks.

    ...

    Attributes
    ----------
    conv_layer_1 : nn.Conv2d
        Convolutional layer for the first convolution operation.
    relu : nn.ReLU
        Rectified linear unit activation function.
    conv_layer_2 : nn.Conv2d
        Convolutional layer for the second convolution operation.
    blocks : nn.ModuleList
        List of encoder blocks.
    """

    def __init__(self, in_channels: int, num_blocks: int, features: int):
        """
        Initializes an encoder for the UNet architecture.

        :param in_channels: The number of input channels of the input data.
        :param num_blocks: The wanted number of encoder blocks.
        :param features: Number of channels in input to the first encoder block, multiplied by 2 for subsequent blocks.
        """
        super(Encoder, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels, features, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.conv_layer_2 = nn.Conv2d(features, features, kernel_size=3, padding='same')
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(EncoderBlock(features * 2 ** i, features * 2 ** (i + 1)))

    def forward(self, x: torch.Tensor) -> (list, torch.Tensor):
        """
        Forward pass through the encoder. The input tensor is passed through two convolutional layers, followed by each
        encoder block. The output tensor from each encoder block is stored in a list and returned, along with the final
        output tensor.

        :param x: The input to the model.
        :return: The output of each encoder block and the final output tensor.
        """
        intermediates = []
        x = self.conv_layer_1(x)
        x = self.relu(x)
        x = self.conv_layer_2(x)
        x = self.relu(x)
        for block in self.blocks:
            intermediates.append(x)
            x = block(x)
        return intermediates, x
