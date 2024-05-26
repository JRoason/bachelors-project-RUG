import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    Encoder block for the UNet architecture.
    The encoder block consists of a max pooling layer, followed by two convolutional layers.

    ...

    Attributes
    ----------
    max_pool : nn.MaxPool2d
        Max pooling layer to downsample the input tensor.
    conv_layer_1 : nn.Conv2d
        Convolutional layer for the first convolution operation.
    relu : nn.ReLU
        Rectified linear unit activation function.
    conv_layer_2 : nn.Conv2d
        Convolutional layer for the second convolution operation.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initializes an encoder block.

        :param in_channels: The input channels of the input tensor.
        :param out_channels: The output channels of the output of encoder block.
        """
        super(EncoderBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.conv_layer_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder block. The input tensor is passed through a max pooling layer, followed by two
        convolutional layers.

        :param x: Output tensor from the previous encoder block or the input tensor.
        :return: The output tensor of the encoder block.
        """
        x = self.max_pool(x)
        x = self.conv_layer_1(x)
        x = self.relu(x)
        x = self.conv_layer_2(x)
        x = self.relu(x)
        return x
