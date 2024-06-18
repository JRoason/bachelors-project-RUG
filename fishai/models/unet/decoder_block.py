import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
from attention_gate import AttentionGate


class DecoderBlock(nn.Module):
    """
    Decoder block for the UNet architecture.
    The decoder block consists of a transposed convolutional layer to upsample the input tensor,
    followed by two convolutional layers.

    ...

    Attributes
    ----------
    upconv : nn.ConvTranspose2d
        Transposed convolutional layer to upsample the input tensor.
    conv_layer_1 : nn.Conv2d
        Convolutional layer for the first convolution operation.
    relu : nn.ReLU
        Rectified linear unit activation function.
    conv_layer_2 : nn.Conv2d
        Convolutional layer for the second convolution operation.

    Methods
    -------
    forward(x, down_tensor)
        Forward pass through the decoder block.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: bool, attention: bool) -> None:
        """
        Initializes a decoder block.

        Parameters
        :param in_channels: The number of input channels of the input tensor.
        :param out_channels: The number of output channels of the decoder block.
        :param dropout: Whether to use dropout in the model.
        """

        super(DecoderBlock, self).__init__()
        self.attention = AttentionGate(out_channels, in_channels) if attention else None
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5) if dropout else None
        self.conv_layer_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.conv_layer_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')


    def forward(self, x: torch.Tensor, down_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder block. The input tensor is upsampled using the transposed convolutional layer,
        and concatenated with the output tensor from the 'corresponding' encoder block. The concatenated tensor is then
        passed through two convolutional layers.

        See diagram in the README, specifically the gray arrows, to better understand the concatenation process and the
        meaning of a 'corresponding' encoder block in the context of the decoder block.

        :param x: Input tensor, from either the last encoder block or the previous decoder block.
        :param down_tensor: The output tensor from the 'corresponding' encoder block.
        :return: The output tensor of the decoder block.
        """

        # Based on https://arxiv.org/pdf/1804.03999, we would have an attention mechanism here, that takes
        # the output tensor from the encoder block and the input tensor to the decoder block, and computes
        # the attention weights. The attention weights are then multiplied with the output tensor from the
        # encoder block to produce the attended tensor. This attended tensor is then concatenated with the
        # input tensor to the decoder block. The rest of the forward pass remains the same.

        if self.attention is not None:
            down_tensor = self.attention(down_tensor, x)

        x = self.upconv(x)
        down_tensor = CenterCrop(x.shape[2:])(down_tensor)
        x = torch.cat([x, down_tensor], dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv_layer_1(x)
        x = self.relu(x)
        x = self.conv_layer_2(x)
        x = self.relu(x)
        return x
