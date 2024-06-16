import torch
import torch.nn as nn

from decoder_block import DecoderBlock


class Decoder(nn.Module):
    """
    Decoder for the UNet architecture.
    The decoder consists of a series of decoder blocks, followed by a final convolutional layer.

    ...

    Attributes
    ----------
    blocks : nn.ModuleList
        List of decoder blocks.
    final_conv : nn.Conv2d
        Convolutional layer for the final convolution operation.
    """

    def __init__(self, num_blocks: int, features: int, dropout: bool) -> None:
        """
        Initializes a decoder for the UNet architecture.

        :param num_blocks: The number of decoder blocks in the decoder.
        :param features: The number of channels in the input to the first encoder block.
        :param dropout: Whether to use dropout in the model.
        """
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                DecoderBlock(features * 2 ** (num_blocks - i), features * 2 ** (num_blocks - i - 1), dropout=dropout))
        self.final_conv = nn.Conv2d(features, 1, kernel_size=1)

    def forward(self, intermediates: list, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder. The input tensor is passed through each decoder block, with the output tensor
        from the corresponding encoder block concatenated to the input tensor of each decoder block. The final output
        tensor is passed through a final convolutional layer to produce the final output tensor.

        :param intermediates: List of output tensors from the encoder blocks.
        :param x: Input tensor from the last encoder block or the previous decoder block.
        :return: The final output tensor of the decoder, containing 5 channels representing 5 days of fish data.
        """
        for i, block in enumerate(self.blocks):
            x = block(x, intermediates[-i - 1])
        return self.final_conv(x)
