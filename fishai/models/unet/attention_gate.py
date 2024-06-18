import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention gate for the UNet architecture.
    The attention gate is used to scale the encoder output before it is concatenated with the decoder output.
    ...

    Attributes
    ----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    """

    def __init__(self, in_channels_down: int, in_channels_up: int) -> None:
        """
        Initializes an attention gate for the UNet architecture.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        """
        super(AttentionGate, self).__init__()
        self.w_g = nn.Conv2d(in_channels_up, in_channels_up, kernel_size=1, stride=1)
        self.w_x = nn.Conv2d(in_channels_down, in_channels_up, kernel_size=2, stride=2)

        self.relu = nn.ReLU()

        self.psi = nn.Conv2d(in_channels_up, 1, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, skip_tensor: torch.Tensor, up_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention gate. The encoder and decoder outputs are passed through their own
        convolutional layers such that their shapes match. Both outputs are then added together and passed through a
        ReLU activation function, before being passed through another convolutional layer. The output tensor is then
        passed through a sigmoid activation function and upsampled to match the shape of the skip connection tensor.
        Finally the output tensor is multiplied with the skip connection tensor.

        :param skip_tensor: The output tensor from the skip connection.
        :param up_tensor: The output tensor from the previous decoder block.
        :return: The final output tensor of the attention gate.
        """

        x = self.w_x(skip_tensor)

        g = self.w_g(up_tensor)

        x = self.relu(x + g)

        x = self.psi(x)

        x = self.sigmoid(x)

        x = self.upsample(x)

        return x * skip_tensor


if __name__ == '__main__':
    encoder_tensor = torch.rand(1, 512, 64, 64)  # Skip connection tensor
    up_conv_tensor = torch.rand(1, 1024, 32, 32)  # Tensor from the previous decoder block

    attention_gate = AttentionGate(512, 1024)

    output = attention_gate(encoder_tensor, up_conv_tensor)

    print(output.shape)
