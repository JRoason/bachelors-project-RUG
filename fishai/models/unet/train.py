import torch
import torch.optim as optim
import sys
import os

from torch.utils.data import DataLoader
from torch.nn import Module

from unet import UNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print(sys.path)

from data.dataloader import FishDataset
from utils.utils import print_train_progress, print_val_progress, print_final


def train(loader: DataLoader, model: Module, criterion: Module, optimizer: optim.Optimizer, device: str, epoch: int,
          epochs: int) -> list:
    """
    Train the UNet model on the fish data.

    :param loader: The data loader for the fish data.
    :param model: The UNet model.
    :param criterion: The loss function.
    :param optimizer: The optimizer.
    :param device: The device to train the model on.
    :param epoch: The current epoch.
    :param epochs: The total number of epochs.
    :return: The losses of the model on the training data.
    """

    losses = []

    model.train()
    for k, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        print_train_progress(epoch, epochs, k, len(loader), loss.item())

    return losses


def validate(loader: DataLoader, model: Module, criterion: Module, device: str, epoch: int, epochs: int) -> list:
    """
    Validate the UNet model on the fish data.

    :param loader: The data loader for the fish data.
    :param model: The UNet model.
    :param criterion: The loss function.
    :param device: The device to validate the model on.
    :param epoch: The current epoch.
    :param epochs: The total number of epochs.
    :return: The losses of the model on the validation data.
    """

    losses = []

    model.eval()
    with torch.no_grad():
        for k, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)
            losses.append(loss.item())

            print_val_progress(epoch, epochs, k, len(loader), loss.item())

    return losses


def test(loader: DataLoader, model: Module, criterion: Module, device: str) -> list:
    """
    Test the UNet model on the fish data.

    :param loader: The data loader for the fish data.
    :param model: The UNet model.
    :param criterion: The loss function.
    :param device: The device to test the model on.
    :return: The losses of the model on the test data.
    """

    losses = []

    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)
            losses.append(loss.item())

    return losses


def train_model(name: str, epochs: int, batch_size: int, learning_rate: float, input_width: int,
                output_width: int, offset_width: int, matrix_structure: str, device: str) -> None:
    """
    Train the UNet model on the fish data.

    :param name: The name of the model.
    :param epochs: The total number of epochs.
    :param batch_size: The batch size.
    :param learning_rate: The learning rate.
    :param input_width: The width of the input window.
    :param output_width: The width of the target window.
    :param offset_width: The offset between the input and target windows.
    :param matrix_structure: The structure of the matrix.
    :param device: The device to train the model on.
    """

    model = UNet(3, 4, 64).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = FishDataset('train', input_width, output_width, offset_width, matrix_structure)
    val_dataset = FishDataset('val', input_width, output_width, offset_width, matrix_structure)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_losses += train(train_loader, model, criterion, optimizer, device, epoch, epochs)
        val_losses += validate(val_loader, model, criterion, device, epoch, epochs)

    # test_dataset = FishDataset('test', input_width, output_width, offset_width, matrix_structure)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #
    # test_losses = test(test_loader, model, criterion, device)
    #
    # print_final(epochs, epochs, sum(train_losses) / len(train_losses), sum(val_losses) / len(val_losses))
    #
    # torch.save(model.state_dict(), f'fishai/models/unet/{name}.pt')
    # torch.save(train_losses, f'fishai/models/unet/{name}_train_losses.pt')
    # torch.save(val_losses, f'fishai/models/unet/{name}_val_losses.pt')
    # torch.save(test_losses, f'fishai/models/unet/{name}_test_losses.pt


if __name__ == '__main__':
    train_model('unet', 1, 32, 0.001, 4, 4, 0, 'diagonal', 'cuda')
