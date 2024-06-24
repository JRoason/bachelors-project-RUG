import torch
import torch.optim as optim
import tqdm
import sys
import os
import datetime
import json

from torch.utils.data import DataLoader
from torch.nn import Module

from unet import UNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.dataloader import FishDataset


def train(loader: DataLoader, model: Module, criterion: Module, optimizer: optim.Optimizer, device: str,
          bar: tqdm.tqdm) -> list:
    """
    Train the UNet model on the fish data.

    :param bar: The batch progress bar.
    :param loader: The data loader for the fish data.
    :param model: The UNet model.
    :param criterion: The loss function.
    :param optimizer: The optimizer.
    :param device: The device to train the model on.
    :return: The losses of the model on the training data.
    """

    losses = []

    model.train()
    for k, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        if k == 0:
            print(target[0, 0, 26:31, 10:30])


        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        bar.update(1)

    return losses


def validate(loader: DataLoader, model: Module, criterion: Module, device: str, bar: tqdm.tqdm) -> list:
    """
    Validate the UNet model on the fish data.

    :param bar: Progress bar
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

            bar.update(1)

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


def train_model(name: str, epochs: int, batch_size: int, learning_rate: float, input_width: int, num_blocks: int,
                output_width: int, offset_width: int, matrix_structure: str, dropout: bool, segmentation: bool,
                early_stopping: bool, attention: bool, device: str, directory_path: str) -> None:
    """
    Train the UNet model on the fish data.

    :param name: The name of the model.
    :param epochs: The total number of epochs.
    :param batch_size: The batch size.
    :param learning_rate: The learning rate.
    :param num_blocks: The number of encoder/decoder blocks in the model.
    :param input_width: The width of the input window.
    :param output_width: The width of the target window.
    :param offset_width: The offset between the input and target windows.
    :param matrix_structure: The structure of the matrix.
    :param dropout: Whether to use dropout.
    :param segmentation: Whether to use 'segmentation maps' of the fish data.
    :param early_stopping: Whether to use early stopping.
    :param attention: Whether to use attention.
    :param device: The device to train the model on.
    :param directory_path: The path to the directory.
    """

    model = UNet(3, num_blocks, 64, dropout, attention).to(device)

    if segmentation:
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = FishDataset('train', input_width, output_width, offset_width, matrix_structure, segmentation)
    val_dataset = FishDataset('val', input_width, output_width, offset_width, matrix_structure, segmentation)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')

    early_stopping_count = None

    if early_stopping:
        early_stopping_count = 0

    train_losses = []
    val_losses = []

    training_status = tqdm.tqdm(total=0, position=0, bar_format='{desc}')
    progress_bar = tqdm.tqdm(total=(len(train_loader) + len(val_loader)) * epochs, position=1, desc='Progress')

    for epoch in range(epochs):
        if len(val_losses) > 0:
            training_status.set_description_str(f'Mean Validation Loss: {val_losses[-1]} - Epoch {epoch}/{epochs}')
        else:
            training_status.set_description_str(f'Training')
        train_losses.append(
            sum(train(train_loader, model, criterion, optimizer, device, progress_bar)) / len(train_loader))
        training_status.set_description_str(f'Mean Training Loss: {train_losses[-1]} - Epoch {epoch + 1}/{epochs}')
        val_loss = sum(validate(val_loader, model, criterion, device, progress_bar)) / len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_count = 0
            with open(f'{os.path.join(directory_path, name)}/{matrix_structure}_best_model.txt', 'w') as f:
                f.write(f'Epoch {epoch + 1}/{epochs}\n')
                f.write(f'Validation Loss: {best_val_loss}\n')
            torch.save(model.state_dict(), f'{os.path.join(directory_path, name)}/{matrix_structure}_best_model.pt')
        val_losses.append(val_loss)
        if early_stopping_count is not None and epoch > 1:
            if val_loss == val_losses[-2]:
                early_stopping_count += 1
            if early_stopping_count == 5:
                training_status.set_description_str(f'Early stopping at epoch {epoch + 1}/{epochs}')
                break
        if epoch == epochs - 1:
            training_status.set_description_str(f'Finished training at epoch {epoch + 1}/{epochs}')

    torch.save(model.state_dict(), f'{os.path.join(directory_path, name)}/{matrix_structure}_final_model.pt')
    torch.save(train_losses, f'{os.path.join(directory_path, name)}/{matrix_structure}_train_losses.pt')
    torch.save(val_losses, f'{os.path.join(directory_path, name)}/{matrix_structure}_val_losses.pt')

    training_status.set_description(f'Model saved')
    training_status.close()
    progress_bar.close()


if __name__ == '__main__':
    assert len(sys.argv) > 1, 'Please provide hyperparameters'
    assert len(sys.argv) == 13, 'Please provide all hyperparameters'
    hyperparameters = {
        'name': sys.argv[1],
        'epochs': int(sys.argv[2]),
        'batch_size': int(sys.argv[3]),
        'learning_rate': float(sys.argv[4]),
        'num_blocks': int(sys.argv[5]),
        'input_width': int(sys.argv[6]),
        'output_width': int(sys.argv[7]),
        'offset_width': int(sys.argv[8]),
        'dropout': sys.argv[9] == 'True',
        'early_stopping': sys.argv[10] == 'True',
        'attention': sys.argv[11] == 'True',
        'segmentation': sys.argv[12] == 'True',
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    }
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    hyperparameters['directory_path'] = dir_path
    if not os.path.exists(os.path.join(dir_path, hyperparameters['name'])):
        os.mkdir(os.path.join(dir_path, hyperparameters['name']))
    model_name = hyperparameters['name']
    with open(f'{os.path.join(dir_path, model_name)}/hyperparameters.txt', 'w') as f:
        json.dump(hyperparameters, f)
    print("Training model, matrix_structure='diagonal'")
    hyperparameters['matrix_structure'] = 'diagonal'
    train_model(**hyperparameters)
    print("Training model, matrix_structure='quadrant'")
    hyperparameters['matrix_structure'] = 'quadrant'
    train_model(**hyperparameters)
    with open(f'{os.path.join(dir_path, model_name)}/finished.txt', 'w') as f:
        f.write('Finished training model\n')
        f.write(f'Training started at {time}\n')
        f.write(f'Training finished at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
