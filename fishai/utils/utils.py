import sys


def print_train_progress(epoch: int, epochs: int, batch: int, batches: int, loss: float) -> None:
    """
    Print the training progress of the model.

    :param epoch: The current epoch.
    :param epochs: The total number of epochs.
    :param batch: The current batch.
    :param batches: The total number of batches.
    :param loss: The current loss.
    """
    t1 = int(float(batch / batches) * 16)
    t2 = 16 - t1 - 1
    t_bar = t1 * '=' + '>' + ' ' * t2
    v_bar = 8 * ' '
    text = "\33[2K\r{:3d}/{}: |{}|[}| - Loss: {:.4f}    ".format(epoch, epochs, t_bar, v_bar, loss)
    sys.stdout.write(text)
    sys.stdout.flush()


def print_val_progress(epoch: int, epochs: int, batch: int, batches: int, loss: float) -> None:
    """
    Print the training progress of the model.

    :param epoch: The current epoch.
    :param epochs: The total number of epochs.
    :param batch: The current batch.
    :param batches: The total number of batches.
    :param loss: The current loss.
    """
    t1 = int(float(batch / batches) * 8)
    t2 = 8 - t1 - 1
    t_bar = t1 * '=' + '>' + ' ' * t2
    v_bar = 16 * ' '
    text = "\33[2K\r{:3d}/{}: |{}|[}| - Loss: {:.4f}    ".format(epoch, epochs, t_bar, v_bar, loss)
    sys.stdout.write(text)
    sys.stdout.flush()


def print_final(epoch: int, epochs: int, mean_train_loss: float, mean_val_loss: float) -> None:
    """
    Print the final training progress of the model.

    :param epoch: The current epoch.
    :param epochs: The total number of epochs.
    :param mean_train_loss: The mean training loss.
    :param mean_val_loss: The mean validation loss.
    """
    text = "\33[2K\r{:3d}/{}: |{}|{}| - Loss: {:.4f} - {:.4f}    ".format(epoch, epochs, 16 * "=",
                                                                          8 * "=", mean_train_loss,
                                                                          mean_val_loss)
    sys.stdout.write(text)
    sys.stdout.flush()

