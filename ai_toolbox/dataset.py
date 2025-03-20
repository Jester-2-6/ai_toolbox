import numpy as np
import tqdm
from collections import OrderedDict
# from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn as nn
import torchvision
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist(batch_size=64, new_transforms=[]):
    all_transforms = [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    new_transforms.extend(all_transforms)
    transform = transforms.Compose(all_transforms)

    # create train and test sets
    mnist_train = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create data loaders for the MNIST dataset
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar10(batch_size=64, new_transforms=[]):
    all_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    all_transforms.extend(new_transforms)
    transform = transforms.Compose(all_transforms)

    # create train and test sets
    cifar10_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Create data loaders for the MNIST dataset
    train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar100(batch_size=64, new_transforms=[], resize=32):
    all_transforms = [
        transforms.Resize(resize),  # ResNet18 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[2,2,2]),
    ]

    new_transforms.extend(all_transforms)
    transform = transforms.Compose(all_transforms)

    # create train and test sets
    cifar_100_train = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    cifar_100_test = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    # Create data loaders for the MNIST dataset
    train_loader = DataLoader(cifar_100_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(cifar_100_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(
    model,
    train_loader,
    test_loader,
    epochs=EPOCHS,
    parellel=True,
    path=None,
    criterion=nn.CrossEntropyLoss(),
):
    # Train the model
    learning_rate = 0.1  # LEARNING_RATE
    optimizer = SGD(model.parameters(), lr=learning_rate)

    if parellel:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")

    model.to(device)

    train_losses = []
    test_losses = []
    prev_acc = 0

    for epoch in range(epochs):
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print("Epoch: {} Loss: {:.6f}\r".format(epoch + 1, loss.item()), end="")

            epoch_loss += loss.item()

        if epoch % STEP_LR == 0:
            learning_rate = learning_rate * LR_SCALER
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

        epoch_loss /= len(train_loader.dataset)
        acc, test_loss = test(model, test_loader, parellel=parellel)
        train_losses.append(epoch_loss)
        test_losses.append(test_loss)
        print(
            "Epoch: {} Loss: {:.6f} Test Loss: {:.6f} Test Accuracy: {:.2f}%".format(
                epoch + 1, epoch_loss, test_loss, acc
            )
        )

        if path is not None and acc - prev_acc > 1:
            print(
                f"Accuracy improved from {prev_acc:.2f}% to {acc:.2f}%. Saving model."
            )
            save_model(model, path)

        prev_acc = acc

    return model



def test(model, test_loader, parellel=False):
    # Test the model
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0

    # if parellel:
    #     model = model.to(device)
    #     model = nn.DataParallel(model)

    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader):
            if parellel:
                data = data.to(device)
                target = target.to(device)
            else:
                data = data.to("cpu")
                target = target.to("cpu")

            output = model(data)

            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / total

    torch.cuda.empty_cache()

    return accuracy, test_loss



def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device=device, parellel=True):
    if parellel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model


def augment_img(img: torch.Tensor, noise: float):
    img += np.random.randn(*img.shape) * noise * 2 - noise
    return np.array(np.clip(img, 0, 1), dtype=np.float32)


def augment_set(x, y, length):
    x_aug, y_aug = [], []

    for i in range(len(x)):
        for j in range(length):
            x_aug.append(augment_img(x[i], 0.002))
            y_aug.append(y[i])

    return np.array(x_aug), np.array(y_aug)


def remove_data_parallel(model):
    old_state_dict = model.state_dict()
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    return model


def loader_to_data(loader):
    x, y = [], []
    for data, target in loader:
        x.append(data)
        y.append(target)
    return torch.cat(x), torch.cat(y)


def data_to_loader(x, y, batch_size=DEFAULT_BATCH_SIZE, shuffle=True):
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def loader_to_noised(loader, noise_level, batch_size=DEFAULT_BATCH_SIZE):
    x, _ = loader_to_data(loader)

    x_noised = x + noise_level * torch.randn_like(x)

    return data_to_loader(x_noised, x, batch_size)
