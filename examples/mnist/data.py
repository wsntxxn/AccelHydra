import torchvision
from torchvision import transforms


def train_dataset(data_root: str):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    return dataset


def val_dataset(data_root: str):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    return dataset
