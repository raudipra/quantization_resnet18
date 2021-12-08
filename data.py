import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from torchvision.datasets import CIFAR10,MNIST
from torchvision.transforms import RandomRotation, ToTensor, Compose, Resize, RandomHorizontalFlip
from torch.utils.data import DataLoader

train_transforms = Compose([Resize(128), RandomHorizontalFlip(), RandomRotation(15), ToTensor()])
val_transforms = Compose([Resize(128), ToTensor()])

train_data = CIFAR10('./', train=True, download=True, transform=train_transforms)
val_data = CIFAR10('./', train=False, download=True, transform=val_transforms)

train_dl = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True, num_workers=2)
val_dl = DataLoader(val_data, batch_size=32, shuffle=True, drop_last=False, num_workers=2)

if __name__ == '__main__':
    print(train_data[0][0].shape)
    print(dir(CIFAR10.meta))
    print(CIFAR10.meta)
