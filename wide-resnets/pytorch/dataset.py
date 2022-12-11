import torch
import torch.utils.data
import torchvision


def load_cifar10(batch_size=128):
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4, padding_mode="symmetric"),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="~/Datasets", train=True, download=True, transform=train_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    )

    valid_dataset = torchvision.datasets.CIFAR10(
        root="~/Datasets", train=False, download=True, transform=test_transforms,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
    )
    return train_loader, valid_loader
