import lightning.pytorch as pl  # updated import
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

class CIFAR10(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        datasets.CIFAR10(root='./.data', train=True, download=True)
        datasets.CIFAR10(root='./.data', train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.cifar10_train = datasets.CIFAR10(root='./.data', train=True, transform=self.transform)
            self.cifar10_val = datasets.CIFAR10(root='./.data', train=False, transform=self.transform)

        if stage == 'test' or stage is None:
            self.cifar10_test = datasets.CIFAR10(root='./.data', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size)
