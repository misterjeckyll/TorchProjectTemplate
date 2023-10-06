import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
import os


class GTSRBDataModule(pl.LightningDataModule):
    def __init__(self, data_dir = './data/dataset', store_dir='./data/store', seed = 42, batch_size: int = 32, num_workers = 8, train_proportion = 0.8):
        super().__init__()
        # set batch size and number of workers
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.store_dir = store_dir
        self.train_prop = train_proportion
        # set up a resize transform
        self.resize = transforms.Resize((224,224))
        # set up a rotation transform from -10 to 10 degrees
        self.rotate = transforms.RandomRotation([-10,10])
        self.transform= transforms.Compose([self.resize, self.rotate, transforms.ToTensor()])

    def prepare_data(self):
        # get the raw dataset files from the store folder and extract it onto the dataset folder
        torchvision.datasets.utils.extract_archive(os.path.join(self.store_dir, 'GTSRB_train.zip'), self.data_dir)
        torchvision.datasets.utils.extract_archive(os.path.join(self.store_dir, 'GTSRB_test.zip'), self.data_dir)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # load the dataset
            gtsrb_full = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform)
            train_size = int(self.train_prop * len(gtsrb_full))
            val_size = len(gtsrb_full) - train_size
            self.train, self.val = random_split(gtsrb_full, [train_size, val_size] , generator=torch.Generator().manual_seed(self.seed))

        if stage == 'test' or stage is None:
            # load the test dataset
            self.test = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=self.transform)

    def train_dataloader(self):
        # sample the full dataset using Dataloader
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        #sample the full dataset using Dataloader
        return DataLoader(self.val, batch_size= self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)