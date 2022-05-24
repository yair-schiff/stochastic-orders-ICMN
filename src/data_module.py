import os
import math

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import build_transforms, get_dataset, get_distribution


class DistributionDataModule(pl.LightningDataModule):
    def __init__(self, distribution_type, distribution_params,
                 batch_size=32, validation_batch_multiplier=1, num_workers=0):
        super().__init__()
        self.distribution_type = distribution_type
        self.distribution_params = distribution_params
        self.batch_size = batch_size
        self.validation_batch_multiplier = validation_batch_multiplier
        self.num_workers = num_workers
        self.distribution = get_distribution(distribution_type, distribution_params)
        self.dataset = TensorDataset(torch.ones(1))  # Dummy dataset to pass to dataloader

    @staticmethod
    def get_data_module_args_as_dict(args):
        return {
            'distribution_type': args.distribution_type,
            'distribution_params': args.distribution_params,
            'batch_size': args.batch_size,
            'validation_batch_multiplier': args.validation_batch_multiplier,
            'num_workers': args.num_workers
        }

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def custom_collate_train_fn(self, _):
        return self.distribution.sample(self.batch_size).squeeze(), 0

    def custom_collate_val_fn(self, _):
        return self.distribution.sample(self.batch_size*self.validation_batch_multiplier).squeeze(), 0

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          collate_fn=self.custom_collate_train_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          collate_fn=self.custom_collate_val_fn)


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, dataset_dir,
                 train_split=0.8, batch_size=32, validation_batch_multiplier=1, num_workers=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.train_split = train_split
        self.batch_size = batch_size
        self.validation_batch_multiplier = validation_batch_multiplier
        self.num_workers = num_workers
        self.train, self.val, self.test = None, None, None

        self.dataset_class = get_dataset(dataset_name)
        self.transforms = build_transforms(dataset_name=dataset_name)

    @staticmethod
    def get_data_module_args_as_dict(args):
        return {
            'dataset_name': args.dataset_name,
            'dataset_dir': args.dataset_dir,
            'train_split': args.train_split,
            'batch_size': args.batch_size,
            'validation_batch_multiplier': args.validation_batch_multiplier,
            'num_workers': args.num_workers
        }

    def prepare_data(self):
        if os.path.isdir(os.path.join(self.dataset_dir, 'train')):
            print(f'{self.dataset_name} train data already downloaded. Skipping.')
        else:
            print(f'Downloading {self.dataset_name} train data...')
            try:
                self.dataset_class(root=os.path.join(self.dataset_dir, 'train'), train=True, download=True)
            except TypeError:
                self.dataset_class(root=os.path.join(self.dataset_dir, 'train'), split='train', download=True)
        if os.path.isdir(os.path.join(self.dataset_dir, 'test')):
            print(f'{self.dataset_name} test data already downloaded. Skipping.')
        else:
            print(f'Downloading {self.dataset_name} test data...')
            try:
                self.dataset_class(root=os.path.join(self.dataset_dir, 'test'), train=False, download=True)
            except TypeError:
                self.dataset_class(root=os.path.join(self.dataset_dir, 'test'), split='test', download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            try:
                full_dataset = self.dataset_class(root=os.path.join(self.dataset_dir, 'train'),
                                                  train=True, transform=self.transforms)
            except TypeError:
                full_dataset = self.dataset_class(root=os.path.join(self.dataset_dir, 'train'),
                                                  split='train', transform=self.transforms)
            train_set_size = math.ceil(len(full_dataset)*self.train_split)
            self.train, self.val = torch.utils.data.random_split(full_dataset,
                                                                 [train_set_size, len(full_dataset) - train_set_size])
        if stage == 'test' or stage is None:
            try:
                self.test = self.dataset_class(root=os.path.join(self.dataset_dir, 'test'),
                                               train=False, transform=self.transforms)
            except TypeError:
                self.test = self.dataset_class(root=os.path.join(self.dataset_dir, 'test'),
                                               split='test', transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size*self.validation_batch_multiplier,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
