import os
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import cfg

# TODO: Define your data path (the directory containing the 4 np array files)
DATA_PATH = './data/'


class FMNIST(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        super(FMNIST, self).__init__()
        # Retrieve all the images and the labels, and store them
        # as class variables.
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        data_point = self.X[index, :]
        data_point = np.array(data_point).astype(np.uint8).reshape(28, 28, 1)

        if self.transforms:
            data_point = self.transforms(data_point)

        if self.y is not None:
            return data_point, self.y[index]

        return data_point


def get_data_loader(set_name, augmentation=False):
    # Create the dataset class tailored to the set (train or test)
    # provided as argument. Use it to create a dataloader. Use the appropriate
    # hyper-parameters from cfg
    # define transform to scale the image to [0, 1]
    if augmentation and set_name=='train':
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(brightness=np.random.uniform(0, 4), contrast=0, saturation=0, hue=0),
             transforms.ToTensor()
             ])
    else:
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor()
             ])
    if set_name == "train":
        images = np.load(DATA_PATH + "train_images.npy")
        labels = np.load(DATA_PATH + "train_labels.npy")
    else:
        images = np.load(DATA_PATH + "test_images.npy")
        labels = np.load(DATA_PATH + "test_labels.npy")

    data = FMNIST(images, labels, transform)

    return DataLoader(data, batch_size=cfg['batch_size'], shuffle=True)


