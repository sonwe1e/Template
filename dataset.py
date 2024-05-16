import os
import glob
import torch
from PIL import Image
import cv2
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from option import get_option

opt = get_option()

train_transform = A.Compose(
    [
        A.Resize(int(opt.image_size / 0.875), int(opt.image_size / 0.875)),
        A.RandomResizedCrop(opt.image_size, opt.image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
)

valid_transform = A.Compose(
    [
        A.Resize(opt.image_size, opt.image_size),
        A.Normalize(),
        ToTensorV2(),
    ]
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, opt, transform=None):
        pass

    def load_image(self, path):
        return None

    def load_images_in_parallel(self):
        with ThreadPoolExecutor(max_workers=24) as executor:
            pass

    def __getitem__(self, index):
        return None

    def __len__(self):
        return len()


def get_dataloader(opt):
    train_dataset = Dataset(phase="train", opt=opt, transform=train_transform)
    valid_dataset = Dataset(phase="valid", opt=opt, transform=valid_transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader
