import os
import glob
import torch
from PIL import Image
import cv2
import json
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from option import get_option

opt = get_option()


class Identity(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Identity, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return img


train_transform = A.Compose(
    [
        A.Resize(opt.image_size, opt.image_size),
        A.RandomResizedCrop(
            opt.image_size,
            opt.image_size,
            scale=(0.64, 1.0),
        ),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.SomeOf(
            [
                ## Color
                A.SomeOf(
                    [
                        A.Sharpen(),
                        A.Posterize(),
                        A.RandomBrightnessContrast(),
                        A.RandomGamma(),
                        A.ColorJitter(),
                    ],
                    n=2,
                ),
                ## CLAHE
                A.CLAHE(),
                ## Noise
                A.GaussNoise(),
                ## Blur
                A.AdvancedBlur(),
                ## Others
                A.ToGray(),
                Identity(),
            ],
            n=opt.aug_m,
        ),
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
        self.data_path = os.path.join(opt.dataset_root, phase)
        with open(os.path.join(opt.dataset_root, "label_mapping.json")) as f:
            self.label_mapping = json.load(f)
        self.image_list = glob.glob(os.path.join(self.data_path, "*/*.png"))
        new_image_list = []

        for idx, image in enumerate(self.image_list):
            if phase == "train":
                if (idx + opt.fold) % opt.num_fold != 0:
                    new_image_list.append(image)
            elif phase == "valid":
                if (idx + opt.fold) % opt.num_fold == 0:
                    new_image_list.append(image)

        self.image_list = new_image_list
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.cvtColor(cv2.imread(self.image_list[index]), cv2.COLOR_BGR2RGB)
        label = self.label_mapping[self.image_list[index].split("/")[-2]]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.image_list)

    def load_image(self, path):
        return None

    def load_images_in_parallel(self):
        with ThreadPoolExecutor(max_workers=24) as executor:
            pass


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


if __name__ == "__main__":
    opt = get_option()
    train_dataloader, valid_dataloader = get_dataloader(opt)

    for i, batch in enumerate(train_dataloader):
        print(batch)
