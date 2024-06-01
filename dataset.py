import os
import glob
import torch
import numpy as np
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor
from option import get_option
from scipy.ndimage import affine_transform
import random

opt = get_option()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, opt):
        self.opt = opt
        self.dataset_root = opt.dataset_root
        self.phase = phase
        self.image_list = [
            i
            for i in glob.glob(
                os.path.join(self.dataset_root, phase, "image", "*.nii.gz")
            )
        ]
        self.label_list = [
            i
            for i in glob.glob(os.path.join(self.dataset_root, phase, "gt", "*.nii.gz"))
        ]
        self.image_list = sorted(self.image_list)
        self.label_list = sorted(self.label_list)
        self.foreground_sampling_prob = opt.foreground_sampling_prob

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label_path = self.label_list[index]
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype(np.float32)
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.int32)
        image = np.clip(image, 0, np.percentile(image, 99.9))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = 2 * (image - 0.5)
        if self.phase == "train" and self.opt.strong_aug:
            image, label = self.rotate90(image, label)
            image, label = self.flip(image, label)
            image = self.add_gaussian_noise(image)
            image = self.bright_adjust(image)
            image = self.contrast_adjust(image)
        patch_size = self.opt.image_size
        x = random.randint(0, label.shape[0] - patch_size[0])
        y = random.randint(0, label.shape[1] - patch_size[1])
        z = random.randint(0, label.shape[2] - patch_size[2])
        image = image[
            x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]
        ]
        label = label[
            x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]
        ]
        image = torch.from_numpy(image.copy()).float().unsqueeze(0)
        label = torch.from_numpy(label.copy()).float().unsqueeze(0)
        return image, label

    def __len__(self):
        return len(self.image_list)

    def rotate90(self, image, label, p=0.5):
        if np.random.rand() < p:
            n = np.random.randint(1, 5)
            dim = np.random.choice([0, 1, 2])
            image = np.rot90(image, n, axes=(dim, (dim + 1) % 3))
            label = np.rot90(label, n, axes=(dim, (dim + 1) % 3))
        return image, label

    def flip(self, image, label, p=0.5):
        if np.random.rand() < p:
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=0)
                label = np.flip(label, axis=0)
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1)
                label = np.flip(label, axis=1)
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=2)
                label = np.flip(label, axis=2)
        return image, label

    def add_gaussian_noise(self, image, p=0.5):
        if np.random.rand() < p:
            noise = np.random.normal(0, 0.3, image.shape)
            image += noise
        return image

    def bright_adjust(self, image, p=0.5):
        if np.random.rand() < p:
            image = image + np.random.uniform(-0.3, 0.3)
        return image

    def contrast_adjust(self, image, p=0.5):
        if np.random.rand() < p:
            image = image * np.random.uniform(0.7, 1.3)
        return image

    def random_affine_transform(
        self,
        image,
        label,
        scale_range=(0.8, 1.2),
        rotation_range=(-10, 10),
        translation_range=(-5, 5),
    ):
        # Generate random scaling factors
        scale = np.random.uniform(scale_range[0], scale_range[1], 3)

        # Generate random rotation angles in degrees
        rotation = np.radians(
            np.random.uniform(rotation_range[0], rotation_range[1], 3)
        )

        # Generate random translations
        translation = np.random.uniform(translation_range[0], translation_range[1], 3)
        scale_matrix = np.diag(scale)
        cos = np.cos(rotation)
        sin = np.sin(rotation)
        rotation_x = np.array([[1, 0, 0], [0, cos[0], -sin[0]], [0, sin[0], cos[0]]])
        rotation_y = np.array([[cos[1], 0, sin[1]], [0, 1, 0], [-sin[1], 0, cos[1]]])
        rotation_z = np.array([[cos[2], -sin[2], 0], [sin[2], cos[2], 0], [0, 0, 1]])

        # Combine transformations
        transformation_matrix = rotation_z @ rotation_y @ rotation_x @ scale_matrix
        transformed_image = affine_transform(
            image, transformation_matrix, offset=translation, order=1, mode="nearest"
        )
        transformed_label = affine_transform(
            label, transformation_matrix, offset=translation, order=1, mode="nearest"
        )

        return transformed_image, transformed_label


def get_dataloader(opt):
    train_dataset = Dataset(phase="train", opt=opt)
    valid_dataset = Dataset(phase="val", opt=opt)
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
    train_dataloader, valid_dataloader = get_dataloader(opt)
    for i, (image, label) in enumerate(train_dataloader):
        print(image.shape, label.shape)

    for i, (image, label) in enumerate(valid_dataloader):
        print(image.shape, label.shape)
