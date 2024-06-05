import os
import glob
import torch
import numpy as np
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor
from option import get_option
import random
from skimage.transform import resize
from tqdm import tqdm

opt = get_option()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, opt):
        self.opt = opt
        self.dataset_root = opt.dataset_root
        self.phase = phase
        self.image_list = glob.glob(
            os.path.join(self.dataset_root, "imagesTr", "*.mha")
        )
        random.shuffle(self.image_list)
        self.label_list = [
            i.replace("imagesTr", "labelsTr").replace("_0000", "")
            for i in self.image_list
        ]

        # Split dataset for training and validation
        split_idx = int(len(self.image_list) * 0.8)
        if phase == "train":
            self.image_list = self.image_list[:split_idx]
            self.label_list = self.label_list[:split_idx]
        else:
            self.image_list = self.image_list[split_idx:]
            self.label_list = self.label_list[split_idx:]

        self.foreground_sampling_prob = opt.foreground_sampling_prob

        # Load data with multithreading and display progress with tqdm
        with ThreadPoolExecutor(max_workers=opt.num_workers * 3) as executor:
            futures_images = {
                executor.submit(self.load_image, img): img for img in self.image_list
            }
            futures_labels = {
                executor.submit(self.load_image, lbl, np.int8): lbl
                for lbl in self.label_list
            }

            self.image_list = [
                future.result()
                for future in tqdm(
                    futures_images, total=len(self.image_list), desc="Loading Images"
                )
            ]
            self.label_list = [
                future.result()
                for future in tqdm(
                    futures_labels, total=len(self.label_list), desc="Loading Labels"
                )
            ]

    def load_image(self, file_path, dtype=np.float32):
        """Helper function to load and process images."""
        image = sitk.ReadImage(file_path)
        array = sitk.GetArrayFromImage(image).astype(dtype)
        return array

    def __getitem__(self, index):
        if self.phase == "train":
            index = random.randint(0, len(self.image_list) - 1)
        image = self.image_list[index]
        label = self.label_list[index]
        label = np.clip(label, 0, 1)
        # CT image preprocessing
        image = np.clip(image, -1000, 500)
        # MRA image preprocessing
        # image = np.clip(image, 0.0, np.percentile(image, 99.9))
        # Normalize image to [-1, 1]
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # image = 2 * (image - 0.5)
        # Standardize image to zero mean and unit variance
        image = (image - np.mean(image)) / np.std(image)
        global_image = image.copy()
        if self.phase == "train":
            if self.opt.strong_aug:
                image, label = self.rotate90(image, label)
                image, label = self.flip(image, label)
                image = self.add_gaussian_noise(image)
                image = self.bright_adjust(image)
                image = self.contrast_adjust(image)

            patch_size = self.opt.image_size
            image, label = pad_to_patch_size(image, label, patch_size)
            x = random.randint(0, label.shape[0] - patch_size[0])
            y = random.randint(0, label.shape[1] - patch_size[1])
            z = random.randint(0, label.shape[2] - patch_size[2])
            image = image[
                x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]
            ]
            label = label[
                x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]
            ]
        else:
            image_size = self.opt.image_size
            crop_size = [size * 1 for size in image_size]
            image, label = pad_to_patch_size(image, label, crop_size)
            image = image[
                (image.shape[0] - crop_size[0])
                // 2 : (image.shape[0] + crop_size[0])
                // 2,
                (image.shape[1] - crop_size[1])
                // 2 : (image.shape[1] + crop_size[1])
                // 2,
                (image.shape[2] - crop_size[2])
                // 2 : (image.shape[2] + crop_size[2])
                // 2,
            ]
            label = label[
                (label.shape[0] - crop_size[0])
                // 2 : (label.shape[0] + crop_size[0])
                // 2,
                (label.shape[1] - crop_size[1])
                // 2 : (label.shape[1] + crop_size[1])
                // 2,
                (label.shape[2] - crop_size[2])
                // 2 : (label.shape[2] + crop_size[2])
                // 2,
            ]
        image = torch.from_numpy(image.copy()).float().unsqueeze(0)
        label = torch.from_numpy(label.copy()).float().unsqueeze(0)
        if opt.enable_global:
            global_image = resize_image(global_image, new_shape=(128, 128, 128))
            global_image = torch.from_numpy(global_image).float().unsqueeze(0)
            return image, label, global_image
        return image, label

    def __len__(self):
        return (
            len(self.image_list) * self.opt.iteration_rate
            if self.phase == "train"
            else len(self.image_list)
        )

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
            noise = np.random.normal(0, 0.1, image.shape)
            image += noise
        return image

    def bright_adjust(self, image, p=0.5):
        if np.random.rand() < p:
            image = image + np.random.uniform(-0.2, 0.2)
        return image

    def contrast_adjust(self, image, p=0.5):
        if np.random.rand() < p:
            image = image * np.random.uniform(0.8, 1.2)
        return image


def pad_to_patch_size(image, label, patch_size):
    padding = []
    for i in range(3):
        pad_before = max(0, (patch_size[i] - image.shape[i]) // 2)
        pad_after = max(0, (patch_size[i] - image.shape[i] + 1) // 2)
        padding.append((pad_before, pad_after))

    image = np.pad(image, padding, mode="constant", constant_values=0)
    label = np.pad(label, padding, mode="constant", constant_values=0)
    return image, label


def resize_image(image, old_spacing=None, new_spacing=None, new_shape=None, order=1):
    assert new_shape is not None or (
        old_spacing is not None and new_spacing is not None
    )
    if new_shape is None:
        new_shape = tuple(
            [
                int(np.round(old_spacing[i] / new_spacing[i] * float(image.shape[i])))
                for i in range(3)
            ]
        )
    resized_image = resize(
        image, new_shape, order=order, mode="edge", cval=0, anti_aliasing=False
    )
    return resized_image


def resize_segmentation(
    segmentation, old_spacing=None, new_spacing=None, new_shape=None, order=0, cval=0
):
    """
    Taken from batchgenerators (https://github.com/MIC-DKFZ/batchgenerators) to prevent dependency


    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    """
    assert new_shape is not None or (
        old_spacing is not None and new_spacing is not None
    )
    if new_shape is None:
        new_shape = tuple(
            [
                int(
                    np.round(
                        old_spacing[i] / new_spacing[i] * float(segmentation.shape[i])
                    )
                )
                for i in range(3)
            ]
        )
    tpe = segmentation.dtype
    assert len(segmentation.shape) == len(
        new_shape
    ), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(
            segmentation,
            new_shape,
            order,
            mode="constant",
            cval=cval,
            clip=True,
            anti_aliasing=False,
        ).astype(tpe)
    else:
        unique_labels = np.unique(segmentation)
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
        for i, c in enumerate(unique_labels):
            reshaped_multihot = resize(
                (segmentation == c).astype(float),
                new_shape,
                order,
                mode="edge",
                clip=True,
                anti_aliasing=False,
            )
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


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
