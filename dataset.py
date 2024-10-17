import torch
from PIL import Image
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
    def __init__(self, phase, opt, train_transform=None, valid_transform=None):
        self.phase = phase
        self.data_path = opt.data_path
        self.transform = train_transform if phase == "train" else valid_transform

    def __getitem__(self, index):
        image = 0
        label = 1
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.image_list)

    def load_image(self, path):
        return None

    def load_images_in_parallel(self):
        with ThreadPoolExecutor(max_workers=24) as executor:
            pass


def get_dataloader(opt):
    train_dataset = Dataset(
        phase="train",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )
    valid_dataset = Dataset(
        phase="valid",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )
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
        print(batch["image"].shape, torch.unique(batch["label"]))
        break
