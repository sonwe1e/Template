import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.option import get_option

opt = get_option(verbose=False)

train_transform = A.Compose(
    [
        A.Resize(opt.image_size, opt.image_size),
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
