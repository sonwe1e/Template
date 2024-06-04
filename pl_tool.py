from sympy import im
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import numpy as np
from utils import BinaryDiceLoss, FocalLoss
from lion_pytorch import Lion

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt
        self.model = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = BinaryDiceLoss()

    def forward(self, local_patch, global_img=None):
        if global_img is None:
            pred = self.model(local_patch)
            return pred
        pred = self.model(local_patch, global_img)
        return pred

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.opt.epochs,
            pct_start=0.15,
            steps_per_epoch=self.len_trainloader,
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        if self.opt.enable_global:
            image, label, global_image = batch
            mixed_image, mixed_global_image, label_a, label_b, lam = (
                self.mixup_patch_img(image, global_image, label, self.opt.alpha)
            )
            pred = self(mixed_image, mixed_global_image)
        else:
            image, label = batch
            mixed_image, label_a, label_b, lam = self.mixup_patch(
                image, label, self.opt.alpha
            )
            pred = self(mixed_image)
        ce_loss = 0
        dice_loss = 0
        image_shape = image.shape[-3:]
        for _, p in enumerate(pred):
            dice_loss += 0.5**_ * (
                self.dice_loss(F.interpolate(F.sigmoid(p), image_shape), label_a) * lam
                + self.dice_loss(F.interpolate(F.sigmoid(p), image_shape), label_b)
                * (1 - lam)
            )
            ce_loss += 0.5**_ * (
                self.criterion(F.interpolate(p, image_shape), label_a) * lam
                + self.criterion(F.interpolate(p, image_shape), label_b) * (1 - lam)
            )
        self.log("train_ce_loss", ce_loss)
        self.log("train_dice_loss", dice_loss)
        self.log("learning_rate", self.scheduler.get_last_lr()[0])
        return 0.5 * ce_loss + 1.5 * dice_loss

    def validation_step(self, batch, batch_idx):
        if self.opt.enable_global:
            image, label, global_image = batch
            pred = self(image, global_image)
        else:
            image, label = batch
            pred = self(image)
        pred = pred[0]
        ce_loss = self.criterion(pred, label)
        dice_loss = self.dice_loss(F.sigmoid(pred), label)
        loss = ce_loss + dice_loss
        self.log("valid_ce_loss", ce_loss)
        self.log("valid_dice_loss", dice_loss)
        self.log("valid_loss", loss)

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def mixup_patch(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_patch_img(self, patch, global_img, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = patch.size()[0]
        index = torch.randperm(batch_size)
        mixed_patch = lam * patch + (1 - lam) * patch[index, :]
        mixed_global_img = lam * global_img + (1 - lam) * global_img[index, :]
        y_a, y_b = y, y[index]
        return mixed_patch, mixed_global_img, y_a, y_b, lam
