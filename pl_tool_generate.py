import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import numpy as np
import torchmetrics as tm
from torch.optim import AdamW
import torchvision

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt
        self.model = model
        self.DNet = torchvision.models.resnet18(num_classes=1)
        self.l1loss = torch.nn.SmoothL1Loss()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.automatic_optimization = False

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        self.optimizer1 = AdamW(
            self.model.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.optimizer2 = AdamW(
            self.DNet.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer1,
            max_lr=self.learning_rate,
            epochs=self.opt.epochs,
            pct_start=0.015,
            steps_per_epoch=self.len_trainloader,
        )
        return (
            {
                "optimizer": self.optimizer1,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "step",
                },
            },
            {"optimizer": self.optimizer2},
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizer_g, optimizer_d = self.optimizers()
        scheduler = self.scheduler

        # train generator
        self.toggle_optimizer(optimizer_g)
        pred = self.model(x)
        pred = torch.clamp(pred, -1, 1)
        valid = torch.ones(x.size(0), 1)
        valid = valid.type_as(x)
        l1loss = self.l1loss(pred, y)
        g_loss = self.adversarial_loss(self.DNet(pred), valid)
        loss = l1loss + 10 * self.scheduler.get_last_lr()[0] * (g_loss)

        self.manual_backward(loss)
        optimizer_g.step()
        scheduler.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)
        valid = torch.ones(x.size(0), 1)
        valid = valid.type_as(x)
        real_loss = self.adversarial_loss(self.DNet(y), valid)
        fake = torch.zeros(x.size(0), 1)
        fake = fake.type_as(x)
        fake_loss = self.adversarial_loss(self.DNet(pred.detach()), fake)
        d_loss = self.scheduler.get_last_lr()[0] * (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        psnr = tm.functional.image.peak_signal_noise_ratio(pred, y)
        ssim = tm.functional.image.structural_similarity_index_measure(pred, y)
        self.log("g_loss", g_loss)
        self.log("d_loss", d_loss)
        self.log("train_l1loss", l1loss)
        self.log("train_psnr", psnr)
        self.log("train_ssim", ssim)
        self.log("learning_rate", self.scheduler.get_last_lr()[0])

    def validation_step(self, batch, batch_idx):
        x, y = batch
        b, c, h, w = x.shape
        size = self.opt.image_size
        pred = torch.ones((b, c, h, w), device=x.device)
        m, n = h // size, w // size
        for i in range(m):
            for j in range(n):
                patch = x[:, :, i * size : (i + 1) * size, j * size : (j + 1) * size]
                patch = self.model(patch)
                pred[:, :, i * size : (i + 1) * size, j * size : (j + 1) * size] = patch
        pred = torch.clamp(pred, -1, 1)
        l1loss = self.l1loss(pred, y)
        psnr = tm.functional.image.peak_signal_noise_ratio(pred, y)
        ssim = tm.functional.image.structural_similarity_index_measure(pred, y)
        self.log("valid_psnr", psnr)
        self.log("valid_ssim", ssim)
        self.log("valid_l1loss", l1loss, prog_bar=True)

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass
