import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import numpy as np

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt
        self.model = model
        self.criterion = torch.nn.L1Loss()

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.opt.epochs,
            pct_start=0.06,
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
        self.log_image("input", batch["image"])
        self.log("train_loss", loss)
        self.log("learning_rate", self.scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("valid_loss", loss)

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass
