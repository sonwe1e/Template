import torch
import seaborn as sns
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt
        self.model = model
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.train_pred = []
        self.train_label = []
        self.valid_pred = []
        self.valid_label = []
        self.train_iou = MeanIoU(num_classes=opt.num_classes, per_class=True)
        self.train_dice = GeneralizedDiceScore(
            num_classes=opt.num_classes, per_class=True
        )
        self.valid_iou = MeanIoU(num_classes=opt.num_classes, per_class=True)
        self.valid_dice = GeneralizedDiceScore(
            num_classes=opt.num_classes, per_class=True
        )

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            betas=(0.9, 0.95),
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.opt.epochs,
            pct_start=0.06,
            steps_per_epoch=self.len_trainloader,
            anneal_strategy="linear",
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        image, label = (batch["image"], batch["label"])
        prediction = self(image)
        ce_loss = self.ce_loss(prediction, label)
        loss = ce_loss
        self.train_pred.append(prediction)
        self.train_label.append(label)
        self.log("train_ce_loss", ce_loss)
        self.log("train_loss", loss)
        self.log("learning_rate", self.scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = (batch["image"], batch["label"])
        prediction = self(image)
        ce_loss = self.ce_loss(prediction, label)
        loss = ce_loss
        self.valid_pred.append(prediction)
        self.valid_label.append(label)
        self.log("valid_ce_loss", ce_loss)
        self.log("valid_loss", loss)

    def on_train_epoch_end(self):
        self.train_pred = torch.cat(self.train_pred, 0)
        self.train_label = torch.cat(self.train_label, 0)
        preds = torch.argmax(self.train_pred, dim=1)

        iou = self.train_iou(preds, self.train_label)
        dice = self.train_dice(preds, self.train_label)
        for i in range(self.opt.num_classes):
            self.log(f"iou/train_iou_{i}", iou[i])
            self.log(f"dice/train_dice_{i}", dice[i])
        self.log("iou/train_iou", iou)
        self.log("dice/train_dice", dice)
        # Reset storage
        self.train_pred = []
        self.train_label = []

    def on_validation_epoch_end(self):
        self.valid_pred = torch.cat(self.valid_pred, 0)
        self.valid_label = torch.cat(self.valid_label, 0)
        preds = torch.argmax(self.valid_pred, dim=1)

        iou = self.valid_iou(preds, self.valid_label)
        dice = self.valid_dice(preds, self.valid_label)

        for i in range(self.opt.num_classes):
            self.log(f"iou/valid_iou_{i}", iou[i])
            self.log(f"dice/valid_dice_{i}", dice[i])
        self.log("iou/valid_iou", iou)
        self.log("dice/valid_dice", dice)
        self.valid_pred = []
        self.valid_label = []
