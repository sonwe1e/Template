import torch
import seaborn as sns
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        """
        初始化 LightningModule。

        Args:
            opt: 包含训练参数的配置对象。
            model: 要训练的 PyTorch 模型。
            len_trainloader: 训练数据加载器的长度，用于学习率调度器。
        """
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt
        self.model = model
        self.ce_loss = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
        # 用于存储训练和验证阶段的预测值和真实标签，在epoch结束时计算指标
        self.train_pred = []
        self.train_label = []
        self.valid_pred = []
        self.valid_label = []
        # 初始化训练和验证集的指标计算器
        self.train_iou = MeanIoU(num_classes=opt.num_classes, per_class=True)
        self.train_dice = GeneralizedDiceScore(
            num_classes=opt.num_classes, per_class=True
        )
        self.valid_iou = MeanIoU(num_classes=opt.num_classes, per_class=True)
        self.valid_dice = GeneralizedDiceScore(
            num_classes=opt.num_classes, per_class=True
        )

    def forward(self, x):
        """
        模型的前向传播。

        Args:
            x: 输入张量。

        Returns:
            模型的预测输出。
        """
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。

        Returns:
            包含优化器和学习率调度器的字典。
        """
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
        """
        单个训练步骤的逻辑。

        Args:
            batch: 一个批次的训练数据。
            batch_idx: 批次的索引。

        Returns:
            训练损失。
        """
        image, label = (batch["image"], batch["label"])
        prediction = self(image)
        ce_loss = self.ce_loss(prediction, label)  # 计算交叉熵损失
        loss = ce_loss  # 总损失等于交叉熵损失
        self.train_pred.append(prediction)  # 存储预测值
        self.train_label.append(label)  # 存储真实标签
        self.log("train_ce_loss", ce_loss)  # 记录训练损失
        self.log("train_loss", loss)  # 记录总损失
        self.log("learning_rate", self.scheduler.get_last_lr()[0])  # 记录学习率
        return loss

    def validation_step(self, batch, batch_idx):
        """
        单个验证步骤的逻辑。

        Args:
            batch: 一个批次的验证数据。
            batch_idx: 批次的索引。
        """
        image, label = (batch["image"], batch["label"])
        prediction = self(image)
        ce_loss = self.ce_loss(prediction, label)  # 计算交叉熵损失
        loss = ce_loss  # 总损失等于交叉熵损失
        self.valid_pred.append(prediction)  # 存储预测值
        self.valid_label.append(label)  # 存储真实标签
        self.log("valid_ce_loss", ce_loss)  # 记录验证集交叉熵损失
        self.log("valid_loss", loss)  # 记录验证集总损失

    def on_train_epoch_end(self):
        """
        在每个训练周期结束时计算并记录训练指标。
        """
        self.train_pred = torch.cat(self.train_pred, 0)  # 将所有批次的预测值拼接在一起
        self.train_label = torch.cat(
            self.train_label, 0
        )  # 将所有批次的真实标签拼接在一起
        preds = torch.argmax(self.train_pred, dim=1)  # 将预测值转换为类别索引

        iou = self.train_iou(preds, self.train_label)  # 计算IOU指标
        dice = self.train_dice(preds, self.train_label)  # 计算Dice指标
        for i in range(self.opt.num_classes):  # 记录每个类别的指标
            self.log(f"iou/train_iou_{i}", iou[i])
            self.log(f"dice/train_dice_{i}", dice[i])
        self.log("iou/train_iou", iou.mean())  # 记录平均IOU
        self.log("dice/train_dice", dice.mean())  # 记录平均Dice
        # 清空存储预测值和标签的列表，为下一个训练周期做准备
        self.train_pred = []
        self.train_label = []

    def on_validation_epoch_end(self):
        """
        在每个验证周期结束时计算并记录验证指标。
        """
        self.valid_pred = torch.cat(self.valid_pred, 0)  # 将所有批次的预测值拼接在一起
        self.valid_label = torch.cat(
            self.valid_label, 0
        )  # 将所有批次的真实标签拼接在一起
        preds = torch.argmax(self.valid_pred, dim=1)  # 将预测值转换为类别索引

        iou = self.valid_iou(preds, self.valid_label)  # 计算IOU指标
        dice = self.valid_dice(preds, self.valid_label)  # 计算Dice指标

        for i in range(self.opt.num_classes):  # 记录每个类别的指标
            self.log(f"iou/valid_iou_{i}", iou[i])
            self.log(f"dice/valid_dice_{i}", dice[i])
        self.log("iou/valid_iou", iou.mean())  # 记录平均IOU
        self.log("dice/valid_dice", dice.mean())  # 记录平均Dice
        # 清空存储预测值和标签的列表，为下一个验证周期做准备
        self.valid_pred = []
        self.valid_label = []
