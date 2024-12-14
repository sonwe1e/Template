import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import numpy as np
import torchmetrics as tm
from torch.optim import AdamW
import torchvision

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    """
    使用对抗训练的 Lightning 模块，用于图像超分辨率任务。

    包含一个生成器模型和一个判别器模型。
    """

    def __init__(self, opt, model, len_trainloader):
        """
        初始化 Lightning 模块。

        Args:
            opt: 包含训练参数的配置对象。
            model: 生成器模型。
            len_trainloader: 训练数据加载器的长度，用于学习率调度器。
        """
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt
        self.model = model  # 生成器模型
        self.DNet = torchvision.models.resnet18(num_classes=1)  # 判别器模型
        self.l1loss = torch.nn.SmoothL1Loss()  # L1平滑损失
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()  # 二元交叉熵损失
        self.automatic_optimization = False  # 关闭自动优化，使用手动优化

    def forward(self, x):
        """
        生成器模型的前向传播。

        Args:
            x: 输入张量。

        Returns:
            生成器模型的预测输出。
        """
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。

        Returns:
            包含生成器和判别器优化器和学习率调度器的字典。
        """
        self.optimizer1 = AdamW(  # 生成器优化器
            self.model.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.optimizer2 = AdamW(  # 判别器优化器
            self.DNet.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(  # 生成器学习率调度器
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
        """
        单个训练步骤的逻辑。

        Args:
            batch: 一个批次的训练数据。
            batch_idx: 批次的索引。
        """
        x, y = batch  # x 是低分辨率图像，y 是高分辨率图像
        optimizer_g, optimizer_d = self.optimizers()  # 获取生成器和判别器优化器
        scheduler = self.scheduler  # 获取学习率调度器

        # 训练生成器
        self.toggle_optimizer(optimizer_g)  # 激活生成器优化器
        pred = self.model(x)  # 生成器生成超分辨率图像
        pred = torch.clamp(pred, -1, 1)  # 将生成图像的像素值限制在 -1 到 1 之间
        valid = torch.ones(x.size(0), 1).type_as(x)  # 创建真标签，用于判别器
        l1loss = self.l1loss(pred, y)  # 计算生成图像和真实图像的L1损失
        g_loss = self.adversarial_loss(self.DNet(pred), valid)  # 计算生成器对抗损失
        loss = l1loss + 10 * self.scheduler.get_last_lr()[0] * (g_loss)  # 总生成器损失

        self.manual_backward(loss)  # 手动计算梯度
        optimizer_g.step()  # 更新生成器参数
        scheduler.step()  # 更新学习率
        optimizer_g.zero_grad()  # 清空生成器梯度
        self.untoggle_optimizer(optimizer_g)  # 关闭生成器优化器

        # 训练判别器
        self.toggle_optimizer(optimizer_d)  # 激活判别器优化器
        valid = torch.ones(x.size(0), 1).type_as(x)  # 创建真标签，用于判别器
        real_loss = self.adversarial_loss(
            self.DNet(y), valid
        )  # 计算真实图像的判别器损失
        fake = torch.zeros(x.size(0), 1).type_as(x)  # 创建假标签，用于判别器
        fake_loss = self.adversarial_loss(
            self.DNet(pred.detach()), fake
        )  # 计算生成图像的判别器损失
        d_loss = (
            self.scheduler.get_last_lr()[0] * (real_loss + fake_loss) / 2
        )  # 总判别器损失
        self.manual_backward(d_loss)  # 手动计算梯度
        optimizer_d.step()  # 更新判别器参数
        optimizer_d.zero_grad()  # 清空判别器梯度
        self.untoggle_optimizer(optimizer_d)  # 关闭判别器优化器

        # 计算评估指标
        psnr = tm.functional.image.peak_signal_noise_ratio(pred, y)  # 计算峰值信噪比
        ssim = tm.functional.image.structural_similarity_index_measure(
            pred, y
        )  # 计算结构相似性
        self.log("g_loss", g_loss)  # 记录生成器对抗损失
        self.log("d_loss", d_loss)  # 记录判别器损失
        self.log("train_l1loss", l1loss)  # 记录训练L1损失
        self.log("train_psnr", psnr)  # 记录训练PSNR
        self.log("train_ssim", ssim)  # 记录训练SSIM
        self.log("learning_rate", self.scheduler.get_last_lr()[0])  # 记录学习率

    def validation_step(self, batch, batch_idx):
        """
        单个验证步骤的逻辑。
        采用分块的方式进行推理。

        Args:
            batch: 一个批次的验证数据。
            batch_idx: 批次的索引。
        """
        x, y = batch  # x 是低分辨率图像，y 是高分辨率图像
        b, c, h, w = x.shape  # 获取输入图像的形状
        size = self.opt.image_size  # 获取图像块的大小
        pred = torch.ones(
            (b, c, h, w), device=x.device
        )  # 创建一个与输入图像大小相同的全1张量，用于存储预测结果
        m, n = h // size, w // size  # 计算输入图像可以分割成多少个图像块
        for i in range(m):
            for j in range(n):
                patch = x[
                    :, :, i * size : (i + 1) * size, j * size : (j + 1) * size
                ]  # 取出当前图像块
                patch = self.model(patch)  # 将图像块输入到生成器模型中
                pred[:, :, i * size : (i + 1) * size, j * size : (j + 1) * size] = (
                    patch  # 将生成器模型的输出放到预测结果张量中
                )
        pred = torch.clamp(pred, -1, 1)  # 将像素值限制在 -1 到 1 之间
        l1loss = self.l1loss(pred, y)  # 计算验证集上的 L1 损失
        psnr = tm.functional.image.peak_signal_noise_ratio(
            pred, y
        )  # 计算验证集上的 PSNR
        ssim = tm.functional.image.structural_similarity_index_measure(
            pred, y
        )  # 计算验证集上的 SSIM
        self.log("valid_psnr", psnr)  # 记录验证集上的 PSNR
        self.log("valid_ssim", ssim)  # 记录验证集上的 SSIM
        self.log(
            "valid_l1loss", l1loss, prog_bar=True
        )  # 记录验证集上的 L1 损失，并显示在进度条上

    def on_train_epoch_end(self):
        """
        在每个训练周期结束时调用。
        目前为空，如有需要可以添加操作。
        """
        pass

    def on_validation_epoch_end(self):
        """
        在每个验证周期结束时调用。
        目前为空，如有需要可以添加操作。
        """
        pass
