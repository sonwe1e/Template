import torch
from option import get_option
from dataset import *
from pl_tool import *
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import torch
import wandb
import timm

torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    opt = get_option()
    """定义网络"""
    model = timm.create_model(
        opt.model_name,
        pretrained=False,
        num_classes=107,
        features_only=False,
        # drop_path_rate=0.2,
        drop_rate=0.2,
    )

    """模型编译"""
    # model = torch.compile(model)

    """导入数据集"""
    train_dataloader, valid_dataloader = get_dataloader(opt)

    """Lightning 模块定义"""
    pl.seed_everything(opt.seed)
    wandb_logger = WandbLogger(
        project=opt.project,
        name=opt.exp_name,
        offline=not opt.save_wandb,
        config=opt,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=[opt.devices],
        strategy="auto",
        max_epochs=opt.epochs,
        precision=opt.precision,
        default_root_dir="./",
        deterministic=False,
        logger=wandb_logger,
        val_check_interval=opt.val_check,
        log_every_n_steps=opt.log_step,
        accumulate_grad_batches=opt.accumulate_grad_batches,
        gradient_clip_val=opt.gradient_clip_val,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join("./checkpoints", opt.exp_name),
                monitor="valid_f1",
                mode="max",
                save_top_k=1,
                save_last=False,
                filename="{epoch}_{valid_f1:.4f}",
            ),
        ],
    )

    # Start training
    trainer.fit(
        LightningModule(opt, model, len(train_dataloader)),
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    wandb.finish()
