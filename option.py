import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="随机种子，用于确保实验的可重复性"
    )
    parser.add_argument(
        "--save_wandb",
        type=bool,
        default=True,
        help="是否将训练过程的指标保存到 wandb",
    )
    parser.add_argument("--project", type=str, default="Test", help="wandb 项目名称")

    # dataset
    parser.add_argument("--data_path", type=str, default="", help="数据集路径")
    parser.add_argument(
        "-is", "--image_size", type=int, default=384, help="输入图像的大小"
    )
    parser.add_argument("--aug_m", type=int, default=2, help="数据增强的强度参数")
    parser.add_argument(
        "--num_classes", type=int, default=1000, help="分类任务的类别数量"
    )
    parser.add_argument("--in_chans", type=int, default=3, help="输入图像的通道数")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="用于数据加载的并行线程数"
    )

    # training setups
    parser.add_argument(
        "--model_name", type=str, default="resnet18d.ra2_in1k", help="使用的模型名称"
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=5e-2,
        help="权重衰减系数，用于防止过拟合",
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=4e-4, help="学习率"
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("-e", "--epochs", type=int, default=500, help="训练的总轮数")

    # experiment
    parser.add_argument(
        "-d",
        "--devices",
        type=int,
        default=0,
        help="使用的 GPU 设备 ID，一般不使用多卡，使用多卡修改为列表 [0, 1]",
    )
    parser.add_argument(
        "-en", "--exp_name", type=str, default="baselinev1", help="实验名称"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="模型恢复的 checkpoint 路径"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        help="训练使用的精度模式，常见选择包括 32，16，bf16-mixed",
    )
    parser.add_argument(
        "--val_check",
        type=float,
        default=1.0,
        help="验证集频率，当容易出现过拟合现象的时候可以设置较小的值找到比较好的 checkpoint",
    )
    parser.add_argument(
        "--log_step",
        type=int,
        default=20,
        help="日志记录的频率，例如每训练 log_step 个 batch 记录一次",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1e6,
        help="梯度裁剪的最大值，用于防止梯度爆炸，默认不使用",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="梯度累积的 batch 数，用于模拟更大的 batch size，默认不使用",
    )

    return parser.parse_args()


def get_option():
    opt = parse_args()
    return opt
