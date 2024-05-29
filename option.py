import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_wandb", type=bool, default=True)
    parser.add_argument("--project", type=str, default="BrainVessel")

    # models

    # augmentations

    # dataset
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/data2/songwei/Data/raw/Dataset012_BrainVessel",
    )
    parser.add_argument("--image_size", type=tuple, default=(128, 128, 128))

    # training setups
    parser.add_argument("-wd", "--weight_decay", type=float, default=2e-4)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-3)
    parser.add_argument("-bs", "--batch_size", type=int, default=2)
    parser.add_argument("-e", "--epochs", type=int, default=300)
    parser.add_argument("--foreground_sampling_prob", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--deep_supervision", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--strong_aug", type=bool, default=False)

    # experiment
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default="baselinev7")
    parser.add_argument("--val_check", type=float, default=1.0)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--gradient_clip_val", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    return parser.parse_args()


def get_option():
    opt = parse_args()
    return opt
