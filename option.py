import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_wandb", type=bool, default=True)
    parser.add_argument("--project", type=str, default="Test")

    # dataset
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("-is", "--image_size", type=int, default=384)
    parser.add_argument("--aug_line", type=list, default=[0.4, 0.95, 1.0])
    parser.add_argument("--aug_m", type=int, default=3)

    # training setups
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-2)
    parser.add_argument("-lr", "--learning_rate", type=float, default=4e-3)
    parser.add_argument("-bs", "--batch_size", type=int, default=8)
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)

    # experiment
    parser.add_argument("-f", "--fold", type=int, default=0)
    parser.add_argument("--num_fold", type=int, default=4)
    parser.add_argument("-d", "--devices", type=int, default=0)
    parser.add_argument("-en", "--exp_name", type=str, default="baselinev1")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--model_name", type=str, default="efficientvit_l3.r384_in1k")
    parser.add_argument("--val_check", type=float, default=1.0)
    parser.add_argument("--log_step", type=int, default=50)
    parser.add_argument("--gradient_clip_val", type=int, default=1e6)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    return parser.parse_args()


def get_option():
    opt = parse_args()
    return opt
