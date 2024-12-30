import argparse
import yaml
from pathlib import Path
import os


def get_option():
    # 首先读取YAML配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建 config.yaml 的绝对路径
    yaml_path = os.path.join(current_dir, "config.yaml")
    if Path(yaml_path).exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
    else:
        yaml_config = {}
    parser = argparse.ArgumentParser()

    # 从 YAML 配置中动态添加参数
    for key, value in yaml_config.items():
        # 可以根据需要添加更多类型的参数和处理逻辑
        if isinstance(value, int):
            arg_type = int
        elif isinstance(value, float):
            arg_type = float
        elif isinstance(value, str):
            arg_type = str
        elif isinstance(value, bool):
            # 对于布尔值，使用 action='store_true' 或 'store_false'
            if value:
                parser.add_argument(
                    f"--{key}", action="store_false", help=f"YAML Config: {key}"
                )
            else:
                parser.add_argument(
                    f"--{key}", action="store_true", help=f"YAML Config: {key}"
                )
            continue  # 跳过后续的 add_argument
        else:
            arg_type = None  # 或者根据你的需要进行处理

        parser.add_argument(
            f"--{key}",
            type=arg_type,
            default=value,
            help=f"YAML Config: {key}, default: {value}",
        )

    # 保留一个单独的 --config 参数，用于指定 YAML 文件的路径
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    # 如果指定了不同的 config 文件，则重新加载配置
    if args.config != "config.yaml":
        yaml_path = Path(args.config)
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
        else:
            print(
                f"Warning: Config file {args.config} not found. Using default values from original config.yaml or command line."
            )

    print("-" * 30)
    print("Current Configuration:")
    for key, value in yaml_config.items():
        print(f"{key}: {value}")
    print("-" * 30)

    return argparse.Namespace(**yaml_config)


if __name__ == "__main__":
    config = parse_args()
    print(config)
