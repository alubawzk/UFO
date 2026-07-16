import argparse
import ast
from datetime import datetime, timedelta
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate the remaining training time from a UFO log.")
    parser.add_argument("log", type=Path, help="Path to the training log file.")
    parser.add_argument("--total-steps", type=int, default=192_000_000, help="Target number of global environment steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_steps <= 0:
        raise SystemExit(f"--total-steps 必须大于 0，当前为: {args.total_steps}")

    latest_metrics = None
    with args.log.open(errors="replace") as log_file:
        for line in log_file:
            if "distributed/global_env_steps" in line and "FPS" in line:
                latest_metrics = line

    if latest_metrics is None:
        raise SystemExit(f"未在日志中找到训练进度记录: {args.log}")

    data = ast.literal_eval(latest_metrics[latest_metrics.index("{") :])
    steps = data["distributed/global_env_steps"]
    fps = data["FPS"]
    if fps <= 0:
        raise SystemExit(f"日志中的 FPS 必须大于 0，当前为: {fps}")

    remaining_seconds = max(args.total_steps - steps, 0) / fps
    finish_time = datetime.now() + timedelta(seconds=remaining_seconds)

    print(f"进度: {steps:,}/{args.total_steps:,} ({steps / args.total_steps:.2%})")
    print(f"速度: {fps:.1f} FPS")
    print(f"预计剩余: {remaining_seconds / 3600:.2f} 小时")
    print(f"理论结束时间: {finish_time:%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
