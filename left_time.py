import ast
from datetime import datetime, timedelta

log_path = "/home/wzk/UFO/ufo_fb_lafan1_mini3_7gpu_buf1500k.log"
total_steps = 192_000_000

with open(log_path, errors="replace") as f:
    lines = [
        line for line in f
        if "distributed/global_env_steps" in line and "FPS" in line
    ]

data = ast.literal_eval(lines[-1][lines[-1].index("{"):])
steps = data["distributed/global_env_steps"]
fps = data["FPS"]
remaining_seconds = max(total_steps - steps, 0) / fps
finish_time = datetime.now() + timedelta(seconds=remaining_seconds)

print(f"进度: {steps:,}/{total_steps:,} ({steps / total_steps:.2%})")
print(f"速度: {fps:.1f} FPS")
print(f"预计剩余: {remaining_seconds / 3600:.2f} 小时")
print(f"理论结束时间: {finish_time:%Y-%m-%d %H:%M:%S}")
