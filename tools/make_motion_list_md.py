import argparse
from pathlib import Path
import joblib


def load_motions(data):
    """
    兼容你的 BFM-Zero 数据格式：
    目前从文件内容看，它大概率是：
    {
        "soma/xxx/motion_name": {
            "root_trans_offset": ...,
            ...
        },
        ...
    }
    所以 motion name 直接取 dict 的 key。
    """
    if isinstance(data, dict):
        return [(i, str(name)) for i, name in enumerate(data.keys())]

    if isinstance(data, list):
        names = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                for key in ["name", "motion_name", "file_name", "filename", "path"]:
                    if key in item:
                        names.append((i, str(item[key])))
                        break
                else:
                    names.append((i, f"motion_{i:06d}"))
            else:
                names.append((i, str(item)))
        return names

    raise TypeError(f"Unsupported data type: {type(data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to input pkl")
    parser.add_argument("--output", required=True, help="path to output md")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    data = joblib.load(input_path)
    motions = load_motions(data)

    lines = []
    lines.append(f"source pkl: `{args.input}`")
    lines.append(f"- total motions: `{len(motions)}`")
    lines.append("")
    lines.append("| id | name |")
    lines.append("|---:|:-----|")

    for idx, name in motions:
        name = str(name).replace("|", "\\|")
        lines.append(f"| {idx} | `{name}` |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved md: {output_path}")
    print(f"Total motions: {len(motions)}")


if __name__ == "__main__":
    main()



    # uv run python tools/make_motion_list_md.py \
#   --input /data/xue/bfmzero/data/mixed_dataset.pkl \
#   --output /data/xue/bfmzero/data/motion_list_mixed_dataset.md