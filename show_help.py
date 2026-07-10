#!/usr/bin/env python3

"""Print welcome message and quickstart hints."""


def main() -> None:
    title = [
        " _    _   ______   ____ ",
        "| |  | | |  ____| / __ \\",
        "| |  | | | |__   | |  | |",
        "| |  | | |  __|  | |  | |",
        "| |__| | | |     | |__| |",
        " \\____/  |_|      \\____/ ",
    ]

    ufo = [
        "                 .-\"\"\"-.",
        "               .'  .-.  '.",
        "              /   (o o)   \\",
        "             |  .-`---'-.  |",
        "             | /  UFO!!! \\ |",
        "              \\|,  ___  ,|/",
        "               \\   ---   /",
        "            _.-'\\  ___  /'-._",
        "          .'___  `-----'  ___'.",
        "         /____/-----------\\____\\",
        "            /_/           \\_\\",
    ]

    pad = max(len(line) for line in title) + 4
    rows = max(len(title), len(ufo))
    for i in range(rows):
        left = title[i] if i < len(title) else ""
        right = ufo[i] if i < len(ufo) else ""
        print(left.ljust(pad) + right)
    print()
    print("UFO is coming, start your training now!")
    print()
    print("Quick help:")
    print("- Training file: humanoidverse/train_mjlab.py")
    print("- Inference files:")
    print("  - humanoidverse/tracking_inference.py")
    print("  - humanoidverse/goal_inference.py")
    print("  - humanoidverse/reward_inference.py")


if __name__ == "__main__":
    main()
