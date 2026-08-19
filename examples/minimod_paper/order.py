#!/usr/bin/env python3
"""Print a deterministic randomized arm order for one allocation."""

import argparse
import random


STANDARD = [
    "default_serial",
    "default_overlap",
    "trigger_serial",
    "trigger_overlap",
    "proxy_serial",
    "proxy_overlap",
]
O5 = ["mirrored_serial", "unmirrored_serial"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("standard", "o5"))
    ap.add_argument("seed")
    args = ap.parse_args()
    arms = list(STANDARD if args.kind == "standard" else O5)
    random.Random(args.seed).shuffle(arms)
    print(",".join(arms))


if __name__ == "__main__":
    main()
