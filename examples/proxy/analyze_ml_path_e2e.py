#!/usr/bin/env python3
"""Validate and compare the real default and GBT-compiled run logs."""

import argparse
import math
import re
import statistics
from pathlib import Path


RESULT = re.compile(
    r"ML_PATH_RESULT runs=(\d+) total_us=([0-9.]+) mean_us=([0-9.]+) "
    r"median_us=([0-9.]+) p25_us=([0-9.]+) p75_us=([0-9.]+) "
    r"staged=(\d+) pushed=(\d+) data=(\w+)")
CHECK = re.compile(r"ML_PATH_CHECKSUM rank=1 fnv64=([0-9a-f]+) data=(\w+)")


def parse(path: Path):
    text = path.read_text()
    result = RESULT.search(text)
    check = CHECK.search(text)
    if not result or not check:
        raise SystemExit(f"{path}: missing result/checksum record")
    runs, total, mean, median, p25, p75, staged, pushed, data = result.groups()
    checksum, check_data = check.groups()
    if data != "OK" or check_data != "OK":
        raise SystemExit(f"{path}: payload verification failed")
    return {
        "runs": int(runs), "total_us": float(total), "mean_us": float(mean),
        "median_us": float(median), "p25_us": float(p25),
        "p75_us": float(p75),
        "staged": int(staged), "pushed": int(pushed), "checksum": checksum,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "logs", nargs="+", type=Path,
        help="one or more DEFAULT GBT log pairs")
    args = ap.parse_args()
    if len(args.logs) % 2:
        raise SystemExit("logs must be DEFAULT GBT pairs")

    results = []
    for pair_index, (default_path, gbt_path) in enumerate(
            zip(args.logs[0::2], args.logs[1::2]), start=1):
        default, gbt = parse(default_path), parse(gbt_path)
        if default["runs"] != gbt["runs"]:
            raise SystemExit(f"pair {pair_index}: sample counts differ")
        if default["checksum"] != gbt["checksum"]:
            raise SystemExit(f"pair {pair_index}: payload checksums differ")
        runs = default["runs"]
        expected_staged = runs * 64
        expected_pushed = runs * (64 + 8)  # puts + one quiet/issuing block
        if (default["staged"], default["pushed"]) != (expected_staged, 0):
            raise SystemExit(
                f"pair {pair_index}: default route counters differ from "
                f"64 staged puts/phase: {default}")
        if (gbt["staged"], gbt["pushed"]) != (0, expected_pushed):
            raise SystemExit(
                f"pair {pair_index}: GBT route counters differ from "
                f"64 puts + 8 quiets/phase: {gbt}")
        speedup = default["median_us"] / gbt["median_us"]
        results.append((default, gbt, speedup))
        print(f"pair {pair_index}:")
        print(f"  default {default['median_us']:.3f} us median "
              f"(IQR {default['p25_us']:.3f}--{default['p75_us']:.3f}; "
              f"staged={default['staged']}, pushed={default['pushed']})")
        print(f"  GBT      {gbt['median_us']:.3f} us median "
              f"(IQR {gbt['p25_us']:.3f}--{gbt['p75_us']:.3f}; "
              f"staged={gbt['staged']}, pushed={gbt['pushed']})")
        print(f"  speedup  {speedup:.4f}x, checksum={default['checksum']}")

    checksums = {x[0]["checksum"] for x in results} | \
                {x[1]["checksum"] for x in results}
    if len(checksums) != 1:
        raise SystemExit("checksums differ across repetitions")
    speedups = [x[2] for x in results]
    default_median = statistics.median(x[0]["median_us"] for x in results)
    gbt_median = statistics.median(x[1]["median_us"] for x in results)
    print(f"aggregate: median(default medians)={default_median:.3f} us, "
          f"median(GBT medians)={gbt_median:.3f} us")
    print(f"aggregate speedup={default_median / gbt_median:.4f}x; "
          f"pair geometric mean={math.prod(speedups) ** (1 / len(speedups)):.4f}x; "
          f"range={min(speedups):.4f}--{max(speedups):.4f}x")
    if any(x <= 1.0 for x in speedups):
        raise SystemExit("GBT did not beat the compiler default in every pair")


if __name__ == "__main__":
    main()
