"""Write an explicit local JSON configuration; does not launch jobs."""
import argparse
import json
from pathlib import Path

from driversi.configuration import read_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = read_mapping(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {args.output}; configure external resources before execution.")


if __name__ == "__main__":
    main()
