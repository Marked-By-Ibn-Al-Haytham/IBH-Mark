#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys


DEFAULT_OUTPUT = "swin_tiny_patch4_window7_224.pth"
DEFAULT_FILE_LINK = "https://drive.google.com/file/d/1TyMf0_uvaxyacMmVzRfqvLLAWSOE2bJR/view?usp=drive_link"


def main() -> int:

    try:
        import gdown
    except ImportError:
        print(
            "Error: gdown is not installed. Run `pip install -r requirements.txt` "
            "or `pip install gdown` first.",
            file=sys.stderr,
        )
        return 1


    output_path = Path(DEFAULT_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Swin checkpoint to: {output_path}")
    gdown.download(
        url=DEFAULT_FILE_LINK,
        output=str(output_path),
        quiet=False,
    )

    if not output_path.exists() or output_path.stat().st_size == 0:
        print("Error: download failed or produced an empty file.", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
