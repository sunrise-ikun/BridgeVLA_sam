"""Scan all .pkl files under the dataset root and report corrupted ones.

Usage:
    python scan_corrupt_pkl.py [data_root]

Default data_root:
    /DATA/disk1/zyz/projects/BridgeVLA_sam/data/bridgevla_data/Real
"""

import os
import pickle
import sys
from pathlib import Path


def scan(data_root: str):
    data_root = Path(data_root)
    if not data_root.is_dir():
        print(f"ERROR: {data_root} is not a directory")
        sys.exit(1)

    total = 0
    corrupt = []

    pkl_files = sorted(data_root.rglob("*.pkl"))
    print(f"Found {len(pkl_files)} .pkl files under {data_root}")

    for p in pkl_files:
        total += 1
        try:
            with open(p, "rb") as f:
                pickle.load(f)
        except Exception as e:
            corrupt.append((str(p), type(e).__name__, str(e)))
            print(f"  CORRUPT: {p}  ({type(e).__name__}: {e})")

    print(f"\nScanned {total} files, found {len(corrupt)} corrupt file(s).")
    if corrupt:
        print("\nCorrupt file list:")
        for path, etype, msg in corrupt:
            print(f"  {path}")
    else:
        print("All files are OK.")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else (
        "/DATA/disk1/zyz/projects/BridgeVLA_sam/data/bridgevla_data/Real"
    )
    scan(root)
