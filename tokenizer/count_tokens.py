
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from streaming.base import StreamingDataset, MDSWriter

def rewrite_shards(src_dir: str,
                   dst_dir: str,
                   batch_size: int = 1024,
                   num_workers: int = 8):
    """
    Stage 1: Read MDS with id+input_ids and write new MDS with only input_ids.
    Shows a per-sample progress bar.
    """
    src = Path(src_dir).resolve()
    dst = Path(dst_dir).resolve()
    if src == dst:
        raise ValueError("Destination must be different from source!")
    if dst.exists() and any(dst.iterdir()):
        raise ValueError("Destination exists and is not empty!")

    ds = StreamingDataset(local=str(src), batch_size=batch_size)
    dst.mkdir(parents=True, exist_ok=True)
    with MDSWriter(out=str(dst), columns={"input_ids": "bytes"}) as writer:
        for sample in tqdm(
            ds,
            total=ds.size,
            desc="Rewriting samples",
            unit="samp",
            miniters=1,        # redraw every sample
            mininterval=0.5,   # but no faster than twice a second
            smoothing=0.1,
        ):
            # MDSWriter expects raw bytes when column is "bytes"
            raw = sample["input_ids"].tobytes()
            writer.write({"input_ids": raw})

def count_tokens_via_offsets(mds_dir: str,
                             chunk_size: int = 10_000_000) -> int:
    """
    Stage 2: mmap each shard's uint32 offset table, diff() and sum() to get
    exact token count. Shows a shard-level bar plus inner per-chunk bar.
    """
    idx_path = Path(mds_dir) / "index.json"
    with open(idx_path, "r") as f:
        idx = json.load(f)

    total_tokens = 0
    for shard in tqdm(
        idx["shards"],
        desc="Shards",
        unit="shard",
        mininterval=0.5,
        smoothing=0.1,
    ):
        fn        = shard["raw_data"]["basename"]
        n_samples = shard["samples"]
        path      = os.path.join(mds_dir, fn)

        offsets = np.memmap(
            path,
            mode="r",
            dtype="<u4",            # uint32 offsets
            shape=(n_samples + 1,),
            offset=0
        )

        shard_bytes = 0
        for start in tqdm(
            range(0, n_samples, chunk_size),
            desc=f"  {fn}",
            unit="chunk",
            leave=False,
            mininterval=0.5,
            smoothing=0.1,
        ):
            end = min(start + chunk_size + 1, n_samples + 1)
            shard_bytes += int(np.diff(offsets[start:end]).sum())

        total_tokens += shard_bytes // 2

    return total_tokens

def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Fast exact MDS token count (with progress bars)"
    )
    p.add_argument("src_dir", help="Original shard dir (with id+input_ids)")
    p.add_argument("dst_dir", help="Empty scratch dir for pure-input_ids shards")
    args = p.parse_args()

    try:
        t0 = time.time()
        rewrite_shards(
            src_dir=args.src_dir,
            dst_dir=args.dst_dir,
            batch_size=1024,
            num_workers=8,
        )
        print(f"\nRewriting took {time.time() - t0:.1f}s")

        t1 = time.time()
        total = count_tokens_via_offsets(args.dst_dir)
        print(f"\nExact total tokens: {total:,}")
        print(f"Counting took {time.time() - t1:.1f}s")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 
    # usage: ./count_tokens.py /data/gigapile/mds_shards /data/gigapile/mds_shards_noid
