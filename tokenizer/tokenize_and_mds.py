import os
import argparse
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from streaming.base import MDSWriter, StreamingDataset

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], add_special_tokens=True)
    return {"input_ids": outputs["input_ids"]}

def write_mds(dataset, output_dir):
    print(f"Writing MDS shards to output directory: {output_dir} ...")
    columns = {"id": "str", "input_ids": "ndarray:uint16"}
    with MDSWriter(
        out=output_dir,
        columns=columns,
        size_limit=1 << 26,
        compression=None,
        hashes=[]
    ) as writer:
        idx = 0
        for sample in dataset:

            if "id" not in sample:
                sample["id"] = str(idx)

            sample["input_ids"] = np.array(sample["input_ids"]).astype(np.uint16)
            writer.write(sample)
            idx += 1
    print("Finished writing MDS shards.")

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize dataset (using an iterable/streaming dataset) and convert to Mosaic Data Shards (MDS)"
    )
    parser.add_argument("--input_dataset_path", type=str, required=True,
                        help="Path to the pre-saved dataset (containing only the 'text' field)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where the Mosaic Data Shards will be saved")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for tokenization mapping")
    args = parser.parse_args()

    print(f"Loading dataset from disk: {args.input_dataset_path}")
    ds = load_from_disk(args.input_dataset_path)
    print(f"Dataset loaded with {len(ds)} samples (memory mapped).")
    
    stream_ds = ds.to_iterable_dataset()
    print("Converted dataset to an iterable (streaming mode).")

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("AI-Sweden-Models/ModernBERT-large", use_fast=True)

    print("Tokenizing dataset in streaming mode...")

    tokenized_stream_ds = stream_ds.map(
        tokenize_function,
        batched=True,
        batch_size=args.batch_size
    )
    print("Tokenization complete.")

    write_mds(tokenized_stream_ds, args.output_dir)

if __name__ == "__main__":
    main()
