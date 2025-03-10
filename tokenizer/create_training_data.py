#!/usr/bin/env python
# coding=utf-8
"""
This script processes the full combined dataset from three sources in parallel:

  • SWEb (local parquet files under /data/sweb/data, organized in YEAR-MONTH subfolders)
  • Nordic Pile (local JSONL files under /data/datasets/nordic_pile/cleaned),
      excluding files whose names end with "_en.jsonl" or contain "commoncrawl"
  • Hermès (from Hugging Face)

Each source is processed in its own process (using multiprocessing). Each process streams
its data (so the entire dataset is never loaded into memory), tokenizes each "text" field using 
the pretrained tokenizer "AI-Sweden-Models/ModernBERT-large", and splits (packs) the tokenized text 
into chunks of exactly 8192 tokens. For a short text, one chunk is produced as:

    [CLS] + tokens + [SEP]

For texts longer than the available capacity (i.e. > BLOCK_SIZE–2 tokens), the text is split so that:
  - The first chunk has [CLS] at the beginning.
  - Intermediate chunks contain tokens only.
  - The final chunk gets a [SEP] at the end (appended if room; otherwise, replacing the last token).

Each chunk is written immediately to an output file (as one JSONL record). After processing all sources,
the outputs are merged into a final dataset and overall statistics (total examples, average/min/max sequence lengths) are computed.
"""

import os
import json
import glob
import re
import random
from itertools import chain
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import concurrent.futures

# ----------------------------
# Configuration
# ----------------------------
BLOCK_SIZE = 8192  # Maximum sequence length (including special tokens)

# SWEb: local directory with YEAR-MONTH subfolders containing parquet files.
SWEB_ROOT = "/data/sweb/data"

# Nordic Pile: local directory with merged JSONL files.
NORDIC_PILE_ROOT = "/data/datasets/nordic_pile/cleaned"

# Hermès dataset from Hugging Face.
HERMES_DATASET_PATH = "timpal0l/hermes_scandeval_dedup_detokenized"

# Nordic Pile file filtering: exclude any JSONL file whose base filename ends with "_en.jsonl"
# or whose full path contains "commoncrawl" (case-insensitive).
# (Adjust these rules as needed.)
# Output file names for each source:
OUTPUT_SWEB = "output_sweb.jsonl"
OUTPUT_NORDIC = "output_nordic.jsonl"
OUTPUT_HERMES = "output_hermes.jsonl"
FINAL_OUTPUT = "full_tokenized_dataset.jsonl"
STATS_FILE = "full_dataset_stats.json"

# Tokenizer name.
TOKENIZER_NAME = "AI-Sweden-Models/ModernBERT-large"

# ----------------------------
# Load the tokenizer (global)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.cls_token is None or tokenizer.sep_token is None:
    raise ValueError("Tokenizer must have both a CLS token and a SEP token defined.")
cls_id = tokenizer.cls_token_id
sep_id = tokenizer.sep_token_id

# ----------------------------
# Tokenization & Chunking Functions
# ----------------------------
def process_sample(token_ids, block_size, cls_id, sep_id):
    """
    Splits a list of token IDs (from one text sample) into chunks of length <= block_size.
    - If len(token_ids) + 2 <= block_size, output one chunk: [CLS] + token_ids + [SEP].
    - Otherwise:
         • First chunk: [CLS] + token_ids[:block_size-1]
         • Intermediate chunks: full chunks (no special tokens)
         • Final chunk: if len(remaining)+1 <= block_size, output remaining + [SEP],
                       otherwise, replace the last token with [SEP].
    Each chunk is truncated to BLOCK_SIZE as a safety check.
    Returns a list of chunks (each a list of ints).
    """
    chunks = []
    if len(token_ids) + 2 <= block_size:
        chunk = [cls_id] + token_ids + [sep_id]
        chunks.append(chunk[:block_size])
    else:
        first_chunk = [cls_id] + token_ids[:block_size - 1]
        chunks.append(first_chunk[:block_size])
        remaining = token_ids[block_size - 1:]
        while len(remaining) > block_size:
            chunk = remaining[:block_size]
            chunks.append(chunk[:block_size])
            remaining = remaining[block_size:]
        if remaining:
            if len(remaining) + 1 <= block_size:
                final_chunk = remaining + [sep_id]
            else:
                final_chunk = remaining[:-1] + [sep_id]
            chunks.append(final_chunk[:block_size])
    return chunks

def tokenize_and_chunk(example):
    """
    Tokenizes the "text" field (without adding special tokens) and splits it into chunks.
    Returns a dictionary with key "input_ids_chunks" containing a list of token ID lists.
    """
    text = example.get("text", "").strip()
    if not text:
        return {"input_ids_chunks": []}
    tokenized = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks = process_sample(tokenized, BLOCK_SIZE, cls_id, sep_id)
    return {"input_ids_chunks": chunks}

# ----------------------------
# File Listing Functions
# ----------------------------
def get_sweb_files(sweb_root):
    """
    Returns a list of all parquet file paths in the SWEb directory.
    """
    return glob.glob(os.path.join(sweb_root, "**", "*.parquet"), recursive=True)

def get_nordic_files(nordic_root):
    """
    Recursively returns a list of all JSONL file paths in Nordic Pile,
    excluding any file whose base filename ends with "_en.jsonl" or whose path contains "commoncrawl" (case-insensitive).
    """
    files = []
    for f in glob.glob(os.path.join(nordic_root, "**", "*.jsonl"), recursive=True):
        fname = os.path.basename(f).lower()
        if fname.endswith("_en.jsonl") or "commoncrawl" in f.lower():
            continue
        files.append(f)
    return files

# ----------------------------
# Generators for Each Source (Streaming)
# ----------------------------
def generator_sweb():
    """Yields examples from SWEb by loading local parquet files."""
    files = get_sweb_files(SWEB_ROOT)
    if files:
        ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
        for ex in ds:
            yield ex

def generator_nordic():
    """Yields examples from Nordic Pile by loading local JSONL files."""
    files = get_nordic_files(NORDIC_PILE_ROOT)
    if files:
        ds = load_dataset("json", data_files=files, split="train", streaming=True)
        for ex in ds:
            yield ex

def generator_hermes():
    """Yields examples from the Hermès dataset (Hugging Face)."""
    ds = load_dataset(HERMES_DATASET_PATH, split="train", streaming=True)
    for ex in ds:
        yield ex

# ----------------------------
# Processing Function for a Source
# ----------------------------
def process_source(generator_func, output_filename):
    """
    Processes examples from a given generator (for one source), tokenizes and chunks them,
    writes each chunk as a JSONL record to output_filename, and collects basic statistics.
    Returns a stats dictionary.
    """
    total_examples = 0
    total_tokens = 0
    min_tokens = float('inf')
    max_tokens = 0
    with open(output_filename, "w", encoding="utf-8") as out_f:
        for example in tqdm(generator_func(), desc=f"Processing {output_filename}"):
            if "text" not in example:
                continue
            tokenized = tokenize_and_chunk(example)
            chunks = tokenized.get("input_ids_chunks", [])
            for chunk in chunks:
                chunk = chunk[:BLOCK_SIZE]  # ensure safety
                out_f.write(json.dumps({"input_ids": chunk}, ensure_ascii=False) + "\n")
                l = len(chunk)
                total_examples += 1
                total_tokens += l
                if l < min_tokens:
                    min_tokens = l
                if l > max_tokens:
                    max_tokens = l
    avg_tokens = total_tokens / total_examples if total_examples > 0 else 0
    return {
        "total_examples": total_examples,
        "avg_tokens": avg_tokens,
        "min_tokens": int(min_tokens) if min_tokens != float('inf') else 0,
        "max_tokens": int(max_tokens)
    }

# ----------------------------
# Merge Output Files and Aggregate Statistics
# ----------------------------
def merge_outputs(output_files, final_output):
    """
    Merges several JSONL output files into one final output file and aggregates statistics.
    Returns aggregated stats.
    """
    total_examples = 0
    total_tokens = 0
    min_tokens = float('inf')
    max_tokens = 0
    with open(final_output, "w", encoding="utf-8") as fout:
        for fname in output_files:
            with open(fname, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    try:
                        record = json.loads(line)
                        tokens = record.get("input_ids", [])
                        l = len(tokens)
                        total_examples += 1
                        total_tokens += l
                        if l < min_tokens:
                            min_tokens = l
                        if l > max_tokens:
                            max_tokens = l
                    except Exception:
                        continue
    avg_tokens = total_tokens / total_examples if total_examples > 0 else 0
    return {
        "total_examples": total_examples,
        "avg_tokens": avg_tokens,
        "min_tokens": int(min_tokens) if min_tokens != float('inf') else 0,
        "max_tokens": int(max_tokens)
    }

# ----------------------------
# Main Execution: Parallel Processing
# ----------------------------
def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = {
            "sweb": executor.submit(process_source, generator_sweb, OUTPUT_SWEB),
            "nordic": executor.submit(process_source, generator_nordic, OUTPUT_NORDIC),
            "hermes": executor.submit(process_source, generator_hermes, OUTPUT_HERMES)
        }
        results = {}
        for key, future in futures.items():
            try:
                stats = future.result()
                results[key] = stats
                print(f"{key} stats: {stats}")
            except Exception as e:
                print(f"Error processing {key}: {e}")
    # Merge outputs from all sources.
    output_files = [OUTPUT_SWEB, OUTPUT_NORDIC, OUTPUT_HERMES]
    merged_stats = merge_outputs(output_files, FINAL_OUTPUT)
    print("Final merged dataset statistics:", merged_stats)
    with open(STATS_FILE, "w", encoding="utf-8") as stat_f:
        json.dump(merged_stats, stat_f, indent=4)
    print(f"Final merged dataset saved to {FINAL_OUTPUT}")
    print(f"Statistics saved to {STATS_FILE}")

if __name__ == "__main__":
    main()
