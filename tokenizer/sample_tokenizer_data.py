import os
import re
import glob
import json
import random

from datasets import load_dataset
from tqdm.auto import tqdm

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------

# 1) SWEb (on disk)
SWEB_ROOT = "/data/sweb/data"              # Where your SWEb subfolders are
DESIRED_LANGS = {"sv", "da", "no", "is"}   # Swedish, Danish, Norwegian, Icelandic
TARGET_SAMPLES_PER_YEAR = 100_000          # How many lines per year from SWEb

# 2) Nordic Pile (on disk)
NORDIC_PILE_ROOT = "/data/datasets/nordic_pile/cleaned"
TARGET_SAMPLES_NORDIC = 1_000_000          # How many lines total from Nordic Pile

# 3) Hermès dataset from Hugging Face
#    (We’ll stream the entire dataset, ~1GB, no sampling limit)
HERMES_DATASET_PATH = "timpal0l/hermes_scandeval_dedup_detokenized"

# 4) We skip any file in Nordic Pile whose path contains these substrings (case-insensitive)
EXCLUDE_KEYWORDS = ("mc4", "oscar")

# 5) Output file for final combined samples
OUTPUT_FILE = "tokenizer_samples.jsonl"

# 6) Checkpoint directory and file prefixes (one per dataset)
CHECKPOINT_DIR = "checkpoints"  # This directory must be writable.
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_SWEB_PREFIX = os.path.join(CHECKPOINT_DIR, "checkpoint_sweb_")  # e.g. checkpoint_sweb_2013.jsonl
CHECKPOINT_NORDIC = os.path.join(CHECKPOINT_DIR, "checkpoint_nordic.jsonl")
CHECKPOINT_HERMES = os.path.join(CHECKPOINT_DIR, "checkpoint_hermes.jsonl")


# ----------------------------------------------------------------
# CHECKPOINT HELPERS
# ----------------------------------------------------------------

def load_checkpoint(ckpt_file):
    """Load checkpoint from a JSONL file. Returns a list of dicts."""
    data = []
    if os.path.exists(ckpt_file):
        print(f"Loading checkpoint from {ckpt_file} ...")
        with open(ckpt_file, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def save_checkpoint(data, ckpt_file):
    """Save a list of dicts to a JSONL file as checkpoint."""
    print(f"Saving checkpoint to {ckpt_file} ...")
    with open(ckpt_file, "w", encoding="utf-8") as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------

def group_sweb_by_year(sweb_root):
    """
    Return a dict: {year: [subfolder paths]}, e.g. "2013" -> ["/data/sweb/data/2013-20", "/data/sweb/data/2013-48", ...]
    """
    year_folders = {}
    for folder_name in sorted(os.listdir(sweb_root)):
        match = re.match(r"^(\d{4})-", folder_name)  # e.g. "2013-20" => year=2013
        if match:
            year = match.group(1)
            abs_path = os.path.join(sweb_root, folder_name)
            if os.path.isdir(abs_path):
                year_folders.setdefault(year, []).append(abs_path)
    return year_folders


def sample_sweb_year(year_subfolders, desired_langs, target_samples=10000, seed=42):
    """
    Streams over all .parquet files in `year_subfolders` (processing each file individually).
    If a file raises an error during loading or iteration, it is skipped.
    Uses reservoir sampling up to `target_samples`.
    Returns a list of dicts: [{"text": ..., "lang": ..., "source": ...}, ...].
    """
    random.seed(seed)
    reservoir = []
    n_seen = 0

    for subfolder in year_subfolders:
        subfolder_name = os.path.basename(subfolder)  # e.g. "2013-20"
        parquet_files = glob.glob(os.path.join(subfolder, "**", "*.parquet"), recursive=True)
        if not parquet_files:
            continue

        for parquet_file in parquet_files:
            try:
                ds_stream = load_dataset(
                    "parquet",
                    data_files=parquet_file,
                    split="train",
                    streaming=True
                )
            except Exception as e:
                print(f"[WARN] Failed to load {parquet_file}:\n  {e}\nSkipping this file.")
                continue

            desc_str = f"SWEb {subfolder_name}/{os.path.basename(parquet_file)}"
            try:
                for example in tqdm(ds_stream, desc=desc_str, leave=True):
                    lang = example.get("language")
                    if lang in desired_langs:
                        txt = example.get("text", "").strip()
                        if not txt:
                            continue
                        n_seen += 1
                        if len(reservoir) < target_samples:
                            reservoir.append({
                                "text": txt,
                                "lang": lang,
                                "source": f"SWEb_{subfolder_name}"
                            })
                        else:
                            r = random.randint(0, n_seen - 1)
                            if r < target_samples:
                                reservoir[r] = {
                                    "text": txt,
                                    "lang": lang,
                                    "source": f"SWEb_{subfolder_name}"
                                }
            except Exception as e:
                print(f"[WARN] Error while iterating file {parquet_file}:\n  {e}\nSkipping remaining examples in this file.")
                continue

    return reservoir


def sample_nordic_pile(root_folder, desired_langs, exclude_keywords, target_samples=200000, seed=42):
    """
    Recursively scans all *.jsonl in `root_folder`.
    Skips any whose path contains e.g. 'mc4' or 'oscar' (case-insensitive).
    Uses reservoir sampling up to `target_samples`,
    returning a list of dicts: [{"text": ..., "lang": ..., "source": ...}, ...].
    """
    random.seed(seed)
    reservoir = []
    n_seen = 0

    jsonl_files = glob.glob(os.path.join(root_folder, "**", "*.jsonl"), recursive=True)
    for fpath in sorted(jsonl_files):
        lower_path = fpath.lower()
        if any(kw in lower_path for kw in exclude_keywords):
            print(f"Skipping (mc4/oscar) file: {fpath}")
            continue

        rel_path = os.path.relpath(fpath, root_folder)
        print(f"Reading Nordic Pile JSONL: {rel_path}")

        with open(fpath, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc=rel_path, mininterval=2):
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                lang = doc.get("lang")
                if lang in desired_langs:
                    text = doc.get("text", "").strip()
                    if not text:
                        continue
                    n_seen += 1
                    if len(reservoir) < target_samples:
                        reservoir.append({
                            "text": text,
                            "lang": lang,
                            "source": f"NordicPile_{rel_path}"
                        })
                    else:
                        r = random.randint(0, n_seen - 1)
                        if r < target_samples:
                            reservoir[r] = {
                                "text": text,
                                "lang": lang,
                                "source": f"NordicPile_{rel_path}"
                            }
    return reservoir


def load_hermes_dataset(dataset_path="timpal0l/hermes_scandeval_dedup_detokenized"):
    """
    Load the entire Hermès dataset in streaming mode from HuggingFace,
    storing all lines (since it's ~1GB).
    There's no explicit language field, so we label "lang" as "unknown".
    """
    ds_stream = load_dataset(dataset_path, split="train", streaming=True)
    hermes_samples = []

    for example in tqdm(ds_stream, desc="HermesDataset", mininterval=2):
        txt = example.get("text", "").strip()
        if txt:
            hermes_samples.append({
                "text": txt,
                "lang": "unknown",
                "source": "Hermes_Scandeval"
            })
    return hermes_samples


# ----------------------------------------------------------------
# MAIN EXECUTION WITH CHECKPOINTING
# ----------------------------------------------------------------

if __name__ == "__main__":
    combined = []

    # --- 1) SWEb Samples ---
    sweb_samples = []
    year_folders = group_sweb_by_year(SWEB_ROOT)
    for year in sorted(year_folders.keys()):
        ckpt_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_sweb_{year}.jsonl")
        # If a checkpoint exists for this year, load it.
        if os.path.exists(ckpt_file):
            year_samples = load_checkpoint(ckpt_file)
        else:
            print(f"\nSampling up to {TARGET_SAMPLES_PER_YEAR} lines for SWEb year {year}...")
            year_samples = sample_sweb_year(
                year_subfolders=year_folders[year],
                desired_langs=DESIRED_LANGS,
                target_samples=TARGET_SAMPLES_PER_YEAR,
                seed=42
            )
            save_checkpoint(year_samples, ckpt_file)
        print(f"  -> Collected {len(year_samples)} lines for year {year}")
        sweb_samples.extend(year_samples)
    print(f"\nTotal SWEb samples: {len(sweb_samples):,}\n")
    combined.extend(sweb_samples)

    # --- 2) Nordic Pile Samples ---
    if os.path.exists(CHECKPOINT_NORDIC):
        nordic_samples = load_checkpoint(CHECKPOINT_NORDIC)
    else:
        print(f"Sampling up to {TARGET_SAMPLES_NORDIC} lines from Nordic Pile (excluding 'mc4'/'oscar')...\n")
        nordic_samples = sample_nordic_pile(
            root_folder=NORDIC_PILE_ROOT,
            desired_langs=DESIRED_LANGS,
            exclude_keywords=EXCLUDE_KEYWORDS,
            target_samples=TARGET_SAMPLES_NORDIC,
            seed=42
        )
        save_checkpoint(nordic_samples, CHECKPOINT_NORDIC)
    print(f"\nTotal Nordic Pile samples (non-excluded): {len(nordic_samples):,}\n")
    combined.extend(nordic_samples)

    # --- 3) Hermès Samples ---
    if os.path.exists(CHECKPOINT_HERMES):
        hermes_samples = load_checkpoint(CHECKPOINT_HERMES)
    else:
        print("Loading entire Hermès dataset from HuggingFace...\n")
        hermes_samples = load_hermes_dataset(HERMES_DATASET_PATH)
        save_checkpoint(hermes_samples, CHECKPOINT_HERMES)
    print(f"Total Hermès samples: {len(hermes_samples):,}\n")
    combined.extend(hermes_samples)

    # --- 4) Combine & Shuffle ---
    random.shuffle(combined)
    print(f"Final combined sample count after shuffle: {len(combined):,}")

    # --- 5) Write Final Combined JSONL ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for item in combined:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done! Wrote {len(combined):,} lines to {OUTPUT_FILE}.")
