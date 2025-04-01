import os
import argparse
import orjson
import concurrent.futures
from itertools import chain
import numpy as np
from datasets import Dataset, concatenate_datasets, disable_progress_bar
from transformers import AutoTokenizer

# Enable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "true"
disable_progress_bar()

from streaming.base import MDSWriter, StreamingDataset

ALLOWED_LANGS = {"sv", "da", "no", "nn", "is"}
REQUIRED_KEYS = {"lang", "keep", "text"}

def get_jsonl_files(dirs, dry_run=False):
    files = []
    for d in dirs:
        for root, _, filenames in os.walk(d):
            for f in filenames:
                if f.endswith(".jsonl"):
                    filepath = os.path.join(root, f)
                    if "oscar" in filepath.lower() or "mc4" in filepath.lower():
                        continue
                    files.append(filepath)
    if dry_run:
        files = files[:1]
    return files

def load_jsonl_file(file_path):
    with open(file_path, "rb") as f:
        for line in f:
            try:
                rec = orjson.loads(line)
            except Exception:
                continue
            if not REQUIRED_KEYS.issubset(rec.keys()):
                continue
            yield {"lang": rec["lang"], "keep": rec["keep"], "text": rec["text"]}

def load_file_dataset(file_path):
    return Dataset.from_generator(lambda: load_jsonl_file(file_path))

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], add_special_tokens=True)
    return {"input_ids": outputs["input_ids"]}

def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_seq_length) * max_seq_length
    return {
        k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }

def finalize_example(batch, indices):
    new_ids = [str(idx) for idx in indices]
    batch["id"] = new_ids
    return batch

def write_mds(dataset, output_dir):
    columns = {"id": "str", "input_ids": "ndarray:uint16"}
    with MDSWriter(
        out=output_dir,
        columns=columns,
        size_limit=1 << 26,
        compression=None,
        hashes=[]
    ) as writer:
        for sample in dataset:
            sample["input_ids"] = np.array(sample["input_ids"]).astype(np.uint16)
            writer.write(sample)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--dry_run", action="store_true", help="Process only one file and 500 samples for testing")
    parser.add_argument("--cores", type=int, default=16, help="Number of processing cores to use")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size for mapping functions")
    args = parser.parse_args()

    global max_seq_length, tokenizer
    max_seq_length = args.max_seq_length
    tokenizer = AutoTokenizer.from_pretrained("AI-Sweden-Models/ModernBERT-large")
    
    files = get_jsonl_files(args.input_dirs, dry_run=args.dry_run)
    if not files:
        raise ValueError("No JSONL files found.")
    
    datasets_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.cores) as executor:
        futures = {executor.submit(load_file_dataset, file): file for file in files}
        for future in concurrent.futures.as_completed(futures):
            try:
                ds = future.result()
                datasets_list.append(ds)
            except Exception:
                pass
    
    if not datasets_list:
        raise ValueError("No datasets loaded successfully.")
    
    raw_datasets = concatenate_datasets(datasets_list)
    raw_datasets = raw_datasets.filter(lambda x: x.get("keep") == 1 and x.get("lang") in ALLOWED_LANGS)
    if args.dry_run:
        n_samples = min(500, len(raw_datasets))
        raw_datasets = raw_datasets.select(range(n_samples))
    raw_datasets = raw_datasets.shuffle(seed=42)
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=["lang", "keep", "text"],
        num_proc=args.cores,
    )
    
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.cores,
    )
    
    tokenized_datasets = tokenized_datasets.map(
        finalize_example,
        with_indices=True,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.cores,
    )
    
    write_mds(tokenized_datasets, args.output_dir)

if __name__ == "__main__":
    main()
