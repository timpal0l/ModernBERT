import os
import argparse
import concurrent.futures
from datasets import load_from_disk, concatenate_datasets, Dataset
import numpy as np
from transformers import AutoTokenizer

from streaming.base import MDSWriter, StreamingDataset

def get_parquet_files(dirs, dry_run=False):
    print("Scanning for Parquet files in the provided directories...")
    files = []
    for d in dirs:
        for root, _, filenames in os.walk(d):
            for f in filenames:
                if f.endswith(".parquet"):
                    filepath = os.path.join(root, f)
                    files.append(filepath)
    if dry_run:
        files = files[:1]
    print(f"Found {len(files)} Parquet files.")
    return files

def load_parquet_dataset(file_path):
    print(f"Loading dataset from Parquet file: {file_path}")
    ds =  Dataset.from_parquet(file_path)
    print(f"Finished loading dataset from Parquet file: {file_path}.")
    return ds

def remove_extra_columns(ds, columns_to_keep):
    """Remove all columns from ds that are not in columns_to_keep."""
    cols_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
    print("Removing extra columns:", cols_to_remove)
    return ds.remove_columns(cols_to_remove)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nordic_dataset_path", type=str, required=True,
                        help="Path to the pre-saved Nordic Pile filtered dataset (which currently has many fields).")
    parser.add_argument("--parquet_dirs", nargs="+", default=[],
                        help="Directories containing SWEB Parquet files (assumed already cleaned).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the final concatenated dataset.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process only one Parquet file and limit samples for testing.")
    parser.add_argument("--cores", type=int, default=16,
                        help="Number of processing cores to use (for loading Parquet files).")
    args = parser.parse_args()

    # Load the pre-filtered Nordic dataset from disk.
    print(f"Loading Nordic dataset from {args.nordic_dataset_path} ...")
    nordic_ds = load_from_disk(args.nordic_dataset_path)
    print(f"Nordic dataset loaded; contains {len(nordic_ds)} samples with columns: {nordic_ds.column_names}")

    swb_datasets = []
    if args.parquet_dirs:
        parquet_files = []
        for d in args.parquet_dirs:
            parquet_files.extend(get_parquet_files([d], dry_run=args.dry_run))
        if parquet_files:
            print("Loading SWEB Parquet datasets concurrently...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.cores) as executor:
                futures = {executor.submit(load_parquet_dataset, file): file for file in parquet_files}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        ds = future.result()
                        swb_datasets.append(ds)
                    except Exception as e:
                        print(f"Error loading Parquet file {futures[future]}: {e}")
            print(f"Loaded {len(swb_datasets)} SWEB datasets.")
        else:
            print("No SWEB Parquet files found in the provided directories.")
    else:
        print("No SWEB Parquet directories provided.")

    # If SWEB datasets were loaded, remove extra columns so that only "text" remains.
    if swb_datasets:
        print("Processing SWEB datasets to keep only the 'text' field...")
        processed_sweb = [remove_extra_columns(ds, ["text"]) for ds in swb_datasets]
        swb_ds = concatenate_datasets(processed_sweb)
        print(f"SWEB dataset after processing has {len(swb_ds)} samples with columns: {swb_ds.column_names}")
        # Concatenate the Nordic dataset with the SWEB dataset.
        print("Concatenating Nordic and SWEB datasets...")
        final_ds = concatenate_datasets([nordic_ds, swb_ds])
    else:
        print("No SWEB datasets loaded; using only Nordic dataset.")
        final_ds = nordic_ds

    print(f"Final concatenated dataset contains {len(final_ds)} samples with columns: {final_ds.column_names}")

    # Remove extra columns so that ONLY "text" remains.
    print("Removing extra columns. Keeping only 'text'...")
    final_ds = remove_extra_columns(final_ds, ["text"])
    print(f"Final dataset now has columns: {final_ds.column_names}")

    if args.dry_run:
        n_samples = min(500, len(final_ds))
        print(f"Dry run enabled. Selecting the first {n_samples} samples...")
        final_ds = final_ds.select(range(n_samples))
        print(f"Dataset reduced to {len(final_ds)} samples for dry run.")

    print(f"Saving final dataset to disk at {args.output_dir} ...")
    final_ds.save_to_disk(args.output_dir)
    print("Save complete.")

if __name__ == "__main__":
    main()
