# save_hf_to_composer_ckpt.py
import os
import torch
from transformers import AutoConfig, AutoModelForMaskedLM
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

# 1) Pull down the HF repo (caches under HF hub cache)
repo_dir = snapshot_download(
    repo_id="answerdotai/ModernBERT-large",
    local_dir=None,
    local_dir_use_symlinks=False
)

# 2) Load the config and build an uninitialized model
config = AutoConfig.from_pretrained(repo_dir, local_files_only=True)
model = AutoModelForMaskedLM.from_config(config)

# 3) Load weights from safetensors
safetensors_path = os.path.join(repo_dir, "model.safetensors")
state_dict = load_safetensors(safetensors_path)
model.load_state_dict(state_dict, strict=False)

# 4) Wrap into a Composer checkpoint
composer_ckpt = {"state": {"model": model.state_dict()}}

# 5) **Save to your preferred location** (not in HF cache)
output_path = "/data/models/ModernBERT-large/modernbert_large_hf_ckpt.pt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save(composer_ckpt, output_path)

print(f"Composer checkpoint saved to {output_path}")
