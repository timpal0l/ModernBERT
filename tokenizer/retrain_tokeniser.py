import json
from transformers import AutoTokenizer

base_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
vocab_size = len(base_tokenizer)

def get_text_iterator(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "text" in data:
                    yield data["text"]
            except json.JSONDecodeError:
                continue

jsonl_path = "tokenizer_samples.jsonl"

new_tokenizer = base_tokenizer.train_new_from_iterator(
    get_text_iterator(jsonl_path),
    vocab_size=vocab_size
)

new_tokenizer.save_pretrained("modernbert_tokenizer_custom")
