import json
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

# 1. Initialize a BPE tokenizer with an unknown token.
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 2. Set up normalization.
tokenizer.normalizer = normalizers.NFC()  # Preserves case and diacritics.

# 3. Use ByteLevel pre-tokenizer with add_prefix_space=True so that spaces are preserved.
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

# 4. Use ByteLevel decoder to properly reinsert spaces on decoding.
tokenizer.decoder = decoders.ByteLevel()

# 5. Define a BPE trainer with your desired vocabulary size and special tokens.
trainer = trainers.BpeTrainer(
    vocab_size=50368,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# 6. Define an iterator to yield batches from your JSONL file (using the "text" field).
def batch_iterator(file_path, batch_size=1000):
    with open(file_path, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            data = json.loads(line)
            text = data.get("text", "").strip()
            if text:
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# 7. Train the tokenizer on your corpus.
tokenizer.train_from_iterator(batch_iterator("tokenizer_samples.jsonl"), trainer=trainer)

# 8. Wrap the trained tokenizer in a PreTrainedTokenizerFast for HF compatibility.
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    mask_token="[MASK]",
    unk_token="[UNK]",
    clean_up_tokenization_spaces=True,
    model_max_length=8192
)

# 9. Save the tokenizer for later use.
hf_tokenizer.save_pretrained("./trained_bpe_tokenizer")
