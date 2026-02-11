import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import DebertaV2Tokenizer


print("start")
tok = DebertaV2Tokenizer.from_pretrained(
    "microsoft/deberta-v3-base",
    cache_dir="C:/hf_cache",
)
print("loaded")
