import json, torch
from pathlib import Path
from tokenizers import Tokenizer
SRC = "/data0/lexi/babyllava/babylm_docs_100M.jsonl"
OUT = Path("/data0/lexi/babyllava/gptbert_data_v2"); OUT.mkdir(parents=True, exist_ok=True)
tk = Tokenizer.from_file("/data0/lexi/babyllava/babylm_eng_baseline_tokenizer/tokenizer.json")

docs = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        ids = tk.encode(json.loads(line)["text"], add_special_tokens=False).ids  # 官方自己加 CLS
        if len(ids) > 1:
            docs.append(torch.tensor(ids, dtype=torch.short))
valid, train = docs[-4000:], docs[:-4000]
torch.save(train, OUT / "train_100M.bin")
torch.save(valid, OUT / "dev_100M.bin")
print("train docs:", len(train), "valid:", len(valid))