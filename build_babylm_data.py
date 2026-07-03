import os, json, hashlib, random
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

LOCAL_OFFICIAL = "/data0/lexi/babyllava/babylm_eng_clean_99M.json"
OUT  = "/data0/lexi/babyllava/babylm_docs_100M.jsonl"
SEED, TARGET_TOTAL, MIN_WORDS = 42, 100_000_000, 2
random.seed(SEED)

nwords = lambda t: len(t.split())
key    = lambda t: hashlib.md5(" ".join(t.split()).lower().encode()).hexdigest()
out = open(OUT, "w", encoding="utf-8")
total, counts, seen = 0, {"official":0, "fineweb":0, "cosmopedia":0}, set()

def emit(text, source, bucket):
    global total
    text = (text or "").strip()
    if nwords(text) < MIN_WORDS: return
    k = key(text)
    if k in seen: return                 # 去重(含官方内部 ~42% 重复)
    seen.add(k)
    w = nwords(text)
    out.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
    counts[bucket] += w; total += w

print("① 官方语料(去重)...")
docs = json.load(open(LOCAL_OFFICIAL, encoding="utf-8")); random.shuffle(docs)
for it in docs:
    if total >= TARGET_TOTAL: break
    if isinstance(it, dict):
        txt = f"{it.get('q','')} {it.get('a','')}" if it.get("is_qa") else it.get("text", "")
    else:
        txt = str(it)
    emit(txt, "official", "official")
print(f"   官方去重后: {counts['official']/1e6:.2f}M")

remain = TARGET_TOTAL - total
if remain > 1_000_000:
    print(f"② 补齐 {remain/1e6:.2f}M(FineWeb + Cosmopedia 各一半)...")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
                      streaming=True).shuffle(seed=SEED, buffer_size=50000)
    for it in fw:
        if counts["fineweb"] >= remain // 2 or total >= TARGET_TOTAL: break
        emit(it.get("text", ""), "fineweb-edu", "fineweb")
    cs = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train",
                      streaming=True).shuffle(seed=SEED, buffer_size=50000)
    for it in cs:
        if total >= TARGET_TOTAL: break
        emit(it.get("text", ""), "cosmopedia", "cosmopedia")
else:
    print("② 官方已够 100M，纯官方。")

out.close()
print({k: round(v/1e6, 2) for k, v in counts.items()}, "| 总计:", round(total/1e6, 2), "M words")