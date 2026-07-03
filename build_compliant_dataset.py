import os
import json

# ==========================================
# 🚀 核级网络穿透：强制接管 HuggingFace 下载节点
# 必须写在 import datasets 之前！
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_TRUST_TRUSTED_CODE"] = "1" # 防止某些数据集报错

from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 🛠️ 架构师全局配置区
# ==========================================
OUTPUT_JSON = "babylm_100M_human_golden_mix.json"
TARGET_TOTAL_WORDS = 100_000_000 # 100M 绝对上限

# 预算分配策略 (100M 总盘)
BUDGET = {
    "wiki_text": 23_300_000,    # 23.3M: Simple Wiki (世界知识)
    "oasst_text": 23_300_000,   # 23.3M: OpenAssistant (人类对话/逻辑)
    "babylm_text": 23_400_000,  # 23.4M: BabyLM 官方语料 (这里用 BookCorpus 替代演示纯文本)
    "qa_dolly": 15_000_000,     # 15.0M: Dolly-15k (高质量人类 QA 监督微调)
    "vision_coco": 15_000_000   # 15.0M: COCO Captions (图像视觉对齐锚点)
}

def count_words(text):
    """BabyLM 官方标准的词数计算方式 (按空格切分)"""
    if not text: return 0
    return len(str(text).split())

def build_dataset():
    mixed_data = []
    counters = {k: 0 for k in BUDGET.keys()}
    
    print("🚀 启动 2026 BabyLM 国内直连版锻造引擎...")
    print("⚡ 已开启 HF-Mirror 国内高速镜像通道！")

    # ==========================================
    # 1. 💬 注入 SFT 监督问答 (Dolly-15k)
    # ==========================================
    print("\n⏳ [1/5] 正在从镜像站拉取高质量问答 (Dolly-15k)...")
    qa_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    for item in qa_dataset:
        if counters["qa_dolly"] >= BUDGET["qa_dolly"]: break
        
        q = item['instruction'] + (f"\nContext: {item['context']}" if item.get('context') else "")
        a = item['response']
        
        words = count_words(q) + count_words(a)
        if counters["qa_dolly"] + words > BUDGET["qa_dolly"]: continue 
        
        mixed_data.append({"is_qa": True, "q": q, "a": a, "image_idx": -1})
        counters["qa_dolly"] += words
    print(f"✅ QA 注入完毕: {counters['qa_dolly']/1e6:.2f}M 单词")

    # ==========================================
    # 2. 🖼️ 注入 视觉对齐描述 (COCO Captions 纯文本版)
    # ==========================================
    print("\n⏳ [2/5] 正在从镜像站拉取图文描述...")
    # COCO 原版图库太大容易断，这里我们只拉取它的 Caption 文本版 (100%合规)
    vision_dataset = load_dataset("embedding-data/coco_captions", split="train")
    img_idx = 0
    for item in vision_dataset:
        if counters["vision_coco"] >= BUDGET["vision_coco"]: break
        
        caption = item['set'][1] if isinstance(item['set'], list) and len(item['set']) > 1 else ""
        if not caption: continue
            
        words = count_words(caption)
        if counters["vision_coco"] + words > BUDGET["vision_coco"]: break
            
        mixed_data.append({
            "is_qa": False, 
            "text": caption, 
            "image_idx": img_idx # 🔗 这里指向你的 V4_VISUAL_NPY 矩阵的行索引！
        })
        counters["vision_coco"] += words
        img_idx += 1
    print(f"✅ 视觉描述注入完毕: {counters['vision_coco']/1e6:.2f}M 单词")

    # ==========================================
    # 3. 📘 注入 纯文本基石: Simple Wikipedia
    # ==========================================
    print("\n⏳ [3/5] 正在从镜像站拉取纯净世界知识 (Simple Wikipedia)...")
    wiki_dataset = load_dataset("wikipedia", "20220301.simple", split="train")
    for item in wiki_dataset:
        if counters["wiki_text"] >= BUDGET["wiki_text"]: break
        
        text = item['text']
        words = count_words(text)
        if counters["wiki_text"] + words > BUDGET["wiki_text"]: continue
            
        mixed_data.append({"is_qa": False, "text": text, "image_idx": -1})
        counters["wiki_text"] += words
    print(f"✅ Wikipedia 注入完毕: {counters['wiki_text']/1e6:.2f}M 单词")

    # ==========================================
    # 4. 🧠 注入 纯文本基石: OpenAssistant (人类高阶逻辑)
    # ==========================================
    print("\n⏳ [4/5] 正在从镜像站拉取人类高阶对话 (OASST1)...")
    oasst_dataset = load_dataset("OpenAssistant/oasst1", split="train")
    for item in oasst_dataset:
        if counters["oasst_text"] >= BUDGET["oasst_text"]: break
        if item['lang'] != 'en': continue # 仅保留英文
        
        text = item['text']
        words = count_words(text)
        if counters["oasst_text"] + words > BUDGET["oasst_text"]: continue
            
        mixed_data.append({"is_qa": False, "text": text, "image_idx": -1})
        counters["oasst_text"] += words
    print(f"✅ OASST1 注入完毕: {counters['oasst_text']/1e6:.2f}M 单词")

    # ==========================================
    # 5. 🍼 注入 纯文本基石: 高质量书籍/故事补充
    # ==========================================
    print("\n⏳ [5/5] 正在从镜像站拉取故事补充语料 (补齐 100M)...")
    # 因为直接拉取 BabyLM 官方私有库可能需要 Token，
    # 这里使用与官方高度重合的优质故事库 (BookCorpus 极小子集) 作为合规替代方案
    book_dataset = load_dataset("bookcorpus", split="train")
    for item in book_dataset:
        if counters["babylm_text"] >= BUDGET["babylm_text"]: break
        
        text = item['text']
        words = count_words(text)
        if counters["babylm_text"] + words > BUDGET["babylm_text"]: continue
            
        mixed_data.append({"is_qa": False, "text": text, "image_idx": -1})
        counters["babylm_text"] += words

    print(f"✅ 补充语料注入完毕: {counters['babylm_text']/1e6:.2f}M 单词")

    # ==========================================
    # 💾 结算与落盘
    # ==========================================
    total_words = sum(counters.values())
    print(f"\n🎉 数据集构建完成！总计曝光量: {total_words / 1e6:.4f} M 单词 (严格遵守 <100M 赛规)")
    
    print(f"💾 正在打乱数据并保存至 {OUTPUT_JSON}...")
    import random
    random.shuffle(mixed_data) # 打乱顺序，避免灾难性遗忘
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(mixed_data, f, ensure_ascii=False, indent=2)
    print("🚀 所有数据均已通过国内镜像极速下载并缓存完毕！")

if __name__ == "__main__":
    build_dataset()