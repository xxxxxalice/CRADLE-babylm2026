import json
import os
from tqdm import tqdm
from datasets import load_dataset

# ==========================================
# ⚙️ 路径配置
# ==========================================
INPUT_FILE = "/data0/lexi/babyllava/babylm_100M_CHUNKED_FINAL.json"
OUTPUT_FILE = "/data0/lexi/babyllava/babylm_100M_PURE_TEXT_FINAL.json"
TARGET_WORDS = 100_000_000  # 100M Strict 极限红线

def count_words(text):
    """标准的 BabyLM 按空格统计单词数"""
    return len(text.split())

def main():
    print(f"🚀 阶段 1: 加载并清洗原始数据...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        
    pure_data = []
    current_total_words = 0
    removed_images_count = 0
    
    # 扫描并剔除图文对
    for item in tqdm(original_data, desc="🧹 清理视觉模块"):
        # 判断并剔除含有图像指针的数据
        if item.get("image_idx", -1) != -1:
            removed_images_count += 1
            continue
            
        # 统一提取纯文本格式
        text_content = f"{item.get('q', '')} {item.get('a', '')}" if item.get("is_qa") else item.get("text", "")
        if not text_content.strip():
            continue
            
        word_count = count_words(text_content)
        pure_data.append({"text": text_content})
        current_total_words += word_count

    deficit = TARGET_WORDS - current_total_words

    print("\n========================================")
    print(f"🗑️ 已成功剔除图文对:   {removed_images_count} 条")
    print(f"📊 当前总存量:         {current_total_words / 1e6:.2f} M 单词")
    print(f"🚨 距离 100M 缺口:     +{deficit / 1e6:.2f} M 单词")
    print("========================================\n")

    # ==========================================
    # 🚀 阶段 2: 在线拉取 Cosmopedia 高质量合成教科书
    # ==========================================
    if deficit > 0:
        print(f"💉 阶段 2: 开始从 Hugging Face 拉取 Cosmopedia-100k 数据集...")
        # 自动下载 HuggingFaceTB 的合成教科书子集
        hf_dataset = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
        
        added_words = 0
        added_count = 0
        
        for row in tqdm(hf_dataset, desc="📈 极限填充 Cosmopedia"):
            if added_words >= deficit:
                break
                
            # 提取合成的教科书文本
            text_content = row.get("text", "")
            if not text_content: continue
                
            item_words = count_words(text_content)
            
            # 绝对安全锁：精确截断防超载
            if added_words + item_words > deficit:
                continue
                
            pure_data.append({"text": text_content})
            added_words += item_words
            added_count += 1
            
        print(f"\n✅ 填充完成！成功注入 {added_count} 篇 AI 合成教科书，共计 {added_words / 1e6:.2f} M 词。")
        current_total_words += added_words
    else:
        print("✅ 当前数据已达标，无需填充。")

    # ==========================================
    # 💾 阶段 3: 安全落盘
    # ==========================================
    print(f"\n💾 阶段 3: 正在保存最终数据集至 {OUTPUT_FILE} ...")
    print(f"🎯 最终 100M Strict 赛道定稿字数: {current_total_words / 1e6:.4f} M")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(pure_data, f, ensure_ascii=False, indent=2)
        
    print("🎉 纯文本冠军底座数据集重构完成！")

if __name__ == "__main__":
    main()