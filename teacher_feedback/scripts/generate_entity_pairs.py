"""
生成 move_contents 和 ambiref 格式的 SimPER 训练对
- 程序化生成，保证 chosen 100% 正确，rejected 为合理错误答案
- 目标：覆盖 high-numops (3-5) 场景，提升 move_contents 子任务分数
"""
import json
import random
import argparse
from copy import deepcopy
from pathlib import Path

# 与评测数据集相同的词汇表（101个）
ITEMS = [
    "apple", "bag", "ball", "beer", "bell", "bill", "block", "boat", "bomb", "bone",
    "book", "boot", "bottle", "bowl", "brain", "branch", "bread", "brick", "bus", "cake",
    "camera", "car", "card", "cash", "cheese", "chemical", "cigarette", "clock", "coat",
    "coffee", "computer", "cream", "creature", "cross", "crown", "cup", "dish", "disk",
    "document", "dress", "drink", "drug", "egg", "engine", "fan", "fig", "file", "fish",
    "flower", "game", "gift", "glass", "guitar", "hat", "ice", "jacket", "key", "knife",
    "leaf", "letter", "machine", "magazine", "map", "meat", "medicine", "milk", "mirror",
    "newspaper", "note", "painting", "paper", "phone", "picture", "pipe", "plane", "plant",
    "plate", "pot", "radio", "ring", "rock", "rose", "seed", "sheet", "shell", "shirt",
    "shoe", "stone", "string", "suit", "tape", "tea", "television", "ticket", "tie",
    "tissue", "train", "watch", "wheel", "wire"
]

COLORS = ["red", "blue", "green", "yellow", "small", "big"]
NUM_BOXES = 7


def fmt_items(items):
    """将items集合格式化为字符串，空则返回'nothing'"""
    if not items:
        return "nothing"
    sorted_items = sorted(items)
    return " and ".join(f"the {item}" for item in sorted_items)


def fmt_items_adj(items):
    """带形容词的items集合格式化"""
    if not items:
        return "nothing"
    sorted_items = sorted(items)
    return " and ".join(f"the {adj} {noun}" for adj, noun in sorted_items)


class BoxState:
    """模拟箱子状态"""
    def __init__(self, boxes):
        # boxes: list of sets, each set contains item names
        self.boxes = [set(b) for b in boxes]

    def move_contents(self, src, dst):
        """将 src 的全部内容移到 dst"""
        self.boxes[dst].update(self.boxes[src])
        self.boxes[src] = set()

    def remove_item(self, box, item):
        self.boxes[box].discard(item)

    def put_item(self, box, item):
        self.boxes[box].add(item)

    def move_item(self, item, src, dst):
        self.boxes[src].discard(item)
        self.boxes[dst].add(item)

    def move_items(self, items, src, dst):
        for item in items:
            self.boxes[src].discard(item)
            self.boxes[dst].add(item)

    def contents(self, box):
        return frozenset(self.boxes[box])


def generate_move_contents_scenario(rng, num_ops, max_items_per_box=3):
    """
    生成一个 move_contents 场景，返回 (input_prefix, correct_answer, distractors)
    num_ops: 操作数量 (0-5)
    """
    # 初始化箱子（7个），随机填充items
    all_items = rng.sample(ITEMS, k=min(35, len(ITEMS)))
    boxes = []
    item_idx = 0
    for i in range(NUM_BOXES):
        n = rng.randint(0, max_items_per_box)
        n = min(n, len(all_items) - item_idx)
        box_items = set(all_items[item_idx:item_idx + n])
        boxes.append(box_items)
        item_idx += n

    # 确保至少有一些 nothing 箱子（让move_contents操作更有意义）
    nothing_count = sum(1 for b in boxes if not b)
    if nothing_count == 0:
        boxes[rng.randint(0, NUM_BOXES - 1)] = set()

    state = BoxState(boxes)
    initial_boxes = [set(b) for b in boxes]

    # 构建初始prefix
    initial_parts = []
    for i in range(NUM_BOXES):
        contents = fmt_items(initial_boxes[i])
        initial_parts.append(f"Box {i} contains {contents}")
    prefix = ", ".join(initial_parts) + ". "

    # 生成操作序列
    ops_text = []
    ops_done = 0
    attempts = 0

    while ops_done < num_ops and attempts < 50:
        attempts += 1
        op_type = rng.choices(
            ["move_contents", "remove", "put", "move_item"],
            weights=[0.30, 0.35, 0.15, 0.20]
        )[0]

        if op_type == "move_contents":
            # 找有内容的箱子作为src
            src_candidates = [i for i in range(NUM_BOXES) if state.boxes[i]]
            if not src_candidates:
                continue
            src = rng.choice(src_candidates)
            dst_candidates = [i for i in range(NUM_BOXES) if i != src]
            dst = rng.choice(dst_candidates)
            state.move_contents(src, dst)
            ops_text.append(f"Move the contents of Box {src} to Box {dst}.")
            ops_done += 1

        elif op_type == "remove":
            # 找有内容的箱子
            candidates = [(i, item) for i in range(NUM_BOXES) for item in state.boxes[i]]
            if not candidates:
                continue
            # 有时移除多个items
            n_remove = rng.randint(1, min(3, max(1, len(candidates))))
            to_remove = rng.sample(candidates, min(n_remove, len(candidates)))
            # 按箱子分组
            by_box = {}
            for box, item in to_remove:
                by_box.setdefault(box, []).append(item)

            for box, items in by_box.items():
                items_sorted = sorted(items)
                for item in items:
                    state.remove_item(box, item)
                items_str = " and ".join(f"the {item}" for item in items_sorted)
                ops_text.append(f"Remove {items_str} from Box {box}.")
                ops_done += 1
                if ops_done >= num_ops:
                    break

        elif op_type == "put":
            # 向某箱子添加新items
            dst = rng.randint(0, NUM_BOXES - 1)
            # 选择当前不在任何箱子中的items
            used = set()
            for b in state.boxes:
                used.update(b)
            available = [item for item in ITEMS if item not in used]
            if not available:
                continue
            n_put = rng.randint(1, min(3, len(available)))
            new_items = rng.sample(available, n_put)
            items_str = " and ".join(f"the {item}" for item in sorted(new_items))
            for item in new_items:
                state.put_item(dst, item)
            ops_text.append(f"Put {items_str} into Box {dst}.")
            ops_done += 1

        elif op_type == "move_item":
            # 移动单个或多个items
            candidates = [(i, item) for i in range(NUM_BOXES) for item in state.boxes[i]]
            if not candidates:
                continue
            src_box, item = rng.choice(candidates)
            dst_box_candidates = [i for i in range(NUM_BOXES) if i != src_box]
            dst_box = rng.choice(dst_box_candidates)
            # 有时移动多个
            same_box_items = list(state.boxes[src_box])
            n_move = rng.randint(1, min(2, len(same_box_items)))
            items_to_move = rng.sample(same_box_items, n_move)
            items_str = " and ".join(f"the {it}" for it in sorted(items_to_move))
            state.move_items(items_to_move, src_box, dst_box)
            ops_text.append(f"Move {items_str} from Box {src_box} to Box {dst_box}.")
            ops_done += 1

    if ops_done < num_ops:
        return None  # 生成失败

    # 选择目标箱子（提问哪个箱子）
    prefix += " ".join(ops_text) + " "
    query_box = rng.randint(0, NUM_BOXES - 1)
    input_prefix = prefix + f"Box {query_box} contains "

    correct_contents = state.contents(query_box)
    chosen = fmt_items(correct_contents) + "."

    # 生成干扰项（rejected）— 文献指导：rejected 应体现时序状态错误
    distractors = []

    # 策略1（最佳）：初始内容（操作前的旧状态）— 模型常见错误：忽略操作
    init_content = fmt_items(initial_boxes[query_box]) + "."
    if init_content != chosen:
        distractors.append(init_content)

    # 策略2：其他最近被操作的箱子内容（混淆box reference）
    other_boxes = [i for i in range(NUM_BOXES) if i != query_box]
    rng.shuffle(other_boxes)
    for other in other_boxes:
        d = fmt_items(state.contents(other)) + "."
        if d != chosen and d not in distractors:
            distractors.append(d)
        if len(distractors) >= 3:
            break

    # 策略3：混合错误（把移走的items留在了原来的箱子，或多记了一个item）
    if len(distractors) < 2 and correct_contents:
        wrong_items = set(correct_contents)
        all_used = set()
        for b in state.boxes:
            all_used.update(b)
        extra = [it for it in ITEMS if it not in all_used]
        if extra and rng.random() < 0.5:
            wrong_items.add(rng.choice(extra))  # 多记了个item
        elif correct_contents:
            wrong_items.discard(rng.choice(list(correct_contents)))  # 漏记了个item
        if wrong_items != correct_contents:
            d = fmt_items(wrong_items) + "."
            if d != chosen and d not in distractors:
                distractors.append(d)

    if not distractors:
        return None

    return input_prefix, chosen, distractors


class AdjBoxState:
    """模拟带形容词的箱子状态，items形如 (adj, noun)"""
    def __init__(self, boxes):
        # boxes: list of sets of (adj, noun) tuples
        self.boxes = [set(b) for b in boxes]

    def remove_by_noun(self, box_idx, noun):
        """通过noun引用删除（可能有歧义，找第一个匹配）"""
        to_remove = None
        for item in self.boxes[box_idx]:
            if item[1] == noun:
                to_remove = item
                break
        if to_remove:
            self.boxes[box_idx].discard(to_remove)
        return to_remove

    def remove_by_adj_noun(self, box_idx, adj, noun):
        self.boxes[box_idx].discard((adj, noun))

    def move_by_noun(self, src, dst, noun):
        item = self.remove_by_noun(src, noun)
        if item:
            self.boxes[dst].add(item)
        return item

    def put_item(self, box_idx, adj, noun):
        self.boxes[box_idx].add((adj, noun))

    def contents(self, box_idx):
        return frozenset(self.boxes[box_idx])


def generate_ambiref_scenario(rng, num_ops, max_items_per_box=3):
    """生成 ambiref 场景（带形容词的歧义引用追踪）"""
    # 生成带形容词的items：(adj, noun)
    nouns = rng.sample(ITEMS, k=min(30, len(ITEMS)))
    adj_items = []
    used_pairs = set()
    for noun in nouns:
        adj = rng.choice(COLORS)
        if (adj, noun) not in used_pairs:
            adj_items.append((adj, noun))
            used_pairs.add((adj, noun))

    # 填充7个箱子
    boxes = []
    item_idx = 0
    for i in range(NUM_BOXES):
        n = rng.randint(0, max_items_per_box)
        n = min(n, len(adj_items) - item_idx)
        box_items = set(adj_items[item_idx:item_idx + n])
        boxes.append(box_items)
        item_idx += n

    state = AdjBoxState(boxes)
    initial_boxes = [set(b) for b in boxes]

    # 构建初始prefix
    initial_parts = []
    for i in range(NUM_BOXES):
        contents = fmt_items_adj(sorted(initial_boxes[i]))
        initial_parts.append(f"Box {i} contains {contents}")
    prefix = ", ".join(initial_parts) + ". "

    # 生成操作
    ops_text = []
    ops_done = 0
    attempts = 0

    while ops_done < num_ops and attempts < 50:
        attempts += 1
        op_type = rng.choices(
            ["remove", "move_item", "put"],
            weights=[0.40, 0.35, 0.25]
        )[0]

        if op_type == "remove":
            candidates = [(i, adj, noun) for i in range(NUM_BOXES) for adj, noun in state.boxes[i]]
            if not candidates:
                continue
            box_idx, adj, noun = rng.choice(candidates)
            # 用不带形容词的方式引用（增加歧义感）
            state.remove_by_adj_noun(box_idx, adj, noun)
            # 检查该箱子中有没有同noun的其他adj item（如有才算ambiguous）
            # 用noun引用会更真实，但需要确保该box中noun唯一才不会歧义
            ops_text.append(f"Remove the {noun} from Box {box_idx}.")
            ops_done += 1

        elif op_type == "move_item":
            candidates = [(i, adj, noun) for i in range(NUM_BOXES) for adj, noun in state.boxes[i]]
            if not candidates:
                continue
            src_box, adj, noun = rng.choice(candidates)
            dst_candidates = [i for i in range(NUM_BOXES) if i != src_box]
            dst_box = rng.choice(dst_candidates)
            state.remove_by_adj_noun(src_box, adj, noun)
            state.boxes[dst_box].add((adj, noun))
            ops_text.append(f"Move the {noun} from Box {src_box} to Box {dst_box}.")
            ops_done += 1

        elif op_type == "put":
            dst = rng.randint(0, NUM_BOXES - 1)
            used_pairs_now = set()
            for b in state.boxes:
                used_pairs_now.update(b)
            available_nouns = [n for n in ITEMS if all(n != pair[1] for pair in used_pairs_now)]
            if not available_nouns:
                continue
            noun = rng.choice(available_nouns)
            adj = rng.choice(COLORS)
            state.put_item(dst, adj, noun)
            ops_text.append(f"Put the {adj} {noun} into Box {dst}.")
            ops_done += 1

    if ops_done < num_ops:
        return None

    # 选择查询箱子
    prefix += " ".join(ops_text) + " "
    query_box = rng.randint(0, NUM_BOXES - 1)
    input_prefix = prefix + f"Box {query_box} contains "

    correct_contents = state.contents(query_box)
    chosen = fmt_items_adj(sorted(correct_contents)) + "."

    # 生成干扰项（5个选项：1正确+4错误）
    distractors = []

    # 策略1：其他箱子内容
    other_boxes = [i for i in range(NUM_BOXES) if i != query_box]
    rng.shuffle(other_boxes)
    for other in other_boxes:
        d = fmt_items_adj(sorted(state.contents(other))) + "."
        if d != chosen and d not in distractors:
            distractors.append(d)
        if len(distractors) >= 4:
            break

    # 策略2：初始内容
    if len(distractors) < 4:
        init_d = fmt_items_adj(sorted(initial_boxes[query_box])) + "."
        if init_d != chosen and init_d not in distractors:
            distractors.append(init_d)

    # 策略3：混合items（交换颜色）
    if len(distractors) < 4 and correct_contents:
        wrong_items = set(correct_contents)
        if wrong_items:
            item_to_change = rng.choice(list(wrong_items))
            wrong_items.discard(item_to_change)
            new_adj = rng.choice([c for c in COLORS if c != item_to_change[0]])
            wrong_items.add((new_adj, item_to_change[1]))
            d = fmt_items_adj(sorted(wrong_items)) + "."
            if d != chosen and d not in distractors:
                distractors.append(d)

    if not distractors:
        return None

    return input_prefix, chosen, distractors[:4]


def generate_dataset(n_pairs_per_type=5000, seed=42):
    """生成完整的训练数据集"""
    rng = random.Random(seed)
    pairs = []

    # move_contents 生成
    mc_target = n_pairs_per_type
    mc_per_ops = mc_target // 6  # 均匀分布在 numops 0-5
    mc_count = 0

    print(f"生成 move_contents 训练对...")
    for ops_level in range(6):
        # 高 numops 多生成些（更难更有价值）
        weight = 1 + ops_level  # 0ops:1份, 5ops:6份
        target_this_level = int(mc_target * weight / 21)  # 21 = 1+2+3+4+5+6

        count_this_level = 0
        attempts = 0
        while count_this_level < target_this_level and attempts < target_this_level * 10:
            attempts += 1
            result = generate_move_contents_scenario(rng, num_ops=ops_level)
            if result is None:
                continue
            input_prefix, chosen, distractors = result
            rejected = distractors[0]

            pair = {
                "chosen": input_prefix + chosen,
                "rejected": input_prefix + rejected,
                "type": "entity_move_contents",
                "numops": ops_level,
                "teacher_correct": True,
                "teacher_certain": True
            }
            pairs.append(pair)
            count_this_level += 1
            mc_count += 1

        print(f"  numops={ops_level}: {count_this_level} 对")

    # ambiref 生成
    print(f"生成 ambiref 训练对...")
    ab_target = n_pairs_per_type
    ab_count = 0

    for ops_level in range(6):
        weight = 1 + ops_level
        target_this_level = int(ab_target * weight / 21)

        count_this_level = 0
        attempts = 0
        while count_this_level < target_this_level and attempts < target_this_level * 10:
            attempts += 1
            result = generate_ambiref_scenario(rng, num_ops=ops_level)
            if result is None:
                continue
            input_prefix, chosen, distractors = result
            rejected = distractors[0]

            pair = {
                "chosen": input_prefix + chosen,
                "rejected": input_prefix + rejected,
                "type": "entity_ambiref",
                "numops": ops_level,
                "teacher_correct": True,
                "teacher_certain": True
            }
            pairs.append(pair)
            count_this_level += 1
            ab_count += 1

        print(f"  numops={ops_level}: {count_this_level} 对")

    rng.shuffle(pairs)
    print(f"\n总计: {len(pairs)} 对 (move_contents: {mc_count}, ambiref: {ab_count})")
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=5000, help="每种类型的训练对数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="/data0/lexi/babyllava/teacher_feedback/data/entity_hard_pairs.jsonl")
    args = parser.parse_args()

    pairs = generate_dataset(n_pairs_per_type=args.n_pairs, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"已保存到: {out_path}")

    # 统计词汇量
    total_chars = sum(len(p["chosen"]) + len(p["rejected"]) for p in pairs)
    est_words = total_chars / 5  # 估算：平均5字符/词
    print(f"估计词汇量: {est_words/1e6:.2f}M words (总曝光影响可忽略)")
