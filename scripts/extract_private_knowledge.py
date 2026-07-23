"""
批量提取脚本：从咨询对话 JSON 中提取临床知识条目 → JSONL
用法：
  python scripts/extract_private_knowledge.py --mode test    # 3份测试
  python scripts/extract_private_knowledge.py --mode full    # 全量
"""

import os, sys, json, re, uuid, time, argparse
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.llm.base import get_llm_adapter
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DATA_ROOTS = [
    r'C:\Users\xt\Desktop\Project\数据集处理\处理数据集1',
    r'C:\Users\xt\Desktop\Project\数据集处理\处理数据集2',
    r'C:\Users\xt\Desktop\Project\数据集处理\处理数据集3',
    r'C:\Users\xt\Desktop\Project\数据集处理\另一个班',
]

OUTPUT_PATH = PROJECT_ROOT / 'data' / 'knowledge' / 'private' / 'clinical_knowledge.jsonl'
PROGRESS_PATH = PROJECT_ROOT / 'scripts' / '.extract_progress.json'

MAX_CONVERSATION_CHARS = 3000  # 单次对话截断长度
SAVE_INTERVAL = 20  # 每 20 份自动存盘


# ---------------------------------------------------------------------------
# 进度显示
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """你是一位资深心理咨询督导师。分析以下心理咨询对话，提取有普遍适用性的临床知识和干预技法。

输出 JSON 数组，每项包含：
- title: 知识条目标题（15字以内）
- content: 临床知识点/技法说明（100-300字，用"来访者""咨询师"替代人称，不含任何真实姓名地名日期）
- tags: 2-4个中文标签

重要规则：
1. 只提有普遍临床价值的知识，不复述对话
2. 绝不输出任何真实姓名、地名、日期、学校名、机构名
3. content 必须独立可读，不依赖对话原文
4. 如果对话没有可提取的临床知识，输出 [] 空数组
5. 每份对话提取 2-5 条
6. 只输出 JSON，不要任何其他文字"""

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def find_conversation_folders(roots: List[str], limit: Optional[int] = None) -> List[str]:
    """遍历数据目录，返回所有含对话 JSON 的子文件夹路径"""
    folders = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        seen = set()
        for r, _, fs in os.walk(root):
            if r in seen:
                continue
            has_json = any(f.endswith('.json') for f in fs)
            if has_json:
                seen.add(r)
                folders.append(r)
    folders.sort()
    if limit:
        folders = folders[:limit]
    return folders


def read_conversation(folder: str) -> Optional[str]:
    """读取子文件夹中的所有对话 JSON，拼接成一段文本"""
    lines = []
    # 优先原始 JSON，没有再用 output
    original = [f for f in os.listdir(folder) if f.endswith('.json') and 'output' not in f.lower()]
    if original:
        json_files = sorted(original)
    else:
        json_files = sorted([f for f in os.listdir(folder) if f.endswith('.json')])

    for fname in json_files:
        fp = os.path.join(folder, fname)
        try:
            with open(fp, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            if not isinstance(item, dict) or 'conversations' not in item:
                continue
            for turn in item['conversations']:
                role = turn.get('from', '?')
                value = turn.get('value', '')
                if role == 'human':
                    label = '来访者'
                elif role == 'assistant':
                    label = '咨询师'
                else:
                    label = role
                lines.append(f"{label}：{value}")

    if not lines:
        return None
    return '\n'.join(lines)


def parse_llm_response(text: str) -> List[Dict]:
    """从 LLM 输出中解析 JSON 数组"""
    # 尝试直接解析
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 尝试找第一个 [ 到最后一个 ]
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


_entry_counter = 0


def next_id() -> str:
    """生成全局唯一递增 ID"""
    global _entry_counter
    _entry_counter += 1
    return f"clinical_{_entry_counter:04d}"


def extract_from_folder(folder: str, llm, index: int) -> List[Dict]:
    """对一个文件夹的对话进行知识提取"""
    conversation = read_conversation(folder)
    if not conversation:
        print(f"  [跳过] 无有效对话内容")
        return []

    # 截断过长对话
    if len(conversation) > MAX_CONVERSATION_CHARS:
        conversation = conversation[:MAX_CONVERSATION_CHARS] + "\n…（后续对话已截断）"

    folder_name = os.path.basename(folder)
    print(f"  [{index}] {folder_name} ({len(conversation)} 字符) …", end=" ", flush=True)

    try:
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=conversation),
        ]
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, 'content') else str(response)
        entries = parse_llm_response(raw)

        # 为每条条目补充唯一 ID 和 source
        for entry in entries:
            entry['id'] = next_id()
            entry.setdefault('source', '临床咨询记录/脱敏')

        print(f"{len(entries)} 条")
        return entries

    except Exception as e:
        print(f"失败: {e}")
        return []


# ---------------------------------------------------------------------------
# 进度显示
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}分{s}秒"
    else:
        h, r = divmod(int(seconds), 3600)
        m, s = divmod(r, 60)
        return f"{h}时{m}分{s}秒"


def print_progress(current: int, total: int, entries: int, skipped: int, failed: int,
                   elapsed: float):
    """打印进度条"""
    pct = current / total * 100 if total else 0
    bar_width = 30
    filled = int(bar_width * current / total) if total else 0
    bar = "#" * filled + "-" * (bar_width - filled)

    # ETA
    if current > 0:
        avg_per_item = elapsed / current
        eta = avg_per_item * (total - current)
        eta_str = format_time(eta)
    else:
        eta_str = "…"

    speed = entries / (elapsed / 60) if elapsed > 0 else 0

    print(f"\r  [{bar}] {pct:5.1f}% | {current}/{total} | "
          f"知识:{entries} 跳过:{skipped} 失败:{failed} | "
          f"耗时:{format_time(elapsed)} | 速度:{speed:.1f}条/分 | 预计剩余:{eta_str}",
          end="", flush=True)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def save_entries(entries: List[Dict], path: Path):
    """保存所有条目到 JSONL"""
    with open(path, 'w', encoding='utf-8') as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')


def load_progress() -> tuple[set, list]:
    """加载断点：返回 (已完成文件夹集合, 已有条目列表)"""
    done = set()
    entries = []
    # 加载已有条目
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception:
            entries = []
    # 加载已完成文件夹
    if PROGRESS_PATH.exists():
        try:
            done = set(json.loads(PROGRESS_PATH.read_text(encoding='utf-8')))
        except Exception:
            done = set()
    return done, entries


def save_progress(done: set, entries: list):
    """保存断点"""
    PROGRESS_PATH.write_text(json.dumps(sorted(done), ensure_ascii=False), encoding='utf-8')
    save_entries(entries, OUTPUT_PATH)


def main():
    global _entry_counter

    parser = argparse.ArgumentParser(description='私有知识库批量提取')
    parser.add_argument('--mode', choices=['test', 'full'], default='test',
                        help='test=3份验证, full=全量')
    args = parser.parse_args()

    limit = 3 if args.mode == 'test' else None
    mode_label = '测试 (3份)' if args.mode == 'test' else '全量'

    print(f"=== 私有知识库提取 - {mode_label} ===")
    print(f"数据源: {len(DATA_ROOTS)} 个目录")
    print(f"输出: {OUTPUT_PATH}\n")

    # 加载断点
    done_folders, all_entries = load_progress()
    # 恢复 ID 计数器
    if all_entries:
        max_id = 0
        for e in all_entries:
            eid = e.get('id', '')
            if eid.startswith('clinical_'):
                try:
                    num = int(eid.split('_')[1])
                    max_id = max(max_id, num)
                except ValueError:
                    pass
        _entry_counter = max_id
    else:
        _entry_counter = 0

    # 查找文件夹
    folders = find_conversation_folders(DATA_ROOTS, limit=limit)
    total = len(folders)

    # 过滤已完成的
    pending = [f for f in folders if f not in done_folders]
    already_done = len(folders) - len(pending)

    if already_done > 0:
        print(f"检测到断点: {already_done} 个已完成, {len(all_entries)} 条已有知识")
        print(f"剩余: {len(pending)} 个文件夹\n")
        if not pending:
            print("全部已完成，无需继续。")
            return
    else:
        if all_entries:
            print(f"检测到 {len(all_entries)} 条旧知识 (无断点记录), 将重新提取全部 {total} 份")
            all_entries.clear()
            _entry_counter = 0
        print(f"找到 {total} 个子文件夹\n")

    # 初始化 LLM
    print("初始化 LLM …")
    llm = get_llm_adapter('qwen')
    print("LLM 就绪，开始提取…\n")

    # 逐份提取
    skipped = 0
    failed = 0
    start_time = time.time()
    next_progress_save = SAVE_INTERVAL  # 下次存盘时的已处理数

    for i, folder in enumerate(pending):
        entries = extract_from_folder(folder, llm, index=already_done + i + 1)
        if not entries:
            skipped += 1
        all_entries.extend(entries)
        time.sleep(0.3)

        # 标记完成
        done_folders.add(folder)

        # 进度条
        current = already_done + i + 1
        print_progress(current, total, len(all_entries), skipped, failed,
                       time.time() - start_time)

        # 定期自动保存（含断点）
        if current % SAVE_INTERVAL == 0:
            save_progress(done_folders, all_entries)

    # 最终保存
    save_progress(done_folders, all_entries)
    # 处理完后删除进度文件（干净收尾）
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
    print()  # 换行

    elapsed = time.time() - start_time
    print(f"\n=== 完成 ===")
    print(f"处理: {total} 份 (跳过: {skipped}, 失败: {failed})")
    print(f"产出: {len(all_entries)} 条知识")
    print(f"耗时: {format_time(elapsed)}")
    print(f"输出: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
