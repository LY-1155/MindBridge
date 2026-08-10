"""周医生"情景→回应"风格参考库索引构建脚本

从真实访谈语料中：
  1. 脱敏（core.privacy.desensitize）—— 去掉姓名/电话/机构名/地名等 PII
  2. 蒸馏成 (human 发言 → 周医生回应) 配对样本
  3. 嵌入建 Chroma 索引（collection: zhou_style）

用法：
  python scripts/build_zhou_style_index.py --dry-run     # 只看统计，不建索引
  python scripts/build_zhou_style_index.py --max-samples 30000
  python scripts/build_zhou_style_index.py                # 全量建索引

隐私说明：原始访谈数据（含真实患者信息）永不进入索引。
建索引前逐轮脱敏；索引文本只保留"患者发言 + 医生回应"的泛化内容。
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.rag.chroma_store import ChromaStore
from core.rag.embedder import create_embedder
from core.privacy.desensitize import desensitize_turn

INTERVIEW_ROOT = PROJECT_ROOT / "心理医生访谈数据"
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "data" / "knowledge" / "chroma_zhou_style")

MIN_HUMAN_LEN = 4
MIN_DOCTOR_LEN = 8

# 无意义的语气词/标点（去掉这些后若剩余不足 N 字则跳过）
_NOISE_RE = re.compile(r"[\s，。、,.！？!?；;：:·…——“”‘’（）()\[\]{}　]+")
_PURE_FILLER = set("嗯啊哦呃唉哈哎呐咦哟嚯哦哦切嘶")

# 跳过含这些标记的 assistant 发言（列表/结构化输出，非对话）
_DOCTOR_SKIP_MARKERS = ("```", "http://", "https://", "治疗建议：", "干预方案：")


def iter_json_files(root: Path):
    """遍历访谈语料下所有 json 文件，容错跳过格式问题。"""
    if not root.exists():
        print(f"[错误] 访谈数据目录不存在: {root}")
        return
    for p in sorted(root.rglob("*.json")):
        yield p


def extract_turns(data):
    """从解析后的 JSON 提取 (human_text, doctor_text) 配对。

    结构：list[{conversations: [{from, value}, ...]}]。
    对每个 human 发言，配其后面最近的 assistant 发言。
    """
    if isinstance(data, list):
        for item in data:
            convs = item.get("conversations", []) if isinstance(item, dict) else None
            if not convs:
                continue
            prev_human = None
            for turn in convs:
                if not isinstance(turn, dict):
                    continue
                role = turn.get("from")
                value = turn.get("value")
                if not isinstance(value, str) or not value.strip():
                    continue
                if role == "human":
                    prev_human = value.strip()
                elif role == "assistant" and prev_human:
                    yield prev_human, value.strip()
                    prev_human = None


def is_noise(turn_text: str, min_len: int) -> bool:
    """判断一段话是否是无信息量的噪声（太短/纯语气词/纯标点）。"""
    if len(turn_text) < min_len:
        return True
    stripped = _NOISE_RE.sub("", turn_text)
    if len(stripped) < 2:
        return True
    # 纯语气词（去掉后为空）
    core = "".join(ch for ch in stripped if ch not in _PURE_FILLER)
    return len(core) < 1


def build_samples(dry_run: bool = False, max_samples: int | None = None):
    """扫描 + 脱敏 + 蒸馏，返回样本列表与统计。"""
    stats = {"files": 0, "parse_fail": 0, "raw_pairs": 0,
             "filtered_short": 0, "filtered_noise": 0, "filtered_doctor_marker": 0,
             "after_dedup": 0}
    samples: list[dict] = []
    seen: set[str] = set()

    for p in iter_json_files(INTERVIEW_ROOT):
        stats["files"] += 1
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            stats["parse_fail"] += 1
            continue

        for human_raw, doctor_raw in extract_turns(data):
            stats["raw_pairs"] += 1
            if max_samples and len(samples) >= max_samples:
                return samples, stats

            # 过滤噪声
            if len(human_raw) < MIN_HUMAN_LEN or len(doctor_raw) < MIN_DOCTOR_LEN:
                stats["filtered_short"] += 1
                continue
            if is_noise(human_raw, 4) or is_noise(doctor_raw, 8):
                stats["filtered_noise"] += 1
                continue
            if any(m in doctor_raw for m in _DOCTOR_SKIP_MARKERS):
                stats["filtered_doctor_marker"] += 1
                continue

            # 脱敏
            human = desensitize_turn("human", human_raw)
            doctor = desensitize_turn("assistant", doctor_raw)
            if not human or not doctor:
                continue

            key = hashlib.sha1((human + "|" + doctor).encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            stats["after_dedup"] += 1

            samples.append({
                "id": f"zhou_{key[:12]}",
                "human": human,
                "doctor": doctor,
            })

    return samples, stats


def build_chroma(samples: list[dict], backend: str = "api", dry_run: bool = False):
    """构建 zhou_style Chroma 索引。"""
    if not samples:
        print("\n[提示] 没有可建索引的样本")
        return
    if dry_run:
        print(f"\n[dry-run] 将建索引 {len(samples)} 条 → {CHROMA_PERSIST_DIR}")
        return

    print(f"\n[建索引] 样本数 {len(samples)}，目标 {CHROMA_PERSIST_DIR}")
    # 自定义 batch_size：Ollama embedding 批处理吞吐更高（实测 50 条/次 ≈ 单条 0.15-0.2s）
    if backend == "local":
        from core.rag.embedder import BGEM3Embedding
        embedder = BGEM3Embedding(batch_size=25)
    else:
        from core.rag.embedder import QianwenEmbedding
        embedder = QianwenEmbedding(batch_size=50)
    store = ChromaStore(
        collection_name="zhou_style",
        embedding_fn=embedder,
        persist_dir=CHROMA_PERSIST_DIR,
    )

    batch_size = 25
    t0 = time.time()
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        ids = [s["id"] for s in batch]
        texts = [f"患者：{s['human']}\n医生：{s['doctor']}" for s in batch]
        metadatas = [{"source": "zhou_interviews", "category": "style_sample"} for _ in batch]
        store.add(ids=ids, texts=texts, metadatas=metadatas)
        pct = min(100, int((i + len(batch)) / len(samples) * 100))
        elapsed = time.time() - t0
        print(f"\r  进度: {pct}% ({min(i + len(batch), len(samples))}/{len(samples)}) "
              f"耗时 {elapsed:.0f}s", end="", flush=True)

    print(f"\n  完成: {len(samples)} 条已写入 {CHROMA_PERSIST_DIR}")
    print(f"  总计耗时 {time.time() - t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="周医生风格参考库索引构建")
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不建索引")
    parser.add_argument("--max-samples", type=int, default=8000,
                        help="最多蒸馏的样本数（默认 8000 ≈ 30 分钟；全量可传 0）")
    parser.add_argument("--backend", choices=["api", "local"], default="api",
                        help="Embedding 后端: api (Ollama/百炼) 或 local (BGE-M3)")
    args = parser.parse_args()

    print("=== 周医生风格参考库索引构建 ===\n")

    samples, stats = build_samples(dry_run=args.dry_run, max_samples=args.max_samples)

    print(f"文件数:          {stats['files']}（解析失败 {stats['parse_fail']}）")
    print(f"原始配对轮数:    {stats['raw_pairs']}")
    print(f"过滤-过短:       {stats['filtered_short']}")
    print(f"过滤-噪声:       {stats['filtered_noise']}")
    print(f"过滤-非对话内容: {stats['filtered_doctor_marker']}")
    print(f"去重后样本数:    {stats['after_dedup']}")

    if not samples:
        print("\n[错误] 没有可用样本，检查访谈数据目录")
        return

    # 抽样展示脱敏后的样本
    print("\n--- 脱敏后样本示例 ---")
    for s in samples[:3]:
        print(f"[患者] {s['human'][:50]}")
        print(f"[医生] {s['doctor'][:60]}")
        print()

    build_chroma(samples, backend=args.backend, dry_run=args.dry_run)
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
