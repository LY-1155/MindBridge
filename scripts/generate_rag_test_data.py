"""RAG 检索评估 — 合成测试数据生成（B 方案）。

从知识库 JSONL 文档反向生成用户查询：
  - 每篇文档 → LLM 生成 3-5 条该文档能回答的自然语言问题
  - 输出 JSONL：{query, relevant_doc_ids, expected_categories, expected_source}

用法：
  python scripts/generate_rag_test_data.py --sample 50 --output data/eval/synthetic_queries.jsonl
  python scripts/generate_rag_test_data.py --sample 10 --dry-run  # 用 mock LLM 快速测试
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SOURCES = _PROJECT_ROOT / "data" / "knowledge" / "sources.json"

BATCH_SYSTEM_PROMPT = """你是一个心理健康咨询评估系统的测试数据生成器。你的任务是为给定的知识文档生成自然的中文用户查询。

## 要求
- 对每篇文档（以文档 ID 标识），生成 {queries_per_doc} 条自然的中文用户提问
- 提问必须模拟真实用户的语气：可能很口语化、简短、模糊、带情绪
- 提问可以是信息类（"XX是什么"）或求助类（"我总是XX怎么办"）
- 每条提问应该让这篇文档成为回答该提问的最佳选项之一
- 输出严格 JSON，不要任何额外文字

## 输出格式
{{
  "doc_xxx": ["提问1", "提问2", "提问3", "提问4"],
  "doc_yyy": ["提问1", "提问2", "提问3", "提问4"]
}}"""


def _load_sources(sources_path: Path) -> dict:
    if not sources_path.exists():
        raise FileNotFoundError(f"sources.json 不存在: {sources_path}")
    return json.loads(sources_path.read_text(encoding="utf-8"))


def _category_from_filename(filename: str) -> str:
    """从文件名推导知识类别。与 KnowledgeRetriever._category_from_filename 逻辑一致。"""
    stem = Path(filename).stem
    for suffix in ["_knowledge", "_info", "_data"]:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem


def load_all_documents(sources_path: Path | None = None) -> List[dict]:
    """加载所有 JSONL 文档。

    Returns:
        [{id, title, content, tags, source, category}, ...]
    """
    config = _load_sources(sources_path or _DEFAULT_SOURCES)
    docs = []

    for layer in config.get("layers", []):
        if not layer.get("enabled") or layer.get("type") != "local":
            continue

        source_label = "private" if "private" in layer["id"] else "public"
        data_dir = Path(layer["path"])
        if not data_dir.is_absolute():
            data_dir = _PROJECT_ROOT / data_dir

        if not data_dir.exists():
            logger.warning("数据目录不存在，跳过: %s", data_dir)
            continue

        for f in sorted(data_dir.glob("*.jsonl")):
            category = _category_from_filename(f.name)
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                docs.append({
                    "id": doc.get("id", f"{source_label}_{len(docs)}"),
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "tags": doc.get("tags", []),
                    "source": source_label,
                    "category": category,
                })

    logger.info("共加载 %d 篇文档", len(docs))
    return docs


def _build_doc_display(doc: dict, max_content: int = 600) -> str:
    """构建给 LLM 看的文档文本片段。"""
    content = doc["content"][:max_content]
    if len(doc["content"]) > max_content:
        content += "..."
    return (
        f'[Doc ID: {doc["id"]} | Category: {doc["category"]} | Source: {doc["source"]}]\n'
        f'Title: {doc["title"]}\n'
        f'Tags: {", ".join(doc.get("tags", []))}\n'
        f'Content: {content}'
    )


def _make_llm():
    """创建 LLM 实例（高 max_tokens 避免 JSON 被截断）。"""
    from core.llm.base import get_llm_adapter, LLMConfig
    return get_llm_adapter("qwen", config=LLMConfig(max_tokens=4096))


def generate_queries_for_batch(
    llm,
    docs: List[dict],
    queries_per_doc: int = 4,
) -> List[dict]:
    """对一批文档调用 LLM 生成查询。

    Args:
        llm: LangChain ChatModel 或兼容接口
        docs: 文档列表
        queries_per_doc: 每篇文档生成的查询数

    Returns:
        [{query, relevant_doc_ids, expected_categories, expected_source}, ...]
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    docs_text = "\n\n---\n\n".join(_build_doc_display(d) for d in docs)
    doc_ids = [d["id"] for d in docs]

    system = BATCH_SYSTEM_PROMPT.format(queries_per_doc=queries_per_doc)
    user = f"为以下文档生成查询：\n\n{docs_text}"

    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    # 解析 JSON
    try:
        data = _parse_llm_json(raw)
    except Exception as e:
        logger.warning("LLM 输出解析失败: %s\n%s", e, raw[:500])
        return []

    test_cases = []
    for doc_id in doc_ids:
        queries = data.get(doc_id, [])
        if not queries:
            continue
        doc = next((d for d in docs if d["id"] == doc_id), None)
        if doc is None:
            continue
        for q in queries:
            test_cases.append({
                "query": q.strip(),
                "relevant_doc_ids": [doc_id],
                "expected_categories": [doc["category"]],
                "expected_source": doc["source"],
            })

    return test_cases


def _parse_llm_json(raw: str) -> dict:
    """从 LLM 输出提取 JSON，容错处理截断和重复输出。"""
    raw = raw.strip()

    # 去掉 markdown 代码块包裹
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else lines[1] if len(lines) > 1 else raw

    # 去重：LLM 有时会重复输出两次 JSON（第一次被截断，第二次完整）
    # 找到最后一个完整的 "} 之后的内容是有效 JSON 起始位置"
    raw = _deduplicate_json(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 截断修复：尝试补齐被截断的 JSON
    result = _repair_truncated_json(raw)
    if result is not None:
        return result

    # 尝试提取第一个 { } 块
    import re
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            result = _repair_truncated_json(match.group(0))
            if result is not None:
                return result

    # 最后尝试：从 raw 中找到最后一个完整的顶层 key-value 对，丢弃后面截断的部分
    result = _extract_complete_entries(raw)
    if result is not None:
        logger.info("部分解析成功: %d 个文档的查询被保留", len(result))
        return result

    raise ValueError(f"无法解析 LLM 输出: {raw[:200]}")


def _deduplicate_json(raw: str) -> str:
    """当 LLM 输出出现 JSON 重复时，取最后一段完整的 JSON。"""
    import re
    # 找所有匹配的顶层 JSON 对象（以 { 开头，到匹配的 } 结束）
    # 简单策略：找最后一个 } 作为结束，取从它往回的第一个完整 JSON
    closes = [m.end() for m in re.finditer(r'\}', raw)]
    if len(closes) >= 2:
        # 检查从不同位置开始的 JSON 结构
        for close_pos in reversed(closes):
            # 从 close_pos 往前找最近的 {
            prefix = raw[:close_pos]
            depth = 0
            start_pos = -1
            for i in range(len(prefix) - 1, -1, -1):
                if prefix[i] == '}':
                    depth += 1
                elif prefix[i] == '{':
                    depth -= 1
                    if depth < 0:
                        start_pos = i
                        break
            if start_pos >= 0:
                candidate = raw[start_pos:close_pos]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
    return raw


def _repair_truncated_json(raw: str) -> dict | None:
    """尝试修复被截断的 JSON：补齐缺失的括号和引号。"""
    import re
    # 找到最后一个完整的 key-value 对
    # 简单策略：找到最后一个 ", 后加 } 试试
    candidates = []
    raw_stripped = raw.strip()
    if raw_stripped.endswith('"'):
        candidates.append(raw_stripped + "\n}")
    if not raw_stripped.endswith('}'):
        # 截断在字符串中间：去掉最后一个不完整的字符串
        last_complete_entry = re.sub(r',\s*"[^"]*$', '', raw_stripped, count=1)
        last_complete_entry = re.sub(r',\s*$', '', last_complete_entry)
        candidates.append(last_complete_entry + "\n}")

    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            pass
    return None


def _extract_complete_entries(raw: str) -> dict | None:
    """从截断的 JSON 中提取已完成的顶层条目。"""
    import re
    # 找所有合法的 "key": [...] 对
    result = {}
    # 匹配格式: "doc_id": [...完整数组...]
    pattern = r'"([\w_-]+)"\s*:\s*\[((?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*)\]'
    matches = re.findall(pattern, raw)
    for key, array_str in matches:
        try:
            arr = json.loads(f"[{array_str}]")
            result[key] = arr
        except json.JSONDecodeError:
            continue
    return result if result else None


def _make_fake_llm():
    """假 LLM，用于 --dry-run 快速验证。"""
    import re
    class FakeGenLLM:
        def invoke(self, messages):
            from langchain_core.messages import AIMessage
            # 从 HumanMessage 里提取文档 ID，为每个生成假查询
            content = messages[-1].content
            doc_ids = re.findall(r'Doc ID: ([\w_-]+)', content)
            result = {}
            for i, did in enumerate(doc_ids):
                result[did] = [
                    f"[dry-run] 测试查询 {i+1}-1",
                    f"[dry-run] 测试查询 {i+1}-2",
                    f"[dry-run] 测试查询 {i+1}-3",
                    f"[dry-run] 测试查询 {i+1}-4",
                ]
            return AIMessage(content=json.dumps(result, ensure_ascii=False))
    return FakeGenLLM()


def main():
    parser = argparse.ArgumentParser(description="RAG 合成测试数据生成")
    parser.add_argument("--sample", type=int, default=50,
                        help="抽样文档数量（默认 50，0 表示全部）")
    parser.add_argument("--queries-per-doc", type=int, default=3,
                        help="每篇文档生成查询数（默认 3）")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="每批发送给 LLM 的文档数（默认 3）")
    parser.add_argument("--output", type=str, default="data/eval/synthetic_queries.jsonl",
                        help="输出路径（相对于项目根目录）")
    parser.add_argument("--dry-run", action="store_true",
                        help="使用假 LLM 快速测试格式")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载文档
    docs = load_all_documents(_DEFAULT_SOURCES)
    if not docs:
        logger.error("没有加载到任何文档，退出")
        return

    # 抽样
    if args.sample > 0 and args.sample < len(docs):
        docs = random.sample(docs, args.sample)
    logger.info("抽样 %d 篇文档用于生成", len(docs))

    # LLM
    if args.dry_run:
        llm = _make_fake_llm()
        logger.info("Dry-run 模式：使用假 LLM")
    else:
        llm = _make_llm()

    # 分批生成
    batch_size = args.batch_size
    all_cases = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        logger.info("生成批次 %d/%d (%d 篇文档)",
                    i // batch_size + 1, (len(docs) + batch_size - 1) // batch_size, len(batch))
        try:
            cases = generate_queries_for_batch(llm, batch, args.queries_per_doc)
            all_cases.extend(cases)
            logger.info("  产出 %d 条查询", len(cases))
        except Exception:
            logger.error("批次生成失败，跳过", exc_info=True)

    # 输出
    output_path = _PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    logger.info("生成完成: %d 条查询 → %s", len(all_cases), output_path)


if __name__ == "__main__":
    main()
