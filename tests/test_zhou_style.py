"""ZhouStyleRetriever 单元测试（注入 fake store，不依赖真实 Chroma/Ollama）"""

from modules.intervention.rag.zhou_style import ZhouStyleRetriever


class FakeStore:
    """模拟 ChromaStore.search() 返回结果的假实现。"""

    def __init__(self, results=None):
        self._results = results or []

    def search(self, query, top_k=3):
        return self._results[:top_k]


def _sample(text="患者：孩子不上学，我跟他爸天天为这事吵\n医生：是不是这个事让你们俩第一次坐下来了？",
            score=0.9):
    return {"id": "zhou_x", "text": text, "score": score}


# ── 检索 ────────────────────────────────────────────────────

def test_retrieve_empty_query():
    ret = ZhouStyleRetriever(chroma_store=FakeStore([_sample()]))
    assert ret.retrieve("") == []
    assert ret.retrieve("   ") == []


def test_retrieve_parses_human_doctor():
    ret = ZhouStyleRetriever(chroma_store=FakeStore([_sample()]))
    hits = ret.retrieve("孩子不上学")
    assert len(hits) == 1
    assert "孩子不上学" in hits[0]["human"]
    assert "坐下来了" in hits[0]["doctor"]


def test_retrieve_skips_non_doctor_text():
    """不含'医生：'分隔符的文本应被跳过。"""
    ret = ZhouStyleRetriever(chroma_store=FakeStore([
        {"id": "bad", "text": "纯知识条目没有分隔符", "score": 0.9},
    ]))
    assert ret.retrieve("测试") == []


def test_retrieve_sorted_and_capped():
    ret = ZhouStyleRetriever(chroma_store=FakeStore([
        _sample(score=0.8), _sample(score=0.95), _sample(score=0.5),
        _sample(score=0.7), _sample(score=0.6),
    ]))
    hits = ret.retrieve("测试", top_k=3)
    assert len(hits) == 3
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


# ── 格式化为 prompt 段 ───────────────────────────────────────

def test_format_for_prompt_empty():
    assert ZhouStyleRetriever(chroma_store=FakeStore()).format_for_prompt([]) == ""


def test_format_for_prompt_content():
    ret = ZhouStyleRetriever(chroma_store=FakeStore())
    text = ret.format_for_prompt([
        {"human": "孩子不上学，我跟他爸吵架", "doctor": "是不是这个事让你们俩第一次坐下来谈了？"},
    ], max_items=2)
    assert "虚构化" in text            # 隐私声明
    assert "孩子不上学" in text
    assert "坐下来" in text
    assert "周医生会回" in text


def test_format_for_prompt_caps_items():
    ret = ZhouStyleRetriever(chroma_store=FakeStore())
    items = [
        {"human": f"h{i}", "doctor": f"d{i}"} for i in range(4)
    ]
    text = ret.format_for_prompt(items, max_items=2)
    assert text.count("周医生会回") == 2


# ── 禁用态安全降级 ──────────────────────────────────────────

def test_disabled_returns_empty():
    """store 加载失败后 disabled：retrieve 返回空、is_available False，不抛异常。"""
    ret = ZhouStyleRetriever(chroma_store=None)
    ret._disabled = True
    assert ret.retrieve("测试") == []
    assert ret.is_available() is False
