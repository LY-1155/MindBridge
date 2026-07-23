"""ONNX 文本情绪推理引擎。

使用 ONNX Runtime 进行中文情绪分类推理，
替代关键词匹配方案，支持语义级情绪判断和意图检测。

未安装 onnxruntime 或模型文件不存在时，is_ready=False，
调用方应回退到 KeywordEmotionEngine。
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

import numpy as np

from modules.emotion.base import IntentLabel, TextEmotionEngine, TextEmotionResult, _SUPPORTED_EMOTIONS

# 默认标签映射：模型输出索引 → 契约情绪标签
# 适配 ModelScope structbert 情绪分类模型（7 类：恐惧/愤怒/厌恶/喜好/悲伤/高兴/惊讶）
_DEFAULT_LABEL_MAP: Dict[int, str] = {
    0: "fear",       # 恐惧
    1: "anger",      # 愤怒
    2: "stress",     # 厌恶（近似映射）
    3: "happiness",  # 喜好
    4: "sadness",    # 悲伤
    5: "happiness",  # 高兴（与喜好合并）
    6: "anxiety",    # 惊讶 → anxiety（近似）
    # 注意：此模型不产出 neutral、confusion 标签
}

# -- 关键词复核：ONNX 模型无独立 anxiety 类，fear 类会吃掉"紧张焦虑"文本 --
_ANXIETY_SIGNALS = [
    "焦虑", "紧张", "担心", "不安", "心慌", "忐忑", "烦躁",
    "社交恐惧", "社恐", "怯场", "胆怯",
]
_FEAR_SIGNALS = [
    "恐惧", "恐怖", "惊悚", "毛骨悚然", "吓人", "噩梦",
    "死亡", "自杀", "自残", "伤害", "暴力",
]
_NEG_PREFIX = r"(没有|不|并非|算不上|谈不上|没那么|不怎么|不太)"

# 意图检测规则
_INFORMATION_PATTERNS = [
    r"(什么是|什么叫|怎么|如何|为什么|是什么|有哪些|介绍|解释|说明|定义|含义|意思|区别|关系|分类)",
    r"[?？]",
]
_EMOTION_PATTERNS = [
    r"(我觉得|我感觉|我好|我很|我真|真的太|太.*了|啊$|呢$|吧$|心里|心口)",
    r"(最近|一直|总是|老是|天天).*(?:焦虑|难过|伤心|烦|累|困|怕|慌|崩溃|绝望|堵|压抑)",
    r"(睡不着|吃不下|没兴趣|不想|懒得|受不了|撑不住)",
]


class ONNXEmotionEngine:
    """基于 ONNX 模型的中文文本情绪分类引擎。

    使用前需先运行 scripts/download_emotion_model.py 下载并导出 ONNX 模型。
    引擎加载失败时 is_ready=False，不影响服务启动（由 factory 降级）。
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        label_map: Optional[Dict[int, str]] = None,
    ) -> None:
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path or os.path.dirname(model_path)
        self._label_map = label_map or _DEFAULT_LABEL_MAP
        self._session = None
        self._tokenizer = None
        self._ready = False
        self._load()

    def _load(self) -> None:
        """尝试加载 ONNX 会话和 tokenizer。失败时不抛异常。"""
        try:
            import onnxruntime as ort

            if not os.path.exists(self._model_path):
                return

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = 2  # Ryzen 9 8940: 2 线程避免过度竞争

            self._session = ort.InferenceSession(
                self._model_path,
                sess_opts,
                providers=["CPUExecutionProvider"],
            )

            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
            self._ready = True
        except Exception:
            self._ready = False

    def predict(self, text: str) -> TextEmotionResult:
        if not self._ready:
            raise RuntimeError("ONNX engine not ready — model not loaded")

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128,
        )

        # ONNX 输入：convert_dtype + 确保 int64
        # 只传模型需要的字段，tokenizer 可能多产 token_type_ids
        model_input_names = {inp.name for inp in self._session.get_inputs()}
        onnx_inputs = {}
        for k, v in inputs.items():
            if k not in model_input_names:
                continue
            if hasattr(v, "numpy"):
                v = v.numpy()
            if hasattr(v, "dtype") and v.dtype != np.int64:
                v = v.astype(np.int64)
            onnx_inputs[k] = v

        logits = self._session.run(None, onnx_inputs)[0]
        probs = self._softmax(logits[0])

        # 构建 all_emotions
        n_labels = len(probs)
        all_emotions: Dict[str, float] = {}
        for i in range(n_labels):
            label = self._label_map.get(i, "neutral")
            all_emotions[label] = float(probs[i])

        # 主情绪
        primary_idx = int(probs.argmax())
        primary = self._label_map.get(primary_idx, "neutral")
        confidence = float(probs[primary_idx])

        # 意图检测
        intent = self._detect_intent(text, primary, confidence)

        # 关键词复核：ONNX 模型无独立 anxiety 类，"紧张焦虑"文本常误入 fear
        if primary == "fear":
            primary = self._refine_fear_vs_anxiety(text, primary, all_emotions)

        # 低置信度修正：happiness < 0.5 且含负面信号 → 大概率是 sadness/stress
        if primary == "happiness" and confidence < 0.5:
            corrected = self._refine_false_happiness(text, all_emotions)
            if corrected:
                primary = corrected

        # 低置信度 + 信息提问 → 回退 neutral（模型无 neutral 分类时的补偿）
        if confidence < 0.5 and intent == "information":
            primary = "neutral"
            # 同步修正 all_emotions，否则融合时 anger 分布会反拉回错误的主情绪
            all_emotions = {"neutral": 0.6, **{k: max(v * 0.3, 0.01) for k, v in all_emotions.items()}}

        return TextEmotionResult(
            primary_emotion=primary,
            confidence=round(confidence, 4),
            all_emotions=all_emotions,
            intent=intent,
            hit_count=0,  # ONNX 模式下无关键词命中数
            model_name="onnx_emotion_classifier",
        )

    def _detect_intent(self, text: str, primary: str, confidence: float) -> IntentLabel:
        """基于规则 + 模型信号的意图检测。

        优先检测信息提问（避免误入情绪干预），其次检测情感表达。
        """
        # 信息提问：疑问句式 → 即使有情绪词也不按情感处理
        for pat in _INFORMATION_PATTERNS:
            if re.search(pat, text):
                return "information"

        # 强情绪信号：高置信度负面情绪 + 第一人称
        if primary in ("anxiety", "sadness", "anger", "fear", "stress") and confidence >= 0.5:
            for pat in _EMOTION_PATTERNS:
                if re.search(pat, text):
                    return "emotion_expression"

        # 纯中性 → 闲聊
        if primary == "neutral" and confidence >= 0.6:
            return "casual_chat"

        return "unknown"

    def _refine_fear_vs_anxiety(self, text: str, primary: str, all_emotions: dict) -> str:
        """关键词复核：有焦虑信号 + 无恐惧硬信号 → fear 退回 anxiety。

        只处理 ONNX 模型将"紧张焦虑"误分类为"恐惧"的场景，
        有恐惧硬信号时保守保留 fear。
        """
        has_anx = self._has_signal(text, _ANXIETY_SIGNALS)
        has_fear = self._has_signal(text, _FEAR_SIGNALS)
        if has_anx and not has_fear:
            all_emotions["anxiety"] = all_emotions.pop("fear", 0.0)
            return "anxiety"
        return primary

    def _refine_false_happiness(self, text: str, all_emotions: dict) -> Optional[str]:
        """低置信度 happiness 修正：识别被误判为"高兴"的消极表达。

        模型看到"是的"（同意句）→ 倾向 happiness，但用户可能是在确认
        负面症状（"是的，我什么都不想做了"）。有负面信号时退回 sadness。
        """
        # 消极信号：否定 + 兴趣/动力相关的表达
        negative_signals = [
            "不想", "不打了", "没意思", "干不好", "做不好", "提不起",
            "累", "难受", "痛苦", "睡不好", "吃不下", "什么都不想做",
            "没劲", "没兴趣", "没精神", "不开心", "烦", "无聊",
        ]
        if self._has_signal(text, negative_signals):
            # 将 happiness 的分数转移给 sadness
            happy_score = all_emotions.pop("happiness", 0.0)
            all_emotions["sadness"] = all_emotions.get("sadness", 0.0) + happy_score
            return "sadness"
        return None

    @staticmethod
    def _has_signal(text: str, words: list) -> bool:
        """任一信号词命中，且前面没有否定前缀。"""
        for w in words:
            if re.search(_NEG_PREFIX + re.escape(w), text):
                continue
            if w in text:
                return True
        return False

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @property
    def model_name(self) -> str:
        return "onnx_emotion_classifier"

    @property
    def is_ready(self) -> bool:
        return self._ready
