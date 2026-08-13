"""
敏感词过滤器 - 增强版
支持：正则匹配、谐音变体、中英文、大小写、日志记录
输出格式: (blocked, level, matched_keywords)
"""

import re
import json
import logging
import os
from typing import Tuple, List, Dict, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class SensitivityFilter:
    _DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "sensitive_words.json")

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = self._DEFAULT_CONFIG
        with open(config_path, 'r', encoding='utf-8') as f:
            self.words_data = json.load(f)

        # 构建词汇集合
        self.level_1_words = self._extract_words("level_1")
        self.level_2_words = self._extract_words("level_2")

        # 构建变体映射 {变体: 原词}
        self.variant_map = self._build_variant_map()

        # 编译正则表达式
        self.patterns = {
            "level_1": self._compile_regex(self.level_1_words),
            "level_2": self._compile_regex(self.level_2_words),
            "variants": self._compile_variants()
        }

        # 防重复触发记录 {user_id: last_trigger_time}
        self.trigger_history: Dict[str, datetime] = {}
        self.cooldown_seconds = 300  # 5分钟冷却期

        logger.info(f"敏感词过滤器初始化完成，一级词数: {len(self.level_1_words)}，二级词数: {len(self.level_2_words)}")

    def _extract_words(self, level: str) -> Set[str]:
        """提取某一级别的所有词汇"""
        words = set()
        for category in self.words_data.get(level, {}):
            words.update(self.words_data[level][category])
        return words

    def _build_variant_map(self) -> Dict[str, str]:
        """构建变体到原词的映射"""
        variant_map = {}
        for variant_group in self.words_data.get("variants", {}).values():
            if isinstance(variant_group[0], list):
                # 格式: [["原词", "变体1", "变体2"], ...]
                for group in variant_group:
                    original = group[0]
                    for variant in group[1:]:
                        variant_map[variant.lower()] = original
        return variant_map

    def _compile_regex(self, words: Set[str]) -> re.Pattern:
        """编译词汇为正则表达式"""
        if not words:
            return re.compile(r'(?!x)x')  # 永不匹配的模式
        # 按长度降序排列，优先匹配长词
        sorted_words = sorted(words, key=len, reverse=True)
        pattern_str = "|".join([re.escape(word) for word in sorted_words])
        return re.compile(pattern_str, re.IGNORECASE)

    def _compile_variants(self) -> re.Pattern:
        """编译变体词为正则表达式"""
        if not self.variant_map:
            return re.compile(r'(?!x)x')
        variants = sorted(self.variant_map.keys(), key=len, reverse=True)
        pattern_str = "|".join([re.escape(v) for v in variants])
        return re.compile(pattern_str, re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        """
        清洗文本，去除特殊字符干扰
        使 "自 杀"、"自.杀"、"自*杀" 等能被匹配
        """
        # 移除非中英文数字的字符
        return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)

    # \u5426\u5b9a\u8bed\u5883\u524d\u7f00\uff1a\u547d\u4e2d\u8bcd\u524d 5 \u5b57\u7b26\u5185\u51fa\u73b0\u4efb\u4e00\u5373\u89c6\u4e3a\u5426\u5b9a\u4fee\u9970\u3002
    # \u523b\u610f\u4e0d\u5305\u542b\u5355\u72ec"\u4e0d"\uff08\u907f\u514d\u8bef\u4f24"\u4e0d\u53bb\u60f3\u81ea\u6740\u7684\u4e8b"\u8fd9\u7c7b\u771f\u5b9e\u610f\u5ff5\u8868\u8fbe\uff09\uff0c
    # \u4e0e modules/assessment/risk_anchors.py \u7684 DENY_KEYWORDS \u8bed\u4e49\u5bf9\u9f50\uff08ADR-0013\uff09\u3002
    _NEGATION_PREFIXES = (
        "\u6ca1\u6709", "\u6ca1", "\u4e0d\u4f1a", "\u4e0d\u80fd", "\u4e0d\u662f", "\u4e0d\u60f3", "\u5e76\u975e", "\u5e76\u6ca1\u6709",
        "\u522b", "\u65e0", "\u672a", "\u4e0d\u66fe", "\u4ece\u672a", "\u4e0d\u518d", "\u518d\u4e0d", "\u4ece\u4e0d", "\u4e0d\u8be5",
        "\u4e0d\u5e94", "\u4e0d\u8981",
    )
    _NEGATION_WINDOW = 5

    def _is_negated(self, text: str, keyword: str) -> bool:
        """\u547d\u4e2d\u8bcd\u5728\u6587\u672c\u4e2d\u6240\u6709\u51fa\u73b0\u4f4d\u7f6e\u662f\u5426\u90fd\u88ab\u5426\u5b9a\u8bcd\u524d\u7f6e\u4fee\u9970\u3002

        "\u5012\u4e5f\u6ca1\u6709\u4f24\u5bb3\u81ea\u5df1" \u2192 True\uff08\u964d\u7ea7\uff09
        "\u6211\u60f3\u4f24\u5bb3\u81ea\u5df1"    \u2192 False\uff08\u4fdd\u6301\u786c\u62e6\u622a\uff09
        "\u6ca1\u6709\u60f3\u8fc7\u8981\u4f24\u5bb3\u81ea\u5df1" \u2192 True\uff08window=5 \u8986\u76d6"\u6ca1\u6709"\uff09
        """
        positions: List[int] = []
        start = 0
        while True:
            idx = text.find(keyword, start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + len(keyword)
        if not positions:
            return False
        return all(self._negated_at(text, idx) for idx in positions)

    def _negated_at(self, text: str, idx: int) -> bool:
        start = max(0, idx - self._NEGATION_WINDOW)
        window = text[start:idx]
        return any(p in window for p in self._NEGATION_PREFIXES)

    def check(self, text: str, user_id: str = None) -> Tuple[bool, int, List[str]]:
        """
        检查文本中的敏感词

        Args:
            text: 待检查的文本
            user_id: 用户ID,用于防重复触发

        Returns:
            Tuple[bool, int, List[str]]: (是否拦截, 风险等级, 命中的敏感词列表)
        """
        matched_keywords = []
        matched_originals = set()

        # 清洗文本
        cleaned_text = self.clean_text(text)

        # 检查变体词
        variant_matches = self.patterns["variants"].findall(cleaned_text.lower())
        for variant in variant_matches:
            if variant in self.variant_map:
                matched_originals.add(self.variant_map[variant])
                matched_keywords.append(f"{self.variant_map[variant]}(变体:{variant})")

        # 检查一级拦截词
        level_1_matches = self.patterns["level_1"].findall(cleaned_text)
        if level_1_matches:
            unique = list(set(level_1_matches))
            # 否定语境：命中的一级词若全部被否定词前置修饰（"倒也没有伤害自己"），
            # 降级为二级警告（记录但不拦截），交给 LLM 语义评估器二次裁决（ADR-0013）。
            # 避免"没有/不会伤害自己"这类否认表达被误判为真实危机。
            negated = [kw for kw in unique if self._is_negated(cleaned_text, kw)]
            non_negated = [kw for kw in unique if kw not in negated]

            if non_negated:
                matched_originals.update(non_negated)
                # 检查是否在冷却期内
                if self._should_trigger_emergency(user_id):
                    logger.warning(f"一级拦截触发！用户: {user_id}, 命中词: {non_negated}, 原文: {text[:50]}...")
                    return True, 1, list(non_negated) + matched_keywords
                else:
                    logger.info(f"一级拦截但处于冷却期，用户: {user_id}")
                    return False, 1, list(non_negated) + matched_keywords

            if negated:
                matched_originals.update(negated)
                logger.info(f"一级词命中但处于否定语境，降级为二级警告: {negated}")
                return False, 2, list(negated) + matched_keywords

        # 检查二级警告词
        level_2_matches = self.patterns["level_2"].findall(cleaned_text)
        if level_2_matches:
            matched_originals.update(level_2_matches)
            logger.info(f"二级警告触发，命中词: {level_2_matches}, 原文: {text[:50]}...")
            return False, 2, list(set(level_2_matches)) + matched_keywords

        # 有变体匹配但无直接匹配
        if matched_keywords:
            # 变体词按二级处理
            return False, 2, matched_keywords

        return False, 0, []

    def _should_trigger_emergency(self, user_id: str = None) -> bool:
        """
        检查是否应该触发紧急推送（防重复触发机制）
        同一用户在冷却期内不重复触发
        """
        if user_id is None:
            return True

        now = datetime.now()
        if user_id in self.trigger_history:
            elapsed = (now - self.trigger_history[user_id]).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False

        self.trigger_history[user_id] = now
        return True

    def get_stats(self) -> Dict:
        """获取词库统计信息"""
        return {
            "level_1_count": len(self.level_1_words),
            "level_2_count": len(self.level_2_words),
            "variant_count": len(self.variant_map),
            "categories": {
                "level_1": list(self.words_data.get("level_1", {}).keys()),
                "level_2": list(self.words_data.get("level_2", {}).keys())
            }
        }


# ============ 单元测试 ============
def run_tests():
    """运行单元测试"""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("敏感词过滤器单元测试")
    print("=" * 60)

    sf = SensitivityFilter()
    test_cases = [
        # (输入文本, 期望拦截, 期望等级, 描述)
        ("我觉得最近压力很大", False, 0, "正常文本"),
        ("我真的不想活了，想去死", False, 2, "纯意念词→软警告"),
        ("感觉人生没意义，好痛苦", False, 2, "二级敏感词"),
        ("自*杀是不对的", False, 2, "谐音变体(带符号)→意念软警告"),
        ("紫砂是一种陶瓷工艺", False, 2, "谐音变体(紫砂)"),
        ("s杀是不好的", False, 2, "字母谐音"),
        ("I want to kill myself", False, 2, "英文意念词→软警告"),
        ("最近心情很压抑", False, 2, "心理关键词"),
        ("你好，今天天气不错", False, 0, "完全正常文本"),
        ("跳lou这种行为很危险", False, 2, "拼音变体"),
        # 计划/即刻/主动意图 → 硬拦（收窄后的 level_1）
        ("我现在就去死", True, 1, "即刻行动短语硬拦"),
        ("我现在就要去死", True, 1, "即刻行动短语硬拦"),
        ("我已经决定结束自己", True, 1, "已决意计划硬拦"),
        ("我要自杀", True, 1, "主动第一人称意图硬拦"),
        ("我想自杀", True, 1, "主动第一人称意图硬拦"),
        ("遗书写好了", True, 1, "准备/计划硬拦"),
        # 否定语境降级（ADR-0013 风险词命中≠危机触发）
        ("我没有伤害自己，只是很难受", False, 2, "否定语境降级(没有伤害自己)"),
        ("倒也没有伤害自己", False, 2, "否定语境降级(倒也没有)"),
        ("我不会想自杀的", False, 2, "否定语境降级(不会)"),
        ("我还没有决定结束自己", False, 2, "否定语境降级(还没有决定)"),
        ("我不会现在就去死", False, 2, "否定语境降级(不会)"),
        ("我想伤害自己", True, 1, "肯定语境保持硬拦"),
    ]

    passed = 0
    failed = 0

    for i, (text, expect_blocked, expect_level, desc) in enumerate(test_cases, 1):
        blocked, level, keywords = sf.check(text, user_id=f"test_user_{i}")
        status = "✓" if (blocked == expect_blocked and level == expect_level) else "✗"

        if blocked == expect_blocked and level == expect_level:
            passed += 1
        else:
            failed += 1

        print(f"\n测试 {i}: {desc}")
        print(f"  输入: {text}")
        print(f"  结果: 拦截={blocked}, 等级={level}, 命中词={keywords}")
        print(f"  期望: 拦截={expect_blocked}, 等级={expect_level}")
        print(f"  状态: {status}")

    print("\n" + "=" * 60)
    print(f"测试结果: 通过 {passed}/{len(test_cases)}, 失败 {failed}/{len(test_cases)}")
    print("=" * 60)

    # 打印词库统计
    stats = sf.get_stats()
    print(f"\n词库统计:")
    print(f"  一级敏感词: {stats['level_1_count']} 个")
    print(f"  二级敏感词: {stats['level_2_count']} 个")
    print(f"  变体映射: {stats['variant_count']} 个")


if __name__ == "__main__":
    run_tests()
