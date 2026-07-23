"""
测试声音列表功能
"""

import asyncio
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from multimodal.tts import TextToSpeech


async def test():
    try:
        voices = await TextToSpeech.list_voices("zh")
        print(f"找到 {len(voices)} 个中文声音")
        for v in voices[:5]:
            print(f"  {v.short_name}: {v.gender}, {v.locale}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(test())
