"""
测试edge_tts声音列表
"""

import asyncio
import edge_tts


async def test_list_voices():
    voices = await edge_tts.list_voices()
    
    print(f"总共 {len(voices)} 个声音")
    print("\n前5个声音的详细信息:")
    
    for i, v in enumerate(voices[:5]):
        print(f"\n声音 {i+1}:")
        print(f"  键: {list(v.keys())}")
        for k, val in v.items():
            print(f"  {k}: {val}")


if __name__ == "__main__":
    asyncio.run(test_list_voices())
