import asyncio
import os
import sys
import uuid

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.memory import TherapySessionMemory, SessionManager
from config.settings import settings
from pipeline.orchestrator import run_pipeline
from schemas.contracts.v1 import PipelineInput
from utils import format_thought_chain_output, format_emotion_analysis, get_timestamp


class InteractiveTherapySession:
    def __init__(self, use_mock: bool = True):
        self.session_id = str(uuid.uuid4())[:8]
        self.use_mock = use_mock

        if use_mock:
            print("使用Mock模式进行测试")
        else:
            print(f"连接模型: {settings.MODEL_NAME}")
            print(f"API地址: {settings.OPENAI_API_BASE}")

        self.session = SessionManager.get_session(self.session_id)

    async def chat(self, user_input: str) -> None:
        print(f"\n[{get_timestamp()}] 用户: {user_input}")

        output = run_pipeline(PipelineInput(
            text=user_input,
            session_id=self.session_id,
        ))

        safety = output.safety
        emotion = output.emotion
        route = output.route
        intervention = output.intervention

        if safety.get("blocked") or safety.get("level", 0) >= 2:
            print("\n⚠️ 安全警报已触发")

        print(f"  [路由] {route.get('route', '?')} (confidence={route.get('confidence', 0):.2f})")
        print(f"  [情绪] {emotion.get('primary_emotion', '?')} intensity={emotion.get('intensity', 0):.1f}")

        if intervention.get("chain_of_thought"):
            print(f"  [思维链] {intervention['chain_of_thought'][:120]}...")

        actions = intervention.get("action_items", [])
        if actions:
            print(f"  [建议] {', '.join(actions)}")

        print(f"\n[{get_timestamp()}] AI咨询师: {intervention.get('reply', '')}")

    def show_session_info(self) -> None:
        trend = self.session.get_emotion_trend()
        print("\n" + "=" * 50)
        print("会话信息")
        print("=" * 50)
        print(f"会话ID: {self.session_id}")
        print(f"消息数量: {self.session.metadata.message_count}")
        print(f"情绪趋势: {trend['trend']}")
        print(f"平均情绪强度: {trend['average_intensity']:.1f}")
        if self.session.metadata.key_topics:
            print(f"关键话题: {', '.join(self.session.metadata.key_topics)}")


async def main():
    print("\n" + "=" * 60)
    print("心理咨询AI助手 - 交互式测试")
    print("=" * 60)
    print("\n命令说明:")
    print("  - 直接输入文字进行对话")
    print("  - 输入 'info' 查看会话信息")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'new' 开始新会话")
    print("=" * 60)

    use_mock = input("\n是否使用Mock模型测试? (y/n, 默认y): ").strip().lower()
    use_mock = use_mock != 'n'

    session = InteractiveTherapySession(use_mock=use_mock)

    print(f"\n新会话已创建，ID: {session.session_id}")
    print("开始对话吧！（输入 'quit' 退出）")

    while True:
        try:
            user_input = input("\n你: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                session.show_session_info()
                print("\n感谢使用，再见！")
                break

            if user_input.lower() == 'info':
                session.show_session_info()
                continue

            if user_input.lower() == 'new':
                session = InteractiveTherapySession(use_mock=use_mock)
                print(f"\n新会话已创建，ID: {session.session_id}")
                continue

            await session.chat(user_input)

        except KeyboardInterrupt:
            print("\n\n会话已中断")
            session.show_session_info()
            break
        except Exception as e:
            print(f"\n错误: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
