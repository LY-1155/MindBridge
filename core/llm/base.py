"""
LLM适配器模块
=============

这个模块负责与大语言模型(LLM)进行交互。
它提供了一个统一的接口，可以连接不同的模型（如Qwen、GPT等）。

核心概念：
- Adapter（适配器）：一种设计模式，让不同模型的接口统一化
- LangChain：一个用于构建LLM应用的框架，提供了标准化的消息格式和模型接口
"""

from abc import ABC, abstractmethod  # ABC是抽象基类，用于定义接口规范
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from pydantic import BaseModel, Field  # Pydantic用于数据验证和设置管理
from langchain_core.language_models import BaseChatModel  # LangChain的聊天模型基类
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage  # 消息类型
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI  # OpenAI兼容的聊天模型

from config.settings import settings  # 导入配置


class LLMConfig(BaseModel):
    """
    LLM配置类
    
    这个类定义了连接大模型需要的所有配置参数。
    使用Pydantic的Field来设置默认值，这些默认值来自settings配置。
    
    属性说明：
        model_name: 模型名称，如 "qwen2.5-7b-instruct"
        temperature: 温度参数，控制输出的随机性（0-1，越高越随机）
        max_tokens: 最大生成token数，控制回复长度
        api_key: API密钥，用于认证
        api_base: API基础地址，可以是本地部署的地址
        streaming: 是否启用流式输出（逐字显示）
        timeout: 请求超时时间（秒）
    """
    model_name: str = Field(default_factory=lambda: settings.MODEL_NAME)
    temperature: float = Field(default_factory=lambda: settings.TEMPERATURE)
    max_tokens: int = Field(default_factory=lambda: settings.MAX_TOKENS)
    api_key: str = Field(default_factory=lambda: settings.OPENAI_API_KEY)
    api_base: str = Field(default_factory=lambda: settings.OPENAI_API_BASE)
    streaming: bool = True
    timeout: int = 60
    # 透传给 ChatOpenAI 的额外参数（如 {"extra_body": {"enable_thinking": False}}）。
    # 默认 None，不影响现有调用方；用于按调用关闭 qwen3.x 思考模式以降低延迟。
    model_kwargs: Optional[Dict[str, Any]] = None


class BaseLLMAdapter(ABC):
    """
    LLM适配器基类（抽象类）
    
    这是一个抽象基类，定义了所有LLM适配器必须实现的接口。
    使用抽象类的好处是：不同的模型可以有不同的实现，但对外接口统一。
    
    设计模式：模板方法模式
    - 基类定义算法骨架（invoke, ainvoke等方法）
    - 子类实现具体细节（_create_llm方法）
    
    属性：
        config: LLM配置对象
        _llm: LangChain聊天模型实例（延迟初始化）
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化适配器
        
        Args:
            config: 可选的配置对象，如果不提供则使用默认配置
        """
        self.config = config or LLMConfig()
        self._llm: Optional[BaseChatModel] = None  # 私有属性，存储模型实例

    @abstractmethod
    def _create_llm(self) -> BaseChatModel:
        """
        创建LLM实例（抽象方法）
        
        这是一个抽象方法，子类必须实现。
        不同的模型有不同的创建方式，所以由子类决定具体实现。
        
        Returns:
            BaseChatModel: LangChain的聊天模型实例
        """
        pass

    @property
    def llm(self) -> BaseChatModel:
        """
        获取LLM实例（属性访问器）
        
        使用@property装饰器，让llm像属性一样访问，但实际是方法调用。
        实现了"延迟初始化"模式：只有在真正需要时才创建模型实例。
        
        Returns:
            BaseChatModel: LangChain的聊天模型实例
        """
        if self._llm is None:
            # 第一次访问时才创建实例
            self._llm = self._create_llm()
        return self._llm

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """
        同步调用模型
        
        发送消息列表给模型，等待完整回复。
        
        Args:
            messages: 消息列表，包含对话历史
            **kwargs: 额外参数，会传递给模型
            
        Returns:
            AIMessage: 模型的回复消息
        """
        return self.llm.invoke(messages, **kwargs)

    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """
        异步调用模型
        
        与invoke类似，但是异步的。在等待模型回复时不会阻塞程序，
        可以同时处理其他任务。这对于Web服务器特别重要。
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Returns:
            AIMessage: 模型的回复消息
        """
        return await self.llm.ainvoke(messages, **kwargs)

    def stream(self, messages: List[BaseMessage], **kwargs) -> AsyncIterator[str]:
        """
        流式调用模型（同步版本）
        
        流式输出：模型生成一个字就返回一个字，而不是等全部生成完。
        这样用户可以更快看到回复开始，体验更好。
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Returns:
            AsyncIterator[str]: 返回一个迭代器，每次yield一个文本片段
        """
        return self.llm.stream(messages, **kwargs)

    async def astream(self, messages: List[BaseMessage], **kwargs) -> AsyncIterator[str]:
        """
        流式调用模型（异步版本）
        
        异步迭代器，适合在异步环境中使用。
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Yields:
            str: 每次yield模型生成的一个文本片段
        """
        async for chunk in self.llm.astream(messages, **kwargs):
            # chunk是模型返回的一个片段，我们提取其中的content（文本内容）
            yield chunk.content


class OpenAICompatibleAdapter(BaseLLMAdapter):
    """
    OpenAI兼容适配器
    
    这个适配器可以连接任何兼容OpenAI API的模型服务。
    包括：OpenAI官方API、Azure OpenAI、vLLM部署的模型、本地部署的Qwen等。
    
    继承自BaseLLMAdapter，只需要实现_create_llm方法。
    """
    
    def _create_llm(self) -> BaseChatModel:
        """
        创建OpenAI兼容的聊天模型实例
        
        使用LangChain提供的ChatOpenAI类，它实现了OpenAI API的调用。
        
        Returns:
            BaseChatModel: 配置好的聊天模型实例
        """
        return ChatOpenAI(
            model=self.config.model_name,        # 模型名称
            temperature=self.config.temperature,  # 温度参数
            max_tokens=self.config.max_tokens,    # 最大token数
            api_key=self.config.api_key,          # API密钥
            base_url=self.config.api_base,        # API地址（可以是本地地址）
            streaming=self.config.streaming,      # 是否流式输出
            timeout=self.config.timeout,          # 超时时间
            model_kwargs=self.config.model_kwargs or {},  # 透传额外参数（关闭思考等）
        )


class QwenAdapter(OpenAICompatibleAdapter):
    """
    Qwen模型适配器
    
    专门为阿里Qwen系列模型设计的适配器。
    继承自OpenAICompatibleAdapter，因为Qwen支持OpenAI兼容接口。
    
    添加了chat_with_system方法，用于带系统提示词的对话。
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化Qwen适配器
        
        Args:
            config: 可选的配置对象
        """
        super().__init__(config)  # 调用父类初始化
        self.default_system_prompt = "你是一个有帮助的AI助手。"

    async def chat_with_system(
        self,
        user_input: str,
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        带系统提示词的对话
        
        这是最常用的对话方法，包含三个部分：
        1. 系统提示词：定义AI的角色和行为规范
        2. 对话历史：之前的对话内容
        3. 用户输入：当前用户说的话
        
        Args:
            user_input: 用户当前输入
            system_prompt: 系统提示词，定义AI的角色
            history: 对话历史，格式如 [{"role": "user", "content": "..."}, ...]
            
        Returns:
            str: 模型的回复文本
        """
        # 构建消息列表
        # LangChain使用不同的消息类型来区分角色
        messages = [SystemMessage(content=system_prompt)]  # 系统消息

        # 添加对话历史
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # 添加当前用户输入
        messages.append(HumanMessage(content=user_input))

        # 异步调用模型
        response = await self.ainvoke(messages)
        return response.content  # 返回回复文本

    async def stream_chat_with_system(
        self,
        user_input: str,
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        """
        带系统提示词的流式对话
        
        与chat_with_system类似，但是流式输出。
        适合需要实时显示回复的场景（如聊天界面）。
        
        Args:
            user_input: 用户当前输入
            system_prompt: 系统提示词
            history: 对话历史
            
        Yields:
            str: 每次yield一个文本片段
        """
        # 构建消息列表（与chat_with_system相同）
        messages = [SystemMessage(content=system_prompt)]

        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_input))

        # 流式调用模型
        async for chunk in self.astream(messages):
            yield chunk


class MockLLMAdapter(BaseLLMAdapter):
    """
    模拟LLM适配器（用于测试）
    
    这个适配器不连接真实的模型，而是返回预设的回复。
    用于在没有模型服务的情况下测试代码逻辑。
    
    使用场景：
    - 开发调试时不需要启动模型服务
    - 单元测试时模拟模型行为
    - 演示功能时快速响应
    """
    
    def _create_llm(self) -> BaseChatModel:
        """
        创建模拟的聊天模型
        
        使用LangChain提供的FakeListChatModel，它会按顺序返回预设的回复。
        
        Returns:
            BaseChatModel: 模拟的聊天模型实例
        """
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        return FakeListChatModel(
            responses=[
                # 预设的回复列表，会按顺序返回
                "我理解你现在的感受。这听起来确实是一个困难的处境。你能告诉我更多关于你的感受吗？",
                "我听到了你的担忧。这是一种很正常的反应。让我们一起探索一下，是什么让你感到这样？",
                "感谢你与我分享这些。你的感受是完全可以理解的。你觉得有什么可以帮助你感觉好一点吗？",
            ]
        )

    async def chat_with_system(
        self,
        user_input: str,
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        模拟带系统提示词的对话
        
        与QwenAdapter.chat_with_system接口相同，但返回预设回复。
        这样测试代码可以无缝切换真实模型和模拟模型。
        """
        messages = [SystemMessage(content=system_prompt)]
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_input))
        response = await self.ainvoke(messages)
        return response.content

    async def stream_chat_with_system(
        self,
        user_input: str,
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        """
        模拟流式对话
        
        直接返回完整回复，模拟流式输出效果。
        """
        response = await self.chat_with_system(user_input, system_prompt, history)
        yield response  # 一次性返回完整内容


def get_llm_adapter(adapter_type: str = "openai_compatible", config: Optional[LLMConfig] = None) -> BaseLLMAdapter:
    """
    获取LLM适配器的工厂函数
    
    这是一个工厂函数，根据类型返回对应的适配器实例。
    使用工厂模式的好处是：调用者不需要知道具体创建哪个类，
    只需要指定类型字符串即可。
    
    Args:
        adapter_type: 适配器类型
            - "openai_compatible": OpenAI兼容适配器
            - "qwen": Qwen模型适配器
            - "mock": 模拟适配器（测试用）
        config: 可选的配置对象
        
    Returns:
        BaseLLMAdapter: 对应类型的适配器实例
        
    Example:
        >>> adapter = get_llm_adapter("qwen")
        >>> response = await adapter.chat_with_system("你好", "你是一个助手")
    """
    # 类型到类的映射字典
    adapters = {
        "openai_compatible": OpenAICompatibleAdapter,
        "qwen": QwenAdapter,
        "mock": MockLLMAdapter,
    }

    # 获取对应的类，如果类型不存在则使用默认的OpenAI兼容适配器
    adapter_class = adapters.get(adapter_type, OpenAICompatibleAdapter)
    
    # 创建并返回实例
    return adapter_class(config=config)
