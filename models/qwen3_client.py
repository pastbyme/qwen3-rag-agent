"""
Qwen3模型API调用封装
支持两种模式：
1. API模式（vLLM/Ollama等OpenAI兼容API）
2. 模拟模式（Mock）- 无需任何外部API服务，用于演示
"""
import json
import logging
import random
from typing import List, Dict, Optional, Union
from openai import OpenAI, APIConnectionError

from config import config

logger = logging.getLogger(__name__)


# ========== 模拟响应数据库 ==========
MOCK_CLASSIFY_RESPONSES = {
    "qwen3": '''{
        "task_type": "simple_qa",
        "complexity": 0.3,
        "required_tools": ["retrieve_knowledge"],
        "query_intent": "了解Qwen3模型的基本信息",
        "sub_questions": [],
        "suggested_top_k": 5,
        "suggested_threshold": 0.65,
        "requires_web_search": false,
        "confidence": 0.92
    }''',
    "rag_agent": '''{
        "task_type": "multi_hop",
        "complexity": 0.75,
        "required_tools": ["retrieve_knowledge"],
        "query_intent": "比较RAG和Agent技术",
        "sub_questions": ["什么是RAG技术？", "什么是Agent技术？", "RAG和Agent如何结合使用？"],
        "suggested_top_k": 8,
        "suggested_threshold": 0.55,
        "requires_web_search": false,
        "confidence": 0.85
    }''',
    "calculation": '''{
        "task_type": "calculation",
        "complexity": 0.4,
        "required_tools": ["calculate"],
        "query_intent": "数学计算",
        "sub_questions": [],
        "suggested_top_k": 3,
        "suggested_threshold": 0.7,
        "requires_web_search": false,
        "confidence": 0.95
    }''',
    "default": '''{
        "task_type": "simple_qa",
        "complexity": 0.5,
        "required_tools": ["retrieve_knowledge"],
        "query_intent": "一般问答",
        "sub_questions": [],
        "suggested_top_k": 5,
        "suggested_threshold": 0.65,
        "requires_web_search": false,
        "confidence": 0.8
    }''',
    "summarization": '''{
        "task_type": "summarization",
        "complexity": 0.5,
        "required_tools": ["retrieve_knowledge", "summarize"],
        "query_intent": "文本摘要",
        "sub_questions": [],
        "suggested_top_k": 5,
        "suggested_threshold": 0.65,
        "requires_web_search": false,
        "confidence": 0.85
    }''',
    "comparison": '''{
        "task_type": "comparison",
        "complexity": 0.65,
        "required_tools": ["retrieve_knowledge"],
        "query_intent": "比较分析",
        "sub_questions": ["比较对象A的特点", "比较对象B的特点", "A和B的异同点"],
        "suggested_top_k": 7,
        "suggested_threshold": 0.6,
        "requires_web_search": false,
        "confidence": 0.82
    }''',
}

MOCK_ANSWER_RESPONSES = {
    "qwen3": "Qwen3是阿里巴巴通义千问团队开发的大语言模型系列，支持多语言理解、文本生成、工具使用等多种能力。它还支持思考模式和非思考模式的动态切换。@confidence=0.95",
    "rag_agent": "RAG（检索增强生成）通过从知识库检索相关信息来增强大模型生成能力。Agent则能自主决策、调用工具。两者结合：Agent负责决策何时检索、如何优化检索参数，RAG提供知识支撑，形成'Agent+ RAG'的智能问答架构。@confidence=0.88",
    "calculation": "计算结果为25。具体计算过程：(2+3)*5 = 5*5 = 25 @confidence=0.99",
    "default": "基于提供的上下文信息，您的查询涉及多个方面。具体来说，Qwen3模型在RAG和Agent技术的结合应用上展现出了强大的能力。@confidence=0.80",
    "summarization": "根据提供的文本，主要内容涉及人工智能和大语言模型技术的核心概念、RAG检索增强生成架构以及Agent智能体系统的应用。@confidence=0.85",
    "comparison": "比较分析如下：两者都是AI领域的重要技术。RAG侧重于外部知识检索与融合，Agent侧重于自主决策与工具调用。在实际应用中，两者常常结合使用。@confidence=0.82",
}


def get_mock_classify_response(query: str) -> str:
    """根据查询内容智能返回模拟分类结果"""
    q = query.lower()
    if any(k in q for k in ["qwen3", "qwen"]):
        return MOCK_CLASSIFY_RESPONSES["qwen3"]
    elif any(k in q for k in ["计算", "+", "-", "*", "/"]):
        return MOCK_CLASSIFY_RESPONSES["calculation"]
    elif any(k in q for k in ["rag", "agent"]) and any(k in q for k in ["区别", "联系", "结合", "比较"]):
        return MOCK_CLASSIFY_RESPONSES["rag_agent"]
    elif any(k in q for k in ["摘要", "总结", "概括"]):
        return MOCK_CLASSIFY_RESPONSES["summarization"]
    elif any(k in q for k in ["比较", "区别", "差异", "对比"]):
        return MOCK_CLASSIFY_RESPONSES["comparison"]
    return MOCK_CLASSIFY_RESPONSES["default"]


def get_mock_answer_response(query: str, context_count: int = 0) -> str:
    """根据查询内容智能返回模拟回答"""
    q = query.lower()
    if any(k in q for k in ["qwen3", "qwen"]):
        return MOCK_ANSWER_RESPONSES["qwen3"]
    elif any(k in q for k in ["计算", "+", "-", "*", "/", "="]):
        return MOCK_ANSWER_RESPONSES["calculation"]
    elif "rag" in q and "agent" in q:
        return MOCK_ANSWER_RESPONSES["rag_agent"]
    elif any(k in q for k in ["摘要", "总结"]):
        return MOCK_ANSWER_RESPONSES["summarization"]
    elif any(k in q for k in ["比较", "区别"]):
        return MOCK_ANSWER_RESPONSES["comparison"]
    return MOCK_ANSWER_RESPONSES["default"]


class Qwen3Client:
    """Qwen3模型客户端封装"""

    def __init__(self, model_config=None):
        self.model_config = model_config or config.model
        self.use_mock = self.model_config.use_mock  # 是否启用模拟模式

        if not self.use_mock:
            # 真实API模式
            try:
                self.client = OpenAI(
                    base_url=self.model_config.api_base,
                    api_key=self.model_config.api_key,
                )
                logger.info(f"API模式连接: {self.model_config.api_base}")
            except Exception as e:
                logger.warning(f"API初始化失败，自动切换到模拟模式: {e}")
                self.use_mock = True
                self.client = None
        else:
            logger.info("🎭 模拟模式已启用（无需任何API服务）")
            self.client = None

    def _build_messages(
        self,
        system_prompt: str,
        user_query: str,
        history: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        if tool_results:
            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.get("tool_call_id", ""),
                    "content": json.dumps(result.get("content", ""), ensure_ascii=False)
                })
        messages.append({"role": "user", "content": user_query})
        return messages

    def chat(
        self,
        system_prompt: str,
        user_query: str,
        history: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> Dict:
        """对话接口 - 根据模式自动选择API调用或模拟"""

        if self.use_mock:
            return self._mock_chat(system_prompt, user_query, history)

        return self._api_chat(system_prompt, user_query, history, tool_results, tools,
                              temperature, max_tokens, enable_thinking)

    def _mock_chat(self, system_prompt: str, user_query: str, history=None) -> Dict:
        """🎭 模拟模式 - 根据提示词类型返回模拟响应"""
        prompt_lower = system_prompt.lower()

        # === 任务分类器 ===
        if "任务类型" in prompt_lower or "任务分析" in prompt_lower or "分类" in prompt_lower:
            content = get_mock_classify_response(user_query)

        # === 问题分解 ===
        elif "分解" in prompt_lower or "子问题" in prompt_lower:
            content = '["子问题1: 了解相关概念", "子问题2: 分析关系", "子问题3: 综合回答"]'

        # === 答案生成（含置信度） ===
        elif "置信度" in prompt_lower or "上下文信息" in prompt_lower or "@confidence" in prompt_lower:
            content = get_mock_answer_response(user_query)

        # === 质量评估 ===
        elif "质量评估" in prompt_lower or "评分" in prompt_lower:
            content = '{"accuracy": 0.85, "completeness": 0.78, "relevance": 0.90, "overall": 0.84}'

        else:
            content = get_mock_answer_response(user_query)

        # 思考内容（模拟）
        thinking = f"【模拟思考】分析问题: {user_query}\n1. 理解用户意图\n2. 检索相关知识\n3. 组织回答"

        return {
            "content": content,
            "tool_calls": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "finish_reason": "stop",
            "thinking": thinking if (self.model_config.enable_thinking and
                                      enable_thinking if 'enable_thinking' in locals() else True) else None
        }

    def _api_chat(self, system_prompt, user_query, history, tool_results, tools,
                  temperature, max_tokens, enable_thinking):
        """🌐 真实API模式"""
        messages = self._build_messages(system_prompt, user_query, history, tool_results)

        kwargs = {
            "model": self.model_config.model_name,
            "messages": messages,
            "temperature": temperature or self.model_config.temperature,
            "max_tokens": max_tokens or self.model_config.max_tokens,
            "top_p": self.model_config.top_p,
        }

        if enable_thinking is None:
            enable_thinking = self.model_config.enable_thinking
        if enable_thinking:
            kwargs["extra_body"] = {"enable_thinking": True}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**kwargs)
            result = {
                "content": response.choices[0].message.content or "",
                "tool_calls": [],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "finish_reason": response.choices[0].finish_reason if response.choices else "",
            }
            if response.choices[0].message.tool_calls:
                for tc in response.choices[0].message.tool_calls:
                    tool_call = {
                        "id": tc.id,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        "type": tc.type,
                    }
                    result["tool_calls"].append(tool_call)
            if hasattr(response.choices[0].message, 'reasoning_content'):
                result["thinking"] = response.choices[0].message.reasoning_content
            return result

        except APIConnectionError as e:
            logger.warning(f"API连接失败，自动切换到模拟模式: {e}")
            self.use_mock = True
            return self._mock_chat(system_prompt, user_query, history)
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            self.use_mock = True
            return self._mock_chat(system_prompt, user_query, history)

    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        """获取嵌入向量（模拟模式下返回随机向量）"""
        if self.use_mock:
            return [[random.random() for _ in range(768)] for _ in texts]
        try:
            response = self.client.embeddings.create(
                model="text-embedding-v2",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception:
            from sentence_transformers import SentenceTransformer
            try:
                model = SentenceTransformer(config.rag.embedding_model)
                return model.encode(texts).tolist()
            except:
                return [[random.random() for _ in range(768)] for _ in texts]


_qwen3_client = None


def get_qwen3_client() -> Qwen3Client:
    global _qwen3_client
    if _qwen3_client is None:
        _qwen3_client = Qwen3Client()
    return _qwen3_client