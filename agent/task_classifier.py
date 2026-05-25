"""
Agent任务分类器
"""
import json
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from models.qwen3_client import get_qwen3_client
from config import config

logger = logging.getLogger(__name__)


class TaskType(Enum):
    SIMPLE_QA = "simple_qa"
    MULTI_HOP_REASONING = "multi_hop"
    CALCULATION = "calculation"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    CREATIVE_WRITING = "creative"
    UNKNOWN = "unknown"


@dataclass
class TaskAnalysis:
    task_type: TaskType
    complexity: float
    required_tools: List[str]
    query_intent: str
    sub_questions: List[str]
    suggested_top_k: int
    suggested_threshold: float
    requires_web_search: bool
    confidence: float


class TaskClassifier:

    def __init__(self):
        self.client = get_qwen3_client()

    def classify(self, query: str) -> TaskAnalysis:
        prompt = """你是一个智能任务分析专家。请分析用户的问题，并输出JSON格式的分析结果。

任务类型定义：
- simple_qa: 可以直接从知识库检索到答案的事实性问题
- multi_hop: 需要多步推理、整合多个信息源的复杂问题
- calculation: 需要进行数学计算的问题
- summarization: 需要对长文本进行摘要总结
- comparison: 需要比较多个实体或概念的问题
- creative: 创意写作、生成类任务

输出格式（严格JSON）：
{
    "task_type": "任务类型",
    "complexity": 0.5,
    "required_tools": ["工具名列表"],
    "query_intent": "查询意图描述",
    "sub_questions": ["子问题1", "子问题2"],
    "suggested_top_k": 5,
    "suggested_threshold": 0.65,
    "requires_web_search": false,
    "confidence": 0.9
}

可用工具说明：
- retrieve_knowledge: 知识库检索
- web_search: 网络搜索
- calculate: 数学计算
- summarize: 文本摘要
- knowledge_graph_query: 知识图谱查询

请分析用户问题并返回JSON。"""

        result = self.client.chat(
            system_prompt=prompt,
            user_query=query,
            temperature=0.3,
            max_tokens=1024,
            enable_thinking=False,
        )

        try:
            content = result["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            analysis_dict = json.loads(content.strip())
            task_analysis = TaskAnalysis(
                task_type=TaskType(analysis_dict.get("task_type", "unknown")),
                complexity=analysis_dict.get("complexity", 0.5),
                required_tools=analysis_dict.get("required_tools", []),
                query_intent=analysis_dict.get("query_intent", ""),
                sub_questions=analysis_dict.get("sub_questions", []),
                suggested_top_k=analysis_dict.get("suggested_top_k", config.rag.default_top_k),
                suggested_threshold=analysis_dict.get("suggested_threshold", config.rag.default_similarity_threshold),
                requires_web_search=analysis_dict.get("requires_web_search", False),
                confidence=analysis_dict.get("confidence", 0.8),
            )
            task_analysis.suggested_top_k = max(
                config.rag.top_k_range[0],
                min(config.rag.top_k_range[1], task_analysis.suggested_top_k)
            )
            task_analysis.suggested_threshold = max(
                config.rag.similarity_threshold_range[0],
                min(config.rag.similarity_threshold_range[1], task_analysis.suggested_threshold)
            )
            return task_analysis
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"任务分类解析失败: {e}")
            return TaskAnalysis(
                task_type=TaskType.UNKNOWN,
                complexity=0.5,
                required_tools=["retrieve_knowledge"],
                query_intent=query,
                sub_questions=[],
                suggested_top_k=config.rag.default_top_k,
                suggested_threshold=config.rag.default_similarity_threshold,
                requires_web_search=False,
                confidence=0.5,
            )

    def decompose_question(self, query: str) -> List[str]:
        prompt = f"""请将以下复杂问题分解为多个简单的子问题，以便逐步回答。
每个子问题应当独立可检索，且按逻辑顺序排列。

问题：{query}

请以JSON数组格式输出：["子问题1", "子问题2", ...]"""

        result = self.client.chat(
            system_prompt="你是一个问题分解专家。",
            user_query=prompt,
            temperature=0.3,
            max_tokens=1024,
            enable_thinking=False,
        )
        try:
            content = result["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            sub_questions = json.loads(content.strip())
            if isinstance(sub_questions, list):
                return sub_questions
        except Exception as e:
            logger.warning(f"问题分解失败: {e}")
        return [query]
