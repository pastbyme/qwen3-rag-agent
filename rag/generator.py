"""
RAG生成模块 - 基于检索结果的答案生成
支持重试机制、置信度评估
"""
import json
import logging
import re
from typing import List, Dict, Optional, Any
from datetime import datetime

from config import config
from models.qwen3_client import get_qwen3_client

logger = logging.getLogger(__name__)


class Generator:
    """RAG答案生成器"""

    def __init__(self, client=None):
        self.client = client or get_qwen3_client()
        self.model_config = config.model
        self.max_retries = config.agent.max_retries

    def generate(
        self,
        query: str,
        context_docs: List[Dict] = None,
        history: List[Dict] = None,
        task_type: str = "simple_qa",
        complexity: float = 0.5,
    ) -> Dict:
        """
        基于上下文生成答案

        Args:
            query: 用户问题
            context_docs: 检索到的上下文文档列表（每项为 dict，含 content/text/source/score 等字段）
            history: 对话历史
            task_type: 任务类型
            complexity: 任务复杂度

        Returns:
            dict: 包含 answer, confidence, thinking 等字段
        """
        # ===== 修复2: 参数类型检查 =====
        if context_docs is not None and not isinstance(context_docs, list):
            logger.warning(f"context_docs 类型错误: {type(context_docs)}，已重置为空列表")
            context_docs = []
        # ================================

        # 构建上下文文本
        context_text = ""
        if context_docs:
            for i, doc in enumerate(context_docs, 1):
                content = doc.get("content", doc.get("text", ""))
                source = doc.get("source", doc.get("metadata", {}).get("source", f"文档{i}"))
                score = doc.get("score", doc.get("similarity", 0))
                context_text += f"[来源{i}: {source} (相关度: {score:.3f})]\n{content}\n\n"

        # 构建系统提示词
        system_prompt = (
            "你是一个基于RAG（检索增强生成）的智能问答助手。"
            "请根据以下提供的上下文信息回答问题。\n\n"
            "要求：\n"
            "1. 优先使用提供的上下文信息回答问题\n"
            "2. 如果上下文信息不足，请明确说明\n"
            "3. 在回答末尾添加 @confidence=X.XX 表示置信度（0-1）\n"
            "4. 保持回答简洁准确\n\n"
            f"任务类型: {task_type}\n"
            f"复杂度: {complexity:.2f}\n"
        )

        if context_text:
            system_prompt += f"\n【上下文信息】\n{context_text}\n"
        else:
            system_prompt += "\n【注意】当前没有检索到相关上下文，请基于自身知识回答，并适当降低置信度。\n"

        # 调用模型
        try:
            result = self.client.chat(
                system_prompt=system_prompt,
                user_query=query,
                history=history,
                temperature=0.7 if complexity < 0.6 else 0.8,
                max_tokens=self.model_config.max_tokens,
            )
        except Exception as e:
            logger.error(f"生成失败: {e}")
            result = {
                "content": f"生成回答时出错: {str(e)} @confidence=0.1",
                "tool_calls": [],
                "usage": {"total_tokens": 0},
                "finish_reason": "error",
            }

        content = result.get("content", "")
        thinking = result.get("thinking", "")

        # 解析置信度
        confidence = self._extract_confidence(content)

        # 去除置信度标记
        clean_content = re.sub(r'\s*@confidence=[0-9.]+', '', content).strip()

        return {
            "answer": clean_content,
            "confidence": confidence,
            "thinking": thinking,
            "raw_content": content,
            "usage": result.get("usage", {"total_tokens": 0}),
            "context_count": len(context_docs) if context_docs else 0,
        }

    def _extract_confidence(self, content: str) -> float:
        """从回答中提取置信度"""
        match = re.search(r'@confidence=([0-9.]+)', content)
        if match:
            return float(match.group(1))
        return 0.7  # 默认置信度

    def generate_with_retry(
        self,
        query: str,
        initial_context: List[Dict] = None,
        task_type: str = "simple_qa",
        complexity: float = 0.5,
        history: List[Dict] = None,
        **kwargs,
    ) -> Dict:
        """
        带重试机制的生成

        Args:
            query: 用户问题
            initial_context: 初始检索上下文（必须为 List[Dict]）
            task_type: 任务类型
            complexity: 任务复杂度
            history: 对话历史
            **kwargs: 其他参数（被滤除，只保留 generate 支持的参数）

        Returns:
            dict: 生成结果
        """
        # ===== 修复3: 类型检查 + 过滤无用参数 =====
        if initial_context is not None and not isinstance(initial_context, list):
            logger.warning(f"initial_context 类型错误: {type(initial_context)}，已重置为 None")
            initial_context = None
        # ==========================================

        current_context = initial_context or []

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"生成尝试 {attempt}/{self.max_retries}")

            try:
                # 只传 generate() 支持的参数
                result = self.generate(
                    query=query,
                    context_docs=current_context,
                    history=history,
                    task_type=task_type,
                    complexity=complexity,
                )

                confidence = result.get("confidence", 0)

                # 置信度达标或已达最大重试次数
                if confidence >= config.agent.confidence_threshold or attempt >= self.max_retries:
                    result["retries"] = attempt
                    result["context_used"] = len(current_context)
                    return result

                # 置信度不足，尝试优化上下文后重试
                logger.info(
                    f"置信度 {confidence:.3f} 低于阈值 "
                    f"{config.agent.confidence_threshold}，尝试重试"
                )

            except Exception as e:
                logger.error(f"生成尝试 {attempt} 失败: {e}")
                if attempt >= self.max_retries:
                    return {
                        "answer": f"经过多次尝试后仍无法生成满意回答: {str(e)}",
                        "confidence": 0.1,
                        "retries": attempt,
                        "context_used": len(current_context),
                        "error": str(e),
                    }

        # 保底返回
        return {
            "answer": "无法生成回答",
            "confidence": 0.0,
            "retries": self.max_retries,
            "context_used": len(current_context),
        }


class RetryManager:
    """重试管理器（兼容旧接口）"""

    def __init__(self, generator: Generator):
        self.generator = generator

    def generate_with_retry(
        self,
        query: str,
        initial_context: List[Dict] = None,
        task_type: str = "simple_qa",
        complexity: float = 0.5,
        history: List[Dict] = None,
        **kwargs,
    ) -> Dict:
        """带重试机制的生成（兼容接口，过滤不支持的参数）"""
        return self.generator.generate_with_retry(
            query=query,
            initial_context=initial_context,
            task_type=task_type,
            complexity=complexity,
            history=history,
        )


# 全局实例
_generator = None


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator