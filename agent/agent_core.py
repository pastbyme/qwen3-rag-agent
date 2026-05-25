"""
Agent核心引擎
"""
import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from config import config
from models.qwen3_client import get_qwen3_client
from agent.task_classifier import TaskClassifier, TaskType
from agent.tool_registry import tool_registry
from agent.memory import memory_manager
from rag.retriever import get_retriever
from rag.generator import Generator, RetryManager

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    step_number: int
    action: str
    input: Any
    output: Any
    duration: float
    confidence: float = 0.0


@dataclass
class AgentResult:
    final_answer: str
    steps: List[AgentStep]
    total_duration: float
    confidence: float
    tool_calls: List[Dict]
    memory_used: bool
    retries: int
    metadata: Dict = field(default_factory=dict)


class AgentCore:

    def __init__(self):
        self.client = get_qwen3_client()
        self.classifier = TaskClassifier()
        self.generator = Generator()
        self.retry_manager = RetryManager(generator=self.generator)
        self.retriever = None

    def _get_retriever(self):
        if self.retriever is None:
            self.retriever = get_retriever()
        return self.retriever

    def process(self, query: str, session_id: Optional[str] = None) -> AgentResult:
        start_time = time.time()
        steps = []
        tool_calls_log = []

        # 步骤1：任务分类
        step_start = time.time()
        task_analysis = self.classifier.classify(query)
        steps.append(AgentStep(
            step_number=1, action="task_classification",
            input=query, output=task_analysis,
            duration=time.time() - step_start,
            confidence=task_analysis.confidence,
        ))
        logger.info(
            f"任务分类: {task_analysis.task_type.value}, "
            f"复杂度={task_analysis.complexity:.2f}"
        )

        # 步骤2：记忆检索
        step_start = time.time()
        memory_context = ""
        memory_used = False
        if config.agent.memory_enabled:
            memory_context = memory_manager.get_context_for_query(query)
            if memory_context:
                memory_used = True
        steps.append(AgentStep(
            step_number=2, action="memory_retrieval",
            input=query,
            output={"memory_found": memory_used},
            duration=time.time() - step_start,
        ))

        # 步骤3：动态RAG检索
        step_start = time.time()
        retriever = self._get_retriever()
        top_k = task_analysis.suggested_top_k
        threshold = task_analysis.suggested_threshold
        if task_analysis.complexity > 0.7:
            top_k = min(top_k + 3, config.rag.top_k_range[1])
            threshold = max(threshold - 0.05, config.rag.similarity_threshold_range[0])

        # ===== 修复1：all_contexts 改为 List[Dict] =====
        all_contexts: List[Dict] = []
        if task_analysis.task_type == TaskType.MULTI_HOP_REASONING:
            sub_questions = task_analysis.sub_questions
            if not sub_questions:
                sub_questions = self.classifier.decompose_question(query)
            for sub_q in sub_questions:
                sub_docs = retriever.retrieve(
                    sub_q, top_k=max(3, top_k // 2),
                    similarity_threshold=threshold,
                )
                for doc in sub_docs:
                    all_contexts.append({
                        "content": doc.content,
                        "source": getattr(doc, "source", f"子问题检索: {sub_q}"),
                        "score": getattr(doc, "score", getattr(doc, "similarity", 0.5)),
                    })
        else:
            docs = retriever.retrieve(
                query, top_k=top_k,
                similarity_threshold=threshold,
            )
            for doc in docs:
                all_contexts.append({
                    "content": doc.content,
                    "source": getattr(doc, "source", f"检索结果"),
                    "score": getattr(doc, "score", getattr(doc, "similarity", 0.5)),
                })
        # ==============================================

        steps.append(AgentStep(
            step_number=3, action="rag_retrieval",
            input={"query": query, "top_k": top_k, "threshold": threshold},
            output={"documents_found": len(all_contexts)},
            duration=time.time() - step_start,
        ))

        # 步骤4：工具执行
        step_start = time.time()
        tools_needed = task_analysis.required_tools
        tool_results = []
        if task_analysis.task_type in [
            TaskType.CALCULATION,
            TaskType.SUMMARIZATION,
            TaskType.COMPARISON,
            TaskType.MULTI_HOP_REASONING,
        ]:
            for tool_name in tools_needed:
                tool = tool_registry.get_tool(tool_name)
                if tool:
                    logger.info(f"执行工具: {tool_name}")
                    if tool_name == "calculate":
                        result = tool_registry.execute_tool(
                            tool_name, {"expression": query}
                        )
                    elif tool_name == "summarize":
                        # 取前3条上下文拼接作为待摘要文本
                        context_text = " ".join(
                            [d["content"] for d in all_contexts[:3]]
                        )
                        result = tool_registry.execute_tool(
                            tool_name,
                            {"text": context_text, "max_length": 200},
                        )
                    elif tool_name == "web_search":
                        result = tool_registry.execute_tool(
                            tool_name, {"query": query, "max_results": 5}
                        )
                    elif tool_name == "knowledge_graph_query":
                        entity = query.split("的")[0] if "的" in query else query
                        result = tool_registry.execute_tool(
                            tool_name, {"entity": entity}
                        )
                    else:
                        result = {
                            "status": "skipped",
                            "message": "已在步骤3中执行",
                        }
                    tool_results.append({"tool": tool_name, "result": result})
                    tool_calls_log.append({
                        "tool": tool_name,
                        "arguments": {},
                        "result_status": result.get("status", "unknown"),
                    })
        steps.append(AgentStep(
            step_number=4, action="tool_execution",
            input={"tools": tools_needed},
            output={"tool_results": tool_results},
            duration=time.time() - step_start,
        ))

        # 步骤5：答案生成（含重试）
        step_start = time.time()

        # ===== 修复2：enhanced_context 统一为 List[Dict] =====
        enhanced_context: List[Dict] = list(all_contexts)  # 已经是 List[Dict]

        for tr in tool_results:
            if tr["result"].get("status") == "success":
                if "summary" in tr["result"]:
                    enhanced_context.append({
                        "content": f"[摘要结果]: {tr['result']['summary']}",
                        "source": f"工具: {tr['tool']}",
                        "score": 1.0,
                    })
                elif "result" in tr["result"]:
                    result_val = tr["result"]["result"]
                    enhanced_context.append({
                        "content": (
                            f"[计算结果]: {result_val}"
                            if not isinstance(result_val, str)
                            else result_val
                        ),
                        "source": f"工具: {tr['tool']}",
                        "score": 1.0,
                    })
                elif "results" in tr["result"]:
                    results_str = json.dumps(
                        tr["result"]["results"][:2], ensure_ascii=False
                    )
                    enhanced_context.append({
                        "content": f"[搜索/查询结果]: {results_str}",
                        "source": f"工具: {tr['tool']}",
                        "score": 1.0,
                    })

        if memory_context:
            enhanced_context.append({
                "content": memory_context,
                "source": "记忆模块",
                "score": 0.8,
            })
        # ====================================================

        # ===== 修复3：去掉多余的 top_k/threshold 参数 =====
        generate_result = self.retry_manager.generate_with_retry(
            query=query,
            initial_context=enhanced_context,
            # ← top_k 和 threshold 被移除，generate_with_retry 不接收它们
        )
        # ===================================================

        steps.append(AgentStep(
            step_number=5, action="answer_generation",
            input={"context_count": len(enhanced_context)},
            output={
                "answer": generate_result["answer"][:100],
                "confidence": generate_result["confidence"],
            },
            duration=time.time() - step_start,
            confidence=generate_result["confidence"],
        ))

        # 步骤6：记忆存储
        if config.agent.memory_enabled:
            memory_manager.store_interaction(
                question=query,
                answer=generate_result["answer"],
                metadata={
                    "task_type": task_analysis.task_type.value,
                    "confidence": generate_result["confidence"],
                    "tools_used": tools_needed,
                },
            )

        total_duration = time.time() - start_time
        return AgentResult(
            final_answer=generate_result["answer"],
            steps=steps,
            total_duration=total_duration,
            confidence=generate_result["confidence"],
            tool_calls=tool_calls_log,
            memory_used=memory_used,
            retries=generate_result.get("attempts_used", 1) - 1,
            metadata={
                "task_type": task_analysis.task_type.value,
                "complexity": task_analysis.complexity,
                "sub_questions": task_analysis.sub_questions,
                "top_k_used": top_k,
                "threshold_used": threshold,
                "retry_history": generate_result.get("retry_history", []),
                "total_tokens": generate_result.get("usage", {}).get("total_tokens", 0),
            },
        )