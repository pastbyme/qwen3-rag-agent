"""
Agent评估指标体系
"""
import time
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from config import config
from agent.agent_core import AgentCore
from agent.memory import memory_manager

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    accuracy: float = 0.0
    avg_response_time: float = 0.0
    multi_hop_solving_rate: float = 0.0
    tool_calling_accuracy: float = 0.0
    memory_efficiency: float = 0.0
    retry_success_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_tokens_used: float = 0.0
    detail: Dict = field(default_factory=dict)


class TestDataLoader:

    @staticmethod
    def load_test_questions() -> List[Dict]:
        test_data_path = config.test_data_path
        if not test_data_path or not __import__('os').path.exists(test_data_path):
            logger.warning(f"测试数据文件不存在: {test_data_path}，使用默认测试集")
            return TestDataLoader._create_default_test_set()
        try:
            with open(test_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"加载测试数据失败: {e}")
            return TestDataLoader._create_default_test_set()

    @staticmethod
    def _create_default_test_set() -> List[Dict]:
        return [
            {"question": "什么是Qwen3？", "type": "simple_qa",
             "expected_answer": ["大语言模型", "Qwen", "阿里巴巴"], "expected_tools": ["retrieve_knowledge"]},
            {"question": "RAG技术如何提升大语言模型的效果？", "type": "multi_hop",
             "expected_answer": ["检索", "生成", "增强"], "expected_tools": ["retrieve_knowledge"]},
            {"question": "计算 (2+3)*5 的结果是多少？", "type": "calculation",
             "expected_answer": ["25"], "expected_tools": ["calculate"]},
            {"question": "Python和Transformer有什么关系？", "type": "comparison",
             "expected_answer": ["编程语言", "神经网络", "架构"], "expected_tools": ["retrieve_knowledge"]},
            {"question": "Qwen3的Agent能力如何通过函数调用实现与外部工具的交互？", "type": "multi_hop",
             "expected_answer": ["函数调用", "工具", "agent"], "expected_tools": ["retrieve_knowledge"]},
        ]


class Evaluator:

    def __init__(self):
        self.agent = AgentCore()
        self.test_data = TestDataLoader.load_test_questions()

    def run_evaluation(self) -> EvaluationResult:
        logger.info(f"开始评估，测试问题数: {len(self.test_data)}")
        results = []
        for item in self.test_data:
            query = item["question"]
            question_type = item["type"]
            logger.info(f"评估问题 [{question_type}]: {query[:50]}...")

            # 增加控制台视觉分割
            print(f"\n▶ 正在评估 [{question_type}]: {query}")

            start_time = time.time()
            agent_result = self.agent.process(query)
            response_time = time.time() - start_time

            # ===== 修复 1：新增评估过程中的终端打印逻辑 =====
            print(f"🤖 Agent: {agent_result.final_answer}")
            print(f"\n📊 详情:")
            print(f"  置信度: {agent_result.confidence:.2%}")
            print(f"  响应时间: {response_time:.2f}s")
            print(f"  执行步骤: {len(agent_result.steps)}")
            print(f"  工具调用: {[tc.get('tool', 'unknown') for tc in agent_result.tool_calls]}")
            print(f"  重试次数: {agent_result.retries}")
            print(f"  使用记忆: {'✅' if agent_result.memory_used else '❌'}")
            print("-" * 60)
            # ===============================================

            accuracy = self._evaluate_accuracy(agent_result.final_answer, item.get("expected_answer", []))
            tool_accuracy = self._evaluate_tool_accuracy(agent_result.tool_calls, item.get("expected_tools", []))
            results.append({"question": query, "type": question_type, "answer": agent_result.final_answer,
                            "accuracy": accuracy, "response_time": response_time, "confidence": agent_result.confidence,
                            "tool_accuracy": tool_accuracy, "memory_used": agent_result.memory_used,
                            "retries": agent_result.retries, "tool_calls": agent_result.tool_calls,
                            "steps": len(agent_result.steps),
                            "tokens_used": agent_result.metadata.get("total_tokens", 0)})
        return self._compute_overall_metrics(results)

    def _evaluate_accuracy(self, answer: str, expected_keywords: List[str]) -> float:
        if not expected_keywords:
            return 0.5
        answer_lower = answer.lower()
        match_count = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        return match_count / len(expected_keywords)

    def _evaluate_tool_accuracy(self, actual_calls: List[Dict], expected_tools: List[str]) -> float:
        if not expected_tools:
            return 1.0
        actual_tools = [tc.get("tool", "") for tc in actual_calls]
        if not actual_tools:
            return 0.0
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        true_positives = len(expected_set & actual_set)
        precision = true_positives / max(len(actual_set), 1)
        recall = true_positives / max(len(expected_set), 1)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        return f1

    def _compute_overall_metrics(self, results: List[Dict]) -> EvaluationResult:
        if not results:
            return EvaluationResult()
        eval_result = EvaluationResult()
        accuracies = [r["accuracy"] for r in results]
        eval_result.accuracy = float(np.mean(accuracies))
        response_times = [r["response_time"] for r in results]
        eval_result.avg_response_time = float(np.mean(response_times))
        multi_hop_results = [r for r in results if r["type"] == "multi_hop"]
        if multi_hop_results:
            eval_result.multi_hop_solving_rate = float(np.mean([r["accuracy"] for r in multi_hop_results]))
        tool_accuracies = [r["tool_accuracy"] for r in results if r["tool_calls"]]
        if tool_accuracies:
            eval_result.tool_calling_accuracy = float(np.mean(tool_accuracies))
        memory_used_count = sum(1 for r in results if r["memory_used"])
        eval_result.memory_efficiency = memory_used_count / len(results)
        retry_counts = [r["retries"] for r in results]
        eval_result.retry_success_rate = 1.0 - (sum(retry_counts) / (len(results) * config.agent.max_retries))
        confidences = [r["confidence"] for r in results]
        eval_result.avg_confidence = float(np.mean(confidences))
        tokens = [r.get("tokens_used", 0) for r in results]
        eval_result.avg_tokens_used = float(np.mean(tokens))
        eval_result.detail = {"per_question_results": results,
                              "accuracy_by_type": self._group_by_type(results, "accuracy"),
                              "response_time_by_type": self._group_by_type(results, "response_time")}
        return eval_result

    def _group_by_type(self, results: List[Dict], metric: str) -> Dict:
        groups = defaultdict(list)
        for r in results:
            groups[r["type"]].append(r[metric])
        return {qtype: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "count": len(vals)}
                for qtype, vals in groups.items()}


def print_evaluation_report(eval_result: EvaluationResult):
    print("\n" + "="*60)
    print("            Agent系统评估报告")
    print("="*60)
    print(f"\n📊 总体指标:")
    print(f"  回答准确率:          {eval_result.accuracy:.2%}")
    print(f"  平均响应时间:        {eval_result.avg_response_time:.2f}s")
    print(f"  多跳问题解决率:      {eval_result.multi_hop_solving_rate:.2%}")
    print(f"  工具调用准确率:      {eval_result.tool_calling_accuracy:.2%}")
    print(f"  记忆利用效率:        {eval_result.memory_efficiency:.2%}")
    print(f"  重试成功率:          {eval_result.retry_success_rate:.2%}")
    print(f"  平均置信度:          {eval_result.avg_confidence:.2%}")
    print(f"  平均Token使用量:     {eval_result.avg_tokens_used:.0f}")
    print(f"\n📈 按问题类型分析:")
    for qtype, metrics in eval_result.detail.get("accuracy_by_type", {}).items():
        print(f"  {qtype}: 准确率={metrics['mean']:.2%} ± {metrics['std']:.2%} ({metrics['count']}条)")
    print(f"\n⏱️ 响应时间分析:")
    for qtype, metrics in eval_result.detail.get("response_time_by_type", {}).items():
        print(f"  {qtype}: {metrics['mean']:.2f}s ± {metrics['std']:.2f}s ({metrics['count']}条)")
    print("\n" + "="*60)
