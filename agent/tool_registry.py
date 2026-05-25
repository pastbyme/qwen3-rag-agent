"""
Agent工具注册与选择
"""
import re
import logging
import math
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass

# ===== 修复1：改用绝对导入 =====
from rag.retriever import Retriever
from models.qwen3_client import get_qwen3_client
# ==============================

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict
    func: Callable
    enabled: bool = True
    usage_count: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.usage_count, 1)

    def to_openai_tool(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


def tool_retrieve_knowledge(query: str, top_k: int = 5, **kwargs) -> Dict:
    """工具1：知识检索"""
    retriever = Retriever()
    results = retriever.retrieve(query, top_k=top_k)
    return {
        "status": "success",
        "results": [r.content for r in results],
        "total_found": len(results),
    }


def tool_web_search(query: str, max_results: int = 5) -> Dict:
    """工具2：网络搜索（模拟）"""
    mock_results = [
        {"title": f"搜索结果1: {query}", "snippet": f"关于'{query}'的相关信息...（模拟搜索结果）",
         "url": f"https://example.com/search?q={query}", "relevance_score": 0.85}
    ]
    logger.info(f"[模拟网络搜索] 查询: {query}")
    return {"status": "success", "results": mock_results, "total_found": len(mock_results)}


def tool_calculate(expression: str) -> Dict:
    """工具3：数学计算"""
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "sqrt": math.sqrt,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "pi": math.pi, "e": math.e,
    }
    cleaned = re.sub(r'[^0-9+\-*/().,%\s^a-zA-Z_]', '', expression)
    try:
        result = eval(cleaned, {"__builtins__": {}}, allowed_names)
        return {"status": "success", "expression": expression, "result": result}
    except Exception as e:
        return {"status": "error", "expression": expression, "error": str(e)}


def tool_summarize(text: str, max_length: int = 200) -> Dict:
    """工具4：文本摘要"""
    client = get_qwen3_client()
    prompt = f"请对以下文本进行不超过{max_length}字的摘要：\n\n{text}"
    result = client.chat(
        system_prompt="你是一个专业的文本摘要助手。",
        user_query=prompt,
        max_tokens=max_length * 2,
    )
    return {
        "status": "success",
        "summary": result["content"],
        "original_length": len(text),
    }


def tool_knowledge_graph_query(entity: str, relation_type: Optional[str] = None) -> Dict:
    """工具5：知识图谱查询（模拟）"""
    mock_relations = [
        {"entity": entity, "relation": "属于", "target": "人工智能领域"},
        {"entity": entity, "relation": "相关技术", "target": "机器学习、深度学习"},
    ]
    return {
        "status": "success",
        "entity": entity,
        "relations": mock_relations,
        "total_relations": len(mock_relations),
    }


class ToolRegistry:

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        self.register(Tool(
            name="retrieve_knowledge",
            description="从本地知识库检索与查询相关的文档片段，适用于回答事实性问题",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询语句"},
                    "top_k": {"type": "integer", "description": "返回的文档数量", "default": 5},
                },
                "required": ["query"],
            },
            func=tool_retrieve_knowledge,
        ))
        self.register(Tool(
            name="web_search",
            description="搜索互联网获取最新信息",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询语句"},
                    "max_results": {"type": "integer", "description": "最大返回结果数", "default": 5},
                },
                "required": ["query"],
            },
            func=tool_web_search,
        ))
        self.register(Tool(
            name="calculate",
            description="执行数学表达式计算",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["expression"],
            },
            func=tool_calculate,
        ))
        self.register(Tool(
            name="summarize",
            description="对长文本进行摘要总结",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "需要摘要的文本"},
                    "max_length": {"type": "integer", "description": "摘要最大字数", "default": 200},
                },
                "required": ["text"],
            },
            func=tool_summarize,
        ))
        self.register(Tool(
            name="knowledge_graph_query",
            description="查询知识图谱中实体的关系信息",
            parameters={
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "实体名称"},
                    "relation_type": {"type": "string", "description": "关系类型过滤（可选）"},
                },
                "required": ["entity"],
            },
            func=tool_knowledge_graph_query,
        ))

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_enabled_tools(self) -> List[Tool]:
        return [t for t in self._tools.values() if t.enabled]

    def get_openai_tools(self) -> List[Dict]:
        return [t.to_openai_tool() for t in self.get_enabled_tools()]

    def execute_tool(self, name: str, arguments: Dict) -> Dict:
        tool = self.get_tool(name)
        if not tool:
            return {"status": "error", "error": f"未知工具: {name}"}
        if not tool.enabled:
            return {"status": "error", "error": f"工具已禁用: {name}"}
        tool.usage_count += 1
        try:
            result = tool.func(**arguments)
            if result.get("status") == "success":
                tool.success_count += 1
            return result
        except Exception as e:
            logger.error(f"工具执行失败 [{name}]: {e}")
            return {"status": "error", "error": str(e)}

    def get_tool_statistics(self) -> Dict:
        stats = {}
        for name, tool in self._tools.items():
            stats[name] = {
                "usage_count": tool.usage_count,
                "success_count": tool.success_count,
                "success_rate": tool.success_rate,
            }
        return stats


tool_registry = ToolRegistry()