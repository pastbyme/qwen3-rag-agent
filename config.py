"""
Qwen3 Agent RAG 系统配置
"""
from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ModelConfig:
    """Qwen3模型配置"""
    model_name: str = "qwen3:8b"
    api_base: str = "http://localhost:11434/v1"
    api_key: str = "EMPTY"

    use_mock: bool = False  # True=模拟模式, False=真实API模式

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    enable_thinking: bool = True


@dataclass
class RAGConfig:
    """RAG检索配置"""
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 128
    default_top_k: int = 5
    default_similarity_threshold: float = 0.65
    top_k_range: tuple = (1, 20)
    similarity_threshold_range: tuple = (0.3, 0.95)


@dataclass
class AgentConfig:
    """Agent配置"""
    available_tools: List[str] = field(default_factory=lambda: [
        "retrieve_knowledge",
        "web_search",
        "calculate",
        "summarize",
        "knowledge_graph_query",
    ])
    confidence_threshold: float = 0.7
    max_retries: int = 3
    max_reasoning_steps: int = 10
    memory_enabled: bool = True
    memory_ttl: int = 3600
    max_memory_items: int = 100


@dataclass
class GlobalConfig:
    """全局配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data_dir: str = "./data"
    document_dir: str = "./data/documents"
    vector_store_path: str = "./data/vector_store"
    test_data_path: str = "./data/test_questions.json"
    log_level: str = "INFO"
    log_file: str = "./logs/experiment3.log"


config = GlobalConfig()

os.makedirs(config.document_dir, exist_ok=True)
os.makedirs(config.vector_store_path, exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./results", exist_ok=True)