"""
RAG检索器（支持动态参数调整）
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from config import config

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    content: str
    source: str
    score: float
    metadata: Dict = None


class Retriever:

    def __init__(self):
        self.embedding_model = self._load_embedding_model()
        self.documents = self._load_documents()
        self.document_embeddings = self._compute_embeddings()

    def _load_embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.rag.embedding_model)
            logger.info(f"嵌入模型加载完成: {config.rag.embedding_model}")
            return model
        except Exception as e:
            logger.warning(f"嵌入模型加载失败: {e}，使用随机嵌入")
            return None

    def _load_documents(self) -> List[Dict]:
        documents = []
        if not os.path.exists(config.document_dir):
            logger.warning(f"文档目录不存在: {config.document_dir}")
            return self._create_sample_documents()
        for filename in os.listdir(config.document_dir):
            if filename.endswith(".txt") or filename.endswith(".md"):
                filepath = os.path.join(config.document_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    chunks = self._chunk_text(content)
                    for i, chunk in enumerate(chunks):
                        documents.append({"content": chunk, "source": filename, "chunk_index": i})
                except Exception as e:
                    logger.error(f"读取文档失败 {filepath}: {e}")
        logger.info(f"加载文档: {len(documents)} 个片段")
        return documents

    def _create_sample_documents(self) -> List[Dict]:
        sample_docs = [
            {"content": "Qwen3是阿里巴巴通义千问团队开发的大语言模型系列，"
                        "支持多语言理解、文本生成、工具使用等多种能力。"
                        "Qwen3支持思考模式和非思考模式的动态切换。",
             "source": "qwen3_intro.txt", "chunk_index": 0},
            {"content": "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的"
                        "AI架构，通过从知识库中检索相关信息来增强语言模型的生成能力。"
                        "RAG可以有效减少大模型的幻觉问题。",
             "source": "rag_intro.txt", "chunk_index": 0},
            {"content": "Agent是指能够自主感知环境、制定计划并使用工具完成任务的"
                        "智能体系统。在大语言模型领域，Agent可以通过函数调用机制"
                        "与外部工具交互，实现复杂任务的自动化。",
             "source": "agent_intro.txt", "chunk_index": 0},
            {"content": "Python是一种高级编程语言，广泛应用于人工智能、数据科学、"
                        "Web开发等领域。Python以其简洁的语法和丰富的生态系统而闻名。",
             "source": "python_intro.txt", "chunk_index": 0},
            {"content": "Transformer是一种基于注意力机制的神经网络架构，"
                        "由Vaswani等人在2017年提出。它是大语言模型的核心基础，"
                        "BERT、GPT等模型都基于Transformer架构。",
             "source": "transformer_intro.txt", "chunk_index": 0},
        ]
        os.makedirs(config.document_dir, exist_ok=True)
        for doc in sample_docs:
            filepath = os.path.join(config.document_dir, doc["source"])
            if not os.path.exists(filepath):
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(doc["content"])
        return sample_docs

    def _chunk_text(self, text: str) -> List[str]:
        chunk_size = config.rag.chunk_size
        overlap = config.rag.chunk_overlap
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def _compute_embeddings(self) -> np.ndarray:
        if not self.documents:
            return np.array([])
        contents = [doc["content"] for doc in self.documents]
        if self.embedding_model:
            embeddings = self.embedding_model.encode(contents, show_progress_bar=False)
        else:
            embeddings = np.random.rand(len(contents), 768)
        return embeddings

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 similarity_threshold: Optional[float] = None) -> List[RetrievedDocument]:
        top_k = top_k or config.rag.default_top_k
        similarity_threshold = similarity_threshold or config.rag.default_similarity_threshold
        if not self.documents:
            return []
        if self.embedding_model:
            query_embedding = self.embedding_model.encode([query])
        else:
            query_embedding = np.random.rand(1, 768)
        similarities = np.dot(self.document_embeddings, query_embedding.T).flatten()
        if similarities.max() - similarities.min() > 0:
            similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        sorted_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in sorted_indices:
            if len(results) >= top_k:
                break
            score = float(similarities[idx])
            if score < similarity_threshold:
                continue
            doc = self.documents[idx]
            results.append(RetrievedDocument(content=doc["content"], source=doc["source"], score=score,
                                            metadata={"chunk_index": doc.get("chunk_index", 0)}))
        logger.info(f"检索完成: query='{query[:50]}...', top_k={top_k}, threshold={threshold:.2f}, 结果数={len(results)}")
        return results


_retriever = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
