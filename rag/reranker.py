"""
重排序器
对检索结果进行二次排序，提升相关性
"""
import logging
from typing import List

from rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class Reranker:

    def __init__(self):
        self.model = None

    def rerank(self, query: str, documents: List[RetrievedDocument], top_k: int = 5) -> List[RetrievedDocument]:
        """使用交叉编码器重排序（简化版：基于长度和关键词密度）"""
        if not documents:
            return []

        query_terms = set(query.lower().split())
        scored_docs = []

        for doc in documents:
            # 关键词匹配得分
            content_lower = doc.content.lower()
            keyword_score = sum(1 for term in query_terms if term in content_lower) / max(len(query_terms), 1)

            # 长度奖励（适中的长度）
            length_score = 1.0 - abs(len(doc.content) - 300) / 500
            length_score = max(0, min(1, length_score))

            # 综合得分
            combined = 0.7 * doc.score + 0.2 * keyword_score + 0.1 * length_score
            scored_docs.append((combined, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        reranked = [doc for _, doc in scored_docs[:top_k]]
        for doc in reranked:
            doc.score = scored_docs[reranked.index(doc)][0]

        return reranked


reranker = Reranker()
