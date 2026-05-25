"""
Agent记忆管理
"""
import json
import time
import logging
import hashlib
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from collections import OrderedDict

from config import config

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    key: str
    content: Dict
    session_id: str
    timestamp: float
    ttl: float
    access_count: int = 0
    relevance_score: float = 0.0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    @property
    def importance(self) -> float:
        recency = 1.0 / (1.0 + (time.time() - self.timestamp) / 3600)
        frequency = min(1.0, self.access_count / 10.0)
        return 0.6 * recency + 0.4 * frequency


class MemoryManager:

    def __init__(self):
        self._memories: Dict[str, MemoryItem] = OrderedDict()
        self._sessions: Dict[str, List[str]] = {}
        self._current_session_id: str = f"session_{int(time.time())}"
        self.enabled = config.agent.memory_enabled
        self.max_items = config.agent.max_memory_items
        self.ttl = config.agent.memory_ttl

    def _generate_key(self, content: Dict) -> str:
        content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content_str.encode()).hexdigest()

    def store(self, key: str, content: Dict, session_id: Optional[str] = None) -> str:
        if not self.enabled:
            return ""
        session_id = session_id or self._current_session_id
        memory = MemoryItem(key=key, content=content, session_id=session_id, timestamp=time.time(), ttl=self.ttl)
        self._memories[key] = memory
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(key)
        if len(self._memories) > self.max_items:
            self._evict_lru()
        return key

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        if not self.enabled:
            return []
        query_lower = query.lower()
        candidates = []
        for memory in self._memories.values():
            if memory.is_expired:
                continue
            content_str = json.dumps(memory.content, ensure_ascii=False).lower()
            overlap = len(set(query_lower.split()) & set(content_str.split()))
            relevance = overlap / max(len(query_lower.split()), 1)
            memory.relevance_score = relevance
            memory.access_count += 1
            if relevance > 0:
                candidates.append(memory)
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)
        return candidates[:top_k]

    def get_context_for_query(self, query: str) -> str:
        memories = self.retrieve(query, top_k=3)
        if not memories:
            return ""
        context_parts = ["【历史记忆】"]
        for m in memories:
            content = m.content
            if "question" in content and "answer" in content:
                context_parts.append(f"之前回答过类似问题：Q: {content['question']} -> A: {content['answer'][:100]}...")
            elif "key_fact" in content:
                context_parts.append(f"已知事实：{content['key_fact']}")
        return "\n".join(context_parts)

    def store_interaction(self, question: str, answer: str, metadata: Optional[Dict] = None):
        content = {"question": question, "answer": answer, "metadata": metadata or {}, "timestamp": time.time()}
        key = self._generate_key({"question": question})
        self.store(key, content)

    def store_fact(self, key_fact: str, source: str = ""):
        content = {"key_fact": key_fact, "source": source, "timestamp": time.time()}
        key = self._generate_key({"fact": key_fact})
        self.store(key, content)

    def _evict_lru(self):
        sorted_memories = sorted(self._memories.values(), key=lambda x: (x.access_count, x.importance))
        if sorted_memories:
            evict_key = sorted_memories[0].key
            del self._memories[evict_key]

    def clear(self):
        self._memories.clear()
        self._sessions.clear()
        logger.info("记忆已清空")

    def get_statistics(self) -> Dict:
        total = len(self._memories)
        expired = sum(1 for m in self._memories.values() if m.is_expired)
        return {"total_memories": total, "active_memories": total - expired, "expired_memories": expired,
                "sessions": len(self._sessions), "enabled": self.enabled}


memory_manager = MemoryManager()
