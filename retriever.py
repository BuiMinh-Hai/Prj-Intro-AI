"""
retriever.py — Vị trí 2: Advanced RAG Developer (Reranking & Hybrid Search)
Nhiệm vụ:
  - Tích hợp BM25 để bắt chính xác tên thuốc/mã bệnh
  - Hybrid Search: trộn điểm BM25 + Vector Search
  - Reranking bằng Cross-Encoder (bge-reranker-v2-m3)
  - Trả về Top 3-5 đoạn context chất lượng nhất
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from vector_db import MedicalVectorDB

# ─────────────────────────── Cấu hình ───────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cross-Encoder model tốt cho đa ngôn ngữ + tiếng Việt
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Số lượng tài liệu lấy ra ban đầu trước khi rerank
INITIAL_DENSE_TOP_K = 20
INITIAL_BM25_TOP_K  = 20

# Số tài liệu sau rerank gửi cho LLM
FINAL_TOP_K = 5

# Trọng số RRF / hybrid
ALPHA_DENSE = 0.6   # tỷ trọng điểm vector search
ALPHA_BM25  = 0.4   # tỷ trọng điểm BM25


# ─────────────────────────── BM25 Index ──────────────────────────
class BM25Index:
    """
    Cài đặt BM25 thuần Python — không cần thư viện ngoài,
    phù hợp với corpus y khoa tiếng Việt.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus: List[str] = []
        self.tokenized: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tách từ đơn giản cho tiếng Việt (theo khoảng trắng + chuẩn hóa)."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def fit(self, corpus: List[str]) -> None:
        """Build BM25 index từ danh sách đoạn văn."""
        self.corpus = corpus
        self.N = len(corpus)
        self.tokenized = [self._tokenize(doc) for doc in corpus]
        self.doc_freqs = [Counter(tokens) for tokens in self.tokenized]

        lengths = [len(t) for t in self.tokenized]
        self.avgdl = sum(lengths) / self.N if self.N else 1.0

        # Tính IDF
        df: Counter = Counter()
        for tf in self.doc_freqs:
            for term in tf:
                df[term] += 1

        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.N - freq + 0.5) / (freq + 0.5) + 1
            )

    def get_scores(self, query: str) -> np.ndarray:
        """Tính BM25 score cho tất cả documents với query."""
        query_terms = self._tokenize(query)
        scores = np.zeros(self.N)

        for term in query_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf in enumerate(self.doc_freqs):
                freq = tf.get(term, 0)
                dl = len(self.tokenized[i])
                numerator   = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i]  += idf * numerator / denominator

        return scores

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Returns: Danh sách (doc_index, bm25_score) sắp xếp giảm dần.
        """
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ─────────────────────────── Hybrid Retriever ────────────────────
class HybridRetriever:
    """
    Pipeline tìm kiếm lai ghép Dense Vector Search + BM25,
    sau đó Rerank bằng Cross-Encoder.
    """

    def __init__(
        self,
        vector_db: MedicalVectorDB,
        reranker_model: str = RERANKER_MODEL,
    ):
        self.vector_db = vector_db
        self._bm25: Optional[BM25Index] = None
        self._bm25_corpus: List[Dict[str, Any]] = []   # [{id, text, metadata}]
        self._reranker: Optional[CrossEncoder] = None
        self._reranker_model = reranker_model

    # ── Build BM25 Index từ corpus đã có trong ChromaDB ──────────
    def build_bm25_index(self) -> None:
        """
        Kéo toàn bộ document text từ ChromaDB để build BM25 index.
        Gọi một lần sau khi ChromaDB đã được populate.
        """
        logger.info("🔨 Đang build BM25 index từ ChromaDB ...")
        coll = self.vector_db._collection
        if coll is None:
            self.vector_db.get_or_create_collection()
            coll = self.vector_db._collection

        total = coll.count()
        if total == 0:
            logger.warning("⚠️  ChromaDB rỗng — BM25 index không có dữ liệu.")
            return

        # Lấy tất cả documents (ChromaDB hỗ trợ get với limit)
        FETCH_BATCH = 5000
        all_docs: List[Dict[str, Any]] = []
        offset = 0

        while offset < total:
            result = coll.get(
                limit=FETCH_BATCH,
                offset=offset,
                include=["documents", "metadatas"],
            )
            for doc_id, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            ):
                all_docs.append({"id": doc_id, "text": text, "metadata": meta})
            offset += FETCH_BATCH

        self._bm25_corpus = all_docs
        corpus_texts = [d["text"] for d in all_docs]

        self._bm25 = BM25Index()
        self._bm25.fit(corpus_texts)
        logger.info(f"✅ BM25 index sẵn sàng: {len(all_docs)} documents.")

    # ── Lazy-load Cross-Encoder ───────────────────────────────────
    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            logger.info(f"🔄 Đang tải Cross-Encoder: {self._reranker_model} ...")
            self._reranker = CrossEncoder(self._reranker_model, max_length=512)
            logger.info("✅ Cross-Encoder sẵn sàng.")
        return self._reranker

    # ── Normalize scores về [0, 1] ───────────────────────────────
    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        if not scores:
            return scores
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    # ── Dense Search ──────────────────────────────────────────────
    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self.vector_db.dense_search(query, top_k=top_k)

    # ── BM25 Search ───────────────────────────────────────────────
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self._bm25 is None or not self._bm25_corpus:
            logger.warning("⚠️  BM25 chưa được build. Gọi build_bm25_index() trước.")
            return []

        ranked = self._bm25.search(query, top_k=top_k)
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            doc = self._bm25_corpus[idx]
            results.append(
                {
                    "id":       doc["id"],
                    "text":     doc["text"],
                    "metadata": doc["metadata"],
                    "score":    score,
                }
            )
        return results

    # ── Hybrid Merge (RRF + weighted score) ───────────────────────
    def _hybrid_merge(
        self,
        dense_hits: List[Dict[str, Any]],
        bm25_hits:  List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Trộn điểm Dense và BM25 bằng weighted average sau khi normalize.
        Loại bỏ trùng lặp theo id.
        """
        # Normalize
        dense_scores = self._normalize([h["score"] for h in dense_hits])
        bm25_scores  = self._normalize([h["score"] for h in bm25_hits])

        merged: Dict[str, Dict[str, Any]] = {}

        for hit, score in zip(dense_hits, dense_scores):
            doc_id = hit["id"]
            merged[doc_id] = {**hit, "hybrid_score": ALPHA_DENSE * score}

        for hit, score in zip(bm25_hits, bm25_scores):
            doc_id = hit["id"]
            if doc_id in merged:
                merged[doc_id]["hybrid_score"] += ALPHA_BM25 * score
            else:
                merged[doc_id] = {**hit, "hybrid_score": ALPHA_BM25 * score}

        sorted_hits = sorted(
            merged.values(), key=lambda x: x["hybrid_score"], reverse=True
        )
        return sorted_hits

    # ── Rerank bằng Cross-Encoder ─────────────────────────────────
    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = FINAL_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Dùng Cross-Encoder để chấm điểm lại candidates, giữ top_k."""
        if not candidates:
            return []

        reranker = self._get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = reranker.predict(pairs, show_progress_bar=False)

        for candidate, ce_score in zip(candidates, ce_scores):
            candidate["rerank_score"] = float(ce_score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    # ── Public API ────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        final_top_k: int = FINAL_TOP_K,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Full pipeline: Dense → BM25 → Hybrid Merge → Rerank → Top-K

        Args:
            query:       Câu hỏi y khoa của người dùng
            final_top_k: Số context trả về cuối cùng (3-5)
            use_rerank:  Có chạy Cross-Encoder hay không

        Returns:
            Danh sách dict: {id, text, metadata, hybrid_score, rerank_score}
        """
        logger.info(f"🔍 Hybrid Retrieval cho query: '{query[:80]}...'")

        # 1. Dense retrieval
        dense_hits = self._dense_search(query, top_k=INITIAL_DENSE_TOP_K)
        logger.info(f"   Dense hits: {len(dense_hits)}")

        # 2. BM25 retrieval
        bm25_hits = self._bm25_search(query, top_k=INITIAL_BM25_TOP_K)
        logger.info(f"   BM25 hits:  {len(bm25_hits)}")

        # 3. Hybrid merge
        merged = self._hybrid_merge(dense_hits, bm25_hits)
        logger.info(f"   Sau merge: {len(merged)} candidates")

        # 4. Reranking
        if use_rerank and merged:
            final = self._rerank(query, merged, top_k=final_top_k)
            logger.info(f"   Sau rerank: {len(final)} kết quả cuối")
        else:
            final = merged[:final_top_k]

        return final

    def format_context(self, hits: List[Dict[str, Any]]) -> str:
        """
        Định dạng danh sách hits thành chuỗi context đưa vào prompt.
        """
        parts = []
        for i, hit in enumerate(hits, 1):
            score_info = ""
            if "rerank_score" in hit:
                score_info = f" [rerank={hit['rerank_score']:.3f}]"
            parts.append(f"[Tài liệu {i}{score_info}]\n{hit['text']}")
        return "\n\n".join(parts)


# ─────────────────────────── Factory ─────────────────────────────
def build_retriever(vector_db: MedicalVectorDB) -> HybridRetriever:
    """
    Khởi tạo HybridRetriever đã có BM25 index.
    """
    retriever = HybridRetriever(vector_db)
    retriever.build_bm25_index()
    return retriever


# ─────────────────────────── Main ────────────────────────────────
if __name__ == "__main__":
    from vector_db import build_database

    print("=== Test Hybrid Retriever ===")
    db = build_database(force_rebuild=False)
    retriever = build_retriever(db)

    test_queries = [
        "Triệu chứng của bệnh tiểu đường type 2 là gì?",
        "Thuốc Metformin dùng để điều trị bệnh gì?",
        "Cách xử lý khi bị ngộ độc thực phẩm tại nhà?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        hits = retriever.retrieve(q, final_top_k=3, use_rerank=True)
        print(f"Số kết quả: {len(hits)}")
        for i, h in enumerate(hits, 1):
            print(f"\n  [{i}] rerank={h.get('rerank_score', 0):.4f} | hybrid={h.get('hybrid_score', 0):.4f}")
            print(f"      {h['text'][:200]}...")
