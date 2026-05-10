"""
vector_db.py — Vị trí 1: RAG Developer (Vector Database & Core Retrieval)
Nhiệm vụ:
  - Xử lý tiền dữ liệu (Text chunking) từ bộ urnus11/Vietnamese-Healthcare
  - Khởi tạo ChromaDB và nạp dữ liệu
  - Cấu hình embedding model (keepitreal/vietnamese-sbert hoặc PhoBERT)
  - Hàm Dense Retrieval cơ bản
"""

import os
import re
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─────────────────────────── Cấu hình ───────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Thư mục lưu ChromaDB local
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "medrag_textbooks"

# Embedding model tốt nhất cho tiếng Việt (vẫn dùng cho đa ngôn ngữ)
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"

# Tham số chunking
CHUNK_SIZE = 512          # tokens / ký tự tối đa mỗi đoạn
CHUNK_OVERLAP = 64        # overlap để giữ ngữ cảnh
BATCH_SIZE = 64           # kích thước batch khi upsert vào ChromaDB

DATASET_NAME = "MedRAG/textbooks"


# ─────────────────────────── Dataclass ───────────────────────────
@dataclass
class Document:
    """Đại diện một đoạn văn bản y khoa đã được chunk."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────── Chunking ────────────────────────────
def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Chia văn bản thành các đoạn nhỏ theo ký tự, có overlap.
    Ưu tiên cắt theo câu (dấu chấm / xuống dòng) để không cắt giữa câu.
    """
    if not text or not text.strip():
        return []

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    # Tách câu theo dấu câu tiếng Việt
    sentences = re.split(r"(?<=[.!?।\n])\s+", text)

    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Nếu câu quá dài, cắt cứng
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size - chunk_overlap):
                    sub = sentence[i: i + chunk_size]
                    if sub.strip():
                        chunks.append(sub.strip())
                current_chunk = ""
            else:
                # Carry-over overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-chunk_overlap // 10:]) if words else ""
                current_chunk = (overlap_text + " " + sentence).strip()

    if current_chunk:
        chunks.append(current_chunk)

    return [c for c in chunks if len(c.strip()) > 20]


def preprocess_dataset(dataset_name: str = DATASET_NAME) -> List[Document]:
    """
    Tải bộ dữ liệu Vietnamese-Healthcare từ HuggingFace và chunk thành Documents.
    Dataset thường có dạng: {'question': ..., 'answer': ..., 'context': ...}
    """
    logger.info(f"📥 Đang tải dataset: {dataset_name} ...")
    try:
        ds = load_dataset(dataset_name, split="train[:500]")  # Lấy 500 mẫu để test nhanh
    except Exception as e:
        logger.error(f"Không tải được dataset: {e}")
        raise

    documents: List[Document] = []
    doc_counter = 0

    logger.info(f"✅ Tải thành công {len(ds)} mẫu. Bắt đầu chunking ...")

    for idx, row in enumerate(tqdm(ds, desc="Chunking")):
        # Trích xuất dữ liệu từ nhiều tên cột phổ biến
        question = row.get("question", "") or row.get("input", "") or ""
        answer   = row.get("answer",   "") or row.get("output", "") or ""
        context  = row.get("context",  "") or row.get("content", "") or row.get("text", "") or ""
        title    = row.get("title",    "") or row.get("heading", "") or ""

        full_text_parts = []
        
        # Lắp ghép nội dung
        if title and context:
            full_text_parts.append(f"{title}\n{context}")
        elif context:
            full_text_parts.append(context)
            
        if question and answer:
            full_text_parts.append(f"Câu hỏi: {question}\nTrả lời: {answer}")
        elif answer:
            full_text_parts.append(answer)
            
        # Nếu vẫn không có gì, lấy ngẫu nhiên các cột chứa text dài
        if not full_text_parts:
            for k, v in row.items():
                if isinstance(v, str) and len(v.strip()) > 20:
                    full_text_parts.append(v.strip())

        for part in full_text_parts:
            chunks = split_text_into_chunks(part)
            for chunk_idx, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        doc_id=f"doc_{idx}_part_{chunk_idx}_{doc_counter}",
                        text=chunk,
                        metadata={
                            "source_idx": idx,
                            "chunk_idx": chunk_idx,
                            "question": question[:200] if question else "",
                            "dataset": dataset_name,
                        },
                    )
                )
                doc_counter += 1

    logger.info(f"📄 Tổng số đoạn sau chunking: {len(documents)}")
    return documents


# ─────────────────────────── Embedding ───────────────────────────
class VietnameseSBertEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Custom EmbeddingFunction cho ChromaDB sử dụng vietnamese-sbert.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"🔄 Khởi tạo embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("✅ Embedding model sẵn sàng.")

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        embeddings = self.model.encode(
            input,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


# ─────────────────────────── VectorDB Class ──────────────────────
class MedicalVectorDB:
    """
    Quản lý toàn bộ vòng đời của ChromaDB cho hệ thống y khoa.
    """

    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL_NAME,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self._embedding_fn = VietnameseSBertEmbeddingFunction(embedding_model)

        os.makedirs(db_path, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection: Optional[chromadb.Collection] = None
        logger.info(f"🗄️  ChromaDB khởi tạo tại: {db_path}")

    # ── Tạo / lấy Collection ─────────────────────────────────────
    def get_or_create_collection(self) -> chromadb.Collection:
        """Lấy collection hiện có hoặc tạo mới."""
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        count = self._collection.count()
        logger.info(f"📦 Collection '{self.collection_name}': {count} vectors hiện có.")
        return self._collection

    # ── Nạp dữ liệu ──────────────────────────────────────────────
    def ingest_documents(self, documents: List[Document]) -> None:
        """Nạp danh sách Document vào ChromaDB theo batch."""
        if self._collection is None:
            self.get_or_create_collection()

        total = len(documents)
        logger.info(f"⬆️  Bắt đầu nạp {total} đoạn vào ChromaDB ...")

        for start in tqdm(range(0, total, BATCH_SIZE), desc="Ingesting"):
            batch = documents[start: start + BATCH_SIZE]
            self._collection.upsert(
                ids=[d.doc_id for d in batch],
                documents=[d.text for d in batch],
                metadatas=[d.metadata for d in batch],
            )

        logger.info(f"✅ Hoàn tất! Tổng vector trong DB: {self._collection.count()}")

    # ── Dense Retrieval ───────────────────────────────────────────
    def dense_search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm ngữ nghĩa (Dense Retrieval) theo cosine similarity.

        Returns:
            Danh sách dict gồm: {id, text, metadata, score}
        """
        if self._collection is None:
            self.get_or_create_collection()

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results and results["ids"]:
            for doc_id, text, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB trả về distance (cosine) → score = 1 - distance
                hits.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "metadata": meta,
                        "score": round(1.0 - dist, 6),
                    }
                )

        return hits

    # ── Tiện ích ──────────────────────────────────────────────────
    def count(self) -> int:
        """Số lượng vector trong collection."""
        if self._collection is None:
            self.get_or_create_collection()
        return self._collection.count()

    def delete_collection(self) -> None:
        """Xóa collection (dùng khi muốn tạo lại từ đầu)."""
        self._client.delete_collection(self.collection_name)
        self._collection = None
        logger.warning(f"🗑️  Đã xóa collection '{self.collection_name}'.")

    def is_populated(self) -> bool:
        """Kiểm tra DB đã có dữ liệu chưa."""
        return self.count() > 0


# ─────────────────────────── CLI Helper ──────────────────────────
def build_database(force_rebuild: bool = False) -> MedicalVectorDB:
    """
    Hàm tiện ích: Tạo và nạp dữ liệu vào DB.
    Nếu DB đã có dữ liệu và force_rebuild=False → bỏ qua.
    """
    db = MedicalVectorDB()
    db.get_or_create_collection()

    if db.is_populated() and not force_rebuild:
        logger.info("⚡ DB đã có dữ liệu. Bỏ qua bước ingest (dùng force_rebuild=True để làm lại).")
        return db

    if force_rebuild and db.is_populated():
        logger.warning("♻️  force_rebuild=True — Đang xóa và tạo lại DB ...")
        db.delete_collection()
        db.get_or_create_collection()

    documents = preprocess_dataset()
    db.ingest_documents(documents)
    return db


# ─────────────────────────── Main ────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Khởi tạo Vector DB cho hệ thống y khoa")
    parser.add_argument("--rebuild", action="store_true", help="Xóa và tạo lại DB từ đầu")
    parser.add_argument("--test-query", type=str, default="", help="Thử truy vấn sau khi build")
    args = parser.parse_args()

    db = build_database(force_rebuild=args.rebuild)

    if args.test_query:
        print(f"\n🔍 Thử truy vấn: '{args.test_query}'")
        results = db.dense_search(args.test_query, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"\n--- Kết quả {i} (score={r['score']:.4f}) ---")
            print(r["text"][:300])
