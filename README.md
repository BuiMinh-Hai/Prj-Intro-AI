# 🏥 MedBot — Chatbot Tư Vấn Y Khoa Tiếng Việt

> Qwen2-7B + QLoRA Fine-tuning | ChromaDB RAG | Hybrid Search + Cross-Encoder Reranking

---

## 📁 Cấu trúc dự án

```
prj intro ai/
├── vector_db.py        # Vị trí 1: RAG Developer — ChromaDB & Dense Retrieval
├── retriever.py        # Vị trí 2: Advanced RAG — BM25 + Hybrid + Reranking
├── finetune.py         # Vị trí 3: LLM Engineer — QLoRA Fine-tune Qwen-7B
├── main.py             # Vị trí 4: Core Logic — Terminal UI & Chat Loop
├── evaluate.py         # Vị trí 5&6: Evaluation — ROUGE/METEOR/RAGAS + Charts
├── requirements.txt    # Thư viện cần thiết
├── data/               # Dataset đã format & test set
├── models/             # Checkpoint Qwen-7B fine-tuned
├── chroma_db/          # ChromaDB persistent storage
├── evaluation_results/ # Kết quả đánh giá & biểu đồ
└── logs/               # Session chat logs
```

---

## 🚀 Hướng dẫn chạy

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Vị trí 1 — Khởi tạo Vector Database
```bash
# Tải dataset & nạp vào ChromaDB
python vector_db.py

# Tạo lại DB từ đầu
python vector_db.py --rebuild

# Test truy vấn
python vector_db.py --test-query "Triệu chứng của bệnh tiểu đường"
```

### 3. Vị trí 2 — Test Hybrid Retriever
```bash
python retriever.py
```

### 4. Vị trí 3 — Fine-tune Qwen-7B (cần GPU VRAM ≥ 16GB)
```bash
# Huấn luyện
python finetune.py --train

# Resume từ checkpoint
python finetune.py --train --resume ./models/qwen7b-medical-qlora/checkpoint-200

# Test inference
python finetune.py --test "Thuốc Metformin dùng để điều trị bệnh gì?"
```

### 5. Vị trí 4 — Chạy Chatbot
```bash
python main.py
```
Lệnh trong chat: `/help`, `/clear`, `/history`, `/save`, `/context`, `/exit`

### 6. Vị trí 5&6 — Đánh giá hệ thống
```bash
# Tạo test set (80 ca)
python evaluate.py --create-test

# Chạy đánh giá đầy đủ (ROUGE + METEOR + RAGAS)
python evaluate.py --run-eval

# Chỉ vẽ biểu đồ
python evaluate.py --plot-only
```

---

## ⚙️ Kiến trúc hệ thống

```
Người dùng
    │ query
    ▼
[ Hybrid Retriever ]
  ├─ Dense Search (ChromaDB + vietnamese-sbert)
  ├─ BM25 Sparse Search
  ├─ Score Fusion (α=0.6 Dense + α=0.4 BM25)
  └─ Cross-Encoder Reranking → Top 5 context
    │
    ▼
[ Prompt Builder ]
  ├─ System Prompt (y tế)
  ├─ Short-term Memory (4 lượt)
  └─ RAG Context
    │
    ▼
[ Qwen2-7B + LoRA Adapter ]
    │ response
    ▼
[ Rich Terminal UI ]
```

---

## 📊 Metrics đánh giá
| Metric | Mô tả |
|--------|-------|
| ROUGE-1/2/L | So sánh n-gram với ground truth |
| METEOR | Đánh giá chất lượng sinh văn bản |
| Faithfulness | % câu trả lời có căn cứ trong context |
| Answer Relevance | Độ bám sát câu hỏi gốc |

---

## 🔧 Yêu cầu phần cứng
- **GPU VRAM:** ≥ 16GB (cho fine-tuning QLoRA 4-bit)
- **RAM:** ≥ 32GB
- **Disk:** ≥ 50GB (model + ChromaDB)
- **Môi trường:** Python 3.10+, CUDA 11.8+
