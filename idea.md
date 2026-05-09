
## PHẦN 1: TỔNG QUAN DỰ ÁN

### 1. Chủ đề (Topic)
Xây dựng hệ thống Chatbot tra cứu và tư vấn y khoa tiếng Việt bằng cách tinh chỉnh mô hình ngôn ngữ lớn Qwen-7B (kỹ thuật QLoRA) kết hợp kiến trúc RAG (Retrieval-Augmented Generation) trên nền tảng cơ sở dữ liệu vector ChromaDB.

### 2. Bài toán (Problem Statement)
* **Thách thức:** Xử lý ngôn ngữ tự nhiên trong lĩnh vực y tế đòi hỏi độ chính xác tuyệt đối và khả năng hiểu sâu thuật ngữ chuyên môn tiếng Việt. Các LLM (Large Language Models) general-purpose hiện nay thường thiếu kiến thức chuyên sâu về y tế bản địa và dễ sinh ra "ảo giác" (hallucination) gây nguy hiểm trong tư vấn.
* **Mục tiêu hệ thống:** 
  * Hiểu và phản hồi các câu hỏi y khoa (triệu chứng, đơn thuốc, quy trình chăm sóc) bằng tiếng Việt tự nhiên, chuẩn mực.
  * Truy xuất thông tin minh bạch, chính xác từ bộ dữ liệu y khoa chuẩn để làm cơ sở cho câu trả lời.
  * Hoạt động hiệu suất cao trên hạ tầng phần cứng thực tế với bộ nhớ GPU được tối ưu hóa.

### 3. Đầu vào (Inputs)
* **Cơ sở tri thức (Knowledge Base) & Dữ liệu huấn luyện:** Tập dữ liệu chuyên biệt `urnus11/Vietnamese-Healthcare` (bao gồm các cặp QA y khoa, kiến thức bệnh lý, dược phẩm bằng tiếng Việt).
* **Câu truy vấn (User Query):** Văn bản tiếng Việt nhập từ người dùng cuối/bác sĩ.

### 4. Đầu ra (Outputs)
* **Phản hồi từ Chatbot:** Câu trả lời văn bản đã qua tinh chỉnh để có văn phong của chuyên gia y tế, đi kèm hướng dẫn hoặc cảnh báo cụ thể.
* **Minh chứng (Context):** Trích dẫn các đoạn tài liệu y khoa thực chứng được rút trích từ cơ sở dữ liệu.

### 5. Phương pháp & Mô hình đề xuất (Proposed Methods)
**A. Mô hình ngôn ngữ & Fine-tuning:**
* **Base Model:** Qwen-7B (Alibaba Cloud) – Lựa chọn tối ưu nhờ khả năng hiểu tiếng Việt tốt và hiệu suất suy luận mạnh trong phân khúc 7B parameters.
* **Kỹ thuật QLoRA (Quantized Low-Rank Adaptation):** Tinh chỉnh mô hình với quantization 4-bit, giúp tiết kiệm tối đa VRAM GPU mà vẫn đảm bảo LLM học được văn phong và kiến thức từ bộ `Vietnamese-Healthcare`.

**B. Hệ thống RAG (Retrieval-Augmented Generation):**
* **Vector Database:** Sử dụng ChromaDB nhờ ưu điểm nhẹ, dễ cấu hình và tốc độ truy vấn vector (embeddings) cao.
* **Cơ chế Hybrid Search:** Kết hợp song song Tìm kiếm ngữ nghĩa (Semantic Search qua Dense Embeddings) và Tìm kiếm từ khóa (Sparse Retrieval như BM25) để không bỏ sót các thuật ngữ biệt dược.
* **Reranking:** Dùng mô hình Cross-Encoder để chấm điểm lại sự phù hợp của các tài liệu lấy từ DB, lọc ra Top 3-5 đoạn context chất lượng nhất làm Prompt cho LLM.

### 6. Kết quả dự kiến (Expected Outcomes)
* Một checkpoint mô hình Qwen-7B đã "y tế hóa" hoàn chỉnh.
* Pipeline RAG trơn tru với độ trễ thấp, đảm bảo 100% câu trả lời đều có căn cứ từ DB.
* Hệ thống xử lý tốt các ca bệnh phức tạp (ví dụ: gộp triệu chứng ở query để đối chiếu với phác đồ trong DB).

### 7. Phương pháp Đánh giá (Evaluation Metrics)
* **Đánh giá sinh ngữ (NLG Metrics):** Dùng ROUGE và METEOR so sánh câu trả lời của AI với ground-truth trong dataset `Vietnamese-Healthcare`.
* **Đánh giá RAG (RAGAS Framework):**
  * *Faithfulness:* Đo lường mức độ trung thực, đảm bảo câu trả lời không chứa thông tin bịa đặt ngoài ngữ cảnh (context).
  * *Answer Relevance:* Độ bám sát của câu trả lời so với câu hỏi gốc.
* **Đánh giá hệ thống (System Performance):** Độ trễ suy luận (Inference Latency) và mức tiêu thụ VRAM.

---

## PHẦN 2: PHÂN CÔNG NHIỆM VỤ CHI TIẾT

### Vị trí 1: RAG Developer (Vector Database & Core Retrieval)
*Người lo nền móng lưu trữ tri thức.*
* **Nhiệm vụ:**
  * Xử lý tiền dữ liệu (Text chunking) từ bộ `urnus11/Vietnamese-Healthcare`.
  * Viết script khởi tạo ChromaDB và nạp dữ liệu.
  * Cấu hình embedding model (ví dụ: `keepitreal/vietnamese-sbert` hoặc PhoBERT).
  * Viết hàm Dense Retrieval cơ bản.
* **Output:** File `vector_db.py` với class quản lý load và search vector.

### Vị trí 2: Advanced RAG Developer (Reranking & Hybrid Search)
*Người quyết định tính chính xác của tài liệu y khoa.*
* **Nhiệm vụ:**
  * Tích hợp thuật toán đếm từ khóa (BM25) để bắt chính xác tên thuốc/mã bệnh.
  * Viết logic Hybrid Search: Trộn và cân bằng điểm số giữa BM25 và Vector Search.
  * Tích hợp mô hình Cross-Encoder (VD: `bge-reranker-v2-m3`) để chấm điểm (Rerank) và giữ lại 3-5 đoạn context vàng.
* **Output:** File `retriever.py` đảm nhiệm toàn bộ luồng pipeline tìm kiếm lai.

### Vị trí 3: LLM Engineer (Fine-tuning)
*Người chuyên trách huấn luyện AI.*
* **Nhiệm vụ:**
  * Định dạng lại dataset `Vietnamese-Healthcare` theo chuẩn cấu trúc Prompt (System/User/Assistant).
  * Viết script fine-tune Qwen-7B bằng QLoRA (ưu tiên dùng Unsloth để tăng tốc và giảm VRAM).
  * Quản lý tiến trình train, monitor Loss và lưu trữ file Adapter weights.
* **Output:** Thư mục `models/` chứa checkpoint Qwen-7B đã fine-tune.

### Vị trí 4: Core Logic & Terminal Interface
*Nhạc trưởng ghép nối hệ thống.*
* **Nhiệm vụ:**
  * Thiết kế Prompt Template chuyên dụng cho tư vấn y khoa.
  * Xây dựng giao diện Terminal UI (sử dụng thư viện `Rich` để trang trí màu sắc, format markdown).
  * Cài đặt bộ nhớ ngắn hạn (Memory) để bot nhớ 3-4 câu hội thoại gần nhất.
  * Import code từ Vị trí 1, 2, 3 và tạo vòng lặp chat chính.
* **Output:** File `main.py` chạy trực tiếp hệ thống.

### Vị trí 5 & 6: Evaluation & Researcher (Đánh giá học thuật)
*Người đảm bảo tính an toàn y khoa và viết báo cáo khoa học.*
* **Nhiệm vụ:**
  * Trích xuất/Tạo tập test dataset gồm 50-100 ca tư vấn y khoa khó.
  * Viết script chạy tự động (Automation) để lấy câu trả lời từ hệ thống RAG hoàn chỉnh.
  * Cài đặt và chạy thư viện tính điểm (ROUGE, METEOR, RAGAS).
  * Phân tích số liệu, vẽ biểu đồ so sánh mô hình gốc vs. mô hình đã Fine-tune + RAG.
  * Tổng hợp tài liệu viết Báo cáo dự án và Slide thuyết trình.
* **Output:** Script `evaluate.py`, File Báo cáo (PDF/Word), Slide thuyết trình.