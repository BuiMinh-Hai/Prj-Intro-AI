"""
main.py — Vị trí 4: Core Logic & Terminal Interface
Nhiệm vụ:
  - Prompt Template chuyên dụng cho tư vấn y khoa
  - Terminal UI đẹp với thư viện Rich (màu sắc, markdown)
  - Short-term Memory: nhớ 3-4 lượt hội thoại gần nhất
  - Ghép nối Vị trí 1 (vector_db), 2 (retriever), 3 (finetune)
  - Vòng lặp chat chính
"""

import os
import sys
import json
import logging
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

# Rich UI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich import box

# Project modules
from vector_db import MedicalVectorDB, build_database
from retriever import HybridRetriever, build_retriever

# ─────────────────────────── Cấu hình ───────────────────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

console = Console()

# ── Màu sắc chủ đạo ─────────────────────────────────────────────
COLOR_PRIMARY   = "bright_cyan"
COLOR_ACCENT    = "bright_green"
COLOR_WARNING   = "bright_yellow"
COLOR_ERROR     = "bright_red"
COLOR_MUTED     = "grey62"
COLOR_BOT       = "bright_blue"
COLOR_USER      = "bright_white"

# ── Memory window ────────────────────────────────────────────────
MEMORY_WINDOW   = 4      # số lượt hội thoại gần nhất được nhớ
FINAL_TOP_K     = 5      # số context gửi cho LLM

# ── Model paths ──────────────────────────────────────────────────
ADAPTER_PATH    = "./models/qwen7b-medical-qlora/final_adapter"
USE_FINETUNED   = os.path.exists(ADAPTER_PATH)

# ── Fallback mode ────────────────────────────────────────────────
# Nếu chưa có model fine-tuned, dùng template trả lời cố định (demo mode)
DEMO_MODE       = not USE_FINETUNED


# ─────────────────────────── Prompt Builder ───────────────────────
SYSTEM_PROMPT = """Bạn là MedBot — trợ lý y tế AI chuyên nghiệp tư vấn sức khỏe tiếng Việt.
Nguyên tắc hoạt động:
1. Chỉ trả lời dựa trên tài liệu y khoa đã cung cấp trong [TÀI LIỆU THAM KHẢO]
2. Nếu tài liệu không đủ thông tin, hãy thành thật nói "Tôi không có đủ dữ liệu về vấn đề này"
3. Luôn kết thúc bằng lời khuyên tham khảo bác sĩ chuyên khoa
4. Không đưa ra chẩn đoán chính xác hoặc kê đơn thuốc cụ thể
5. Ngôn ngữ: Tiếng Việt chuẩn mực, thân thiện, dễ hiểu"""


def build_full_prompt(
    user_query: str,
    context: str,
    conversation_history: List[Dict[str, str]],
) -> str:
    """
    Xây dựng prompt đầy đủ gồm:
    - System prompt
    - Lịch sử hội thoại (memory)
    - Context từ RAG
    - Câu hỏi hiện tại
    """
    history_text = ""
    if conversation_history:
        history_text = "\n[LỊCH SỬ HỘI THOẠI GẦN NHẤT]\n"
        for turn in conversation_history:
            history_text += f"Người dùng: {turn['user']}\nMedBot: {turn['bot']}\n\n"

    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"{history_text}"
        f"\n[TÀI LIỆU THAM KHẢO]\n{context}\n\n"
        f"[CÂU HỎI HIỆN TẠI]\n{user_query}\n\n"
        f"[TRẢ LỜI]"
    )
    return prompt


# ─────────────────────────── Short-term Memory ────────────────────
class ConversationMemory:
    """Bộ nhớ ngắn hạn lưu N lượt hội thoại gần nhất."""

    def __init__(self, window: int = MEMORY_WINDOW):
        self.window = window
        self._history: Deque[Dict[str, str]] = deque(maxlen=window)
        self._session_log: List[Dict[str, Any]] = []

    def add(self, user: str, bot: str) -> None:
        self._history.append({"user": user, "bot": bot})
        self._session_log.append(
            {"timestamp": datetime.now().isoformat(), "user": user, "bot": bot}
        )

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def save_session(self, path: str = "./logs/session.jsonl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for entry in self._session_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        console.print(f"[{COLOR_MUTED}]💾 Phiên chat đã lưu tại: {path}[/]")

    def __len__(self) -> int:
        return len(self._history)


# ─────────────────────────── LLM Wrapper ─────────────────────────
class LLMWrapper:
    """
    Wrapper trừu tượng hóa việc gọi LLM.
    Hỗ trợ: Fine-tuned Qwen-7B hoặc Demo mode (template).
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        if DEMO_MODE:
            console.print(
                f"[{COLOR_WARNING}]⚠️  Model chưa được fine-tune. Đang chạy ở DEMO MODE.[/]\n"
                f"[{COLOR_MUTED}]   → Chạy 'python finetune.py --train' để huấn luyện model.[/]"
            )
            self._loaded = True
            return

        with console.status(f"[{COLOR_PRIMARY}]🔄 Đang tải model fine-tuned...[/]", spinner="dots"):
            from finetune import load_finetuned_model
            self._model, self._tokenizer = load_finetuned_model(ADAPTER_PATH)
        self._loaded = True
        console.print(f"[{COLOR_ACCENT}]✅ Model sẵn sàng![/]")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        if not self._loaded:
            self.load()

        if DEMO_MODE:
            return self._demo_response(prompt)

        from finetune import generate_response
        return generate_response(
            self._model,
            self._tokenizer,
            question=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    @staticmethod
    def _demo_response(prompt: str) -> str:
        """Trả lời mẫu khi ở demo mode (chưa có model thật)."""
        context_start = prompt.find("[TÀI LIỆU THAM KHẢO]")
        context_end   = prompt.find("[CÂU HỎI HIỆN TẠI]")
        context = ""
        if context_start != -1 and context_end != -1:
            context = prompt[context_start + len("[TÀI LIỆU THAM KHẢO]"):context_end].strip()

        query_start = prompt.find("[CÂU HỎI HIỆN TẠI]")
        query = ""
        if query_start != -1:
            query = prompt[query_start + len("[CÂU HỎI HIỆN TẠI]"):].replace("[TRẢ LỜI]", "").strip()

        response = f"**[DEMO MODE]** Dựa trên tài liệu y khoa truy xuất được:\n\n"
        if context:
            lines = context.split("\n")
            for line in lines[:5]:
                if line.strip():
                    response += f"> {line.strip()}\n"
        else:
            response += "> Không tìm thấy tài liệu phù hợp trong cơ sở dữ liệu.\n"

        response += (
            f"\n📋 **Nhận xét:** Câu hỏi của bạn về '{query[:100]}...' đã được hệ thống "
            f"tìm kiếm và truy xuất tài liệu liên quan.\n\n"
            f"⚕️ **Lưu ý:** Đây là phản hồi demo. Khi model được fine-tune hoàn chỉnh, "
            f"câu trả lời sẽ chi tiết và chính xác hơn.\n\n"
            f"🏥 **Khuyến nghị:** Vui lòng tham khảo ý kiến bác sĩ chuyên khoa để được tư vấn chính xác."
        )
        return response


# ─────────────────────────── Terminal UI ─────────────────────────
def print_banner() -> None:
    """Hiển thị banner chào mừng."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║          🏥  MedBot — Chatbot Y Khoa Tiếng Việt         ║
║     Qwen-7B + QLoRA Fine-tune | RAG + ChromaDB          ║
╚══════════════════════════════════════════════════════════╝"""
    console.print(f"[{COLOR_PRIMARY}]{banner}[/]")

    status_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    status_table.add_column(style=COLOR_MUTED)
    status_table.add_column(style=COLOR_ACCENT)
    status_table.add_row("🤖 Model:", "Qwen2-7B-Instruct (QLoRA Fine-tuned)" if not DEMO_MODE else "Demo Mode (chưa fine-tune)")
    status_table.add_row("🗄️  Database:", "ChromaDB (Vietnamese-Healthcare)")
    status_table.add_row("🔍 Retrieval:", "Hybrid Search (Dense + BM25) + Cross-Encoder Rerank")
    status_table.add_row("💬 Memory:", f"Short-term ({MEMORY_WINDOW} lượt gần nhất)")
    console.print(status_table)
    console.print()


def print_help() -> None:
    """Hiển thị hướng dẫn lệnh."""
    help_text = """
**Lệnh đặc biệt:**
- `/help` — Hiển thị trợ giúp này
- `/clear` — Xóa lịch sử hội thoại
- `/history` — Xem lịch sử hội thoại hiện tại
- `/save` — Lưu phiên chat ra file
- `/context` — Xem context tài liệu truy xuất lần trước
- `/exit` hoặc `/quit` — Thoát chương trình
"""
    console.print(Panel(Markdown(help_text), title="📖 Trợ giúp", border_style=COLOR_MUTED))


def print_user_message(message: str) -> None:
    console.print()
    console.print(
        Panel(
            message,
            title=f"[{COLOR_USER}]👤 Bạn[/]",
            border_style=COLOR_USER,
            padding=(0, 2),
        )
    )


def print_bot_message(response: str, sources: Optional[List[Dict]] = None) -> None:
    console.print()
    console.print(
        Panel(
            Markdown(response),
            title=f"[{COLOR_BOT}]🤖 MedBot[/]",
            border_style=COLOR_BOT,
            padding=(1, 2),
        )
    )

    if sources:
        source_table = Table(
            title="📚 Nguồn tài liệu tham khảo",
            box=box.SIMPLE_HEAD,
            style=COLOR_MUTED,
            show_lines=True,
        )
        source_table.add_column("#",    width=3,  style=COLOR_ACCENT)
        source_table.add_column("Điểm", width=8)
        source_table.add_column("Đoạn trích", style="white")

        for i, src in enumerate(sources[:3], 1):
            score = src.get("rerank_score", src.get("hybrid_score", 0))
            snippet = src["text"][:150].replace("\n", " ") + "..."
            source_table.add_row(str(i), f"{score:.3f}", snippet)

        console.print(source_table)


def print_thinking() -> Live:
    """Hiển thị spinner khi đang xử lý."""
    spinner = Spinner("dots", text=f" [{COLOR_PRIMARY}]MedBot đang suy nghĩ...[/]")
    return Live(spinner, console=console, refresh_per_second=10)


# ─────────────────────────── Main Chat Loop ───────────────────────
class MedBotChat:
    """
    Điều phối toàn bộ hệ thống:
    VectorDB → Retriever → LLM → Response
    """

    def __init__(self):
        self.memory    = ConversationMemory(window=MEMORY_WINDOW)
        self.llm       = LLMWrapper()
        self.db: Optional[MedicalVectorDB]      = None
        self.retriever: Optional[HybridRetriever] = None
        self._last_context: str = ""
        self._last_hits: List[Dict] = []

    def initialize(self) -> None:
        """Khởi tạo tất cả components."""
        print_banner()

        # 1. Load Vector DB
        with console.status(f"[{COLOR_PRIMARY}]🗄️  Đang khởi tạo Vector Database...[/]", spinner="dots"):
            self.db = build_database(force_rebuild=False)

        # 2. Build Retriever
        with console.status(f"[{COLOR_PRIMARY}]🔍 Đang khởi tạo Hybrid Retriever...[/]", spinner="dots"):
            self.retriever = build_retriever(self.db)

        # 3. Load LLM
        self.llm.load()

        console.print(f"\n[{COLOR_ACCENT}]✅ Hệ thống sẵn sàng! Gõ câu hỏi y tế hoặc /help để xem hướng dẫn.[/]")
        console.print(Rule(style=COLOR_MUTED))

    def process_query(self, user_input: str) -> Tuple[str, List[Dict]]:
        """
        Xử lý câu hỏi của người dùng:
        1. Retrieve context từ RAG
        2. Build prompt với memory
        3. Generate response từ LLM
        """
        # 1. Retrieve
        hits = self.retriever.retrieve(user_input, final_top_k=FINAL_TOP_K, use_rerank=True)
        self._last_hits = hits
        context = self.retriever.format_context(hits) if hits else "Không tìm thấy tài liệu liên quan."
        self._last_context = context

        # 2. Build prompt
        history = self.memory.get_history()
        prompt  = build_full_prompt(user_input, context, history)

        # 3. Generate
        response = self.llm.generate(prompt)
        return response, hits

    def handle_command(self, cmd: str) -> bool:
        """
        Xử lý lệnh đặc biệt.
        Returns True nếu là lệnh hợp lệ.
        """
        cmd = cmd.strip().lower()

        if cmd in ("/exit", "/quit"):
            self.memory.save_session()
            console.print(f"\n[{COLOR_ACCENT}]👋 Cảm ơn bạn đã sử dụng MedBot. Chúc sức khỏe![/]\n")
            sys.exit(0)

        elif cmd == "/help":
            print_help()
            return True

        elif cmd == "/clear":
            self.memory.clear()
            console.print(f"[{COLOR_ACCENT}]🗑️  Đã xóa lịch sử hội thoại.[/]")
            return True

        elif cmd == "/history":
            history = self.memory.get_history()
            if not history:
                console.print(f"[{COLOR_MUTED}]Chưa có lịch sử hội thoại.[/]")
            else:
                for i, turn in enumerate(history, 1):
                    console.print(f"[{COLOR_USER}][{i}] Bạn:[/] {turn['user'][:100]}...")
                    console.print(f"[{COLOR_BOT}]   MedBot:[/] {turn['bot'][:100]}...")
            return True

        elif cmd == "/save":
            self.memory.save_session()
            return True

        elif cmd == "/context":
            if self._last_context:
                console.print(
                    Panel(
                        self._last_context[:2000] + ("..." if len(self._last_context) > 2000 else ""),
                        title="📚 Context RAG lần trước",
                        border_style=COLOR_MUTED,
                    )
                )
            else:
                console.print(f"[{COLOR_MUTED}]Chưa có context nào.[/]")
            return True

        return False

    def run(self) -> None:
        """Vòng lặp chat chính."""
        self.initialize()

        while True:
            try:
                user_input = Prompt.ask(
                    f"\n[{COLOR_USER}]❓ Câu hỏi của bạn[/]"
                ).strip()

                if not user_input:
                    continue

                # Kiểm tra lệnh đặc biệt
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        console.print(f"[{COLOR_WARNING}]Lệnh không hợp lệ. Gõ /help để xem hướng dẫn.[/]")
                    continue

                # Hiển thị tin nhắn người dùng
                print_user_message(user_input)

                # Xử lý query với spinner
                with print_thinking():
                    response, hits = self.process_query(user_input)

                # Hiển thị phản hồi
                print_bot_message(response, sources=hits)

                # Lưu vào memory
                self.memory.add(user=user_input, bot=response)

            except KeyboardInterrupt:
                console.print(f"\n\n[{COLOR_WARNING}]⚠️  Nhấn Ctrl+C lần nữa hoặc gõ /exit để thoát.[/]")
                try:
                    user_input = Prompt.ask(f"[{COLOR_MUTED}]Tiếp tục? (y/n)[/]")
                    if user_input.lower() in ("n", "no", "exit", "quit"):
                        self.memory.save_session()
                        console.print(f"[{COLOR_ACCENT}]👋 Tạm biệt![/]")
                        sys.exit(0)
                except KeyboardInterrupt:
                    self.memory.save_session()
                    console.print(f"\n[{COLOR_ACCENT}]👋 Tạm biệt![/]")
                    sys.exit(0)

            except Exception as e:
                console.print(f"[{COLOR_ERROR}]❌ Lỗi: {e}[/]")
                logging.exception("Lỗi không mong đợi:")


# ─────────────────────────── Entry Point ─────────────────────────
if __name__ == "__main__":
    bot = MedBotChat()
    bot.run()
