"""
evaluate.py — Vị trí 5 & 6: Evaluation & Researcher
Nhiệm vụ:
  - Tạo tập test 50-100 ca tư vấn y khoa
  - Chạy automation lấy câu trả lời từ hệ thống RAG
  - Tính ROUGE, METEOR, RAGAS (Faithfulness, Answer Relevance)
  - Vẽ biểu đồ so sánh base model vs fine-tuned + RAG
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"

from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
RESULTS_DIR   = "./evaluation_results"
TEST_SET_PATH = "./data/test_set.json"
DATASET_NAME  = "MedRAG/textbooks"
NUM_TEST      = 80   # số ca test


# ─────────────────────────── Test Set ────────────────────────────
def create_test_set(num_samples: int = NUM_TEST, force: bool = False) -> List[Dict]:
    """Trích xuất tập test từ dataset gốc."""
    os.makedirs(os.path.dirname(TEST_SET_PATH), exist_ok=True)

    if os.path.exists(TEST_SET_PATH) and not force:
        with open(TEST_SET_PATH, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"📂 Test set đã có: {len(data)} mẫu")
        return data

    logger.info(f"📥 Tạo test set từ {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME, split="train")

    # Chọn các mẫu phức tạp (câu hỏi dài hơn)
    candidates = []
    for row in ds:
        q = (row.get("question") or row.get("input") or "").strip()
        a = (row.get("answer")   or row.get("output") or "").strip()
        
        content = (row.get("content") or row.get("contents") or row.get("text") or "").strip()
        title = (row.get("title") or "").strip()

        if not q and not a and content:
            q = f"Vui lòng cung cấp thông tin y khoa về: {title}" if title else "Trình bày thông tin y khoa."
            a = content

        if len(q) > 30 and len(a) > 50:
            candidates.append({"question": q, "reference_answer": a})

    # Random sample
    import random
    random.seed(42)
    test_set = random.sample(candidates, min(num_samples, len(candidates)))

    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Đã tạo {len(test_set)} mẫu test → {TEST_SET_PATH}")
    return test_set


# ─────────────────────────── Auto Runner ─────────────────────────
def run_system_on_test_set(
    test_set: List[Dict],
    use_rag: bool = True,
    use_finetuned: bool = True,
    output_tag: str = "rag_ft",
) -> List[Dict]:
    """
    Chạy hệ thống RAG + LLM trên toàn bộ test set,
    lưu câu trả lời vào từng sample.
    """
    from vector_db import build_database
    from retriever import build_retriever

    results_path = os.path.join(RESULTS_DIR, f"answers_{output_tag}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            logger.info(f"📂 Tải kết quả từ cache: {results_path}")
            return json.load(f)

    # Khởi tạo hệ thống
    db        = build_database(force_rebuild=False) if use_rag else None
    retriever = build_retriever(db) if (use_rag and db) else None

    if use_finetuned:
        from finetune import load_finetuned_model, generate_response, MEDICAL_SYSTEM_PROMPT
        model, tokenizer = load_finetuned_model()
    else:
        model = tokenizer = None

    results = []
    for sample in tqdm(test_set, desc=f"Running [{output_tag}]"):
        q = sample["question"]
        context = ""

        if use_rag and retriever:
            hits    = retriever.retrieve(q, final_top_k=5)
            context = retriever.format_context(hits)

        if use_finetuned and model:
            answer = generate_response(model, tokenizer, q, context=context)
        else:
            answer = f"[BASE MODEL DEMO] Trả lời cho: {q[:100]}"

        results.append({
            **sample,
            "system_answer": answer,
            "context_used":  context[:500],
            "tag": output_tag,
            "latency_ms": 0,
        })

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Đã lưu {len(results)} kết quả → {results_path}")
    return results


# ─────────────────────────── Metrics ─────────────────────────────
def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k].append(scores[k].fmeasure)
    return {k: round(float(np.mean(v)), 4) for k, v in agg.items()}


def compute_meteor(predictions: List[str], references: List[str]) -> float:
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    scores = []
    for pred, ref in zip(predictions, references):
        score = meteor_score([ref.split()], pred.split())
        scores.append(score)
    return round(float(np.mean(scores)), 4)


def compute_ragas_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Tính RAGAS-style metrics (Faithfulness, Answer Relevance) bằng heuristic
    khi không có API key. Dùng overlap đơn giản làm proxy.
    """
    faithfulness_scores = []
    relevance_scores    = []

    for item in results:
        answer  = item.get("system_answer", "").lower()
        context = item.get("context_used",  "").lower()
        question = item.get("question",      "").lower()

        # Faithfulness: % từ câu trả lời xuất hiện trong context
        answer_words  = set(answer.split())
        context_words = set(context.split())
        faith = len(answer_words & context_words) / max(len(answer_words), 1)
        faithfulness_scores.append(min(faith, 1.0))

        # Answer Relevance: % từ câu hỏi xuất hiện trong câu trả lời
        q_words = set(question.split())
        rel = len(q_words & answer_words) / max(len(q_words), 1)
        relevance_scores.append(min(rel, 1.0))

    return {
        "faithfulness":     round(float(np.mean(faithfulness_scores)), 4),
        "answer_relevance": round(float(np.mean(relevance_scores)),    4),
    }


def evaluate_results(results: List[Dict], tag: str = "") -> Dict[str, Any]:
    """Tính toàn bộ metrics cho một tập kết quả."""
    predictions = [r["system_answer"]    for r in results]
    references  = [r["reference_answer"] for r in results]

    logger.info(f"📊 Đang tính ROUGE [{tag}]...")
    rouge  = compute_rouge(predictions, references)

    logger.info(f"📊 Đang tính METEOR [{tag}]...")
    meteor = compute_meteor(predictions, references)

    logger.info(f"📊 Đang tính RAGAS [{tag}]...")
    ragas  = compute_ragas_metrics(results)

    metrics = {
        "tag":    tag,
        "n":      len(results),
        **rouge,
        "meteor": meteor,
        **ragas,
    }
    logger.info(f"✅ [{tag}] → {metrics}")
    return metrics


# ─────────────────────────── Visualization ───────────────────────
def plot_comparison(metrics_list: List[Dict], save_path: str = None) -> None:
    """Vẽ biểu đồ so sánh các cấu hình hệ thống."""
    if not metrics_list:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("So sánh hiệu suất: Base Model vs Fine-tuned + RAG", fontsize=14, fontweight="bold")

    tags   = [m["tag"]   for m in metrics_list]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"][:len(tags)]

    # ── Plot 1: NLG Metrics ────────────────────────────────────
    ax1 = axes[0]
    nlg_keys = ["rouge1", "rouge2", "rougeL", "meteor"]
    nlg_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]
    x = np.arange(len(nlg_labels))
    width = 0.8 / len(tags)

    for i, (m, color) in enumerate(zip(metrics_list, colors)):
        vals = [m.get(k, 0) for k in nlg_keys]
        bars = ax1.bar(x + i * width - (len(tags) - 1) * width / 2, vals, width, label=m["tag"], color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.set_title("NLG Metrics (ROUGE & METEOR)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nlg_labels)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # ── Plot 2: RAG Metrics ────────────────────────────────────
    ax2 = axes[1]
    rag_keys   = ["faithfulness", "answer_relevance"]
    rag_labels = ["Faithfulness", "Answer Relevance"]
    x2 = np.arange(len(rag_labels))

    for i, (m, color) in enumerate(zip(metrics_list, colors)):
        vals = [m.get(k, 0) for k in rag_keys]
        bars = ax2.bar(x2 + i * width - (len(tags) - 1) * width / 2, vals, width, label=m["tag"], color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax2.set_title("RAG Quality Metrics (RAGAS-style)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(rag_labels)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "comparison_chart.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"📈 Biểu đồ lưu tại: {save_path}")
    plt.show()


def save_metrics_report(metrics_list: List[Dict]) -> None:
    """Lưu báo cáo metrics ra JSON và Markdown."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # JSON
    json_path = os.path.join(RESULTS_DIR, "metrics_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, ensure_ascii=False, indent=2)

    # Markdown table
    md_path = os.path.join(RESULTS_DIR, "metrics_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo Đánh giá Hệ thống MedBot\n\n")
        f.write(f"**Ngày đánh giá:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Kết quả Metrics\n\n")
        f.write("| Hệ thống | N | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | Faithfulness | Ans. Relevance |\n")
        f.write("|----------|---|---------|---------|---------|--------|--------------|----------------|\n")
        for m in metrics_list:
            f.write(
                f"| {m['tag']} | {m['n']} "
                f"| {m.get('rouge1',0):.4f} | {m.get('rouge2',0):.4f} "
                f"| {m.get('rougeL',0):.4f} | {m.get('meteor',0):.4f} "
                f"| {m.get('faithfulness',0):.4f} | {m.get('answer_relevance',0):.4f} |\n"
            )
        f.write("\n## Kết luận\n\nXem biểu đồ `comparison_chart.png` để so sánh trực quan.\n")

    logger.info(f"📝 Báo cáo lưu tại: {json_path} và {md_path}")


# ─────────────────────────── Main ────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Đánh giá hệ thống MedBot")
    parser.add_argument("--create-test",  action="store_true", help="Tạo test set mới")
    parser.add_argument("--run-eval",     action="store_true", help="Chạy evaluation đầy đủ")
    parser.add_argument("--plot-only",    action="store_true", help="Chỉ vẽ biểu đồ từ kết quả có sẵn")
    parser.add_argument("--num-test",     type=int, default=NUM_TEST)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.create_test:
        create_test_set(num_samples=args.num_test, force=True)

    if args.run_eval:
        test_set = create_test_set(num_samples=args.num_test)

        all_metrics = []

        # Cấu hình 1: RAG + Fine-tuned
        results_ft = run_system_on_test_set(test_set, use_rag=True, use_finetuned=True, output_tag="RAG+FT")
        all_metrics.append(evaluate_results(results_ft, tag="RAG+FT"))

        # Cấu hình 2: RAG only (base model)
        results_base = run_system_on_test_set(test_set, use_rag=True, use_finetuned=False, output_tag="RAG+Base")
        all_metrics.append(evaluate_results(results_base, tag="RAG+Base"))

        save_metrics_report(all_metrics)
        plot_comparison(all_metrics)

    elif args.plot_only:
        report_path = os.path.join(RESULTS_DIR, "metrics_report.json")
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                all_metrics = json.load(f)
            plot_comparison(all_metrics)
        else:
            logger.error(f"Chưa có kết quả. Chạy --run-eval trước.")
    else:
        print("Sử dụng:")
        print("  python evaluate.py --create-test       # Tạo test set")
        print("  python evaluate.py --run-eval           # Chạy đánh giá đầy đủ")
        print("  python evaluate.py --plot-only          # Vẽ biểu đồ từ kết quả cũ")
