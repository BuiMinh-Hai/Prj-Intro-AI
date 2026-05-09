"""
finetune.py — Vị trí 3: LLM Engineer (Fine-tuning Qwen-7B với QLoRA)
Nhiệm vụ:
  - Định dạng dataset Vietnamese-Healthcare theo chuẩn System/User/Assistant
  - Fine-tune Qwen-7B bằng QLoRA (dùng Unsloth để tăng tốc)
  - Monitor Loss, lưu Adapter weights vào thư mục models/
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

# ─────────────────────────── Cấu hình ───────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Model ──────────────────────────────────────────────────────
BASE_MODEL_ID = "Qwen/Qwen2-7B-Instruct"   # Qwen2 tốt hơn Qwen1 cho tiếng Việt

# ── Paths ──────────────────────────────────────────────────────
MODEL_OUTPUT_DIR   = "./models/qwen7b-medical-qlora"
LOGS_DIR           = "./models/logs"
DATASET_NAME       = "urnus11/Vietnamese-Healthcare"
FORMATTED_DATA_DIR = "./data/formatted"

# ── QLoRA hyperparameters ───────────────────────────────────────
LORA_R          = 16       # Rank của LoRA adapter
LORA_ALPHA      = 32       # LoRA scaling factor
LORA_DROPOUT    = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training hyperparameters ────────────────────────────────────
MAX_SEQ_LENGTH     = 2048
TRAIN_EPOCHS       = 3
BATCH_SIZE         = 2      # nhỏ để tiết kiệm VRAM
GRAD_ACCUMULATION  = 8      # gradient accumulation steps (effective batch = 16)
LEARNING_RATE      = 2e-4
WARMUP_RATIO       = 0.05
WEIGHT_DECAY       = 0.01
SAVE_STEPS         = 200
EVAL_STEPS         = 100
LOGGING_STEPS      = 25
FP16               = True   # Mixed precision


# ─────────────────────────── System Prompt ───────────────────────
MEDICAL_SYSTEM_PROMPT = """Bạn là một trợ lý y tế AI chuyên nghiệp, được đào tạo chuyên sâu về y khoa Việt Nam. 
Nhiệm vụ của bạn là:
- Cung cấp thông tin y tế chính xác, dựa trên bằng chứng khoa học
- Giải thích triệu chứng, bệnh lý và hướng dẫn điều trị bằng ngôn ngữ dễ hiểu
- Luôn nhắc nhở người dùng tham khảo ý kiến bác sĩ chuyên khoa cho các quyết định y tế quan trọng
- Không cung cấp chẩn đoán chính thức hoặc kê đơn thuốc

Hãy trả lời bằng tiếng Việt chuẩn mực, chuyên nghiệp và có trách nhiệm."""


# ─────────────────────────── Prompt Formatter ────────────────────
def format_chat_prompt(question: str, answer: str, system: str = MEDICAL_SYSTEM_PROMPT) -> str:
    """
    Định dạng câu hỏi/trả lời theo chuẩn ChatML (Qwen2 format).
    <|im_start|>system ... <|im_end|>
    <|im_start|>user ... <|im_end|>
    <|im_start|>assistant ... <|im_end|>
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


def preprocess_and_format_dataset(dataset_name: str = DATASET_NAME) -> Dataset:
    """
    Tải và định dạng lại dataset sang chuẩn ChatML.
    Lưu bản formatted ra đĩa để tái sử dụng.
    """
    os.makedirs(FORMATTED_DATA_DIR, exist_ok=True)
    cache_path = os.path.join(FORMATTED_DATA_DIR, "formatted_dataset")

    if os.path.exists(cache_path):
        logger.info(f"📂 Tải dataset đã format từ cache: {cache_path}")
        return Dataset.load_from_disk(cache_path)

    logger.info(f"📥 Tải dataset gốc: {dataset_name}")
    raw_ds = load_dataset(dataset_name, split="train")
    logger.info(f"✅ {len(raw_ds)} samples tải thành công.")

    formatted_samples: List[Dict[str, str]] = []
    skipped = 0

    for row in raw_ds:
        question = (
            row.get("question") or row.get("input") or row.get("instruction") or ""
        ).strip()
        answer = (
            row.get("answer") or row.get("output") or row.get("response") or ""
        ).strip()

        if not question or not answer:
            skipped += 1
            continue

        # Giới hạn độ dài để tránh OOM
        if len(question) + len(answer) > MAX_SEQ_LENGTH * 3:
            answer = answer[: MAX_SEQ_LENGTH * 2]

        formatted_samples.append(
            {"text": format_chat_prompt(question, answer)}
        )

    logger.info(f"✅ Formatted: {len(formatted_samples)} samples | Bỏ qua: {skipped}")

    dataset = Dataset.from_list(formatted_samples)
    dataset.save_to_disk(cache_path)
    logger.info(f"💾 Đã lưu dataset vào: {cache_path}")
    return dataset


# ─────────────────────────── Model Loader ────────────────────────
def load_base_model_and_tokenizer(
    model_id: str = BASE_MODEL_ID,
    use_4bit: bool = True,
):
    """
    Tải Qwen-7B với quantization 4-bit (BitsAndBytes) để tiết kiệm VRAM.
    """
    logger.info(f"🔄 Đang tải tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config: Optional[BitsAndBytesConfig] = None
    if use_4bit:
        logger.info("⚙️  Cấu hình BitsAndBytes 4-bit quantization ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )

    logger.info(f"🔄 Đang tải model: {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    logger.info("✅ Model và Tokenizer sẵn sàng.")
    return model, tokenizer


# ─────────────────────────── LoRA Config ─────────────────────────
def apply_qlora(model) -> Any:
    """Áp dụng LoRA adapter lên model đã quantize."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────── Training ────────────────────────────
def train(
    use_4bit: bool = True,
    epochs: int = TRAIN_EPOCHS,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """
    Full fine-tuning pipeline:
    1. Tải dataset và format
    2. Tải model + QLoRA
    3. Training với SFTTrainer
    4. Lưu adapter weights
    """
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # 1. Dataset
    dataset = preprocess_and_format_dataset()
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    logger.info(f"📊 Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # 2. Model + QLoRA
    model, tokenizer = load_base_model_and_tokenizer(use_4bit=use_4bit)
    model = apply_qlora(model)
    model.config.use_cache = False   # Bắt buộc khi dùng gradient checkpointing

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        logging_dir=LOGS_DIR,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
    )

    # 4. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,   # Packing để tối ưu tốc độ
    )

    # 5. Huấn luyện
    logger.info("🚀 Bắt đầu fine-tuning ...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 6. Lưu adapter
    final_adapter_path = os.path.join(MODEL_OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    logger.info(f"💾 Đã lưu LoRA adapter tại: {final_adapter_path}")

    # Lưu config training
    config_save = {
        "base_model": BASE_MODEL_ID,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_target_modules": LORA_TARGET_MODULES,
        "epochs": epochs,
        "batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "dataset": DATASET_NAME,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
    }
    config_path = os.path.join(MODEL_OUTPUT_DIR, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_save, f, ensure_ascii=False, indent=2)
    logger.info(f"📝 Training config lưu tại: {config_path}")


# ─────────────────────────── Inference Helper ─────────────────────
def load_finetuned_model(adapter_path: str = os.path.join(MODEL_OUTPUT_DIR, "final_adapter")):
    """
    Tải model đã fine-tune (base + LoRA adapter) cho inference.
    """
    from peft import PeftModel

    logger.info(f"🔄 Tải model fine-tuned từ: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    logger.info("✅ Model fine-tuned sẵn sàng cho inference.")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    context: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """
    Sinh câu trả lời từ model đã fine-tune.
    """
    import torch

    if context:
        user_msg = f"Thông tin tham khảo:\n{context}\n\nCâu hỏi: {question}"
    else:
        user_msg = question

    prompt = format_chat_prompt(user_msg, "")
    # Bỏ phần cuối <|im_end|> để model tự điền
    prompt = prompt.rstrip("<|im_end|>").rstrip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()


# ─────────────────────────── Main ────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen-7B cho y khoa tiếng Việt")
    parser.add_argument("--train",    action="store_true", help="Bắt đầu training")
    parser.add_argument("--no-4bit",  action="store_true", help="Tắt quantization 4-bit")
    parser.add_argument("--epochs",   type=int, default=TRAIN_EPOCHS, help="Số epochs")
    parser.add_argument("--resume",   type=str, default=None, help="Checkpoint để resume")
    parser.add_argument("--test",     type=str, default="",   help="Test inference với câu hỏi")
    args = parser.parse_args()

    if args.train:
        train(
            use_4bit=not args.no_4bit,
            epochs=args.epochs,
            resume_from_checkpoint=args.resume,
        )
    elif args.test:
        model, tokenizer = load_finetuned_model()
        answer = generate_response(model, tokenizer, args.test)
        print(f"\n{'='*60}\nCâu hỏi: {args.test}\n{'='*60}\nTrả lời: {answer}")
    else:
        print("Sử dụng: python finetune.py --train [--no-4bit] [--epochs N] [--resume PATH]")
        print("         python finetune.py --test 'Câu hỏi y tế của bạn'")
