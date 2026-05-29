import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

# =========================
# 0. 路径配置
# =========================

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
dataset_path = "/root/autodl-tmp/llm_ads_project/data/ads_sft_demo"
output_dir = "/root/autodl-tmp/llm_ads_project/outputs/day5_lora_sft"

os.makedirs(output_dir, exist_ok=True)

print("===== 环境检查 =====")
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

# =========================
# 1. 加载 tokenizer
# =========================

print("\n===== 加载 tokenizer =====")
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad_token:", tokenizer.pad_token)
print("eos_token:", tokenizer.eos_token)

# =========================
# 2. 加载 base model
# =========================

print("\n===== 加载 base model =====")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
)

model.to("cuda")
model.config.use_cache = False

print("base model device:", next(model.parameters()).device)
print("base model dtype:", next(model.parameters()).dtype)

# =========================
# 3. 配置 LoRA
# =========================

print("\n===== 配置 LoRA =====")

# Qwen2 常见 attention / MLP 线性层：
# q_proj, k_proj, v_proj, o_proj：attention
# gate_proj, up_proj, down_proj：MLP
# 这里先覆盖主要线性层，效果更明显
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)

print("\n===== LoRA 可训练参数 =====")
model.print_trainable_parameters()

# =========================
# 4. 加载 Dataset
# =========================

print("\n===== 加载 Dataset =====")
dataset = load_from_disk(dataset_path)
print(dataset)

max_length = 512

# =========================
# 5. 构造 assistant-only labels
# =========================

def build_masked_example(example):
    """
    标准 SFT：
    - input_ids：完整 system + user + assistant
    - labels：system/user/assistant起始标记 = -100，assistant answer = token id
    """

    prompt_messages = [
        {
            "role": "system",
            "content": "你是一个广告算法助手，擅长根据广告商品信息分析目标受众、核心卖点和点击倾向。"
        },
        {
            "role": "user",
            "content": f"{example['instruction']}\n\n广告信息：{example['input']}"
        }
    ]

    full_messages = prompt_messages + [
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_tokenized = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )

    full_tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )

    input_ids = full_tokenized["input_ids"]
    attention_mask = full_tokenized["attention_mask"]

    prompt_len = len(prompt_tokenized["input_ids"])

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * min(prompt_len, len(labels))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_len": prompt_len,
        "full_len": len(input_ids),
    }


print("\n===== 构造 LoRA SFT tokenized dataset =====")
tokenized_dataset = dataset.map(
    build_masked_example,
    remove_columns=dataset.column_names,
)

print(tokenized_dataset)

# 检查第一条样本
sample = tokenized_dataset[0]
print("\n===== 检查第一条样本 mask 情况 =====")
print("prompt_len:", sample["prompt_len"])
print("full_len:", sample["full_len"])

valid_label_ids = [x for x in sample["labels"] if x != -100]
print("assistant 参与 loss 的 token 解码：")
print(tokenizer.decode(valid_label_ids, skip_special_tokens=False))

# =========================
# 6. Data Collator
# =========================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

# =========================
# 7. TrainingArguments
# =========================

training_args = TrainingArguments(
    output_dir=output_dir,

    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    # LoRA 训练学习率通常可以比全参 SFT 稍大
    learning_rate=1e-4,

    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=2,

    fp16=False,
    bf16=True,

    report_to="none",
    dataloader_num_workers=0,
    warmup_steps=0,
    remove_unused_columns=False,
)

# =========================
# 8. Trainer
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# =========================
# 9. 开始训练
# =========================

print("\n===== 开始 LoRA SFT 训练 =====")
trainer.train()

# =========================
# 10. 保存 LoRA Adapter
# =========================

final_adapter_dir = os.path.join(output_dir, "final_adapter")
model.save_pretrained(final_adapter_dir)
tokenizer.save_pretrained(final_adapter_dir)

print(f"\n✅ Day5 LoRA SFT 训练完成，Adapter 已保存到：{final_adapter_dir}")
