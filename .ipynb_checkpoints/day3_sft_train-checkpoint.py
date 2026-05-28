import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# =========================
# 0. 基础路径配置
# =========================

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

dataset_path = "/root/autodl-tmp/llm_ads_project/data/ads_sft_demo"
output_dir = "/root/autodl-tmp/llm_ads_project/outputs/day3_sft"

os.makedirs(output_dir, exist_ok=True)

print("===== 环境检查 =====")
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

# =========================
# 1. 加载 tokenizer 和模型
# =========================

print("\n===== 加载 tokenizer =====")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Qwen 一般有 eos_token，但有些模型没有 pad_token
# Trainer batch padding 时需要 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad_token:", tokenizer.pad_token)
print("eos_token:", tokenizer.eos_token)

print("\n===== 加载模型 =====")
model = AutoModelForCausalLM.from_pretrained(
    model_id
)

model.to("cuda")


# 避免训练时 cache 占显存，训练一般关掉
model.config.use_cache = False
print("model device:", next(model.parameters()).device)
print("model dtype:", next(model.parameters()).dtype)
# =========================
# 2. 读取 Day2 数据集
# =========================

print("\n===== 加载 Dataset =====")
dataset = load_from_disk(dataset_path)
print(dataset)
print("\n第一条 text：")
print(dataset[0]["text"])

# =========================
# 3. Tokenize
# =========================

max_length = 512

def tokenize_function(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    # Causal LM 训练中，labels 通常等于 input_ids
    # 模型学习预测下一个 token
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("\n===== Tokenize Dataset =====")
tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
)

print(tokenized_dataset)
print("\n第一条 tokenized 样本 keys:", tokenized_dataset[0].keys())
print("input_ids length:", len(tokenized_dataset[0]["input_ids"]))

# =========================
# 4. Data Collator
# =========================

# mlm=False 表示做 causal language modeling，不是 BERT 那种 mask language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# =========================
# 5. TrainingArguments
# =========================

training_args = TrainingArguments(
    output_dir=output_dir,

    # 10 条数据很小，我们多训几个 epoch，观察 loss 是否能下降
    num_train_epochs=10,

    # 0.5B 模型，batch size 先设 1，稳妥
    per_device_train_batch_size=1,

    # 累积梯度，等效 batch size = 1 * 4 = 4
    gradient_accumulation_steps=4,

    # 学习率稍大一点，方便在小数据上看到 loss 下降
    learning_rate=2e-5,

    # 日志频率
    logging_steps=1,

    # 保存策略
    save_strategy="epoch",
    save_total_limit=2,

    # 混合精度
    fp16=False,
    bf16=True,

    # 训练日志
    report_to="none",

    # DataLoader
    dataloader_num_workers=0,

    # 小数据集不需要太多复杂策略
    warmup_steps=0,

    # 避免一些 warning
    remove_unused_columns=False,
)

# =========================
# 6. Trainer
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# =========================
# 7. 开始训练
# =========================

print("\n===== 开始训练 =====")
trainer.train()

# =========================
# 8. 保存最终模型
# =========================

final_dir = os.path.join(output_dir, "final_model")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

print(f"\n✅ Day3 SFT 训练完成，模型已保存到：{final_dir}")
