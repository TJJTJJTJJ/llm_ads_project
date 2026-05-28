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

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

dataset_path = "/root/autodl-tmp/llm_ads_project/data/ads_sft_demo"
output_dir = "/root/autodl-tmp/llm_ads_project/outputs/day4_sft_masked"

os.makedirs(output_dir, exist_ok=True)

print("===== 环境检查 =====")
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

print("\n===== 加载 tokenizer =====")
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad_token:", tokenizer.pad_token)
print("eos_token:", tokenizer.eos_token)

print("\n===== 加载模型 =====")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
)
model.to("cuda")
model.config.use_cache = False

print("model device:", next(model.parameters()).device)
print("model dtype:", next(model.parameters()).dtype)

print("\n===== 加载 Dataset =====")
dataset = load_from_disk(dataset_path)
print(dataset)

max_length = 512


def build_masked_example(example):
    """
    标准 SFT：
    - input_ids：完整 system + user + assistant
    - labels：system/user 部分 = -100，assistant 部分 = token id
    """

    # 1. prompt_messages 只包含 system + user
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

    # 2. full_messages 包含 system + user + assistant
    full_messages = prompt_messages + [
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    # 3. 渲染 prompt text
    # add_generation_prompt=True 会在末尾加 <|im_start|>assistant
    # 这部分是 assistant 答案之前的全部上下文
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 4. 渲染完整训练 text
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 5. 分别 tokenize
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

    # 6. labels 初始等于 input_ids
    labels = input_ids.copy()

    # 7. mask 掉 prompt 部分
    # prompt 部分包括 system/user 以及 assistant 起始标记
    labels[:prompt_len] = [-100] * min(prompt_len, len(labels))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_len": prompt_len,
        "full_len": len(input_ids),
    }


print("\n===== 构造 masked tokenized dataset =====")
tokenized_dataset = dataset.map(
    build_masked_example,
    remove_columns=dataset.column_names,
)

print(tokenized_dataset)

print("\n===== 检查第一条样本 mask 情况 =====")
sample = tokenized_dataset[0]
print("prompt_len:", sample["prompt_len"])
print("full_len:", sample["full_len"])
print("input_ids length:", len(sample["input_ids"]))
print("labels length:", len(sample["labels"]))

print("\n前 30 个 labels:")
print(sample["labels"][:30])

print("\n从 prompt_len 附近开始的 labels:")
start = max(0, sample["prompt_len"] - 5)
end = min(len(sample["labels"]), sample["prompt_len"] + 20)
print(sample["labels"][start:end])

print("\nassistant 参与 loss 的 token 解码：")
valid_label_ids = [x for x in sample["labels"] if x != -100]
print(tokenizer.decode(valid_label_ids, skip_special_tokens=False))


# 这里使用 DataCollatorForSeq2Seq 更适合已有 labels 且需要 padding 的场景
# 它会把 labels 的 padding 补成 -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("\n===== 开始标准 masked SFT 训练 =====")
trainer.train()

final_dir = os.path.join(output_dir, "final_model")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

print(f"\n✅ Day4 masked SFT 训练完成，模型已保存到：{final_dir}")
