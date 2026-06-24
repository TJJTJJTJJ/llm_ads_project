import os
import torch
import bitsandbytes as bnb

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =========================
# 0. 基础配置
# =========================

model_id = "Qwen/Qwen2.5-7B-Instruct"
output_dir = "/root/autodl-tmp/llm_ads_project/outputs/day9_qwen7b_qlora_sft"
final_adapter_dir = f"{output_dir}/final_adapter"

os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["HF_HUB_CACHE"] = "/root/autodl-tmp/huggingface/hub"
os.environ["HF_HUB_DISABLE_XET"] = "1"

os.makedirs(output_dir, exist_ok=True)

print("===== 环境检查 =====")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("bitsandbytes:", bnb.__version__)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("before load allocated GB:", torch.cuda.memory_allocated() / 1024**3)
    print("before load reserved GB:", torch.cuda.memory_reserved() / 1024**3)

# =========================
# 1. 构造广告 SFT 数据
# =========================

raw_data = [
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：低糖白桃气泡水；卖点：低卡、清爽解腻、适合控糖；投放场景：小红书信息流。",
        "output": "目标受众：控糖人群、健身人群、年轻女性、轻食爱好者。\n核心卖点：低卡、低糖、白桃风味、清爽解腻。\n点击倾向：高。\n原因：商品卖点与健康饮食、控糖和轻负担消费需求高度匹配。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：男士机械腕表；卖点：商务风、防水、夜光表盘；投放场景：资讯 App 信息流。",
        "output": "目标受众：商务男性、职场人士、腕表爱好者。\n核心卖点：商务外观、防水性能、夜光表盘、通勤适用。\n点击倾向：中高。\n原因：商品适合职场和商务场景，但价格、品牌和款式会进一步影响点击。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：儿童益智拼图；卖点：安全环保、训练专注力、亲子互动；投放场景：短视频 App。",
        "output": "目标受众：3-8 岁儿童家长、宝妈宝爸、早教关注人群。\n核心卖点：安全材质、益智启蒙、专注力训练、亲子陪伴。\n点击倾向：高。\n原因：家长对儿童教育和安全材质敏感，短视频场景也适合展示玩法。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：高端游戏显卡；卖点：高帧率、光追、AI 加速；投放场景：游戏社区。",
        "output": "目标受众：硬核游戏玩家、DIY 装机用户、内容创作者、AI 开发者。\n核心卖点：高性能渲染、光线追踪、高帧率体验、AI 加速能力。\n点击倾向：高。\n原因：投放场景与目标用户高度匹配，卖点直接对应游戏和创作需求。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：办公室人体工学椅；卖点：腰托支撑、久坐舒适、可调节扶手；投放场景：职场社区。",
        "output": "目标受众：白领、程序员、居家办公人群、久坐办公用户。\n核心卖点：腰部支撑、久坐舒适、可调节设计、改善坐姿。\n点击倾向：高。\n原因：职场社区用户对久坐健康和办公舒适度有明确需求。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：宠物自动喂食器；卖点：定时投喂、远程控制、防潮储粮；投放场景：宠物内容频道。",
        "output": "目标受众：养猫养狗人群、上班族宠物主人、经常出差的宠物用户。\n核心卖点：自动投喂、远程控制、储粮防潮、减少喂养焦虑。\n点击倾向：高。\n原因：商品解决宠物主人无法按时喂养的痛点，场景匹配度高。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：平价口红套装；卖点：多色号、显白、不易掉色；投放场景：美妆短视频。",
        "output": "目标受众：年轻女性、学生党、彩妆入门用户、美妆爱好者。\n核心卖点：多色可选、显白效果、持妆能力、性价比高。\n点击倾向：高。\n原因：美妆短视频适合展示上嘴效果，平价套装也容易刺激尝试。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：户外露营帐篷；卖点：防雨、防风、快速搭建；投放场景：旅行攻略社区。",
        "output": "目标受众：露营爱好者、户外旅行用户、家庭周末出游人群。\n核心卖点：防雨防风、搭建方便、适合户外场景。\n点击倾向：中高。\n原因：旅行社区用户具备潜在需求，但点击还会受季节、价格和出游计划影响。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：中老年钙片；卖点：补钙、维生素 D、增强骨骼健康；投放场景：健康资讯 App。",
        "output": "目标受众：中老年人、关注骨骼健康人群、子女为父母购买的人群。\n核心卖点：补钙、促进吸收、骨骼健康、日常保健。\n点击倾向：中高。\n原因：健康资讯场景匹配，但保健品点击受品牌信任和功效表达影响较大。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：无线降噪耳机；卖点：主动降噪、长续航、通勤舒适；投放场景：地铁 WiFi 页面。",
        "output": "目标受众：通勤人群、学生、办公族、音乐和播客用户。\n核心卖点：主动降噪、长续航、佩戴舒适、适合通勤。\n点击倾向：高。\n原因：地铁场景与通勤降噪需求高度匹配，用户痛点明确。"
    },
]

dataset = Dataset.from_list(raw_data)

print("\n===== 原始数据 =====")
print(dataset)
print(dataset[0])

# =========================
# 2. 加载 tokenizer
# =========================

print("\n===== 加载 tokenizer =====")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad_token:", tokenizer.pad_token)
print("eos_token:", tokenizer.eos_token)

# =========================
# 3. 预处理：assistant-only labels
# =========================

max_length = 512

def preprocess(example):
    system_message = "你是一个广告算法助手，擅长根据商品信息分析目标受众、核心卖点和点击倾向。"

    user_content = f"{example['instruction']}\n\n{example['input']}"
    assistant_content = example["output"]

    # prompt 部分：system + user + assistant 起始标记
    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]

    # 完整训练样本：system + user + assistant answer
    full_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
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

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    full = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = input_ids.copy()

    prompt_len = len(prompt_ids)

    # prompt 部分不算 loss，只训练 assistant answer
    labels[:prompt_len] = [-100] * min(prompt_len, len(labels))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
)

print("\n===== tokenized dataset =====")
print(tokenized_dataset)
print("first input length:", len(tokenized_dataset[0]["input_ids"]))
print("first labels length:", len(tokenized_dataset[0]["labels"]))
print("first labels non -100:", sum(1 for x in tokenized_dataset[0]["labels"] if x != -100))

# =========================
# 4. 4-bit 加载 base model
# =========================

print("\n===== 4-bit 加载 base model =====")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # 3090 上训练建议 FP16 更稳
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

print("base model loaded")
print("first param dtype:", next(model.parameters()).dtype)

linear4bit_count = 0
for name, module in model.named_modules():
    if isinstance(module, bnb.nn.Linear4bit):
        linear4bit_count += 1

print("Linear4bit modules:", linear4bit_count)

if torch.cuda.is_available():
    print("after base load allocated GB:", torch.cuda.memory_allocated() / 1024**3)
    print("after base load reserved GB:", torch.cuda.memory_reserved() / 1024**3)

# =========================
# 5. prepare model for k-bit training
# =========================

print("\n===== prepare_model_for_kbit_training =====")

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# 可选：开启 gradient checkpointing，进一步省显存
model.gradient_checkpointing_enable()

print("prepare done")

# =========================
# 6. 注入 LoRA
# =========================

print("\n===== 注入 LoRA =====")

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

print("LoRA injected")
model.print_trainable_parameters()

if torch.cuda.is_available():
    print("after LoRA allocated GB:", torch.cuda.memory_allocated() / 1024**3)
    print("after LoRA reserved GB:", torch.cuda.memory_reserved() / 1024**3)

# =========================
# 7. Data collator
# =========================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

# =========================
# 8. TrainingArguments
# =========================

training_args = TrainingArguments(
    output_dir=output_dir,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    num_train_epochs=3,
    learning_rate=1e-4,

    logging_steps=1,
    save_steps=20,
    save_total_limit=2,

    fp16=True,
    bf16=False,

    optim="paged_adamw_8bit",

    gradient_checkpointing=True,

    report_to="none",

    remove_unused_columns=False,
)

# =========================
# 9. Trainer
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# =========================
# 10. 开始训练
# =========================

print("\n===== 开始 QLoRA SFT 训练 =====")
trainer.train()

# =========================
# 11. 保存 adapter
# =========================

print("\n===== 保存 LoRA adapter =====")
model.save_pretrained(final_adapter_dir)
tokenizer.save_pretrained(final_adapter_dir)

print("saved final adapter to:", final_adapter_dir)

if torch.cuda.is_available():
    print("\n===== 训练结束显存 =====")
    print("allocated GB:", torch.cuda.memory_allocated() / 1024**3)
    print("reserved GB:", torch.cuda.memory_reserved() / 1024**3)
    print("max allocated GB:", torch.cuda.max_memory_allocated() / 1024**3)

print("\n===== 完成：7B QLoRA SFT =====")
