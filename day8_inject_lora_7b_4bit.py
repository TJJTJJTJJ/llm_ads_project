import os
import torch
import bitsandbytes as bnb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
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

os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["HF_HUB_CACHE"] = "/root/autodl-tmp/huggingface/hub"
os.environ["HF_HUB_DISABLE_XET"] = "1"

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
# 1. 4-bit 量化配置
# =========================

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("\n===== BitsAndBytesConfig =====")
print(quantization_config)

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
# 3. 4-bit 加载 base model
# =========================

print("\n===== 4-bit 加载 base model =====")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

print("\n===== base model 加载完成 =====")
print("model class:", type(model))
print("model device:", next(model.parameters()).device)
print("first param dtype:", next(model.parameters()).dtype)

linear4bit_count = 0
for name, module in model.named_modules():
    if isinstance(module, bnb.nn.Linear4bit):
        linear4bit_count += 1

print("Linear4bit modules before LoRA:", linear4bit_count)

if torch.cuda.is_available():
    print("after base load allocated GB:", torch.cuda.memory_allocated() / 1024**3)
    print("after base load reserved GB:", torch.cuda.memory_reserved() / 1024**3)
    print("max memory allocated GB:", torch.cuda.max_memory_allocated() / 1024**3)

# =========================
# 4. 为 k-bit training 做准备
# =========================

print("\n===== prepare_model_for_kbit_training =====")

# 训练时建议关闭 cache，否则可能和 gradient checkpointing 冲突
model.config.use_cache = False

model = prepare_model_for_kbit_training(model)

print("prepare_model_for_kbit_training done")

# =========================
# 5. LoRA 配置
# =========================

print("\n===== 创建 LoraConfig =====")

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

print(lora_config)

# =========================
# 6. 注入 LoRA
# =========================

print("\n===== 注入 LoRA adapter =====")

model = get_peft_model(model, lora_config)

print("LoRA injected")

# =========================
# 7. 打印可训练参数
# =========================

print("\n===== 可训练参数 =====")
model.print_trainable_parameters()

# 手动统计一次，方便理解
trainable_params = 0
all_params = 0

for name, param in model.named_parameters():
    num_params = param.numel()
    all_params += num_params
    if param.requires_grad:
        trainable_params += num_params

print("\nmanual count:")
print("trainable params:", trainable_params)
print("all params:", all_params)
print("trainable%:", 100 * trainable_params / all_params)

# =========================
# 8. 检查部分 trainable 参数名
# =========================

print("\n===== 前 30 个可训练参数名 =====")
count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape, param.dtype, param.device)
        count += 1
        if count >= 30:
            break

# =========================
# 9. 再次检查 4-bit 模块
# =========================

linear4bit_count_after = 0
for name, module in model.named_modules():
    if isinstance(module, bnb.nn.Linear4bit):
        linear4bit_count_after += 1

print("\nLinear4bit modules after LoRA:", linear4bit_count_after)

if torch.cuda.is_available():
    print("\n===== LoRA 注入后显存 =====")
    print("after LoRA allocated GB:", torch.cuda.memory_allocated() / 1024**3)
    print("after LoRA reserved GB:", torch.cuda.memory_reserved() / 1024**3)
    print("max memory allocated GB:", torch.cuda.max_memory_allocated() / 1024**3)

print("\n===== 完成：7B 4-bit + LoRA 注入成功 =====")
