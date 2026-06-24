import os
import torch
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# =========================
# 0. 基础配置
# =========================

model_id = "Qwen/Qwen2.5-7B-Instruct"

os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["HF_HUB_CACHE"] = "/root/autodl-tmp/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/huggingface/transformers"
os.environ["HF_HUB_DISABLE_XET"] = "1"

print("===== 环境检查 =====")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("bitsandbytes:", bnb.__version__)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
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
# 3. 4-bit 加载模型
# =========================

print("\n===== 4-bit 加载模型 =====")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

print("\n===== 模型加载完成 =====")
print("model class:", type(model))
print("model device:", next(model.parameters()).device)
print("first param dtype:", next(model.parameters()).dtype)

linear4bit_count = 0
for name, module in model.named_modules():
    if isinstance(module, bnb.nn.Linear4bit):
        linear4bit_count += 1

print("Linear4bit modules:", linear4bit_count)

if torch.cuda.is_available():
    print("after load allocated GB:", torch.cuda.memory_allocated() / 1024**3)
    print("after load reserved GB:", torch.cuda.memory_reserved() / 1024**3)
    print("max memory allocated GB:", torch.cuda.max_memory_allocated() / 1024**3)

# =========================
# 4. 最小推理测试
# =========================

messages = [
    {"role": "system", "content": "你是一个广告算法助手。"},
    {
        "role": "user",
        "content": "商品：低糖白桃气泡水；卖点：低卡、清爽解腻、适合控糖；投放场景：小红书信息流。请判断目标受众、核心卖点和点击倾向。"
    },
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to(model.device)

print("\n===== 开始生成 =====")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("\n===== 生成结果 =====")
print(answer)
