import os
import torch
import bitsandbytes as bnb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel

# =========================
# 0. 基础配置
# =========================

model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_dir = "/root/autodl-tmp/llm_ads_project/outputs/day9_qwen7b_qlora_sft/final_adapter"

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

print("\n===== adapter 路径检查 =====")
print("adapter_dir:", adapter_dir)
print("files:", os.listdir(adapter_dir))

# =========================
# 1. 加载 tokenizer
# =========================

print("\n===== 加载 tokenizer =====")
tokenizer = AutoTokenizer.from_pretrained(
    adapter_dir,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad_token:", tokenizer.pad_token)
print("eos_token:", tokenizer.eos_token)

# =========================
# 2. 4-bit 加载 base model
# =========================

print("\n===== 4-bit 加载 base model =====")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # 3090 推理也可以用 float16，更稳定
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

base_model.eval()

print("base model loaded")
print("base model class:", type(base_model))
print("base model device:", next(base_model.parameters()).device)
print("base first param dtype:", next(base_model.parameters()).dtype)

linear4bit_count = 0
for name, module in base_model.named_modules():
    if isinstance(module, bnb.nn.Linear4bit):
        linear4bit_count += 1

print("Linear4bit modules:", linear4bit_count)

if torch.cuda.is_available():
    print("after base load allocated GB:", torch.cuda.memory_allocated() / 1024**3)
    print("after base load reserved GB:", torch.cuda.memory_reserved() / 1024**3)

# =========================
# 3. 加载 LoRA adapter
# =========================

print("\n===== 加载 LoRA adapter =====")

model = PeftModel.from_pretrained(
    base_model,
    adapter_dir,
)

model.eval()

print("adapter loaded")
print("model class:", type(model))
print("model device:", next(model.parameters()).device)

if torch.cuda.is_available():
    print("after adapter load allocated GB:", torch.cuda.memory_allocated() / 1024**3)
    print("after adapter load reserved GB:", torch.cuda.memory_reserved() / 1024**3)
    print("max memory allocated GB:", torch.cuda.max_memory_allocated() / 1024**3)

# =========================
# 4. 推理函数
# =========================

def generate_answer(instruction, item_info, max_new_tokens=180):
    system_message = "你是一个广告算法助手，擅长根据商品信息分析目标受众、核心卖点和点击倾向。"

    user_content = f"{instruction}\n\n{item_info}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return answer.strip()

# =========================
# 5. 测试样本
# =========================

test_cases = [
    {
        "name": "训练内相似样本：低糖饮料",
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "item_info": "商品：无糖青柠气泡水；卖点：0糖0脂、清爽解腻、适合控糖；投放场景：小红书信息流。",
    },
    {
        "name": "训练内相似样本：办公椅",
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "item_info": "商品：人体工学电脑椅；卖点：腰托支撑、久坐不累、可调节头枕；投放场景：职场社区。",
    },
    {
        "name": "新样本：跑步鞋",
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "item_info": "商品：轻量缓震跑步鞋；卖点：透气网面、缓震回弹、适合日常慢跑；投放场景：运动健身 App。",
    },
]

print("\n===== 开始 adapter 推理 =====")

for case in test_cases:
    print("\n" + "=" * 80)
    print("CASE:", case["name"])
    print("INPUT:", case["item_info"])
    print("-" * 80)

    answer = generate_answer(
        case["instruction"],
        case["item_info"],
    )

    print(answer)

if torch.cuda.is_available():
    print("\n===== 推理结束显存 =====")
    print("allocated GB:", torch.cuda.memory_allocated() / 1024**3)
    print("reserved GB:", torch.cuda.memory_reserved() / 1024**3)
    print("max allocated GB:", torch.cuda.max_memory_allocated() / 1024**3)

print("\n===== 完成：final_adapter 推理验证 =====")
