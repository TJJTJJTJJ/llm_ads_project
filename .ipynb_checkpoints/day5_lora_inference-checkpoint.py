import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "/root/autodl-tmp/llm_ads_project/outputs/day5_lora_sft/final_adapter"

test_cases = [
    {
        "name": "训练集内样本：无糖乌龙茶",
        "input": "商品：9.9元包邮无糖乌龙茶；卖点：0糖0脂、解腻、适合控糖；投放场景：小红书信息流。"
    },
    {
        "name": "相似泛化样本：低糖气泡水",
        "input": "商品：低糖白桃气泡水；卖点：低卡、清爽解腻、适合控糖；投放场景：小红书信息流。"
    },
    {
        "name": "跨品类泛化样本：人体工学办公椅",
        "input": "商品：人体工学办公椅；卖点：腰部支撑、久坐舒适、可调节；投放场景：职场人群信息流。"
    },
]

system_prompt = "你是一个广告算法助手，擅长根据广告商品信息分析目标受众、核心卖点和点击倾向。"
instruction = "根据广告商品信息，判断目标受众、核心卖点和点击倾向。"


print("===== 加载 tokenizer =====")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad_token:", tokenizer.pad_token)
print("eos_token:", tokenizer.eos_token)

print("\n===== 加载 base model =====")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("base model loaded")

print("\n===== 加载 LoRA adapter =====")
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)

model.eval()
print("LoRA adapter loaded")
print("model device:", next(model.parameters()).device)
print("model dtype:", next(model.parameters()).dtype)


def generate_answer(ad_input):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"{instruction}\n\n广告信息：{ad_input}",
        }
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
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


print("\n========== LoRA SFT 推理结果 ==========\n")

for case in test_cases:
    print("=" * 80)
    print(f"案例：{case['name']}")
    print(f"广告信息：{case['input']}")
    print("-" * 80)

    answer = generate_answer(case["input"])
    print(answer)

    print("=" * 80)
    print()
