import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
sft_model_path = "/root/autodl-tmp/llm_ads_project/outputs/day3_sft/final_model"

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
        "name": "跨品类泛化样本：办公椅",
        "input": "商品：人体工学办公椅；卖点：腰部支撑、久坐舒适、可调节；投放场景：职场人群信息流。"
    },
]

system_prompt = "你是一个广告算法助手，擅长根据广告商品信息分析目标受众、核心卖点和点击倾向。"
instruction = "根据广告商品信息，判断目标受众、核心卖点和点击倾向。"


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    return tokenizer, model


def generate_answer(tokenizer, model, ad_input):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"{instruction}\n\n广告信息：{ad_input}"
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
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

    # 只解码新生成部分，避免把 prompt 也打印出来
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer.strip()


def main():
    print("===== 加载 Base Model =====")
    base_tokenizer, base_model = load_model_and_tokenizer(base_model_id)

    print("===== 加载 SFT Model =====")
    sft_tokenizer, sft_model = load_model_and_tokenizer(sft_model_path)

    print("\n\n========== 训练前后对比 ==========\n")

    for case in test_cases:
        print("=" * 80)
        print(f"案例：{case['name']}")
        print(f"广告信息：{case['input']}")
        print("-" * 80)

        base_answer = generate_answer(base_tokenizer, base_model, case["input"])
        sft_answer = generate_answer(sft_tokenizer, sft_model, case["input"])

        print("[Base Model 输出]")
        print(base_answer)

        print("\n[SFT Model 输出]")
        print(sft_answer)

        print("=" * 80)
        print()


if __name__ == "__main__":
    main()
