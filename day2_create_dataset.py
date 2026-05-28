from datasets import Dataset
from transformers import AutoTokenizer

# 1. 定义模型 ID：这里继续使用 Day1 的 Qwen Instruct 模型
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# 2. 加载 tokenizer
# 说明：
# - 今天不加载模型，只加载 tokenizer
# - 因为我们要用 apply_chat_template 把 messages 转成 SFT 训练文本
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. 构造 10 条模拟广告 SFT 样本
# 每条样本包括：
# - instruction：任务说明
# - input：广告商品信息
# - output：期望模型学会输出的结构化结果
raw_data = [
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：9.9元包邮无糖乌龙茶；卖点：0糖0脂、解腻、适合控糖；投放场景：小红书信息流。",
        "output": "目标受众：学生党、上班族、控糖人群；核心卖点：低价、健康、解腻；点击倾向：高。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：便携式筋膜枪；卖点：小巧便携、缓解肌肉酸痛、续航长；投放场景：抖音短视频信息流。",
        "output": "目标受众：健身人群、久坐上班族、运动爱好者；核心卖点：便携、放松恢复、续航长；点击倾向：高。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：儿童护眼台灯；卖点：无频闪、智能调光、保护视力；投放场景：家长人群定向广告。",
        "output": "目标受众：学生家长、小学生家庭、重视视力健康人群；核心卖点：护眼、安全、智能调光；点击倾向：高。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：男士控油洗面奶；卖点：深层清洁、控油、改善黑头；投放场景：电商搜索广告。",
        "output": "目标受众：年轻男性、油皮人群、护肤入门用户；核心卖点：控油、清洁、改善黑头；点击倾向：中高。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：低卡鸡胸肉即食包；卖点：高蛋白、低脂、开袋即食；投放场景：健身内容相关推荐。",
        "output": "目标受众：健身人群、减脂人群、轻食爱好者；核心卖点：高蛋白、低脂、方便；点击倾向：高。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：通勤双肩电脑包；卖点：大容量、防泼水、独立电脑仓；投放场景：职场人群信息流。",
        "output": "目标受众：上班族、程序员、学生党；核心卖点：大容量、防泼水、电脑保护；点击倾向：中高。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：猫咪自动喂食器；卖点：定时投喂、远程控制、防卡粮；投放场景：宠物内容广告。",
        "output": "目标受众：养猫人群、独居上班族、经常出差用户；核心卖点：自动投喂、远程控制、省心；点击倾向：高。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：新手化妆刷套装；卖点：柔软不扎脸、多刷型组合、适合入门；投放场景：美妆种草广告。",
        "output": "目标受众：学生党、新手化妆用户、美妆爱好者；核心卖点：新手友好、套装齐全、柔软肤感；点击倾向：中高。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：家用小型空气炸锅；卖点：少油烹饪、操作简单、适合小家庭；投放场景：家庭生活内容流。",
        "output": "目标受众：年轻家庭、租房人群、厨房新手；核心卖点：少油健康、易操作、小巧实用；点击倾向：中高。"
    },
    {
        "instruction": "根据广告商品信息，判断目标受众、核心卖点和点击倾向。",
        "input": "商品：降噪蓝牙耳机；卖点：主动降噪、长续航、通勤学习适用；投放场景：数码内容信息流。",
        "output": "目标受众：通勤人群、学生党、数码爱好者；核心卖点：降噪、长续航、沉浸体验；点击倾向：高。"
    },
]

# 4. 把 instruction + input + output 转成多轮对话 messages
# 这是为了贴近 Qwen Instruct 模型的 SFT 格式
def build_messages(example):
    user_content = f"{example['instruction']}\n\n广告信息：{example['input']}"
    assistant_content = example["output"]

    messages = [
        {
            "role": "system",
            "content": "你是一个广告算法助手，擅长根据广告商品信息分析目标受众、核心卖点和点击倾向。"
        },
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": assistant_content
        }
    ]
    return messages

# 5. 使用 tokenizer.apply_chat_template 生成最终训练文本
# 注意：
# - 训练数据里已经包含 assistant 的答案，所以 add_generation_prompt=False
# - 这里生成的 text 就是后续 SFT 可以直接喂给模型的文本
def build_text(example):
    messages = build_messages(example)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {
        "messages": messages,
        "text": text
    }

processed_data = []

for item in raw_data:
    processed_item = dict(item)
    processed_item.update(build_text(item))
    processed_data.append(processed_item)

# 6. 转成 Hugging Face Dataset
dataset = Dataset.from_list(processed_data)

# 7. 打印 Dataset 基本信息
print("===== Dataset 基本信息 =====")
print(dataset)

# 8. 打印第一条样本
print("\n===== 第一条原始样本 =====")
print("instruction:", dataset[0]["instruction"])
print("input:", dataset[0]["input"])
print("output:", dataset[0]["output"])

# 9. 打印第一条 messages
print("\n===== 第一条 messages =====")
print(dataset[0]["messages"])

# 10. 打印第一条最终 SFT text
print("\n===== 第一条 SFT text =====")
print(dataset[0]["text"])

# 11. 保存到本地，供 Day3-Day5 训练使用
save_path = "/root/autodl-tmp/llm_ads_project/data/ads_sft_demo"
dataset.save_to_disk(save_path)

print(f"\n✅ Dataset 已保存到：{save_path}")
