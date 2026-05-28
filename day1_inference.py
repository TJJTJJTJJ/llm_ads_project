import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import datasets
print(torch.__version__)
print(transformers.__version__)
print(datasets.__version__)
print(torch.cuda.is_available())


print("1. 正在加载 Qwen2.5-0.5B-Instruct...")
model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
# model_id = "/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.float16, 
    device_map="auto"
)

print("2. 构造 Prompt...")
prompt = "你是一个资深广告优化师。请为一款'9.9元包邮的无糖乌龙茶'写一句吸引年轻人的小红书广告标题。"

# 套用模型的 Chat Template
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("text: ", text)

print("3. 开始推理生成...")
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=50)

# 截取并解码输出
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n🎉 模型输出结果：")
print(response)
