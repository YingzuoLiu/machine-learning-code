from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载 Mistral-7B-Instruct 模型
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

# 初始化上下文记忆列表
context_history = []

# 添加用户输入偏好
user_input = "I like suspenseful movies with tight plots, preferably released in recent years."
context_history.append(f"User: {user_input}")

# 构造初始 prompt
def build_prompt(context):
    return "\n".join(context + ["Assistant: Recommend 5 movies and provide reasons.\n1."])

# 推理函数
def generate_recommendation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 第1轮推荐
prompt_1 = build_prompt(context_history)
response_1 = generate_recommendation(prompt_1)
context_history.append("Assistant: " + response_1.split("Assistant:")[-1].strip())

print("\u2728 Round 1 Recommendation:")
print(response_1)

# 模拟用户反馈
user_feedback = "I've already watched some of those. Can you recommend lesser-known titles with similar suspense?"
context_history.append(f"User: {user_feedback}")

# 第2轮推荐
prompt_2 = build_prompt(context_history)
response_2 = generate_recommendation(prompt_2)
context_history.append("Assistant: " + response_2.split("Assistant:")[-1].strip())

print("\n🔄 Round 2 Recommendation:")
print(response_2)

# 你可以继续添加更多轮次的用户输入并调用相同逻辑进行多轮生成
