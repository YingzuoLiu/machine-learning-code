
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from config import config
import torch

tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"])
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(config["model_name"])
model.eval()

device = torch.device("cuda" if config["use_cuda"] and torch.cuda.is_available() else "cpu")
model = model.to(device)

def extract_keywords(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            do_sample=True,
            top_k=30,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "I love Interstellar and Inception. Recommend something."
    print("Prompt:", prompt)
    print("Extracted Interest:", extract_keywords(prompt))
