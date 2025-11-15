import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Simple inference script:
# - Load the soft-prompt-tuned T5;
# - Translate a few Chinese sentences into English.
#
# Again you can see why Prompt Tuning fits translation well:
# - We still provide a clear text prefix: "translate Chinese to English: ...";
# - Internally, the learned soft prompt is injected before embeddings;
# - Together they consistently force the model into ZH->EN translation mode.


def main():
    model_name = "t5-small"
    peft_dir = "./t5_prompt_tuning_zh2en"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model = PeftModel.from_pretrained(base_model, peft_dir)
    model.to(device)
    model.eval()

    def translate(sent: str) -> str:
        inp = f"translate Chinese to English: {sent}"
        inputs = tokenizer(inp, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=4,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    examples = [
        "我今天有点累。",
        "这个模型是通过 Prompt Tuning 微调得到的。",
        "请帮我预定明天早上的出租车。",
    ]

    for s in examples:
        print("SRC-ZH:", s)
        print("PRED-EN:", translate(s))
        print("-" * 40)


if __name__ == "__main__":
    main()