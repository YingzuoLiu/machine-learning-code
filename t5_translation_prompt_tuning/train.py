import os
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import PromptTuningConfig, TaskType, get_peft_model

# This script shows how to use T5 + Prompt Tuning for Chinese -> English translation.
# -------------------------------------------------------------------------
# Core idea:
# 1. T5 is an Encoder-Decoder model:
#    - Encoder: encodes the source sentence (Chinese) into a semantic representation.
#    - Decoder: generates the target sentence (English) from that representation.
# 2. Translation is a classic sequence-to-sequence task: "understand + generate".
# 3. Prompt Tuning:
#    - We freeze almost all parameters of T5 (encoder & decoder);
#    - We insert a small trainable "soft prompt" in front of the input embeddings;
#    - We only train this soft prompt so that T5 learns:
#        "Whenever I see this soft prefix, I should treat the task as ZH->EN translation".
#
# Why this is a good fit for translation:
# - Very lightweight: only a tiny number of parameters are trained (good for limited GPU);
# - Stable: we do not destroy T5's pre-trained multi-lingual knowledge;
# - Natural for seq2seq: we only need to steer the model into the correct "task mode".


# ===== 1. Build a tiny toy dataset (replace with your real parallel data) =====
def build_toy_dataset():
    """
    In practice you should use a large Chinese-English parallel corpus.
    Here we only use a few hand-crafted examples as a toy demo.
    Each example has:
    - src: source Chinese sentence
    - tgt: target English translation
    """
    train_pairs = [
        {"src": "今天天气很好。", "tgt": "The weather is nice today."},
        {"src": "我喜欢机器学习。", "tgt": "I like machine learning."},
        {"src": "这本书非常有趣。", "tgt": "This book is very interesting."},
        {"src": "他正在学英语。", "tgt": "He is learning English."},
        {"src": "请给我一杯咖啡。", "tgt": "Please give me a cup of coffee."},
        {"src": "我明天要去上班。", "tgt": "I need to go to work tomorrow."},
        {"src": "你来自哪个国家？", "tgt": "Which country are you from?"},
        {"src": "这道题有点难。", "tgt": "This question is a bit difficult."},
        {"src": "我在看一部电影。", "tgt": "I am watching a movie."},
        {"src": "谢谢你的帮助。", "tgt": "Thank you for your help."},
    ]

    eval_pairs = [
        {"src": "我正在学习深度学习。", "tgt": "I am studying deep learning."},
        {"src": "这家餐厅的食物很好吃。", "tgt": "The food in this restaurant is delicious."},
    ]

    train_ds = Dataset.from_list(train_pairs)
    eval_ds = Dataset.from_list(eval_pairs)
    return train_ds, eval_ds, eval_pairs


# ===== 2. Preprocess: add T5 text prefix + tokenize =====
def preprocess(tokenizer, train_ds, eval_ds):
    """
    T5 is trained in a text-to-text style and usually receives an explicit task prefix like:
      - "translate English to German: ..."
      - "summarize: ..."
    Here we use:
      - "translate Chinese to English: {src}"

    On top of this *text* prompt, Prompt Tuning will add a *soft* prompt in the embedding space.
    Text prompt + soft prompt work together to guide the model into translation mode.
    """
    max_source_length = 64
    max_target_length = 64

    def _preprocess_batch(batch):
        inputs = [f"translate Chinese to English: {s}" for s in batch["src"]]
        targets = batch["tgt"]

        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            padding="max_length",
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                padding="max_length",
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(_preprocess_batch, batched=True)
    tokenized_eval = eval_ds.map(_preprocess_batch, batched=True)
    return tokenized_train, tokenized_eval


# ===== 3. Simple eval: print model translations =====
def simple_eval(model, tokenizer, eval_pairs, device):
    model.eval()
    for ex in eval_pairs:
        src = ex["src"]
        tgt = ex["tgt"]
        inp = f"translate Chinese to English: {src}"
        inputs = tokenizer(inp, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=4,
            )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("SRC-ZH:", src)
        print("TGT-EN:", tgt)
        print("PRED  :", pred)
        print("-" * 40)


@dataclass
class DataCollatorForSeq2SeqWithPadding:
    tokenizer: AutoTokenizer
    model: AutoModelForSeq2SeqLM

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding="longest",
        )
        return collator(features)


def main():
    model_name = "t5-small"
    output_dir = "./t5_prompt_tuning_zh2en"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1) Load base T5 and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Why Prompt Tuning here?
    # - We keep encoder/decoder weights frozen so T5's general knowledge remains intact.
    # - We only add a small trainable prompt (num_virtual_tokens) at the input embedding level.
    # - For translation, this is usually enough to adapt to your domain/style without heavy cost.

    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=20,          # length of soft prompt (you can tune this)
        tokenizer_name_or_path=model_name,
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()  # should show only prompt-related params

    model.to(device)

    # 2) Data
    train_ds, eval_ds, eval_pairs = build_toy_dataset()
    tokenized_train, tokenized_eval = preprocess(tokenizer, train_ds, eval_ds)
    data_collator = DataCollatorForSeq2SeqWithPadding(tokenizer=tokenizer, model=model)

    # 3) Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=30,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-3,          # can be larger than full fine-tuning
        weight_decay=0.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    print("Start Prompt Tuning (only soft prompt is trainable)...")
    trainer.train()

    trainer.save_model(output_dir)
    print("Model (PEFT prompt) saved to:", output_dir)

    print("\n==== Eval on small dev set ====\n")
    simple_eval(model, tokenizer, eval_pairs, device)

    # Some extra test sentences
    test_sents = [
        "我喜欢看电影。",
        "这个示例主要是用来演示 Prompt Tuning。",
        "今天下雨了，我还是去上班。",
    ]
    model.eval()
    for s in test_sents:
        inp = f"translate Chinese to English: {s}"
        inputs = tokenizer(inp, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=4,
            )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("SRC-ZH:", s)
        print("PRED-EN:", pred)
        print("-" * 40)


if __name__ == "__main__":
    main()