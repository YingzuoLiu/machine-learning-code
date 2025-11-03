"""
基于GRPO后训练的智能医疗问诊系统
使用Phi-3-mini模型 + LoRA微调
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset
import numpy as np
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass
from torch.utils.data import DataLoader

# ==================== 配置参数 ====================
@dataclass
class GRPOConfig:
    """GRPO训练配置"""
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 512
    temperature: float = 0.7
    kl_coef: float = 0.1  # KL散度系数
    group_size: int = 4  # GRPO组大小
    reward_baseline: str = "group_mean"  # 奖励基线类型


# ==================== 医疗问诊数据集 ====================
class MedicalQADataset:
    """医疗问诊数据集"""
    
    @staticmethod
    def get_sample_data():
        """获取示例医疗问诊数据"""
        return [
            {
                "patient_query": "我最近总是头痛，特别是早上起床的时候，持续了一周了",
                "doctor_response": "您好，根据您描述的症状，我需要了解更多信息。请问：1. 头痛的具体位置是哪里？2. 疼痛程度如何（1-10分）？3. 是否伴有恶心、呕吐？4. 最近是否有睡眠不足或压力过大的情况？5. 是否有发热？",
                "quality_score": 0.9
            },
            {
                "patient_query": "我咳嗽三天了，有黄痰，体温37.8度",
                "doctor_response": "根据您的症状，可能是上呼吸道感染。建议：1. 多喝温水，注意休息；2. 可以服用止咳化痰药物；3. 如果症状持续加重或高烧不退，建议及时就医做进一步检查。同时请告诉我：是否有胸痛或呼吸困难？",
                "quality_score": 0.85
            },
            {
                "patient_query": "肚子疼了两天，拉肚子，吃什么药好？",
                "doctor_response": "您的症状可能是急性肠胃炎。在用药前，我需要确认：1. 疼痛的具体位置？2. 腹泻频率和性状？3. 是否有发热、呕吐？4. 最近饮食情况如何？请注意，我不能直接开药，建议您先到医院就诊，医生会根据具体情况开具处方。",
                "quality_score": 0.88
            },
            {
                "patient_query": "我血压150/95，需要吃药吗？",
                "doctor_response": "您的血压值属于高血压范围（正常值<120/80）。是否需要用药取决于多个因素：1. 年龄和既往病史；2. 是否持续性高血压；3. 是否有其他心血管危险因素。建议：先进行生活方式干预（低盐饮食、适量运动、控制体重），并监测血压一段时间。请尽快到心内科就诊，医生会评估是否需要药物治疗。",
                "quality_score": 0.92
            },
            {
                "patient_query": "我手臂骨折了怎么办",
                "doctor_response": "骨折是紧急情况！请立即采取以下措施：1. 不要移动受伤部位；2. 用硬板或杂志固定患肢；3. 如果有出血，用干净纱布压迫止血；4. 立即前往医院急诊科就诊或拨打120急救。请不要尝试自行复位，以免造成二次伤害。",
                "quality_score": 0.95
            },
            {
                "patient_query": "感冒了，可以吃阿莫西林吗？",
                "doctor_response": "普通感冒多为病毒感染，阿莫西林是抗生素，对病毒无效。不建议自行使用。建议：1. 多休息、多喝水；2. 如有发热可用退烧药；3. 如果症状持续或加重（如高热超过3天、咳嗽加重、呼吸困难），需就医检查是否合并细菌感染，由医生决定是否使用抗生素。",
                "quality_score": 0.90
            }
        ]


# ==================== GRPO奖励模型 ====================
class MedicalRewardModel:
    """医疗问诊奖励模型"""
    
    def __init__(self):
        self.criteria = {
            "completeness": 0.3,      # 信息完整性
            "professionalism": 0.25,  # 专业性
            "safety": 0.25,           # 安全性
            "empathy": 0.2            # 同理心
        }
    
    def calculate_reward(self, query: str, response: str) -> float:
        """
        计算奖励分数
        实际应用中可以训练一个专门的奖励模型
        """
        score = 0.0
        
        # 1. 信息完整性：是否询问关键信息
        key_questions = ["是否", "请问", "告诉我", "哪里", "多久", "程度"]
        completeness = sum(1 for kw in key_questions if kw in response) / len(key_questions)
        score += completeness * self.criteria["completeness"]
        
        # 2. 专业性：是否包含医学术语和建议
        medical_terms = ["症状", "建议", "检查", "治疗", "诊断", "就医", "医生"]
        professionalism = min(sum(1 for term in medical_terms if term in response) / 3, 1.0)
        score += professionalism * self.criteria["professionalism"]
        
        # 3. 安全性：是否有免责说明
        safety_phrases = ["建议就医", "医生", "不能替代", "请咨询", "及时就诊"]
        safety = min(sum(1 for phrase in safety_phrases if phrase in response) / 2, 1.0)
        score += safety * self.criteria["safety"]
        
        # 4. 同理心：是否有关怀性语言
        empathy_words = ["您好", "理解", "不要担心", "注意", "请", "关心"]
        empathy = min(sum(1 for word in empathy_words if word in response) / 2, 1.0)
        score += empathy * self.criteria["empathy"]
        
        # 惩罚过短或过长的回复
        response_length = len(response)
        if response_length < 50:
            score *= 0.7
        elif response_length > 500:
            score *= 0.9
            
        return score


# ==================== GRPO训练器 ====================
class GRPOTrainer:
    """Group Relative Policy Optimization 训练器"""
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载分词器
        print(f"加载分词器: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        print(f"加载模型: {config.model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],  # Phi-3特定
            task_type=TaskType.CAUSAL_LM,
            bias="none"
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()
        
        # 保存参考模型（用于KL散度计算）
        self.ref_model = self.base_model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 奖励模型
        self.reward_model = MedicalRewardModel()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
    
    def format_prompt(self, query: str) -> str:
        """格式化医疗问诊提示"""
        return f"""<|system|>
你是一位专业的医疗助手。请根据患者的症状描述，提供初步的医疗建议。注意：
1. 询问关键信息以更好地理解病情
2. 给出专业的医疗建议
3. 提醒患者及时就医，不能替代医生诊断
4. 语言要专业且有同理心
<|end|>
<|user|>
患者问题：{query}
<|end|>
<|assistant|>
"""
    
    def generate_responses(self, queries: List[str], num_samples: int = 4) -> List[List[str]]:
        """
        为每个查询生成多个响应样本（用于GRPO组比较）
        """
        all_responses = []
        
        for query in queries:
            prompt = self.format_prompt(query)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成多个样本
            responses = []
            for _ in range(num_samples):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=self.config.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                responses.append(response.strip())
            
            all_responses.append(responses)
        
        return all_responses
    
    def compute_kl_divergence(self, query: str, response: str) -> float:
        """计算与参考模型的KL散度"""
        prompt = self.format_prompt(query) + response
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 当前模型的logits
            current_logits = self.model(**inputs).logits
            # 参考模型的logits
            ref_logits = self.ref_model(**inputs).logits
            
            # 计算KL散度
            kl_div = F.kl_div(
                F.log_softmax(current_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction='batchmean'
            )
        
        return kl_div.item()
    
    def grpo_loss(self, query: str, responses: List[str], rewards: List[float]) -> torch.Tensor:
        """
        计算GRPO损失
        GRPO使用组内相对奖励，减少方差
        """
        # 计算组平均奖励作为基线
        baseline = np.mean(rewards)
        advantages = [r - baseline for r in rewards]
        
        # 归一化优势
        if np.std(advantages) > 0:
            advantages = [(a - np.mean(advantages)) / (np.std(advantages) + 1e-8) 
                         for a in advantages]
        
        total_loss = 0
        for response, advantage in zip(responses, advantages):
            # 计算对数概率
            prompt = self.format_prompt(query) + response
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            log_prob = -outputs.loss
            
            # GRPO目标：最大化加权对数概率
            policy_loss = -log_prob * advantage
            
            # 添加KL惩罚
            kl_penalty = self.compute_kl_divergence(query, response)
            
            total_loss += policy_loss + self.config.kl_coef * kl_penalty
        
        return total_loss / len(responses)
    
    def train_step(self, batch_queries: List[str]) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 为每个查询生成多个响应
        all_responses = self.generate_responses(
            batch_queries,
            num_samples=self.config.group_size
        )
        
        # 计算奖励
        all_rewards = []
        for query, responses in zip(batch_queries, all_responses):
            rewards = [self.reward_model.calculate_reward(query, resp) 
                      for resp in responses]
            all_rewards.append(rewards)
        
        # 计算GRPO损失
        total_loss = 0
        for query, responses, rewards in zip(batch_queries, all_responses, all_rewards):
            loss = self.grpo_loss(query, responses, rewards)
            total_loss += loss
        
        avg_loss = total_loss / len(batch_queries)
        
        # 反向传播
        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            "loss": avg_loss.item(),
            "avg_reward": np.mean([np.mean(r) for r in all_rewards])
        }
    
    def train(self, train_data: List[Dict]):
        """完整训练流程"""
        print("\n开始GRPO训练...")
        
        queries = [item["patient_query"] for item in train_data]
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # 批次训练
            total_loss = 0
            total_reward = 0
            num_batches = 0
            
            for i in range(0, len(queries), self.config.batch_size):
                batch_queries = queries[i:i + self.config.batch_size]
                
                metrics = self.train_step(batch_queries)
                total_loss += metrics["loss"]
                total_reward += metrics["avg_reward"]
                num_batches += 1
                
                if num_batches % 5 == 0:
                    print(f"Batch {num_batches}: Loss={metrics['loss']:.4f}, "
                          f"Reward={metrics['avg_reward']:.4f}")
            
            avg_loss = total_loss / num_batches
            avg_reward = total_reward / num_batches
            print(f"Epoch {epoch + 1} 平均: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}")
    
    def save_model(self, output_dir: str):
        """保存模型"""
        print(f"\n保存模型到: {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def inference(self, query: str) -> str:
        """推理生成回复"""
        self.model.eval()
        prompt = self.format_prompt(query)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


# ==================== 主程序 ====================
def main():
    """主函数"""
    print("="*60)
    print("基于GRPO的智能医疗问诊系统")
    print("模型: Phi-3-mini + LoRA")
    print("="*60)
    
    # 1. 初始化配置
    config = GRPOConfig()
    
    # 2. 准备数据
    dataset = MedicalQADataset()
    train_data = dataset.get_sample_data()
    print(f"\n训练数据量: {len(train_data)} 条")
    
    # 3. 初始化训练器
    trainer = GRPOTrainer(config)
    
    # 4. 训练模型
    trainer.train(train_data)
    
    # 5. 保存模型
    trainer.save_model("./medical_chatbot_grpo_lora")
    
    # 6. 测试推理
    print("\n" + "="*60)
    print("测试推理效果")
    print("="*60)
    
    test_queries = [
        "我最近总是失眠，睡不着觉",
        "孩子发烧38.5度，怎么办？",
        "胸口疼痛，呼吸困难"
    ]
    
    for query in test_queries:
        print(f"\n患者: {query}")
        response = trainer.inference(query)
        print(f"医生: {response}")
        print("-" * 60)


if __name__ == "__main__":
    # 注意：实际运行需要GPU和足够的内存
    # 这是完整的实现代码，包含了GRPO算法的核心逻辑
    main()
