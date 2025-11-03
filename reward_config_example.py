import json
from typing import Dict, List

# ==================== 配置方案对比 ====================

# 方案1: 通用医疗问诊（原始版本）
GENERAL_MEDICAL_CONFIG = {
    "name": "通用医疗问诊",
    "criteria_weights": {
        "completeness": 0.3,
        "professionalism": 0.25,
        "safety": 0.25,
        "empathy": 0.2
    },
    "keywords": {
        "inquiry": [
            "是否", "请问", "告诉我", "哪里", "多久", "程度"
        ],
        "medical_terms": [
            "症状", "建议", "检查", "治疗", "诊断", "就医", "医生"
        ],
        "safety_warnings": [
            "建议就医", "医生", "不能替代", "请咨询", "及时就诊"
        ],
        "empathy_words": [
            "您好", "理解", "不要担心", "注意", "请", "关心"
        ]
    }
}


# 方案2: 急诊场景（更注重安全性和紧急性）
EMERGENCY_CONFIG = {
    "name": "急诊场景",
    "criteria_weights": {
        "completeness": 0.2,    # 降低完整性要求
        "professionalism": 0.2,
        "safety": 0.45,         # 大幅提高安全性权重
        "empathy": 0.15
    },
    "keywords": {
        "inquiry": [
            "是否", "有无", "程度", "时间", "位置"
        ],
        "medical_terms": [
            "生命体征", "意识", "呼吸", "心跳", "出血", "休克", "过敏"
        ],
        "safety_warnings": [
            "立即就医", "拨打120", "紧急", "危险", "急诊",
            "不要延误", "刻不容缓", "马上", "救护车"
        ],
        "empathy_words": [
            "保持冷静", "不要慌张", "先稳定", "马上帮助"
        ]
    },
    "special_rules": {
        "urgent_keywords": ["胸痛", "呼吸困难", "大出血", "昏迷", "抽搐"],
        "urgent_keyword_bonus": 0.3  # 出现紧急关键词额外加分
    }
}


# 方案3: 儿科场景（更注重家长沟通）
PEDIATRIC_CONFIG = {
    "name": "儿科问诊",
    "criteria_weights": {
        "completeness": 0.35,   # 需要详细了解儿童情况
        "professionalism": 0.2,
        "safety": 0.25,
        "empathy": 0.2
    },
    "keywords": {
        "inquiry": [
            "孩子年龄", "体重", "精神状态", "饮食", "大小便", 
            "睡眠", "疫苗接种", "生长发育"
        ],
        "medical_terms": [
            "儿童", "婴幼儿", "发育", "生长曲线", "疫苗", "喂养"
        ],
        "safety_warnings": [
            "儿科就诊", "儿童医院", "不要自行用药", "遵医嘱", "观察变化"
        ],
        "empathy_words": [
            "理解您的担心", "不要太着急", "孩子会好的", 
            "家长注意", "耐心观察", "您做得很好"
        ]
    }
}


# 方案4: 慢病管理（糖尿病、高血压等）
CHRONIC_DISEASE_CONFIG = {
    "name": "慢病管理",
    "criteria_weights": {
        "completeness": 0.3,
        "professionalism": 0.35,  # 更强调专业指导
        "safety": 0.2,
        "empathy": 0.15
    },
    "keywords": {
        "inquiry": [
            "血糖", "血压", "用药依从性", "饮食控制", "运动情况",
            "监测频率", "并发症", "复查时间"
        ],
        "medical_terms": [
            "血糖控制", "糖化血红蛋白", "血压达标", "生活方式干预",
            "药物调整", "并发症筛查", "定期复查"
        ],
        "safety_warnings": [
            "遵医嘱用药", "定期监测", "不要擅自停药", "及时复诊"
        ],
        "empathy_words": [
            "坚持很重要", "您做得不错", "继续努力", 
            "慢病需要长期管理", "我们一起努力"
        ]
    }
}


# 方案5: 心理咨询场景
MENTAL_HEALTH_CONFIG = {
    "name": "心理咨询",
    "criteria_weights": {
        "completeness": 0.25,
        "professionalism": 0.25,
        "safety": 0.2,
        "empathy": 0.3         # 大幅提高同理心权重
    },
    "keywords": {
        "inquiry": [
            "感受", "情绪", "想法", "困扰", "持续时间",
            "影响程度", "应对方式", "支持系统"
        ],
        "medical_terms": [
            "情绪", "压力", "焦虑", "抑郁", "心理咨询", "心理评估"
        ],
        "safety_warnings": [
            "心理咨询师", "专业心理治疗", "心理科就诊",
            "危机干预", "自杀风险评估"
        ],
        "empathy_words": [
            "我理解", "感同身受", "您很勇敢", "愿意倾听",
            "您并不孤单", "这很不容易", "给自己一些时间",
            "您的感受是真实的", "我们一起面对"
        ]
    },
    "special_rules": {
        "crisis_keywords": ["想死", "自杀", "活不下去", "结束生命"],
        "crisis_response_required": True
    }
}


# ==================== 使用示例 ====================

def load_config(config_dict: Dict) -> Dict:
    """加载配置"""
    print(f"✓ 加载配置: {config_dict['name']}")
    print(f"  权重分配: {config_dict['criteria_weights']}")
    print(f"  关键词类别: {len(config_dict['keywords'])} 类")
    return config_dict


def customize_keywords():
    """自定义关键词示例"""
    
    # 示例1: 从现有配置修改
    my_config = GENERAL_MEDICAL_CONFIG.copy()
    my_config["name"] = "我的自定义配置"
    
    # 添加更多询问关键词
    my_config["keywords"]["inquiry"].extend([
        "什么原因", "如何引起", "能否", "可不可以"
    ])
    
    # 调整权重
    my_config["criteria_weights"]["safety"] = 0.35  # 更重视安全性
    my_config["criteria_weights"]["empathy"] = 0.15
    
    return my_config


def save_config_to_file(config: Dict, filename: str):
    """保存配置到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"✓ 配置已保存到: {filename}")


def load_config_from_file(filename: str) -> Dict:
    """从JSON文件加载配置"""
    with open(filename, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"✓ 从文件加载配置: {filename}")
    return config


# ==================== 动态关键词生成 ====================

def generate_keywords_from_domain(domain: str) -> List[str]:
    """
    根据医疗领域自动生成关键词
    实际应用中可以从医学知识图谱或词典中提取
    """
    domain_keywords = {
        "心血管": ["胸痛", "心悸", "气短", "血压", "心电图", "冠心病"],
        "呼吸系统": ["咳嗽", "咳痰", "呼吸困难", "胸闷", "肺部", "支气管"],
        "消化系统": ["腹痛", "恶心", "呕吐", "腹泻", "便血", "胃镜"],
        "神经系统": ["头痛", "头晕", "麻木", "无力", "意识", "CT", "MRI"],
    }
    return domain_keywords.get(domain, [])


def build_dynamic_config(domains: List[str]) -> Dict:
    """
    根据多个医疗领域动态构建配置
    """
    config = {
        "name": f"动态配置-{'+'.join(domains)}",
        "criteria_weights": GENERAL_MEDICAL_CONFIG["criteria_weights"].copy(),
        "keywords": {
            "inquiry": GENERAL_MEDICAL_CONFIG["keywords"]["inquiry"].copy(),
            "medical_terms": [],
            "safety_warnings": GENERAL_MEDICAL_CONFIG["keywords"]["safety_warnings"].copy(),
            "empathy_words": GENERAL_MEDICAL_CONFIG["keywords"]["empathy_words"].copy(),
        }
    }
    
    # 为每个领域添加专业术语
    for domain in domains:
        config["keywords"]["medical_terms"].extend(
            generate_keywords_from_domain(domain)
        )
    
    return config


# ==================== 主程序示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("奖励模型配置示例")
    print("=" * 60)
    
    # 1. 展示不同场景的配置
    print("\n【场景1】通用医疗问诊")
    print(json.dumps(GENERAL_MEDICAL_CONFIG, ensure_ascii=False, indent=2))
    
    print("\n【场景2】急诊场景（注意安全性权重提高到45%）")
    print(json.dumps(EMERGENCY_CONFIG["criteria_weights"], 
                     ensure_ascii=False, indent=2))
    
    print("\n【场景3】儿科场景（注意儿童特定的询问关键词）")
    print("儿科询问关键词:", PEDIATRIC_CONFIG["keywords"]["inquiry"])
    
    print("\n【场景4】心理咨询（同理心权重30%）")
    print("同理心词汇:", MENTAL_HEALTH_CONFIG["keywords"]["empathy_words"])
    
    # 2. 自定义配置示例
    print("\n" + "=" * 60)
    print("自定义配置示例")
    print("=" * 60)
    
    custom_config = customize_keywords()
    print(f"自定义配置名称: {custom_config['name']}")
    print(f"修改后的询问关键词数量: {len(custom_config['keywords']['inquiry'])}")
    
    # 3. 保存和加载配置
    print("\n" + "=" * 60)
    print("保存和加载配置")
    print("=" * 60)
    
    save_config_to_file(EMERGENCY_CONFIG, "emergency_config.json")
    # loaded = load_config_from_file("emergency_config.json")
    
    # 4. 动态生成配置
    print("\n" + "=" * 60)
    print("动态配置生成")
    print("=" * 60)
    
    dynamic_config = build_dynamic_config(["心血管", "呼吸系统"])
    print(f"动态生成的配置: {dynamic_config['name']}")
    print(f"医学术语: {dynamic_config['keywords']['medical_terms']}")
    
    # 5. 实际使用建议
    print("\n" + "=" * 60)
    print("使用建议")
    print("=" * 60)
    print("""
1. 根据实际业务场景选择或自定义配置
2. 关键词需要覆盖：
   - 询问信息的引导词（"是否"、"请问"等）
   - 专业医学术语（根据科室定制）
   - 安全免责声明（法律合规要求）
   - 同理心表达（提升用户体验）

3. 权重调整原则：
   - 急诊场景：提高安全性权重
   - 慢病管理：提高专业性权重
   - 心理咨询：提高同理心权重
   - 儿科问诊：提高完整性权重

4. 迭代优化：
   - 收集真实问诊数据
   - 人工标注质量分数
   - 分析高分/低分回复的特征
   - 更新关键词列表
   - 调整权重配置

5. 进阶方案：
   - 使用sentence-transformers做语义匹配
   - 训练专门的BERT分类器
   - 使用GPT-4作为评判模型
   - 结合人工反馈持续优化
    """)
