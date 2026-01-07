import numpy as np

def should_fallback(cos_scores):
    if not cos_scores:
        return True

    mean_cos = np.mean(cos_scores)
    top1 = cos_scores[0]
    gap = top1 - mean_cos

    # 1. 整体相似度太低
    if mean_cos < 0.25:
        return True

    # 2. 区分度不足
    if gap < 0.05:
        return True

    return False
