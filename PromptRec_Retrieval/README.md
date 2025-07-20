
# PromptRec Retrieval Module

This module combines DistilGPT2 + TF-IDF to support LLM-based retrieval-style recommendation using MovieLens 100K.

## ✅ Structure

1. User inputs natural language prompt.
2. GPT2 converts it into interest keywords.
3. TF-IDF retrieves top-K similar movie titles from `u.item`.

## 📦 Usage

1. Download MovieLens 100K and place `u.item` in the root.
2. Install dependencies:
```bash
pip install torch transformers pandas scikit-learn
```

3. Run full pipeline:
```bash
python run_pipeline.py
```
