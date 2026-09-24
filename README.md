# SEAL-KG: Neuro-Symbolic Mental Health Dialogue System

A hybrid pipeline combining symbolic Knowledge Graphs with Qwen language models and a Selective Abstention Layer (SEAL) to mitigate hallucinations in mental health query analysis.

## Motivation

Pure generative language models frequently hallucinate or generate unsafe guidance when handling clinical queries. Conversely, purely symbolic knowledge bases cannot naturally understand ambiguous user descriptions.

SEAL-KG combines the reasoning transparency of a symbolic knowledge graph with the natural language understanding of Qwen. When the neural model's prediction diverges from verified knowledge graph relations or confidence falls below safe thresholds, the system explicitly abstains rather than returning an unverified response.

## How It Works

1. **Entity & Symptom Extraction**: Extracts mentions of clinical symptoms and severity markers from user text.
2. **Symbolic Verification**: Queries an RDF knowledge graph for verified entity associations, risk categories, and clinical relations.
3. **Language Model Inference**: Uses Qwen (via Hugging Face Transformers) to generate candidate responses and confidence distributions.
4. **Selective Abstention (SEAL)**: Computes an agreement score between neural predictions and symbolic graph facts. If confidence is insufficient or a conflict is detected, an abstention token (`<|ABSTAIN|>`) is returned with safe referral guidance.

## Setup

```bash
git clone https://github.com/ehtesum/llmHybrid.git
cd llmHybrid
pip install -r requirements.txt
```

## Usage

Run evaluation benchmark:
```bash
python main.py --evaluate
```

Interactive query mode:
```bash
python main.py --interactive
```

Enable Qwen GPU inference:
```bash
python main.py --interactive --use_qwen
```
