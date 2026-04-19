---
title: Dialogue Summarizer
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# Dialogue Summarizer — Fine-tuned Phi-3-mini with LoRA

A fine-tuned language model that condenses multi-turn messenger-style conversations into concise summaries. Given a short dialogue between two or more people, the model produces a one-paragraph summary of what was discussed and agreed upon.

**[Live demo on HuggingFace Spaces](https://huggingface.co/spaces/rotemso23/dialogue-summarizer)** | **[Model on HuggingFace Hub](https://huggingface.co/rotemso23/dialogsum-phi3-lora)**

---

## What it does

Paste a conversation like this:

```
Amanda: I baked cookies. Do you want some?
Jerry: Sure! What kind?
Amanda: Chocolate chip. I'll bring them to the office tomorrow.
Jerry: Amazing, I can't wait. Thanks Amanda!
Amanda: No problem :)
```

And get a summary like:

```
Amanda baked chocolate chip cookies and will bring them to the office tomorrow for Jerry.
```

---

## Task and dataset

The model is trained on **[DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum)** — a public dataset of ~14,460 real-world messenger-style conversations paired with human-written summaries. Conversations cover everyday topics: scheduling, shopping, making plans, and more. The informal language, abbreviations, and varied structure make this a realistic and challenging summarization task.

- Train: 12,460 examples (subsampled to 4,000 for efficient fine-tuning)
- Validation: 500 examples
- Test: 819 examples

---

## Why LoRA?

Fine-tuning a 3.8B-parameter model end-to-end requires updating billions of weights — impractical on a single GPU. **LoRA (Low-Rank Adaptation)** solves this by freezing all base model weights and injecting small trainable rank decomposition matrices into the attention layers.

In practice: instead of updating 3.8B parameters, we update only ~8M (about 0.2% of the total). The base model's knowledge is preserved, and the adapter learns the task-specific pattern — in this case, generating concise summaries from conversations.

This is the dominant approach to LLM fine-tuning in industry: faster, cheaper, and produces adapters small enough to store and share alongside any base model.

---

## Tech stack

| Layer | Tool |
|---|---|
| Base model | `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters) |
| Fine-tuning | LoRA via `peft` (`r=16`, `alpha=32`, target: `qkv_proj`, `o_proj`) |
| Training framework | HuggingFace `Trainer` + `transformers` |
| Quantization | 4-bit NF4 (BitsAndBytes) — fits on a free Colab T4 GPU |
| Dataset | `knkarthick/dialogsum` via HuggingFace `datasets` |
| Evaluation | ROUGE-1 / ROUGE-2 / ROUGE-L (`rouge_score`) |
| Experiment tracking | MLflow |
| Model hosting | HuggingFace Hub |
| Demo | Gradio on HuggingFace Spaces |

---

## Training setup

- **Hardware**: Google Colab free tier (NVIDIA T4 GPU, 16 GB VRAM)
- **Epochs**: 3
- **Effective batch size**: 16 (batch size 2 × gradient accumulation 8)
- **Learning rate**: 2e-4
- **Precision**: fp16
- **Best checkpoint**: selected by lowest validation loss

Training took approximately 1.5 hours on a T4 GPU.

The prompt format used throughout training and inference:

```
<|user|>
Summarize the following conversation in a few sentences.

Conversation:
{dialogue}
<|end|>
<|assistant|>
{summary}
<|end|>
```

Loss is computed only on the assistant turn (the summary), not on the instruction or the conversation.

---

## Results

Evaluated on the full DialogSum test set (819 examples).

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Phi-3-mini zero-shot (baseline) | 0.312 | 0.101 | 0.238 |
| Phi-3-mini fine-tuned with LoRA | **0.474** | **0.215** | **0.391** |
| **Improvement** | +52% | +114% | +64% |

The fine-tuned model substantially outperforms the zero-shot baseline across all metrics. ROUGE-2 more than doubles, reflecting that the model learned not just to produce topically relevant text but to match the n-gram structure of human-written summaries.

---

## Qualitative examples

**Example 1**

> **Conversation:**
> Amanda: I baked cookies. Do you want some?
> Jerry: Sure! What kind?
> Amanda: Chocolate chip. I'll bring them to the office tomorrow.
> Jerry: Amazing, I can't wait. Thanks Amanda!
> Amanda: No problem :)
>
> **Reference summary:** Amanda baked chocolate chip cookies and will bring some to the office tomorrow for Jerry.
>
> **Model summary:** Amanda baked chocolate chip cookies and will bring them to the office tomorrow. Jerry is looking forward to it.

---

**Example 2**

> **Conversation:**
> Eric: Could you send me the project report by Friday?
> Maria: Sure, I'll have it done by Thursday so you have time to review.
> Eric: Perfect. Let me know if you need anything from my side.
> Maria: Will do, thanks!
>
> **Reference summary:** Maria will send Eric the project report by Thursday so he has time to review it before Friday.
>
> **Model summary:** Maria will send Eric the project report by Thursday. Eric offered to help if needed.

---

**Example 3**

> **Conversation:**
> Tom: Hey, are we still on for lunch today?
> Linda: Sorry, something came up at work. Can we do tomorrow instead?
> Tom: No problem, same time?
> Linda: Yes, 1pm works. See you then!
>
> **Reference summary:** Linda had to cancel today's lunch with Tom due to work. They rescheduled for tomorrow at 1pm.
>
> **Model summary:** Tom and Linda rescheduled their lunch from today to tomorrow at 1pm because Linda had something come up at work.

---

## How to run locally

**Prerequisites**: Python 3.10+, a CUDA-capable GPU (for training), or CPU (for inference only).

```bash
git clone https://github.com/rotemso23/dialogue-summarizer.git
cd dialogue-summarizer
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your HuggingFace token:

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=your_token_here
```

**Run the demo locally:**

```bash
python app.py
```

**Run evaluation (requires GPU):**

```bash
python src/evaluate.py
```

---

## Project structure

```
dialogue-summarizer/
├── src/
│   ├── data.py        — dataset loading, formatting, tokenization
│   ├── model.py       — base model + LoRA config setup
│   ├── train.py       — Trainer setup, MLflow logging, training loop
│   ├── evaluate.py    — ROUGE evaluation, baseline vs. fine-tuned comparison
│   └── infer.py       — single-conversation summarization (used by app.py)
├── app.py             — Gradio demo UI
├── notebooks/
│   ├── train_colab.ipynb    — Colab training notebook
│   └── evaluate_colab.ipynb — Colab evaluation notebook
└── requirements.txt
```

---

## Author

Rotem Sofer — M.Sc. Biomedical Engineering, Technion (2026)
[rotemso23@gmail.com](mailto:rotemso23@gmail.com)
