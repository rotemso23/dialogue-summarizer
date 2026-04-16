# CLAUDE.md — Dialogue Summarization LLM Fine-Tuning (Portfolio Project B)

## What this project is

A fine-tuned small language model for dialogue summarization, trained on [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) — a public dataset of messenger-style conversations paired with human-written summaries. The task: given a short multi-turn conversation, generate a concise summary of what was discussed and agreed upon.

The model is fine-tuned with LoRA (via PEFT) on top of a small base model (Phi-3-mini), tracked with MLflow, and published to HuggingFace Hub. A lightweight Gradio demo is deployed on HuggingFace Spaces — anyone can paste a conversation and get a summary instantly.

This is a CV portfolio project. The deliverable is a working fine-tuned model on HuggingFace Hub + a clean GitHub repo + a live demo.

## Motivation / context

Rotem is a fresh M.Sc. graduate (Biomedical Engineering, Technion, 2026) seeking AI/ML roles in Israel. After Project A (RAG pipeline), the main remaining gaps are: LLM fine-tuning, LoRA/PEFT, the HuggingFace Trainer/Transformers ecosystem, experiment tracking, and model publishing. This project closes all of them.

Rotem already understands training loops, loss functions, and gradient-based optimization from the M.Sc. thesis. This project is mostly the HuggingFace + LoRA API layer on top of that existing foundation.

The DialogSum task was chosen because:
- The demo is immediately understandable to any recruiter or hiring manager — no domain knowledge required
- It is a generation task (not just classification), which shows the model learned to produce fluent text
- The use case is directly relevant to real products (Zoom, Slack, Teams, WhatsApp all have or want auto-summarization)
- ROUGE metrics give a clean quantitative before/after story

## Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Base model | `microsoft/Phi-3-mini-4k-instruct` (3.8B) | Small enough for free Colab T4, strong enough to show real fine-tuning gains |
| Fine-tuning method | LoRA via `peft` | Industry standard for parameter-efficient fine-tuning; appears in every JD |
| Training | HuggingFace `transformers` + `Trainer` API | Closes HuggingFace Trainer gap directly |
| Dataset | DialogSum (`knkarthick/dialogsum` on HF Hub) | Public, practical, instant demo appeal |
| Evaluation | ROUGE-1 / ROUGE-2 / ROUGE-L | Standard summarization metrics; easy before/after comparison |
| Experiment tracking | MLflow | Closes experiment tracking gap; widely required |
| Model hosting | HuggingFace Hub | Free, recognized, shows end-to-end ownership |
| Demo UI | Gradio | Native to HuggingFace Spaces; ideal for text-in / text-out demos |
| Deployment | HuggingFace Spaces | Free, same ecosystem as Hub |

## Project structure

```
dialogue-summarizer/
├── CLAUDE.md               ← this file
├── README.md               ← write last, CV-quality
├── requirements.txt
├── .env.example            ← HF_TOKEN placeholder (never commit real tokens)
├── .gitignore
├── src/
│   ├── data.py             ← dataset loading, formatting, tokenization
│   ├── model.py            ← base model + LoRA config setup
│   ├── train.py            ← Trainer setup, MLflow logging, training loop
│   ├── evaluate.py         ← ROUGE scores on test split, baseline vs. fine-tuned
│   └── infer.py            ← single-conversation summarization function (used by app.py)
├── app.py                  ← Gradio demo UI (entry point for HF Spaces)
├── mlflow_runs/            ← local MLflow artifact store (in .gitignore)
└── notebooks/
    └── exploration.ipynb   ← optional: loss curves, ROUGE breakdown, example outputs
```

## Dataset: DialogSum

- Load via: `datasets.load_dataset("knkarthick/dialogsum")`
- ~14,460 examples total: pre-split into train (12,460) / validation (500) / test (1,500)
- Each example has: `dialogue` (multi-turn conversation) and `summary` (one short paragraph)
- Conversations are messenger-style: informal language, abbreviations, emojis — realistic and varied
- No custom splitting needed — use the dataset's built-in splits

## Prompt format

Use a consistent instruction format throughout training and inference:

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

Use the Phi-3 chat template — call `tokenizer.apply_chat_template(messages, tokenize=False)` rather than building the string by hand. This ensures the model sees the same special tokens it was pre-trained with.

For training, the loss should only be computed on the assistant turn (the summary), not the prompt. Use a data collator that masks the input tokens — `DataCollatorForSeq2Seq` or manual label masking with `-100`.

## Build phases — do these in order

### Phase 1: Data preparation (`src/data.py`)
- Load DialogSum from HuggingFace Hub using the built-in train/validation/test splits
- Format each example into the prompt template above
- Tokenize with truncation (max_length=1024) — dialogue + summary must fit
- Mask prompt tokens in labels with `-100` so loss is computed only on the summary
- Add a CLI: `python src/data.py` that prints split sizes and one formatted example
- Sanity check: verify that the decoded labels (ignoring -100) match the original summary

### Phase 2: Model + LoRA setup (`src/model.py`)
- Load `microsoft/Phi-3-mini-4k-instruct` in 4-bit quantization (BitsAndBytes `load_in_4bit=True`) so it fits on a single T4 GPU
- Apply LoRA config via `peft.LoraConfig`:
  - `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
  - Target modules: `q_proj`, `v_proj` (standard for attention-only LoRA)
  - `task_type=CAUSAL_LM`
- Call `get_peft_model(model, lora_config)` and print trainable parameter count
- Expected: ~1–2% of total parameters are trainable — print this as a sanity check

### Phase 3: Training (`src/train.py`)
- Use HuggingFace `Trainer` with `TrainingArguments`:
  - `num_train_epochs=3`
  - `per_device_train_batch_size=4` (adjust based on GPU memory)
  - `gradient_accumulation_steps=4` (effective batch size = 16)
  - `learning_rate=2e-4`
  - `fp16=True`
  - `evaluation_strategy="epoch"`
  - `save_strategy="epoch"`
  - `load_best_model_at_end=True` (best by eval loss)
- Log to MLflow: start a run, log all hyperparameters and per-epoch train/val loss
- After training: push the LoRA adapter to HuggingFace Hub via `model.push_to_hub("rotemso23/dialogsum-phi3-lora")`
- Also push the tokenizer: `tokenizer.push_to_hub("rotemso23/dialogsum-phi3-lora")`
- CLI: `python src/train.py` — runs full training and pushes to Hub on completion

### Phase 4: Evaluation (`src/evaluate.py`)
- Load the fine-tuned adapter from Hub (or from local checkpoint)
- Run inference on the full test split (819 examples)
- Compute ROUGE-1, ROUGE-2, ROUGE-L using the `rouge_score` library
- Compare against a zero-shot baseline: same model, no fine-tuning, same prompt
- Save results to `evaluation_results.json` — these numbers go in the README
- Target: fine-tuned model should clearly outperform zero-shot baseline on ROUGE-L
- Also print 3–5 qualitative examples (dialogue → reference summary → model summary) for the README

### Phase 5: Inference function (`src/infer.py`)
- Single function: `summarize(dialogue: str) -> str`
- Returns the generated summary as a plain string
- Loads the adapter from Hub (cached after first call)
- Generation settings: `max_new_tokens=128`, `do_sample=False` (greedy for reproducibility)
- This is what `app.py` calls — keep it clean and import-safe

### Phase 6: Gradio demo (`app.py`)
- One large text area input: the conversation
- One text output: the generated summary
- A "Summarize" button
- Pre-populate the input with one example from the test set so it's not blank on load
- Keep it simple — this is a demo, not a product

### Phase 7: Deployment (HuggingFace Spaces)
- Create a new Space: Gradio SDK
- Add `HF_TOKEN` as a Space secret
- The model loads from Hub — no model weights committed to the Space repo
- `requirements.txt` must pin versions to ensure reproducibility
- First load will be slow (model download) — acceptable for a demo Space

### Phase 8: README + GitHub
- README must include: what it does, the task, the dataset, tech stack, training setup, ROUGE results table (baseline vs. fine-tuned), 2–3 qualitative examples, how to run locally, link to live demo, link to model on Hub
- This README is what a recruiter will read — write it to show you understand what LoRA is and why it matters, not just that you ran a script
- The qualitative examples are important: show a conversation and the model's summary side by side

## Key conventions

- All secrets via environment variables (`.env` file locally, Space secrets in HF). Never hardcoded.
- `mlflow_runs/` in `.gitignore`
- Type hints on all functions
- Each module has an `if __name__ == "__main__"` block for standalone testing
- Git commit after each phase is complete
- Use `transformers>=4.40` — Phi-3 support was added in 4.40

## Compute notes

- This project requires a GPU. Use Google Colab (free T4) for training.
- Training Phase 3 on a T4 with the above settings: approximately 1–2 hours for 3 epochs on 12k examples. Run overnight or use Colab Pro if needed.
- If T4 OOMs: reduce `per_device_train_batch_size` to 2 and increase `gradient_accumulation_steps` to 8.
- Alternatively, use `microsoft/phi-2` (2.7B) — slightly older, but fits more comfortably on T4 and trains faster.
- Do NOT try to run training on CPU — it will take many hours.

## Honest gaps this project closes (for CV)

After completing this project, Rotem can honestly claim:
- LLM fine-tuning with LoRA / PEFT
- HuggingFace Transformers ecosystem (Trainer, tokenizers, datasets)
- Parameter-efficient fine-tuning (understands rank, alpha, target modules)
- 4-bit quantization (BitsAndBytes)
- Experiment tracking with MLflow
- ROUGE evaluation methodology
- HuggingFace Hub model publishing
- HuggingFace Spaces deployment (Gradio)
- Baseline vs. fine-tuned model comparison and evaluation

## What NOT to overclaim on the CV

- This is not a production-grade model — describe it as "a fine-tuned summarization model" or "portfolio fine-tuning project"
- LoRA only trains adapter weights, not the full model — describe it accurately as "parameter-efficient fine-tuning with LoRA", not "trained from scratch"
- The evaluation set is the DialogSum test split, not a novel benchmark — describe results as "DialogSum test set ROUGE scores"
- ROUGE measures n-gram overlap, not semantic quality — do not claim the model produces "perfect" summaries
