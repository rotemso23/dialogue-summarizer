"""
src/train.py — Training loop for Phi-3-mini + LoRA on DialogSum.

Loads the model and dataset, runs 3 epochs with HuggingFace Trainer,
logs hyperparameters and per-epoch metrics to MLflow, then pushes
the LoRA adapter and tokenizer to HuggingFace Hub.

Run: python src/train.py
Requires: HF_TOKEN in .env (for Hub push), GPU (T4 or better).
"""

from __future__ import annotations

import os

import mlflow
from dotenv import load_dotenv
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.data import DEFAULT_MAX_LENGTH, make_data_collator, prepare_datasets
from src.model import (
    HUB_REPO,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_ID,
    load_model_and_tokenizer,
    print_trainable_parameters,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs/dialogsum-phi3-lora"
MLFLOW_EXPERIMENT = "dialogsum-phi3-lora"

HYPERPARAMS: dict = {
    "model_id": MODEL_ID,
    "hub_repo": HUB_REPO,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,   # effective batch size = 16
    "learning_rate": 2e-4,
    "fp16": True,
    "max_length": DEFAULT_MAX_LENGTH,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "lora_target_modules": ",".join(LORA_TARGET_MODULES),
}


# ---------------------------------------------------------------------------
# MLflow callback
# ---------------------------------------------------------------------------

class MLflowEpochCallback(TrainerCallback):
    """
    Log train loss, eval loss, and learning rate to MLflow at the end of
    each epoch. The Trainer already computes these — this callback just
    forwards them to the active MLflow run.

    Uses `on_evaluate` (fires after each eval pass) rather than `on_epoch_end`
    because eval metrics are only available after evaluation completes.
    """

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ) -> None:
        if not state.is_world_process_zero:
            return
        step = state.global_step
        epoch = int(state.epoch) if state.epoch else step
        log: dict[str, float] = {"epoch": float(epoch)}
        for key in ("eval_loss", "eval_runtime", "train_loss"):
            if key in metrics:
                log[key] = metrics[key]
        # learning rate is in the last log history entry
        for entry in reversed(state.log_history):
            if "learning_rate" in entry:
                log["learning_rate"] = entry["learning_rate"]
                break
        mlflow.log_metrics(log, step=step)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict,
        **kwargs,
    ) -> None:
        """Also forward step-level train loss so the MLflow chart is smooth."""
        if not state.is_world_process_zero:
            return
        step = state.global_step
        metrics: dict[str, float] = {}
        for key in ("loss", "learning_rate", "grad_norm"):
            if key in logs:
                metrics[key] = logs[key]
        if metrics:
            mlflow.log_metrics(metrics, step=step)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train() -> None:
    """
    Full training pipeline:
        1. Load model + tokenizer (4-bit quant + LoRA).
        2. Tokenize DialogSum train/val splits.
        3. Run Trainer for 3 epochs, logging to MLflow.
        4. Push adapter + tokenizer to HuggingFace Hub.
    """
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN not set. Copy .env.example to .env and add your token."
        )

    # --- Model + tokenizer ---
    print("Loading model and tokenizer ...")
    model, tokenizer = load_model_and_tokenizer()
    print("\nTrainable parameters:")
    print_trainable_parameters(model)

    # --- Data ---
    print("\nPreparing datasets ...")
    train_ds, val_ds, _ = prepare_datasets(tokenizer)
    collator = make_data_collator(tokenizer)
    print(f"  train: {len(train_ds):,}  |  val: {len(val_ds):,}")

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=HYPERPARAMS["num_train_epochs"],
        per_device_train_batch_size=HYPERPARAMS["per_device_train_batch_size"],
        gradient_accumulation_steps=HYPERPARAMS["gradient_accumulation_steps"],
        learning_rate=HYPERPARAMS["learning_rate"],
        fp16=HYPERPARAMS["fp16"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",           # MLflow handled manually via callback
        dataloader_pin_memory=False, # avoids issues with quantized models on some setups
    )

    # --- MLflow run ---
    # Use a local file store explicitly — avoids path-encoding issues on
    # Windows when the username contains non-ASCII characters.
    mlflow.set_tracking_uri("mlflow_runs")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="phi3-lora-dialogsum"):
        mlflow.log_params(HYPERPARAMS)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            callbacks=[MLflowEpochCallback()],
        )

        print("\nStarting training ...")
        trainer.train()

        # Log final eval loss explicitly
        final_metrics = trainer.evaluate()
        mlflow.log_metrics(
            {"final_eval_loss": final_metrics["eval_loss"]},
            step=trainer.state.global_step,
        )
        print(f"\nFinal eval loss: {final_metrics['eval_loss']:.4f}")

    # --- Push to Hub ---
    print(f"\nPushing adapter to Hub: {HUB_REPO} ...")
    model.push_to_hub(HUB_REPO, token=hf_token)

    print(f"Pushing tokenizer to Hub: {HUB_REPO} ...")
    tokenizer.push_to_hub(HUB_REPO, token=hf_token)

    print(f"\nDone. Model published at: https://huggingface.co/{HUB_REPO}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
