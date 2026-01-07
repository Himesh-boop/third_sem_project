import os
import time
import torch
import json
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)

# -----------------------------
# Paths
# -----------------------------
dataset_base = "third_sem_project/tokenizer/Hierarchical_Approach/tokenized_dataset_hierarchical"
output_dir = "third_sem_project/freud_model"

# -----------------------------
# Load datasets
# -----------------------------
print("Loading tokenized datasets...")
train_dataset = load_from_disk(f"{dataset_base}/train")
eval_dataset = load_from_disk(f"{dataset_base}/validation")

# -----------------------------
# Load tokenizer
# -----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(f"{dataset_base}/tokenizer")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Load GPT-Neo model
# -----------------------------
print("Loading GPT-Neo-125M model...")
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-125M",
    torch_dtype=torch.float32
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# Check first example
print("First training example:")
print(train_dataset[0])

# Check type and nesting
print("\nType of input_ids:", type(train_dataset[0]['input_ids']))
print("Length of input_ids:", len(train_dataset[0]['input_ids']))

if any(isinstance(i, list) for i in train_dataset[0]['input_ids']):
    print("Warning: input_ids contain nested lists!")
else:
    print("input_ids are flat lists (good).")


# -----------------------------
# Data collator with padding
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8    # optional, improves GPU efficiency
)

# -----------------------------
# Kaggle-friendly progress callback
# -----------------------------
class KaggleProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"Training started. Total steps: {state.max_steps}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            eval_loss = logs.get("eval_loss")
            elapsed = time.time() - self.start_time
            steps_done = step
            steps_left = state.max_steps - steps_done
            eta_sec = (elapsed / steps_done) * steps_left if steps_done > 0 else 0
            eta_min = eta_sec / 60

            msg = f"[Step {step}/{state.max_steps}] Loss: {loss:.4f}" if loss is not None else f"[Step {step}/{state.max_steps}]"
            if lr is not None:
                msg += f", LR: {lr:.2e}"
            if eval_loss is not None:
                msg += f", Eval Loss: {eval_loss:.4f}"
            msg += f", ETA: {eta_min:.1f} min"
            print(msg)

# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=f"{output_dir}/checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=False,
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_num_workers=0,  # FIXED: Changed from 2 to 0 to avoid worker issues
    gradient_checkpointing=True,
    optim="adamw_torch"
)

# -----------------------------
# Initialize Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[KaggleProgressCallback()]
)

# -----------------------------
# Train the model
# -----------------------------
print("Starting training...")
trainer.train()

# -----------------------------
# Save final model and tokenizer
# -----------------------------
print(f"Saving final model and tokenizer to {output_dir} ...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# -----------------------------
# Save final metrics
# -----------------------------
final_metrics = trainer.evaluate()
with open(f"{output_dir}/training_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

print("Training complete!")
print(f"Final evaluation metrics: {final_metrics}")