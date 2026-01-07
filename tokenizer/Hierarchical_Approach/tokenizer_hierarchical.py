import json
import os
from datasets import Dataset
from transformers import AutoTokenizer


def tokenize_hierarchical_dataset(
    input_file="preprocessed_data_hierarchical.json",
    output_dir="tokenized_dataset_hierarchical",
    model_name="EleutherAI/gpt-neo-125M",
    max_length=768,
    test_split=0.2,
    seed=42,
):
    # -----------------------------
    # Load raw data
    # -----------------------------
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {input_file}")

    # -----------------------------
    # Load tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(data)

    # -----------------------------
    # Tokenization function
    # -----------------------------
    def tokenize_function(batch):
        input_ids_batch = []
        labels_batch = []
        attention_masks = []

        for system, user, assistant in zip(
            batch["SYSTEM"], batch["User"], batch["Assistant"]
        ):
            # ---- Normalize texts ----
            system_text = system.strip() + "\n\n"
            user_text = f"User: {user.strip()}\n\n"
            assistant_text = assistant.strip()

            # ---- Explicit answer separation ----
            # Everything before <ANSWER> is context only
            # Only text inside <ANSWER>...</ANSWER> is trained
            if "<ANSWER>" in assistant_text:
                answer = assistant_text.split("<ANSWER>")[-1]
                answer = answer.replace("</ANSWER>", "").strip()
            else:
                # Fallback safety
                answer = assistant_text.strip()

            # ---- Final prompt format ----
            full_prompt = (
                system_text
                + user_text
                + "Assistant:\n<ANSWER>\n"
            )

            answer_text = answer + "\n</ANSWER>"

            # ---- Tokenize ----
            prompt_ids = tokenizer(
                full_prompt, add_special_tokens=False
            ).input_ids

            answer_ids = tokenizer(
                answer_text, add_special_tokens=False
            ).input_ids

            input_ids = prompt_ids + answer_ids

            # ---- Labels: mask everything except answer ----
            labels = [-100] * len(prompt_ids) + answer_ids

            # ---- Truncation (CRITICAL FIX) ----
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

            # ---- Padding ----
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len

            attention_mask = [
                0 if token_id == tokenizer.pad_token_id else 1
                for token_id in input_ids
            ]

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_masks.append(attention_mask)

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_masks,
        }

    # -----------------------------
    # Apply tokenization
    # -----------------------------
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing hierarchical dataset",
    )

    # -----------------------------
    # Train / Validation split
    # -----------------------------
    split_dataset = tokenized_dataset.train_test_split(
        test_size=test_split,
        seed=seed
    )

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # -----------------------------
    # Save to disk
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)

    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    val_dataset.save_to_disk(os.path.join(output_dir, "validation"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    stats = {
        "total_samples": len(data),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "model": model_name,
        "max_length": max_length,
        "loss_masking": "assistant_answer_only",
        "answer_format": "<ANSWER> ... </ANSWER>",
        "hierarchy_preserved": True,
        "task": "hierarchical_emotion_dialogue"
    }

    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Tokenized dataset saved to: {output_dir}")

    return train_dataset, val_dataset


if __name__ == "__main__":
    tokenize_hierarchical_dataset()
