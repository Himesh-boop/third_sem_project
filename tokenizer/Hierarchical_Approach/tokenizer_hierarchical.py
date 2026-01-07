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
    seed=42):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {input_file}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(data)

    def tokenize_function(batch):
        input_ids_batch = []
        labels_batch = []
        attention_masks = []

        for system, user, assistant in zip(
            batch["SYSTEM"], batch["User"], batch["Assistant"]):
            system_text = system.strip() + "\n\n"
            user_text = f"User: {user.strip()}\n\nAssistant:"
            assistant_text = " " + assistant.strip()

            system_ids = tokenizer(system_text, add_special_tokens=False).input_ids
            user_ids = tokenizer(user_text, add_special_tokens=False).input_ids
            assistant_ids = tokenizer(assistant_text, add_special_tokens=False).input_ids

            input_ids = system_ids + user_ids + assistant_ids
            labels = [-100] * len(system_ids) + [-100] * len(user_ids) + assistant_ids

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len

            attention_mask = [1 if id != tokenizer.pad_token_id else 0 for id in input_ids]

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_masks.append(attention_mask)

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_masks,
        }

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    split_dataset = tokenized_dataset.train_test_split(
        test_size=test_split,
        seed=seed)

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

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
        "loss_masking": "assistant_only",
        "format": "SYSTEM → User → Assistant",
        "task": "hierarchical_emotion"
    }

    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Tokenized dataset saved to: {output_dir}")

    return train_dataset, val_dataset


if __name__ == "__main__":
    tokenize_hierarchical_dataset()
