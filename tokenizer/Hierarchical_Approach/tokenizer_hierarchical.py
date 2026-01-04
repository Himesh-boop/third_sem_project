import json
import os
from datasets import Dataset
from transformers import AutoTokenizer


def tokenize_hierarchical_dataset(
    input_file="rejected_data_hierarchical.json",
    output_dir="tokenized_rejected_dataset_hierarchical",
    model_name="EleutherAI/gpt-neo-125M",
    max_length=768,
    test_split=0.2
):
    
    print(f"\nLoading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    print(f"\nLoading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\nCreating dataset...")
    dataset = Dataset.from_list([
        {
            "SYSTEM": item["SYSTEM"],
            "User": item["User"],
            "Assistant": item["Assistant"]
        }
        for item in data
    ])
    
    def tokenize_function(example):
        system_text = example["SYSTEM"].strip() + "\n\n"
        user_text = f"User: {example['User'].strip()}\n\nAssistant:"
        assistant_text = " " + example["Assistant"].strip()
        
        system_ids = tokenizer(
            system_text,
            add_special_tokens=False
        )["input_ids"]
        
        user_ids = tokenizer(
            user_text,
            add_special_tokens=False
        )["input_ids"]
        
        assistant_ids = tokenizer(
            assistant_text,
            add_special_tokens=False
        )["input_ids"]
        
        input_ids = system_ids + user_ids + assistant_ids
        
        labels = (
            [-100] * len(system_ids)
            + [-100] * len(user_ids)
            + assistant_ids
        )
        
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
    
    print("Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=["SYSTEM", "User", "Assistant"],
        desc="Tokenizing"
    )
    
    split_dataset = tokenized_dataset.train_test_split(
        test_size=test_split,
        seed=42
    )
    
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    print(f"\nSplit: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.save_to_disk(f"{output_dir}/train")
    val_dataset.save_to_disk(f"{output_dir}/validation")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")
    
    stats = {
        "total_samples": len(data),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "max_length": max_length,
        "classification": "hierarchical_emotion",
        "label_masking": "assistant_only",
        "fields_used": ["SYSTEM", "User", "Assistant"]
    }
    
    with open(f"{output_dir}/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSaved to {output_dir}/")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Tokenizer config saved")
    print("\nTokenization complete\n")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    tokenize_hierarchical_dataset()