import json
import random
import os
from transformers import AutoTokenizer
from datasets import Dataset

def parse_dataset(input_file="Dataset.json", output_file="preprocessed_data.json"):
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {input_file} not found!")
        return None

    samples = []
    skipped = []

    for intent in data["intents"]:
        tag = intent.get("tag", "unknown")
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])
        
        if not responses:
            skipped.append(tag)
            continue
        
        for pattern in patterns:
            response = random.choice(responses)
            samples.append({"text": f"User: {pattern}\nAssistant: {response}"})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(samples)} conversation samples")
    print(f"Skipped {len(skipped)} intents (no responses)")

    return samples

def tokenize_data(
    input_file="preprocessed_data.json",
    output_dir="tokenized_dataset",
    model_name="EleutherAI/gpt-neo-125M",
    max_length=512,
    test_split=0.1
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(data)

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    split = tokenized_dataset.train_test_split(test_size=test_split, seed=42)

    train = split["train"]
    val = split["test"]

    os.makedirs(output_dir, exist_ok=True)
    train.save_to_disk(f"{output_dir}/train")
    val.save_to_disk(f"{output_dir}/validation")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")

    stats = {
        "total_samples": len(data),
        "train_samples": len(train),
        "val_samples": len(val),
        "max_length": max_length,
        "model_name": model_name
    }
    with open(f"{output_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Tokenization complete. Train: {len(train)}, Validation: {len(val)}")
    return train, val, tokenizer

def main():
    print("=== Starting Data Pipeline ===")
    parsed = parse_dataset()
    if parsed is None or len(parsed) < 50:
        print("Warning: Very few samples. Training may not be effective.")
        return
    tokenize_data()
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
