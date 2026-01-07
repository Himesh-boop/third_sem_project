from untokenized import dataset
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token 

hf_dataset = Dataset.from_list(dataset)

def tokenize_fn(example):
    tok = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

tokenized_dataset = hf_dataset.map(tokenize_fn, batched=False, remove_columns=["text"])

with open("tokenized_data.txt", "w") as f:
    for item in tokenized_dataset:
        f.write(str(item) + "\n\n")

print(tokenized_dataset[0].keys())