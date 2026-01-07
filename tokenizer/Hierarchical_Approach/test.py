#to test the tokenized dataset

from datasets import load_from_disk
from transformers import AutoTokenizer

dataset = load_from_disk("tokenized_dataset_hierarchical/train")

tokenizer = AutoTokenizer.from_pretrained("tokenized_dataset_hierarchical/tokenizer")

decoded_text = tokenizer.decode(
    dataset[10]["input_ids"],
    skip_special_tokens=True
)

print(decoded_text)