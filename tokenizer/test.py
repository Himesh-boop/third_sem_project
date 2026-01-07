#to test the tokenized dataset

from datasets import load_from_disk
from transformers import AutoTokenizer

dataset = load_from_disk("tokenized_dataset/train")

tokenizer = AutoTokenizer.from_pretrained("tokenized_dataset/tokenizer")

decoded_text = tokenizer.decode(
    dataset[0]["input_ids"],
    skip_special_tokens=True
)

print(decoded_text)
