import json, random, os
from transformers import AutoTokenizer
from datasets import Dataset

# --- Templates ---
TEMPLATE_GENERAL = """SYSTEM: You are Freud, a helpful mental health assistant.
CORE PRINCIPLES:
- Listen, validate emotions
- Don't diagnose
- Don't give medical advice
- Encourage professional help
- Stay supportive

RESPONSE:
1. Acknowledge
2. Normalize
3. Ask open question
4. Reassure

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_CRISIS = """SYSTEM: You are Freud, trained for crisis situations.
If suicide/self-harm is mentioned:
- Express concern
- Provide helpline numbers
- Encourage professional help
- Stay calm and supportive

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_INFO = """SYSTEM: You are Freud, giving mental health info in a supportive way.
- Give factual info
- Distinguish general info vs medical advice
- Keep it simple
- End supportively

User: {user_input}
Assistant: {assistant_response}"""

# --- Intent classifier ---
class IntentClassifier:
    def __init__(self):
        self.crisis_keywords = ['suicide','kill myself','end it all','want to die','self harm','hurt myself','end my life','no reason to live']
        self.info_keywords = ['what is','what are','define','definition','explain','tell me about','how to','can you explain','difference between']
        self.educational_tags = ['mental_health_info','depression_info','anxiety_info','therapy_info','medication_info','professional_help']

    def classify(self, user_input, tag=None):
        text = user_input.lower()
        if any(k in text for k in self.crisis_keywords):
            return 'crisis'
        if tag and any(t in tag for t in self.educational_tags):
            return 'info'
        if any(k in text for k in self.info_keywords):
            return 'info'
        return 'general'

# --- Create structured dataset ---
def create_structured_dataset(input_file="Dataset.json", output_file="preprocessed_data_structured.json"):
    print("Creating structured dataset...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    classifier = IntentClassifier()
    sample_data = []
    skipped = []
    stats = {'general':0,'crisis':0,'info':0,'total':0}

    for intent in data['intents']:
        tag = intent.get("tag","unknown")
        patterns = intent.get("patterns",[])
        responses = intent.get("responses",[])
        if not responses:
            skipped.append(tag)
            continue
        for pattern in patterns:
            response = random.choice(responses)
            t = classifier.classify(pattern, tag)
            template = TEMPLATE_CRISIS if t=='crisis' else TEMPLATE_INFO if t=='info' else TEMPLATE_GENERAL
            structured = template.format(user_input=pattern, assistant_response=response)
            sample_data.append({"text":structured,"intent_type":t,"original_tag":tag})
            stats[t] += 1
            stats['total'] += 1

    with open(output_file,"w",encoding="utf-8") as f:
        json.dump(sample_data,f,indent=2,ensure_ascii=False)

    print(f"Done! {stats}")
    return sample_data, stats

# --- Tokenize ---
def tokenize_structured_data(input_file="preprocessed_data_structured.json", output_dir="tokenized_dataset_structured", model_name="EleutherAI/gpt-neo-125M", max_length=768, test_split=0.2):
    print("Tokenizing dataset...")
    with open(input_file,"r",encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [{"text":d["text"]} for d in data]
    dataset = Dataset.from_list(texts)

    def tok(examples):
        t = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
        t["labels"] = t["input_ids"].copy()
        return t

    tokenized = dataset.map(tok, batched=True, remove_columns=["text"])
    split = tokenized.train_test_split(test_size=test_split, seed=42)
    train = split["train"]
    val = split["test"]

    os.makedirs(output_dir, exist_ok=True)
    train.save_to_disk(f"{output_dir}/train")
    val.save_to_disk(f"{output_dir}/validation")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")
    stats = {"total_samples":len(data),"train_samples":len(train),"val_samples":len(val),"max_length":max_length,"model_name":model_name}
    with open(f"{output_dir}/stats.json","w") as f:
        json.dump(stats,f,indent=2)

    print("Tokenization done!")
    return train, val

# --- Verify sample prompts ---
def verify_structured_prompts(file_path="preprocessed_data_structured.json", num_samples=3):
    with open(file_path,"r",encoding="utf-8") as f:
        data = json.load(f)
    shown = set()
    count = 0
    for item in data:
        t = item.get("intent_type","general")
        if t not in shown and count<num_samples:
            print(f"\n--- {t.upper()} ---")
            print(item["text"][:400]+"...")
            shown.add(t)
            count += 1
        if count>=num_samples:
            break

# --- Main ---
def main():
    print("Starting pipeline...")
    create_structured_dataset()
    verify_structured_prompts()
    tokenize_structured_data()
    print("Pipeline complete!")

if __name__=="__main__":
    main()
