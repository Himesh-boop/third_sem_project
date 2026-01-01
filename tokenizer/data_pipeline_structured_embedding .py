import json
import random
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer
from datasets import Dataset


TEMPLATE_GENERAL = """SYSTEM:
You are Freud, a supportive and thoughtful mental health assistant.
Your tone should be calm, respectful, and non-judgmental.
Do not rush to solutions unless the user asks for advice.

User:
{user_input}

Assistant:
{assistant_response}
"""


TEMPLATE_SADNESS = """SYSTEM:
You are Freud, responding to emotional sadness.
Acknowledge the user's feelings first.
Use empathy and validation before offering gentle perspective.
Avoid minimizing emotions or forcing positivity.

User:
{user_input}

Assistant:
{assistant_response}
"""


TEMPLATE_ANXIETY = """SYSTEM:
You are Freud, helping a user experiencing anxiety or worry.
Your tone should be grounding, slow, and reassuring.
Encourage clarity, breathing, or present-moment awareness if appropriate.
Do not overwhelm the user with too many suggestions.

User:
{user_input}

Assistant:
{assistant_response}
"""


TEMPLATE_ANGER = """SYSTEM:
You are Freud, responding to anger or frustration.
Validate the emotion without endorsing harmful actions.
Encourage reflection, emotional regulation, and safe expression.
Remain calm even if the user sounds aggressive.

User:
{user_input}

Assistant:
{assistant_response}
"""


TEMPLATE_LONELINESS = """SYSTEM:
You are Freud, responding to loneliness or emotional isolation.
Offer warmth, understanding, and a sense of connection.
Avoid sounding transactional or robotic.
Encourage the idea that the user is heard and not alone.

User:
{user_input}

Assistant:
{assistant_response}
"""


TEMPLATE_GRIEF = """SYSTEM:
You are Freud, supporting someone dealing with loss or grief.
Respect the depth and personal nature of grief.
Do not rush healing or suggest "moving on."
Use compassionate, slow, and gentle language.

User:
{user_input}

Assistant:
{assistant_response}
"""


TEMPLATE_SELF_ESTEEM = """SYSTEM:
You are Freud, responding to concerns about self-worth or confidence.
Avoid comparisons and absolute judgments.
Gently challenge negative self-beliefs when appropriate.
Reinforce the idea of intrinsic value.

User:
{user_input}

Assistant:
{assistant_response}
"""


TEMPLATE_BURNOUT = """SYSTEM:
You are Freud, responding to mental exhaustion, stress, or burnout.
Acknowledge fatigue and emotional overload.
Encourage rest, boundaries, and self-compassion.
Avoid productivity pressure or guilt-inducing language.

User:
{user_input}

Assistant:
{assistant_response}
"""


TEMPLATE_INFO = """SYSTEM:
You are Freud, providing general mental health information.
Be clear, accurate, and neutral.
Avoid diagnosing the user.
Explain concepts in simple and accessible language.

User:
{user_input}

Assistant:
{assistant_response}
"""


TEMPLATE_CRISIS = """SYSTEM:
You are Freud, prioritizing user safety and emotional support.
Remain calm, caring, and serious.
Encourage reaching out to trusted people or local support services.
Do NOT provide instructions, methods, or graphic descriptions.
Focus on care, presence, and support.

User:
{user_input}

Assistant:
{assistant_response}
"""

TEMPLATE_MAP = {
    "crisis": TEMPLATE_CRISIS,
    "sadness": TEMPLATE_SADNESS,
    "anxiety": TEMPLATE_ANXIETY,
    "anger": TEMPLATE_ANGER,
    "loneliness": TEMPLATE_LONELINESS,
    "grief": TEMPLATE_GRIEF,
    "self_esteem": TEMPLATE_SELF_ESTEEM,
    "burnout": TEMPLATE_BURNOUT,
    "info": TEMPLATE_INFO,
    "general": TEMPLATE_GENERAL
}

class SemanticEmotionClassifier:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        self.examples = {
            "crisis": [
                "I want to end my life",
                "I don't want to live anymore",
                "Everything feels pointless",
                "I want to hurt myself"
            ],
            "sadness": [
                "I feel very sad",
                "I feel empty inside",
                "I feel hopeless"
            ],
            "anxiety": [
                "I feel anxious",
                "I can't stop worrying",
                "My thoughts are racing"
            ],
            "anger": [
                "I'm very angry",
                "I feel furious",
                "I can't control my anger"
            ],
            "loneliness": [
                "I feel alone",
                "Nobody understands me",
                "I feel isolated"
            ],
            "grief": [
                "I lost someone close",
                "I'm grieving",
                "I can't move on from this loss"
            ],
            "self_esteem": [
                "I feel worthless",
                "I hate myself",
                "I'm not good enough"
            ],
            "burnout": [
                "I'm exhausted",
                "I'm mentally drained",
                "I'm burned out"
            ],
            "info": [
                "What is depression",
                "Explain anxiety",
                "What does therapy do"
            ],
            "general": [
                "I'm having a bad day",
                "I need someone to talk to",
                "Can you help me"
            ]
        }

        self.embeddings = {
            k: self.model.encode(v, show_progress_bar=False)
            for k, v in self.examples.items()
        }

    def classify(self, text):
        text_emb = self.model.encode([text], show_progress_bar=False)[0]

        scores = {}
        for label, emb in self.embeddings.items():
            sim = cosine_similarity(text_emb.reshape(1, -1), emb)[0]
            scores[label] = float(np.max(sim))

        if scores["crisis"] >= 0.65:
            return "crisis"

        return max(scores, key=scores.get)

def create_structured_dataset(
    input_file="Dataset.json",
    output_file="preprocessed_data_structured.json"
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    classifier = SemanticEmotionClassifier()
    samples = []

    for intent in data["intents"]:
        tag = intent.get("tag", "unknown")
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])

        if not responses:
            continue

        for pattern in patterns:
            emotion = classifier.classify(pattern)
            template = TEMPLATE_MAP.get(emotion, TEMPLATE_GENERAL)
            response = random.choice(responses)

            structured = template.format(
                user_input=pattern,
                assistant_response=response
            )

            samples.append({
                "text": structured,
                "emotion": emotion,
                "original_tag": tag
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Structured dataset created: {len(samples)} samples")


def tokenize_dataset(
    input_file="preprocessed_data_structured.json",
    output_dir="tokenized_dataset",
    model_name="EleutherAI/gpt-neo-125M",
    max_length=768,
    test_split=0.2
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list([{"text": d["text"]} for d in data])

    def tokenize_fn(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    split = tokenized.train_test_split(test_size=test_split, seed=42)

    os.makedirs(output_dir, exist_ok=True)
    split["train"].save_to_disk(f"{output_dir}/train")
    split["test"].save_to_disk(f"{output_dir}/validation")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")

    print("Tokenization complete")

def main():
    create_structured_dataset()
    tokenize_dataset()
    print("Pipeline finished successfully")


if __name__ == "__main__":
    main()
