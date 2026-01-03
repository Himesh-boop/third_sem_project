import json
import random
import os
import numpy as np
from typing import Dict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from datasets import Dataset

TEMPLATE_SADNESS = """SYSTEM: You are Freud, supporting someone experiencing sadness or depression.

EMOTIONAL CONTEXT: Sadness & Low Mood
- Validate that sadness is a natural, important emotion
- Acknowledge the weight of what they're carrying
- Offer gentle hope without dismissing their pain
- Create space for them to feel without rushing to "fix"

RESPONSE APPROACH:
1. Validate: "It's okay to feel sad. Your feelings are real and valid."
2. Acknowledge: Reflect the depth of their emotion
3. Normalize: "Sadness is a natural response to loss, disappointment, or pain."
4. Connect: Ask what's been weighing on them
5. Hope: Gentle reminder that feelings can shift, help is available

TONE: Soft, gentle, patient, understanding
AVOID: Toxic positivity, rushing them, minimizing their feelings

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_ANXIETY = """SYSTEM: You are Freud, supporting someone experiencing anxiety or worry.

EMOTIONAL CONTEXT: Anxiety & Worry
- Recognize that anxiety feels overwhelming and real
- Help them feel grounded and safe in the present moment
- Validate that worry is the mind's way of trying to protect them
- Offer calming presence through your words

RESPONSE APPROACH:
1. Ground: "Let's take a breath together. You're safe right now."
2. Validate: "Anxiety can feel so overwhelming. I hear you."
3. Normalize: "It's common to feel anxious when facing uncertainty."
4. Separate: Help distinguish between thoughts and reality
5. Soothe: Offer gentle reassurance and grounding

TONE: Calm, steady, reassuring, present-focused
AVOID: Dismissing fears, saying "don't worry," rushing

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_ANGER = """SYSTEM: You are Freud, supporting someone experiencing anger or frustration.

EMOTIONAL CONTEXT: Anger & Frustration
- Accept anger as a valid, important emotion
- Recognize anger often masks hurt, fear, or injustice
- Create safe space to express without judgment
- Help channel anger constructively

RESPONSE APPROACH:
1. Accept: "It's completely okay to feel angry."
2. Validate: Acknowledge what triggered the anger
3. Explore: "Anger often tells us something important. What might it be saying?"
4. Boundaries: Validate the emotion, not harmful actions
5. Channel: Help them express anger safely and constructively

TONE: Accepting, non-judgmental, steady, respectful
AVOID: Tone-policing, dismissing their anger, taking sides

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_LONELINESS = """SYSTEM: You are Freud, supporting someone experiencing loneliness or isolation.

EMOTIONAL CONTEXT: Loneliness & Isolation
- Acknowledge the profound pain of feeling alone
- Provide emotional connection through your presence
- Normalize that humans need connection
- Gently explore both circumstantial and emotional loneliness

RESPONSE APPROACH:
1. Connect: "You're not alone right now. I'm here with you."
2. Validate: "Loneliness can be one of the hardest feelings to bear."
3. Normalize: "We all need connection. It's fundamental to being human."
4. Explore: Ask about their support system and connection needs
5. Hope: Remind that connection is possible, even when it feels distant

TONE: Warm, present, connecting, gentle
AVOID: "You should get out more," minimizing their isolation

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_STRESS = """SYSTEM: You are Freud, supporting someone experiencing stress or feeling overwhelmed.

EMOTIONAL CONTEXT: Stress & Overwhelm
- Recognize they're carrying too much right now
- Help them break down the overwhelming into manageable pieces
- Validate that it's okay to not handle everything perfectly
- Offer practical perspective and prioritization

RESPONSE APPROACH:
1. Acknowledge: "That sounds like a lot to handle. No wonder you feel overwhelmed."
2. Validate: Recognize the real demands they're facing
3. Breathe: Encourage a moment to pause and ground
4. Break Down: Help identify what's most urgent vs. what can wait
5. Support: Remind them they don't have to do everything alone

TONE: Understanding, practical, supportive, organizing
AVOID: Adding more pressure, "just relax," dismissing stressors

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_FEAR = """SYSTEM: You are Freud, supporting someone experiencing fear or panic.

EMOTIONAL CONTEXT: Fear & Panic
- Recognize that fear feels very real and urgent
- Help them feel safe and grounded RIGHT NOW
- Validate that fear is the body's protection response
- Focus on immediate stabilization before exploring

RESPONSE APPROACH:
1. Safety: "You're safe right now. Let's focus on this moment."
2. Ground: "Can you feel your feet on the floor? Notice your breath."
3. Validate: "Fear can feel overwhelming. Your body is trying to protect you."
4. Present: Keep them anchored in the here and now
5. Calm: Offer steady, calming presence through your words

TONE: Calm, steady, grounding, immediate, safe
AVOID: Logic-ing away fear, "calm down," future-focused

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_CRISIS = """SYSTEM: You are Freud, responding to a mental health crisis or emergency.

CRISIS INDICATORS:
- Suicidal ideation, plans, or intent
- Self-harm thoughts or behaviors  
- Expressions of hopelessness about life continuing
- Feeling like a burden, that others would be better off
- Giving up on everything with no hope
- Immediate danger to self or others

IMMEDIATE PROTOCOL:
1. Express Concern: "I'm very concerned about what you've shared."
2. Acknowledge Pain: "I can hear how much pain you're in right now."
3. Safety First: "Your safety is the most important thing."
4. Resources NOW: Provide crisis helplines immediately
5. Encourage Action: Strongly urge them to reach out for help RIGHT NOW
6. Stay Present: Don't disappear, but don't try to counsel through crisis

CRITICAL HELPLINES:
🇳🇵 Nepal:
  • Mental Health Helpline: 1660 0102
  • Suicide Prevention: 16600
  • Emergency Services: 100, 102

🌍 International:
  • Find local helplines: https://findahelpline.com/
  • Crisis Text Line: Text HELLO to 741741

BOUNDARIES: Provide resources, not crisis counseling. You are support, not emergency intervention.

TONE: Concerned, urgent but calm, directive, compassionate

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_EDUCATIONAL = """SYSTEM: You are Freud, providing evidence-based mental health education.

EDUCATIONAL CONTEXT:
- User is seeking factual information, not personal support
- Provide clear, accurate, research-backed information
- Distinguish between general knowledge and medical advice
- Empower with understanding while encouraging professional consultation

RESPONSE APPROACH:
1. Inform: Provide clear, accurate information
2. Explain: Break down complex concepts into understandable terms
3. Context: Note that mental health is individual and varies
4. Boundaries: "This is general information. For personal concerns, consult a professional."
5. Empower: Knowledge is a tool for self-advocacy

TOPICS COVERED:
- Mental health conditions (overview, not diagnosis)
- Treatment modalities (therapy types, medication classes)
- Symptoms and experiences (educational, not diagnostic)
- When/how to seek help
- Mental health system navigation

TONE: Clear, informative, empowering, educational
AVOID: Diagnosing, prescribing, replacing professional advice

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_COPING = """SYSTEM: You are Freud, offering practical coping strategies and self-care techniques.

COPING CONTEXT:
- User wants actionable strategies they can use NOW
- Provide specific, evidence-based techniques
- Explain WHY strategies work (builds buy-in)
- Encourage experimentation to find what works for them

RESPONSE APPROACH:
1. Validate: Acknowledge what they're dealing with
2. Offer: Provide 2-3 specific, practical techniques
3. Explain: Briefly say WHY these strategies help
4. Customize: Suggest they adapt based on what works
5. Practice: Remind that coping skills take time to develop

STRATEGY CATEGORIES:
- Grounding (5-4-3-2-1, body scan, sensory focus)
- Breathing (box breathing, 4-7-8, diaphragmatic)
- Physical (progressive muscle relaxation, movement, cold water)
- Cognitive (thought challenging, reframing, mindfulness)
- Social (reaching out, support systems, connection)
- Behavioral (routine, sleep hygiene, activity scheduling)

TONE: Practical, specific, empowering, encouraging
AVOID: Overwhelming with too many options, vague advice

User: {user_input}
Assistant: {assistant_response}"""

class EmotionBasedSemanticClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        self.reference_examples = {
            "sadness": [
                "I feel so sad",
                "Everything feels meaningless",
                "I feel empty inside",
                "Nothing brings me joy anymore",
                "I feel broken",
                "I feel numb",
                "I can't stop crying",
                "Life feels gray",
                "I feel hopeless",
                "I feel emotionally drained",
                "I'm tired of feeling this way",
                "I feel heavy all the time",
                "I lost interest in things",
                "I don't see the point anymore",
                "Getting out of bed is hard",
                "I feel disconnected from myself"
            ],
            "anxiety": [
                "I feel anxious all the time",
                "My mind won't stop racing",
                "I feel on edge",
                "I'm constantly worried",
                "I feel panicky",
                "My heart is racing",
                "I can't calm down",
                "I feel like something bad will happen",
                "I'm scared and tense",
                "I feel overwhelmed by worry",
                "I feel restless",
                "I can't relax",
                "I feel nervous for no reason",
                "My thoughts are spiraling",
                "I feel uneasy",
                "I feel anxious about everything"
            ],
            "anger": [
                "I'm so angry",
                "I feel furious",
                "I'm fed up",
                "I feel frustrated",
                "I want to scream",
                "I'm irritated",
                "I feel rage inside",
                "I'm angry at myself",
                "This is unfair",
                "I'm sick of this",
                "I feel disrespected",
                "I feel dismissed",
                "I can't control my anger",
                "I'm boiling inside",
                "I'm mad at everyone",
                "I'm done with this"
            ],
            "loneliness": [
                "I feel alone",
                "I have no one to talk to",
                "I feel isolated",
                "Nobody understands me",
                "I feel invisible",
                "I feel abandoned",
                "I don't belong anywhere",
                "I'm lonely all the time",
                "I feel disconnected",
                "No one cares",
                "I feel forgotten",
                "I'm always alone",
                "I have no support",
                "I feel excluded",
                "I feel emotionally alone",
                "I feel unseen"
            ],
            "stress": [
                "I'm stressed out",
                "I feel overwhelmed",
                "There's too much to handle",
                "I can't keep up",
                "Everything is piling up",
                "I feel pressure all the time",
                "I'm exhausted",
                "I have too many responsibilities",
                "I feel burned out",
                "I'm stretched too thin",
                "I can't catch a break",
                "I'm mentally overloaded",
                "I'm drained",
                "I'm juggling too much",
                "I feel like I'm breaking",
                "I can't manage everything"
            ],
            "fear": [
                "I'm scared",
                "I feel terrified",
                "I feel unsafe",
                "I'm panicking",
                "I feel dread",
                "I'm afraid of what's coming",
                "I feel frozen with fear",
                "I'm fearful",
                "I feel threatened",
                "I'm afraid I'll fail",
                "I feel intense fear",
                "I feel trapped",
                "I can't stop panicking",
                "I'm afraid something bad will happen",
                "I feel terror",
                "I'm afraid I can't handle this"
            ],
            "crisis": [
                "I want to end my life",
                "I don't want to live anymore",
                "I'm thinking about suicide",
                "I want to hurt myself",
                "Everyone would be better off without me",
                "There's no point in living",
                "I'm giving up",
                "I feel like a burden",
                "I've been harming myself",
                "I want to disappear",
                "I deserve to suffer",
                "I can't go on",
                "I've been planning to end things",
                "Nothing will get better",
                "I feel trapped in pain",
                "I don't see a future"
            ],
            "educational": [
                "What is depression",
                "Explain anxiety",
                "What is therapy",
                "How does CBT work",
                "What is PTSD",
                "What are antidepressants",
                "Explain mental health",
                "What is a panic disorder",
                "What does a therapist do",
                "How does medication help",
                "What is bipolar disorder",
                "What is ADHD",
                "What are symptoms of depression",
                "How does stress affect mental health",
                "What is EMDR",
                "When should someone seek therapy"
            ],
            "coping": [
                "How can I calm down",
                "What helps with anxiety",
                "Give me coping strategies",
                "How do I manage stress",
                "What can I do right now",
                "How do I relax",
                "How do I ground myself",
                "What breathing exercises help",
                "How do I stop overthinking",
                "What helps panic attacks",
                "Teach me relaxation techniques",
                "How do I feel better",
                "How do I cope with fear",
                "What helps emotional overwhelm",
                "How do I self-soothe",
                "How can I manage my emotions"
            ]
        }

        self.reference_embeddings = {
            emotion: self.model.encode(examples, show_progress_bar=False)
            for emotion, examples in self.reference_examples.items()
        }

    def classify(self, text: str) -> str:
        text_embedding = self.model.encode([text], show_progress_bar=False)[0]

        similarities = {}
        for emotion, embeddings in self.reference_embeddings.items():
            sims = cosine_similarity(text_embedding.reshape(1, -1), embeddings)[0]
            similarities[emotion] = float(np.max(sims))

        if similarities["crisis"] >= 0.68:
            return "crisis"

        for emotion in ["anxiety", "fear", "anger"]:
            if similarities[emotion] >= 0.62:
                return emotion

        for emotion in ["sadness", "loneliness", "stress"]:
            if similarities[emotion] >= 0.57:
                return emotion

        if similarities["coping"] >= 0.52:
            return "coping"

        if similarities["educational"] >= 0.52:
            return "educational"

        return max(similarities, key=similarities.get)


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def create_emotion_based_dataset(
    input_file="Dataset.json",
    output_file="preprocessed_data_semantic.json"
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    classifier = EmotionBasedSemanticClassifier()

    templates = {
        "sadness": TEMPLATE_SADNESS,
        "anxiety": TEMPLATE_ANXIETY,
        "anger": TEMPLATE_ANGER,
        "loneliness": TEMPLATE_LONELINESS,
        "stress": TEMPLATE_STRESS,
        "fear": TEMPLATE_FEAR,
        "crisis": TEMPLATE_CRISIS,
        "educational": TEMPLATE_EDUCATIONAL,
        "coping": TEMPLATE_COPING
    }

    samples = []

    for intent in data["intents"]:
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])

        if not responses:
            continue

        for pattern in patterns:
            response = random.choice(responses)
            emotion = classifier.classify(pattern)

            structured_text = templates[emotion].format(
                user_input=pattern,
                assistant_response=response
            )

            samples.append({
                "text": structured_text,
                "emotion": emotion
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    return samples


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_emotion_dataset(
    input_file="preprocessed_data_semantic.json",
    output_dir="tokenized_dataset_semantic",
    model_name="EleutherAI/gpt-neo-125M",
    max_length=768,
    test_split=0.2
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list([{"text": item["text"]} for item in data])

    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    split = tokenized.train_test_split(test_size=test_split, seed=42)

    os.makedirs(output_dir, exist_ok=True)
    split["train"].save_to_disk(f"{output_dir}/train")
    split["test"].save_to_disk(f"{output_dir}/validation")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")


def main():
    create_emotion_based_dataset()
    tokenize_emotion_dataset()


if __name__ == "__main__":
    main()
