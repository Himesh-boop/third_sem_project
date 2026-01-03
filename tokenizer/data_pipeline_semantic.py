import json
import random
import numpy as np
import os
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
            'greeting': [
                "hi",
                "hello",
                "hey",
                "hey there",
                "hello there",
                "good morning",
                "good afternoon",
                "good evening",
                "is anyone there",
                "are you there",
                "hello freud",
                "hi freud",
                "yo what's up",
                "just saying hi",
                "checking in",
                "can we talk",
                "are you available"
            ],
            'sadness': [
                # Direct expressions
                "I feel so sad",
                "I'm feeling depressed",
                "I feel hopeless and empty inside",
                "Everything feels meaningless",
                "I can't stop crying",
                "I feel numb and disconnected",
                "Nothing brings me joy anymore",
                "I feel like I'm drowning in sadness",
                "The sadness is overwhelming",
                "I feel broken and worthless",
                "Life feels gray and colorless",
                "I'm tired of feeling this way",
                
                # Contextual expressions
                "I've lost interest in everything I used to love",
                "I don't see the point of anything anymore",
                "I feel heavy, like I'm carrying the weight of the world",
                "I can't remember the last time I felt happy",
                "Getting out of bed feels impossible",
                "I feel like I'm just going through the motions"
            ],
            
            'anxiety': [
                # Direct expressions
                "I feel anxious all the time",
                "I'm constantly worried about everything",
                "My anxiety is overwhelming",
                "I can't stop worrying",
                "I feel panicky and on edge",
                "My mind won't stop racing",
                "I'm terrified something bad will happen",
                "I feel like I'm going to have a panic attack",
                "The worry never stops",
                "I'm scared and I don't know why",
                "Everything makes me nervous",
                "I feel like something terrible is going to happen",
                
                # Physical symptoms
                "My heart is racing and I can't breathe",
                "I feel shaky and dizzy",
                "I can't calm down no matter what I do",
                "I feel like I'm losing control",
                "The anxiety is paralyzing me",
                "I'm anxious about being anxious"
            ],
            
            'anger': [
                # Direct expressions
                "I'm so angry right now",
                "I feel furious and frustrated",
                "I'm fed up with everything",
                "I'm so mad I could scream",
                "Everything is making me angry",
                "I feel rage building inside me",
                "I'm irritated by everything",
                "I want to break something",
                "I'm sick of this situation",
                "I'm angry at myself",
                "I'm furious at how unfair this is",
                "I can't control my anger",
                
                # Contextual expressions
                "Why does this always happen to me",
                "I'm tired of being treated this way",
                "No one understands how frustrating this is",
                "I'm done with people disappointing me",
                "I feel disrespected and dismissed",
                "I'm angry that no one listens"
            ],
            
            'loneliness': [
                # Direct expressions
                "I feel so alone",
                "Nobody understands me",
                "I feel isolated and disconnected",
                "I have no one to talk to",
                "I'm surrounded by people but feel lonely",
                "I feel invisible to everyone",
                "No one cares about me",
                "I'm all by myself",
                "I feel abandoned and forgotten",
                "Everyone has someone except me",
                "I'm lonely all the time",
                "I feel like I don't belong anywhere",
                
                # Social expressions
                "I have no real friends",
                "I don't fit in with anyone",
                "I'm always the outsider",
                "People don't reach out to me",
                "I feel disconnected from everyone around me",
                "I'm alone in this struggle"
            ],
            
            'stress': [
                # Direct expressions
                "I'm so stressed out",
                "I feel overwhelmed by everything",
                "There's too much on my plate",
                "I can't handle all of this",
                "The pressure is too much",
                "I'm stressed about work and life",
                "Everything is piling up",
                "I have too many responsibilities",
                "I'm drowning in stress",
                "I feel like I'm about to break",
                "The stress never ends",
                "I'm exhausted from all the pressure",
                
                # Situational stress
                "I have too many deadlines",
                "I'm juggling too many things at once",
                "I don't have time for anything",
                "Everything is demanding my attention",
                "I feel pulled in a million directions",
                "I can't keep up with everything"
            ],
            
            'fear': [
                # Direct expressions
                "I'm scared",
                "I feel terrified",
                "I'm afraid of what might happen",
                "Fear is consuming me",
                "I'm panicking",
                "I feel like something bad is about to happen",
                "I'm frightened and don't know why",
                "I'm having a panic attack",
                "I can't shake this feeling of dread",
                "I'm paralyzed by fear",
                "I feel unsafe",
                "Terror is taking over",
                
                # Specific fears
                "I'm scared I'm going to fail",
                "I'm afraid of losing everything",
                "I'm terrified of being alone",
                "I'm scared I'm not good enough",
                "I fear the worst will happen",
                "I'm afraid I can't handle this"
            ],
            
            'crisis': [
                # Suicidal ideation
                "I want to end my life",
                "I don't want to live anymore",
                "I'm thinking about suicide",
                "I want to kill myself",
                "I'm planning to end things",
                "I've been researching ways to die",
                
                # Hopelessness
                "Life has no meaning anymore",
                "There's no point in continuing",
                "I'm giving up on everything",
                "Nothing will ever get better",
                "I can't see any way forward",
                "Why keep going when nothing matters",
                
                # Burden thoughts
                "Everyone would be better off without me",
                "I'm a burden to everyone",
                "My family deserves better than me",
                "I'm just making things worse for everyone",
                
                # Self-harm
                "I want to hurt myself",
                "I'm thinking about cutting",
                "I need to punish myself",
                "I deserve to suffer",
                "I've been harming myself",
                "I can't stop hurting myself"
            ],
            
            'educational': [
                # Definition questions
                "What is depression",
                "Define anxiety disorder",
                "Explain PTSD to me",
                "What does bipolar disorder mean",
                "Tell me about OCD",
                "What is schizophrenia",
                "Explain panic disorder",
                "What are eating disorders",
                "Define borderline personality disorder",
                "What is ADHD",
                
                # Mechanism questions
                "How does therapy work",
                "What do antidepressants do",
                "Explain cognitive behavioral therapy",
                "How does EMDR work",
                "What happens in psychotherapy",
                "How do SSRIs work",
                
                # Comparison questions
                "What's the difference between sadness and depression",
                "How is anxiety different from stress",
                "What's the difference between psychiatrist and psychologist",
                "Is sadness the same as depression",
                
                # Information seeking
                "Tell me about mental health treatment",
                "What are the symptoms of depression",
                "How do I know if I need therapy",
                "When should I see a therapist"
            ],
            
            'coping': [
                # Technique requests
                "How can I calm down",
                "What helps with anxiety",
                "I need coping strategies",
                "How do I deal with stress",
                "What can I do to feel better",
                "Give me techniques for managing panic",
                "How can I stop overthinking",
                "What breathing exercises help",
                "I need grounding techniques",
                "How do I relax when anxious",
                
                # Specific situations
                "How do I handle a panic attack",
                "What should I do when I feel overwhelmed",
                "How can I sleep better",
                "What helps when I'm spiraling",
                "How do I calm my racing thoughts",
                "What can I do right now to feel better",
                
                # Skill building
                "Teach me relaxation techniques",
                "I want to learn mindfulness",
                "Show me how to practice self-care",
                "Help me build coping skills"
            ]
        }

        self.reference_embeddings = {}
        total_examples = 0
        for emotion, examples in self.reference_examples.items():
            embeddings = self.model.encode(examples, show_progress_bar=False)
            self.reference_embeddings[emotion] = embeddings
            total_examples += len(examples)
            
    def classify(
        self,
        text: str,
        threshold_crisis: float = 0.68,
        threshold_high: float = 0.62,
        threshold_medium: float = 0.57,
        threshold_low: float = 0.52
    ) -> str:
        
        # STEP 1: Semantic classification
        text_embedding = self.model.encode([text], show_progress_bar=False)[0]
        
        similarities = {}
        for emotion, ref_embeddings in self.reference_embeddings.items():
            sims = cosine_similarity(
                text_embedding.reshape(1, -1),
                ref_embeddings
            )[0]
            similarities[emotion] = float(np.max(sims))
        
        max_similarity = max(similarities.values())
        
        if max_similarity < 0.50: 
            return 'educational'  
        
        if similarities['greeting'] >= 0.55:
            return 'educational'
        
        if similarities['crisis'] >= threshold_crisis:
            return 'crisis'
        
        high_intensity = ['anxiety', 'fear', 'anger']
        for emotion in high_intensity:
            if similarities[emotion] >= threshold_high:
                return emotion
        
        medium_intensity = ['sadness', 'loneliness', 'stress']
        for emotion in medium_intensity:
            if similarities[emotion] >= threshold_medium:
                return emotion
        
        if similarities['coping'] >= threshold_low:
            return 'coping'
        
        if similarities['educational'] >= threshold_low:
            return 'educational'
        
        return max(similarities, key=similarities.get)
    
    def classify_with_details(self, text: str) -> Dict:
        
        text_embedding = self.model.encode([text], show_progress_bar=False)[0]
        
        similarities = {}
        for emotion, ref_embeddings in self.reference_embeddings.items():
            sims = cosine_similarity(
                text_embedding.reshape(1, -1),
                ref_embeddings
            )[0]
            similarities[emotion] = {
                'max_similarity': float(np.max(sims)),
                'avg_similarity': float(np.mean(sims)),
                'top_3_avg': float(np.mean(np.sort(sims)[-3:]))
            }
        
        classification = self.classify(text)
        
        return {
            'classification': classification,
            'similarities': similarities,
            'confidence': similarities[classification]['max_similarity'],
            'top_3_emotions': sorted(
                [(k, v['max_similarity']) for k, v in similarities.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }
    
def create_emotion_based_dataset(
    input_file="Dataset.json",
    output_file="preprocessed_data_semantic.json"
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)


    classifier = EmotionBasedSemanticClassifier()
    
    templates = {
        'sadness': TEMPLATE_SADNESS,
        'anxiety': TEMPLATE_ANXIETY,
        'anger': TEMPLATE_ANGER,
        'loneliness': TEMPLATE_LONELINESS,
        'stress': TEMPLATE_STRESS,
        'fear': TEMPLATE_FEAR,
        'crisis': TEMPLATE_CRISIS,
        'educational': TEMPLATE_EDUCATIONAL,
        'coping': TEMPLATE_COPING
    }
    
    sample_data = []
    stats = {emotion: 0 for emotion in templates.keys()}
    stats['total'] = 0

    print(f"\nProcessing with emotion-based classification...")

    # Build flat list of all (pattern, response, tag) entries so we can
    # randomly choose which examples to display for demo purposes.
    all_entries = []
    for intent in data["intents"]:
        tag = intent.get("tag", "unknown")
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])
        if not responses:
            continue
        for pattern in patterns:
            response = random.choice(responses)
            all_entries.append((pattern, response, tag))

    num_to_demo = min(12, len(all_entries))
    demo_indices = set(random.sample(range(len(all_entries)), num_to_demo)) if num_to_demo > 0 else set()
    print(f"   (Showing {num_to_demo} random examples)\n")

    demo_shown = 0
    for idx, (pattern, response, tag) in enumerate(all_entries):
        result = classifier.classify_with_details(pattern)
        emotion = result['classification']

        if idx in demo_indices:
            print(f"Example {demo_shown + 1}:")
            print(f"  Input: \"{pattern[:55]}...\"" if len(pattern) > 55 else f"  Input: \"{pattern}\"")
            print(f"  Top 3 matches:")
            for rank, (em, score) in enumerate(result['top_3_emotions'], 1):
                marker = "→" if em == emotion else " "
                print(f"    {marker} {rank}. {em:12s}: {score:.3f}")
            print(f"  Classified as: {emotion.upper()}")
            print()
            demo_shown += 1

        if emotion == 'greeting':
            emotion = 'educational'

        template = templates[emotion]
        structured_text = template.format(
            user_input=pattern,
            assistant_response=response
        )

        sample = {
            "text": structured_text,
            "emotion": emotion,
            "original_tag": tag,
            "confidence": result['confidence']
        }

        sample_data.append(sample)
        stats[emotion] += 1
        stats['total'] += 1
    
    print(f"\nSaving to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print("EMOTION-BASED DATASET CREATED")
    print(f"\nDistribution Across 9 Emotions:")
    print(f"   Total: {stats['total']} samples\n")
    
    for emotion in templates.keys():
        count = stats[emotion]
        pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {emotion:12s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print(f"\n Output: {output_file}")
    
    return sample_data, stats

def tokenize_emotion_dataset(
    input_file="preprocessed_data_semantic.json",
    output_dir="tokenized_dataset_semantic",
    model_name="EleutherAI/gpt-neo-125M",
    max_length=768,
    test_split=0.2
):
  
    print("TOKENIZING EMOTION-BASED DATASET")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    print(f"\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    texts = [{"text": item["text"]} for item in data]
    dataset = Dataset.from_list(texts)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    print(f"Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    split_dataset = tokenized_dataset.train_test_split(test_size=test_split, seed=42)
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
        "templates": 9,
        "classification": "emotion_based_semantic"
    }
    
    with open(f"{output_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved to {output_dir}/\n")
    
    return train_dataset, val_dataset

def main():
    sample_data, stats = create_emotion_based_dataset()
    train_data, val_data = tokenize_emotion_dataset()
    
if __name__ == "__main__":
    main()