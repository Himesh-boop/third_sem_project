import json
import random
import numpy as np
import pandas as pd
import os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
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

TEMPLATE_POSITIVE = """SYSTEM: You are Freud, supporting someone experiencing positive emotions.

EMOTIONAL CONTEXT: Joy, Happiness & Positive Emotions
- Celebrate their positive feelings genuinely
- Help them savor and appreciate the moment
- Encourage them to notice what contributed to this feeling
- Validate that positive emotions are as important as difficult ones

RESPONSE APPROACH:
1. Celebrate: "That's wonderful! I'm so glad you're feeling this way."
2. Explore: "What's contributing to these positive feelings?"
3. Savor: Help them fully experience and appreciate the moment
4. Reflect: "What does this tell you about what brings you joy?"
5. Encourage: Support them in seeking more of what makes them feel good

TONE: Warm, celebratory, genuine, encouraging
AVOID: Dampening their joy, warning about it ending, toxic positivity

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_GUILT = """SYSTEM: You are Freud, supporting someone experiencing guilt or shame.

EMOTIONAL CONTEXT: Guilt, Shame & Self-Blame
- Distinguish between healthy guilt (motivates change) and toxic shame (attacks self-worth)
- Validate that guilt shows they care about their impact on others
- Help separate actions from identity - they did something wrong, they aren't wrong
- Encourage self-compassion and making amends where appropriate

RESPONSE APPROACH:
1. Validate: "Guilt can be so heavy. It shows you care about doing right."
2. Separate: "What you did isn't who you are. You're more than this moment."
3. Explore: "What's this guilt trying to tell you?"
4. Compassion: Encourage self-forgiveness and learning
5. Action: Help them consider constructive steps forward

TONE: Gentle, compassionate, non-judgmental, understanding
AVOID: Minimizing their feelings, harsh judgment, "you shouldn't feel guilty"

User: {user_input}
Assistant: {assistant_response}"""

TEMPLATE_NEUTRAL = """SYSTEM: You are Freud, having a casual, non-crisis conversation.

INTERACTION CONTEXT: Casual Conversation
- This is general chat, questions about you, or lighthearted interaction
- Be warm, personable, and human-like
- Answer questions naturally and conversationally
- Maintain therapeutic presence without being overly clinical

RESPONSE APPROACH:
1. Engage: Respond naturally to their question or comment
2. Warmth: Keep your tone friendly and approachable
3. Openness: Be willing to chat while remaining boundaried
4. Redirect: If appropriate, gently turn conversation toward their wellbeing
5. Presence: Show you're here as a supportive companion

TONE: Friendly, conversational, warm, natural
AVOID: Being robotic, overly formal, dismissive

User: {user_input}
Assistant: {assistant_response}"""

class EmotionBasedSemanticClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        self.reference_examples = {
            'greeting': [
                "hi", "hello", "hey", "hey there", "hello there",
                "good morning", "good afternoon", "good evening",
                "is anyone there", "are you there", "hello freud",
                "hi freud", "yo what's up", "just saying hi",
                "checking in", "can we talk", "are you available",
                "greetings", "howdy", "yo", "hey you"
            ],
            
            'sadness': [
                # Direct expressions
                "I feel so sad", "I'm feeling depressed",
                "I feel hopeless and empty inside",
                "Everything feels meaningless",
                "I can't stop crying", "I feel numb and disconnected",
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
                "I feel like I'm just going through the motions",
                
                # Worthlessness (EXPANDED)
                "I feel like a failure",
                "I'm not good enough",
                "I don't deserve good things",
                "I'm worthless",
                "I always mess everything up",
                "Everyone is better than me",
                "I don't matter to anyone",
                "Nothing I do is ever enough",
                "I disappoint everyone",
                "I'm such a mess",
                "Why can't I do anything right",
                "I hate myself",
                "I feel empty inside",
                "Life has lost its meaning",
                "I'm exhausted from pretending I'm okay"
            ],
            
            'anxiety': [
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
                "My heart is racing and I can't breathe",
                "I feel shaky and dizzy",
                "I can't calm down no matter what I do",
                "I feel like I'm losing control",
                "The anxiety is paralyzing me",
                "I'm anxious about being anxious",
                # Sleep/worry related (EXPANDED)
                "I'm worried about how this is affecting me",
                "I can't sleep because I'm so worried",
                "My anxiety is keeping me up at night",
                "I'm constantly on edge",
                "What if everything goes wrong",
                "I can't stop thinking about the worst case scenario",
                "My chest feels tight with worry",
                "I'm having trouble breathing from anxiety"
            ],
            
            'anger': [
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
                "Why does this always happen to me",
                "I'm tired of being treated this way",
                "No one understands how frustrating this is",
                "I'm done with people disappointing me",
                "I feel disrespected and dismissed",
                "I'm angry that no one listens",
                # Criticism/frustration (EXPANDED)
                "You're so inefficient",
                "This is completely unacceptable",
                "I'm fed up with this nonsense",
                "Why can't people just do their job",
                "I'm tired of incompetence",
                "This makes me so mad",
                "I've had enough of this",
                "I'm at my breaking point with anger"
            ],
            
            'loneliness': [
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
                "I have no real friends",
                "I don't fit in with anyone",
                "I'm always the outsider",
                "People don't reach out to me",
                "I feel disconnected from everyone around me",
                "I'm alone in this struggle",
                "I wish I had someone to talk to",
                "Nobody really knows me",
                "I feel like no one gets me"
            ],
            
            'stress': [
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
                "I have too many deadlines",
                "I'm juggling too many things at once",
                "I don't have time for anything",
                "Everything is demanding my attention",
                "I feel pulled in a million directions",
                "I can't keep up with everything",
                "I'm burnt out",
                "The pressure is crushing me"
            ],
            
            'fear': [
                "I'm scared", "I feel terrified",
                "I'm afraid of what might happen",
                "Fear is consuming me", "I'm panicking",
                "I feel like something bad is about to happen",
                "I'm frightened and don't know why",
                "I'm having a panic attack",
                "I can't shake this feeling of dread",
                "I'm paralyzed by fear", "I feel unsafe",
                "Terror is taking over",
                "I'm scared I'm going to fail",
                "I'm afraid of losing everything",
                "I'm terrified of being alone",
                "I'm scared I'm not good enough",
                "I fear the worst will happen",
                "I'm afraid I can't handle this",
                "I'm really frightened",
                "This scares me so much"
            ],
            
            'crisis': [
                # Suicidal ideation
                "I want to end my life",
                "I don't want to live anymore",
                "I'm thinking about suicide",
                "I want to kill myself",
                "I'm planning to end things",
                "I've been researching ways to die",
                "I want to die",
                "I wish I was dead",
                "I'm going to kill myself",
                
                # Hopelessness
                "Life has no meaning anymore",
                "There's no point in continuing",
                "I'm giving up on everything",
                "Nothing will ever get better",
                "I can't see any way forward",
                "Why keep going when nothing matters",
                "I have no reason to live",
                "Everything is hopeless",
                
                # Burden thoughts
                "Everyone would be better off without me",
                "I'm a burden to everyone",
                "My family deserves better than me",
                "I'm just making things worse for everyone",
                "People would be happier if I was gone",
                "I'm worthless and should just disappear",
                
                # Self-harm
                "I want to hurt myself", "I'm thinking about cutting",
                "I need to punish myself", "I deserve to suffer",
                "I've been harming myself", "I can't stop hurting myself",
                "I cut myself", "I hurt myself on purpose"
            ],
            
            'guilt': [  # NEW CATEGORY
                "I feel so guilty",
                "I'm ashamed of myself",
                "I don't deserve happiness",
                "I'm a terrible person",
                "I feel like I've let everyone down",
                "I'm so ashamed of what I did",
                "I can't forgive myself",
                "I deserve to feel this bad",
                "I've disappointed everyone",
                "I'm not worthy of love",
                "I feel like I'm a bad person",
                "I should have done better",
                "This is all my fault",
                "I'm responsible for everything going wrong",
                "I don't deserve good things",
                "I'm filled with shame",
                "I hate myself for this",
                "I can't stop blaming myself",
                "I feel like I ruined everything",
                "I'm such a bad person"
            ],
            
            'positive': [  # NEW CATEGORY
                "I'm on top of the world",
                "I'm feeling so happy",
                "I feel amazing today",
                "I'm so excited",
                "I'm feeling great",
                "Life is wonderful right now",
                "I'm feeling jolly",
                "I'm in such a good mood",
                "Everything is going well",
                "I feel fantastic",
                "I'm thrilled about this",
                "I'm feeling blessed",
                "I'm so grateful",
                "Things are looking up",
                "I'm really proud of myself",
                "I feel energized and alive",
                "I'm having a great day",
                "I feel so good",
                "Everything is awesome",
                "I'm so happy I could cry",
                "Life is good",
                "I'm in a great place",
                "I feel wonderful",
                "This is the best day ever",
                "delighted", "blissful", "rapturous", "ecstatic", 
                "thrilled", "overjoyed", "elated", "jubilant",
                "on top of the world", "feeling amazing"
            ],
            
            'neutral': [  # NEW CATEGORY
                "Do you have a fixed location",
                "What's your name",
                "How do you work",
                "Tell me about yourself",
                "What can you do",
                "Are you a real person",
                "Who created you",
                "What are your capabilities",
                "Can you help me with something",
                "I have a question",
                "Let's talk about something else",
                "I'm not interested in this topic",
                "Can we switch topics",
                "Tell me something interesting",
                "What's the weather like",
                "Do you speak other languages",
                "Bonjour", "Hola", "Namaste",
                "How's it going",
                "What's new",
                "Interesting",
                "I see",
                "Okay"
            ],
            
            'educational': [
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
                "How does therapy work",
                "What do antidepressants do",
                "Explain cognitive behavioral therapy",
                "How does EMDR work",
                "What happens in psychotherapy",
                "How do SSRIs work",
                "What's the difference between sadness and depression",
                "How is anxiety different from stress",
                "What's the difference between psychiatrist and psychologist",
                "Tell me about mental health treatment",
                "What are the symptoms of depression",
                "How do I know if I need therapy",
                "When should I see a therapist",
                "What is CBT",
                "Explain mindfulness to me",
                "What are mental health disorders"
            ],
            
            'coping': [
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
                "How do I handle a panic attack",
                "What should I do when I feel overwhelmed",
                "How can I sleep better",
                "What helps when I'm spiraling",
                "How do I calm my racing thoughts",
                "What can I do right now to feel better",
                "Teach me relaxation techniques",
                "I want to learn mindfulness",
                "Show me how to practice self-care",
                "Help me build coping skills",
                "What are some coping mechanisms",
                "How do I manage my emotions"
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
        threshold_crisis: float = 0.55,      # LOWERED from 0.68
        threshold_high: float = 0.38,        # LOWERED from 0.62
        threshold_medium: float = 0.36,      # LOWERED from 0.57
        threshold_low: float = 0.33          # LOWERED from 0.52
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
        
        if max_similarity < 0.35:  # LOWERED from 0.50
            return 'neutral' 
        
        if similarities['greeting'] >= 0.50:  # LOWERED from 0.55
            return 'neutral'
        
        if similarities['crisis'] >= threshold_crisis:
            return 'crisis'
        
        high_intensity = ['anxiety', 'fear', 'anger']
        for emotion in high_intensity:
            if similarities[emotion] >= threshold_high:
                return emotion
        
        medium_intensity = ['sadness', 'loneliness', 'stress', 'guilt']  # Added guilt
        for emotion in medium_intensity:
            if similarities[emotion] >= threshold_medium:
                return emotion
        
        if similarities['positive'] >= 0.30:
            return 'positive'
                
        if similarities['coping'] >= threshold_low:
            return 'coping'
        
        if similarities['educational'] >= threshold_low:
            return 'educational'

        if similarities['neutral'] >= threshold_low:
            return 'neutral'
        
        best_match = max(similarities, key=similarities.get)
        
        # Prefer emotional categories over neutral/educational for ambiguous cases
        emotional_categories = ['sadness', 'anxiety', 'anger', 'loneliness', 
                                'stress', 'fear', 'guilt', 'positive']
        emotional_scores = {k: v for k, v in similarities.items() 
                           if k in emotional_categories}
        
        if emotional_scores and max(emotional_scores.values()) > 0.35:
            return max(emotional_scores, key=emotional_scores.get)
        
        return best_match
    
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
        'guilt': TEMPLATE_GUILT,        # NEW
        'positive': TEMPLATE_POSITIVE,  # NEW
        'neutral': TEMPLATE_NEUTRAL,    # NEW
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
        raw_system = templates[emotion].split("User: ")[0]
        system = raw_system.replace("SYSTEM:", "").strip()
        user = templates[emotion].split("User: ")[1].split("Assistant: ")[0]
        assistant = templates[emotion].split("Assistant: ")[1]

        sample = {
            "SYSTEM": system,
            "User" : user.replace("{user_input}", pattern),
            "Assistant": assistant.replace("{assistant_response}", response),
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

class DatasetValidator:
    """Validates the quality of emotion-classified dataset"""
    
    def __init__(self, dataset_file: str = "preprocessed_data_semantic.json"):
        with open(dataset_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.classifier = EmotionBasedSemanticClassifier()
    
    # ===================================================================
    # VALIDATION METHOD 1: Distribution Analysis
    # ===================================================================
    def check_distribution(self):
        """Check if emotion categories are balanced"""
        print("\n" + "="*60)
        print("VALIDATION 1: EMOTION DISTRIBUTION")
        print("="*60)
        
        emotions = [item['emotion'] for item in self.data]
        distribution = Counter(emotions)
        total = len(emotions)
        
        print(f"\nTotal samples: {total}\n")
        print(f"{'Emotion':<15} {'Count':<8} {'%':<8} {'Status'}")
        print("-" * 50)
        
        issues = []
        for emotion, count in sorted(distribution.items(), key=lambda x: -x[1]):
            pct = (count / total) * 100
            
            # Flag issues
            status = "✓ OK"
            if count < 50:
                status = "⚠ LOW (need more examples)"
                issues.append(f"{emotion}: only {count} samples")
            elif count > total * 0.3:
                status = "⚠ HIGH (may dominate)"
                issues.append(f"{emotion}: {count} samples ({pct:.1f}%)")
            
            print(f"{emotion:<15} {count:<8} {pct:>5.1f}%   {status}")
        
        if issues:
            print("\n⚠ DISTRIBUTION ISSUES:")
            for issue in issues:
                print(f"  • {issue}")
            print("\n💡 Recommendation: Balance by adding/removing samples")
        else:
            print("\n✅ Distribution looks good!")
        
        return distribution
    
    # ===================================================================
    # VALIDATION METHOD 2: Confidence Analysis
    # ===================================================================
    def check_confidence_scores(self, low_threshold: float = 0.45):
        """Check classification confidence scores"""
        print("\n" + "="*60)
        print("VALIDATION 2: CLASSIFICATION CONFIDENCE")
        print("="*60)
        
        confidences = [item['confidence'] for item in self.data]
        
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        low_confidence = [item for item in self.data if item['confidence'] < low_threshold]
        
        print(f"\nConfidence Statistics:")
        print(f"  Average: {avg_conf:.3f}")
        print(f"  Minimum: {min_conf:.3f}")
        print(f"  Maximum: {max_conf:.3f}")
        print(f"  Samples < {low_threshold}: {len(low_confidence)} ({len(low_confidence)/len(self.data)*100:.1f}%)")
        
        if low_confidence:
            print(f"\n⚠ {len(low_confidence)} LOW CONFIDENCE SAMPLES (may be misclassified):")
            print("\nShowing 5 random examples:\n")
            
            for item in random.sample(low_confidence, min(5, len(low_confidence))):
                user_input = item['User'].replace("{user_input}", "").strip()
                print(f"  Input: \"{user_input[:60]}...\"" if len(user_input) > 60 else f"  Input: \"{user_input}\"")
                print(f"  Classified as: {item['emotion'].upper()}")
                print(f"  Confidence: {item['confidence']:.3f}")
                print(f"  Original tag: {item['original_tag']}")
                print()
            
            print("💡 Recommendation: Manually review these samples")
        else:
            print("\n✅ All samples have good confidence!")
        
        return low_confidence
    
    # ===================================================================
    # VALIDATION METHOD 3: Manual Inspection with Random Sampling
    # ===================================================================
    def manual_inspection(self, num_samples: int = 20):
        """Present random samples for human review"""
        print("\n" + "="*60)
        print("VALIDATION 3: MANUAL INSPECTION")
        print("="*60)
        
        print(f"\nReviewing {num_samples} random samples...")
        print("For each sample, verify if classification makes sense.\n")
        
        samples = random.sample(self.data, min(num_samples, len(self.data)))
        
        for i, item in enumerate(samples, 1):
            user_input = item['User'].replace("{user_input}", "").strip()
            
            print(f"Sample {i}/{num_samples}")
            print(f"  Input: \"{user_input}\"")
            print(f"  Classified as: {item['emotion'].upper()}")
            print(f"  Confidence: {item['confidence']:.3f}")
            print(f"  Original intent: {item['original_tag']}")
            
            # Re-classify to show alternatives
            result = self.classifier.classify_with_details(user_input)
            print(f"  Top 3 matches:")
            for rank, (em, score) in enumerate(result['top_3_emotions'], 1):
                marker = "→" if em == item['emotion'] else " "
                print(f"    {marker} {rank}. {em:<12}: {score:.3f}")
            
            print()
    
    # ===================================================================
    # VALIDATION METHOD 4: Edge Case Testing
    # ===================================================================
    def test_edge_cases(self):
        """Test problematic inputs"""
        print("\n" + "="*60)
        print("VALIDATION 4: EDGE CASE TESTING")
        print("="*60)
        
        edge_cases = {
            'crisis_implicit': [
                "I don't deserve happiness",
                "Everyone would be better off without me",
                "I'm a burden to everyone"
            ],
            'positive': [
                "I'm on top of the world",
                "I'm feeling so jolly",
                "Life is wonderful"
            ],
            'neutral': [
                "Bonjour",
                "What's your name",
                "Do you have a fixed location"
            ],
            'ambiguous': [
                "I'm worried about how this lack of sleep is affecting me",
                "You're so inefficient to work with",
                "I'm not invested in this topic"
            ]
        }
        
        print("\nTesting edge cases...\n")
        
        for category, inputs in edge_cases.items():
            print(f"{category.upper()}:")
            for inp in inputs:
                result = self.classifier.classify_with_details(inp)
                classification = result['classification']
                confidence = result['confidence']
                
                status = "✓" if confidence > 0.45 else "⚠"
                print(f"  {status} \"{inp[:50]}...\"" if len(inp) > 50 else f"  {status} \"{inp}\"")
                print(f"     → {classification} (conf: {confidence:.3f})")
            print()
    
    # ===================================================================
    # VALIDATION METHOD 5: Misclassification Detection
    # ===================================================================
    def detect_misclassifications(self):
        """Find likely misclassifications using secondary emotion"""
        print("\n" + "="*60)
        print("VALIDATION 5: MISCLASSIFICATION DETECTION")
        print("="*60)
        
        print("\nLooking for samples where 2nd choice is very close to 1st...\n")
        
        suspicious = []
        
        for item in self.data:
            user_input = item['User'].replace("{user_input}", "").strip()
            result = self.classifier.classify_with_details(user_input)
            
            top_2 = result['top_3_emotions'][:2]
            if len(top_2) == 2:
                first_emotion, first_score = top_2[0]
                second_emotion, second_score = top_2[1]
                
                # If difference is < 0.05, classification is uncertain
                if abs(first_score - second_score) < 0.05:
                    suspicious.append({
                        'input': user_input,
                        'classified': item['emotion'],
                        'first': (first_emotion, first_score),
                        'second': (second_emotion, second_score),
                        'diff': first_score - second_score
                    })
        
        if suspicious:
            print(f"Found {len(suspicious)} uncertain classifications:\n")
            
            for item in random.sample(suspicious, min(10, len(suspicious))):
                print(f"  Input: \"{item['input'][:55]}...\"" if len(item['input']) > 55 else f"  Input: \"{item['input']}\"")
                print(f"  Classified as: {item['classified']}")
                print(f"  But could be: {item['second'][0]} (diff: {item['diff']:.3f})")
                print(f"    • {item['first'][0]}: {item['first'][1]:.3f}")
                print(f"    • {item['second'][0]}: {item['second'][1]:.3f}")
                print()
            
            print("💡 Recommendation: Review these and consider:")
            print("   - Adding more diverse reference examples")
            print("   - Creating sub-categories (e.g., guilt separate from sadness)")
            print("   - Manual relabeling if obviously wrong")
        else:
            print("✅ No suspicious classifications found!")
        
        return suspicious
    
    # ===================================================================
    # VALIDATION METHOD 6: Template Coverage Check
    # ===================================================================
    def check_template_coverage(self):
        """Ensure all emotions have sufficient template examples"""
        print("\n" + "="*60)
        print("VALIDATION 6: REFERENCE EXAMPLE COVERAGE")
        print("="*60)
        
        print("\nReference examples per emotion:\n")
        
        for emotion, examples in self.classifier.reference_examples.items():
            count = len(examples)
            status = "✓ Good" if count >= 15 else "⚠ Need more"
            print(f"  {emotion:<15}: {count:>3} examples - {status}")
        
        print("\n💡 Recommendation: Each emotion should have 15-30 diverse examples")
    
    # ===================================================================
    # RUN ALL VALIDATIONS
    # ===================================================================
    def run_all_validations(self):
        """Run complete validation suite"""
        print("\n" + "="*70)
        print(" "*15 + "DATASET QUALITY VALIDATION REPORT")
        print("="*70)
        
        self.check_distribution()
        low_conf = self.check_confidence_scores()
        self.manual_inspection(num_samples=10)
        self.test_edge_cases()
        suspicious = self.detect_misclassifications()
        self.check_template_coverage()
        
        # Summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        issues_found = []
        if low_conf:
            issues_found.append(f"{len(low_conf)} low confidence samples")
        if suspicious:
            issues_found.append(f"{len(suspicious)} uncertain classifications")
        
        if not issues_found:
            print("\n✅ DATASET QUALITY: EXCELLENT")
            print("   No major issues found. Ready for training!")
        else:
            print("\n⚠ ISSUES FOUND:")
            for issue in issues_found:
                print(f"   • {issue}")
            print("\n💡 Review flagged samples and iterate on:")
            print("   1. Add more reference examples for underrepresented emotions")
            print("   2. Fine-tune thresholds if needed")
            print("   3. Manually relabel obvious misclassifications")
       
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
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


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
        "templates": 9,
        "classification": "emotion_based_semantic",
        "label_masking": "assistant_only",
        "fields_used": ["SYSTEM", "User", "Assistant"]
    }

    with open(f"{output_dir}/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved to {output_dir}/\n")

    return train_dataset, val_dataset


def main():
    sample_data, stats = create_emotion_based_dataset()
    validator = DatasetValidator()
    validator.run_all_validations()
    train_data, val_data = tokenize_emotion_dataset()
    
if __name__ == "__main__":
    main()