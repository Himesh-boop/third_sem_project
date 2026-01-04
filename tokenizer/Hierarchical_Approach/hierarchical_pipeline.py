import json
import random
import os
from collections import Counter
from hierarchical_emotion_classifier import HierarchicalEmotionClassifier

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

def load_templates():
    """Load all emotion templates"""
    return {
        'positive': TEMPLATE_POSITIVE,
        'sadness': TEMPLATE_SADNESS,
        'anxiety': TEMPLATE_ANXIETY,
        'anger': TEMPLATE_ANGER,
        'fear': TEMPLATE_FEAR,
        'loneliness': TEMPLATE_LONELINESS,
        'stress': TEMPLATE_STRESS,
        'guilt': TEMPLATE_GUILT,
        'crisis': TEMPLATE_CRISIS,
        'neutral': TEMPLATE_NEUTRAL,
        'educational': TEMPLATE_EDUCATIONAL,
        'coping': TEMPLATE_COPING,
        'greeting': TEMPLATE_NEUTRAL,
        'conversation_control': TEMPLATE_NEUTRAL,
        'feedback': TEMPLATE_POSITIVE
    }


def create_hierarchical_dataset(
    input_file="Dataset.json",
    output_file="preprocessed_data_hierarchical.json",
    rejected_file="rejected_data_hierarchical.json"
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found!")
    
    print(f"\nLoading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("\nInitializing hierarchical classifier...")
    classifier = HierarchicalEmotionClassifier()
    
    templates = load_templates()
    print(f"Loaded {len(templates)} templates")
    
    sample_data = []
    rejected_data = []
    stats = {
        'total': 0,
        'accepted': 0,
        'rejected': 0,
        'by_polarity': Counter(),
        'by_emotion': Counter(),
        'by_intent': Counter(),
        'by_final_category': Counter(),
        'uncertain': 0
    }
    
    all_entries = []
    for intent in data.get("intents", []):
        tag = intent.get("tag", "unknown")
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])
        
        if not responses:
            continue
            
        for pattern in patterns:
            response = random.choice(responses)
            all_entries.append((pattern, response, tag))
    
    print(f"\nProcessing {len(all_entries)} conversation pairs...")
    
    for idx, (pattern, response, tag) in enumerate(all_entries):
        try:
            result = classifier.classify(pattern)
            
            reject_reason = None
            
            if not result.is_certain:
                if result.polarity == 'NEUTRAL':
                    result.final_category = 'neutral'
                else:
                    reject_reason = "low_confidence_emotion"
            
            template_key = result.final_category
            
            if template_key not in templates:
                if result.polarity == 'POSITIVE':
                    template_key = 'positive'
                elif result.polarity == 'NEGATIVE':
                    template_key = 'sadness'
                else:
                    template_key = 'neutral'
            
            template_full = templates.get(template_key, templates['neutral'])
            
            parts = template_full.split("User: ")
            if len(parts) == 2:
                system_part = parts[0].replace("SYSTEM: ", "").strip()
                rest = parts[1]
                user_assistant = rest.split("Assistant: ")
                if len(user_assistant) == 2:
                    user_template = user_assistant[0].strip()
                    assistant_template = user_assistant[1].strip()
                else:
                    user_template = "{user_input}"
                    assistant_template = "{assistant_response}"
            else:
                system_part = template_full
                user_template = "{user_input}"
                assistant_template = "{assistant_response}"
            
            sample = {
                "SYSTEM": system_part,
                "User": user_template.replace("{user_input}", pattern),
                "Assistant": assistant_template.replace("{assistant_response}", response),
                "polarity": result.polarity,
                "polarity_confidence": result.polarity_confidence,
                "emotion": result.emotion,
                "emotion_confidence": result.emotion_confidence,
                "intent": result.intent,
                "intent_confidence": result.intent_confidence,
                "final_category": result.final_category,
                "is_certain": result.is_certain,
                "original_tag": tag,
                "reasoning": result.reasoning
            }
            
            if reject_reason:
                sample['reject_reason'] = reject_reason
                rejected_data.append(sample)
                stats['rejected'] += 1
            else:
                sample_data.append(sample)
                stats['accepted'] += 1
                stats['by_polarity'][result.polarity] += 1
                if result.emotion:
                    stats['by_emotion'][result.emotion] += 1
                if result.intent:
                    stats['by_intent'][result.intent] += 1
                stats['by_final_category'][result.final_category] += 1
                if not result.is_certain:
                    stats['uncertain'] += 1
            
            stats['total'] += 1
            
            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{len(all_entries)}")
        
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            rejected_data.append({
                "User": pattern,
                "Assistant": response,
                "original_tag": tag,
                "reject_reason": "processing_error",
                "error": str(e)
            })
            stats['rejected'] += 1
            continue
    
    print(f"\nSaving {len(sample_data)} accepted samples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saving {len(rejected_data)} rejected samples to {rejected_file}...")
    with open(rejected_file, "w", encoding="utf-8") as f:
        json.dump(rejected_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("HIERARCHICAL DATASET CREATED")
    print("="*80)
    
    print(f"\nTotal processed: {stats['total']}")
    print(f"Accepted: {stats['accepted']} ({stats['accepted']/stats['total']*100:.1f}%)")
    print(f"Rejected: {stats['rejected']} ({stats['rejected']/stats['total']*100:.1f}%)")
    print(f"Uncertain in accepted: {stats['uncertain']} ({stats['uncertain']/stats['accepted']*100:.1f}%)")
    
    print("\nAccepted Polarity Distribution:")
    for polarity, count in stats['by_polarity'].most_common():
        pct = count / stats['accepted'] * 100
        print(f"  {polarity:10s}: {count:4d} ({pct:5.1f}%)")
    
    print("\nAccepted Emotion Distribution:")
    for emotion, count in stats['by_emotion'].most_common():
        pct = count / stats['accepted'] * 100
        print(f"  {emotion:12s}: {count:4d} ({pct:5.1f}%)")
    
    if stats['by_intent']:
        print("\nAccepted Intent Distribution:")
        for intent, count in stats['by_intent'].most_common():
            pct = count / stats['accepted'] * 100
            print(f"  {intent:18s}: {count:4d} ({pct:5.1f}%)")
    
    print("\nAccepted Final Categories:")
    for category, count in stats['by_final_category'].most_common():
        pct = count / stats['accepted'] * 100
        print(f"  {category:18s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nAccepted data saved to: {output_file}")
    print(f"Rejected data saved to: {rejected_file}")
    
    return sample_data, rejected_data, stats


def main():
    import sys
    if not os.path.exists("Dataset.json"):
        print("\nDataset.json not found!")
        sys.exit(1)
    
    try:
        create_hierarchical_dataset(
            input_file="Dataset.json",
            output_file="preprocessed_data_hierarchical.json",
            rejected_file="rejected_data_hierarchical.json"
        )
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()