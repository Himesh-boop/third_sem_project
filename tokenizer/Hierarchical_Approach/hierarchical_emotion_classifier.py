import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

@dataclass
class ClassificationResult:
    """Complete classification result with all layers"""
    polarity: str
    polarity_confidence: float
    emotion: Optional[str]
    emotion_confidence: Optional[float]
    intent: Optional[str]
    intent_confidence: Optional[float]
    final_category: str
    is_certain: bool
    reasoning: str


class HierarchicalEmotionClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
        self.polarity_anchors = self._load_polarity_anchors()
        self.emotion_anchors = self._load_emotion_anchors()
        self.intent_anchors = self._load_intent_anchors()
        
        self.polarity_embeddings = self._compute_embeddings(self.polarity_anchors)
        self.emotion_embeddings = self._compute_embeddings(self.emotion_anchors)
        self.intent_embeddings = self._compute_embeddings(self.intent_anchors)
        
        self.polarity_threshold = 0.45
        self.emotion_threshold = 0.40
        self.intent_threshold = 0.42
        self.margin_threshold = 0.08
        
        self.crisis_keywords = [
            "suicide",
            "kill myself",
            "end my life",
            "want to die",
            "better off dead",
            "self harm",
            "hurt myself",
            "no reason to live",
            "better off without me",
            "i am a burden",
            "life is not worth living",
            "nothing to live for"
        ]
    
    def _load_polarity_anchors(self) -> Dict[str, List[str]]:
        """Polarity detection using emotional anchor sentences"""
        return {
            'POSITIVE': [
                "I feel joyful and happy",
                "I am excited about this",
                "This makes me feel wonderful",
                "I feel blessed and grateful",
                "Everything feels amazing",
                "I am thrilled and delighted",
                "I feel fantastic today",
                "This brings me so much happiness",
                "I feel energized and alive",
                "I am having a great time",
                "I feel content and peaceful",
                "This makes me smile",
                "I feel proud of myself",
                "I am enjoying this moment",
                "I feel optimistic about the future",
                "I'm on top of the world",
                "Life is wonderful right now",
                "I feel ecstatic",
                "Everything is going well",
                "I am so happy I could cry"
            ],
            
            'NEGATIVE': [
                "I feel deeply sad and hurt",
                "I am overwhelmed by sadness",
                "I feel hopeless and empty",
                "Everything feels meaningless",
                "I am so angry and frustrated",
                "I feel terrified and scared",
                "I am anxious and worried",
                "I feel lonely and isolated",
                "I am stressed and overwhelmed",
                "I feel guilty and ashamed",
                "Nothing brings me joy anymore",
                "I feel like a failure",
                "I am drowning in despair",
                "I feel broken inside",
                "I can't handle this anymore",
                "Everything is falling apart",
                "I feel worthless",
                "I am in so much pain",
                "I can't see a way forward",
                "I feel like giving up"
            ],
            
            'NEUTRAL': [
                "I have a question",
                "Can you help me with something",
                "I want to know more about this",
                "Tell me about your capabilities",
                "What can you do",
                "I'm just checking in",
                "Let's talk about something",
                "I want to change the topic",
                "Can we discuss something else",
                "I'm here to chat",
                "What's your name",
                "How do you work",
                "I need information",
                "Can you explain this",
                "I'm curious about something"
            ]
        }
    
    def _load_emotion_anchors(self) -> Dict[str, List[str]]:
        """Negative emotion anchors"""
        return {
            'sadness': [
                "I feel deep sorrow and grief",
                "I am overwhelmed by sadness",
                "I feel hopeless about everything",
                "Nothing brings me joy",
                "I feel empty and numb inside",
                "I am mourning and heartbroken",
                "Life feels colorless",
                "I feel heavy with despair",
                "I lost something precious",
                "I am drowning in sadness"
            ],
            
            'anxiety': [
                "I feel extremely anxious",
                "I am constantly worried",
                "My mind races with fear",
                "I feel panicky and on edge",
                "I can't stop worrying",
                "I feel nervous about everything",
                "I am terrified something bad will happen",
                "I feel like I'm losing control",
                "My anxiety is overwhelming",
                "I can't calm down"
            ],
            
            'anger': [
                "I feel furious and enraged",
                "I am extremely angry",
                "I feel rage building inside",
                "I am fed up with everything",
                "I feel frustrated and irritated",
                "I want to scream in anger",
                "I am sick of this situation",
                "I feel disrespected and dismissed",
                "I am done with this nonsense",
                "I feel infuriated"
            ],
            
            'fear': [
                "I feel terrified and scared",
                "I am paralyzed by fear",
                "I feel unsafe and threatened",
                "I am frightened of what might happen",
                "Fear is consuming me",
                "I feel dread and terror",
                "I am scared for my safety",
                "I feel panic taking over",
                "I am afraid of everything",
                "Fear grips me"
            ],
            
            'loneliness': [
                "I feel profoundly alone",
                "Nobody understands me",
                "I feel isolated from everyone",
                "I have no one to turn to",
                "I feel abandoned and forgotten",
                "I am completely alone",
                "I feel disconnected from the world",
                "Nobody cares about me",
                "I feel invisible to others",
                "I have no real connections"
            ],
            
            'stress': [
                "I feel overwhelmed by pressure",
                "I am drowning in stress",
                "Everything is too much to handle",
                "I am burnt out and exhausted",
                "I feel crushed by responsibilities",
                "I can't keep up with everything",
                "I am under immense pressure",
                "I feel like I'm breaking",
                "I am completely overwhelmed",
                "The stress never ends"
            ],
            
            'guilt': [
                "I feel terrible guilt",
                "I am ashamed of myself",
                "I feel like a bad person",
                "I don't deserve happiness",
                "I feel guilty for what I did",
                "I am filled with shame",
                "I can't forgive myself",
                "I feel like I let everyone down",
                "I deserve to feel this bad",
                "I am responsible for everything wrong"
            ],
            
            'crisis': [
                "I want to end my life",
                "I don't want to live anymore",
                "Everyone would be better off without me",
                "I am planning to hurt myself",
                "I see no reason to continue",
                "I want to die",
                "I am going to kill myself",
                "Life has no meaning anymore",
                "I am a burden to everyone",
                "I can't go on living"
            ]
        }
    
    def _load_intent_anchors(self) -> Dict[str, List[str]]:
        """Intent classification anchors"""
        return {
            'greeting': [
                "Hello, how are you",
                "Hi there, I wanted to say hello",
                "Good morning, I'm checking in",
                "Hey, is anyone there",
                "Greetings, I'm here to talk"
            ],
            
            'educational': [
                "What is depression and how does it work",
                "Can you explain anxiety to me",
                "Tell me about cognitive behavioral therapy",
                "I want to understand mental health",
                "What are the symptoms of PTSD",
                "How does therapy help people",
                "Explain what mindfulness means"
            ],
            
            'coping': [
                "How can I calm down right now",
                "What techniques help with anxiety",
                "I need coping strategies for stress",
                "What can I do to feel better",
                "Give me breathing exercises",
                "How do I handle a panic attack",
                "What are grounding techniques"
            ],
            
            'conversation_control': [
                "Let's change the subject",
                "I don't want to talk about this",
                "Can we discuss something else",
                "I'm not interested in this topic",
                "Let's move on",
                "I'd rather talk about something different"
            ],
            
            'feedback': [
                "Thank you for your help",
                "This has been very helpful",
                "I appreciate your support",
                "Your advice is working",
                "I'm grateful for this conversation"
            ]
        }
    
    def _compute_embeddings(self, anchors: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """Compute average embeddings for each category"""
        embeddings = {}
        for category, sentences in anchors.items():
            sentence_embeddings = self.model.encode(sentences, show_progress_bar=False)
            avg_embedding = np.mean(sentence_embeddings, axis=0)
            embeddings[category] = avg_embedding
        return embeddings
    
    def detect_polarity(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Determine if input is positive/negative/neutral"""
        text_embedding = self.model.encode([text], show_progress_bar=False)[0]
        
        scores = {}
        for polarity, embedding in self.polarity_embeddings.items():
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            scores[polarity] = float(similarity)
        
        top_polarity = max(scores, key=scores.get)
        confidence = scores[top_polarity]
        
        return top_polarity, confidence, scores
    
    def classify_emotion(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Classify negative emotion"""
        text_embedding = self.model.encode([text], show_progress_bar=False)[0]
        
        scores = {}
        for emotion, embedding in self.emotion_embeddings.items():
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            scores[emotion] = float(similarity)
        
        top_emotion = max(scores, key=scores.get)
        confidence = scores[top_emotion]
        
        return top_emotion, confidence, scores
    
    def classify_intent(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """Classify user intent"""
        text_embedding = self.model.encode([text], show_progress_bar=False)[0]
        
        scores = {}
        for intent, embedding in self.intent_embeddings.items():
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            scores[intent] = float(similarity)
        
        top_intent = max(scores, key=scores.get)
        confidence = scores[top_intent]
        
        if confidence >= self.intent_threshold:
            return top_intent, confidence
        
        return None, None
    
    def check_crisis(self, text: str) -> bool:
        """Check for crisis keywords with self-reference"""
        text_lower = text.lower()
        
        if not re.search(r"\b(i|me|my|myself)\b", text_lower):
            return False
        
        return any(keyword in text_lower for keyword in self.crisis_keywords)
    
    def is_confident(self, scores: Dict[str, float], text: str) -> bool:
        """Check if classification confidence meets threshold"""
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) < 2:
            return True
        
        margin = sorted_scores[0] - sorted_scores[1]
        
        if len(text.split()) <= 3:
            return margin >= 0.05
        
        return margin >= self.margin_threshold
    
    def quick_intent_override(self, text: str):
        """Quick pattern matching for common greetings"""
        text = text.lower().strip()
        
        greeting_patterns = [
            r"^hi\b",
            r"^hey\b",
            r"^hello\b",
            r"^hola\b",
            r"^namaste\b",
            r"^bonjour\b",
            r"^ciao\b",
            r"^hey there\b",
            r"^hello there\b"
        ]
        
        for pattern in greeting_patterns:
            if re.match(pattern, text):
                return "greeting"
        
        return None
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Complete hierarchical classification pipeline:
        1. Crisis check
        2. Polarity detection
        3. Emotion classification (if negative)
        4. Intent classification
        5. Confidence gating
        6. Final routing
        """
        quick_intent = self.quick_intent_override(text)
        if quick_intent:
            return ClassificationResult(
                polarity='NEUTRAL',
                polarity_confidence=1.0,
                emotion=None,
                emotion_confidence=None,
                intent=quick_intent,
                intent_confidence=1.0,
                final_category=quick_intent,
                is_certain=True,
                reasoning="Rule-based greeting override"
            )
        
        if self.check_crisis(text):
            return ClassificationResult(
                polarity='NEGATIVE',
                polarity_confidence=1.0,
                emotion='crisis',
                emotion_confidence=1.0,
                intent=None,
                intent_confidence=None,
                final_category='crisis',
                is_certain=True,
                reasoning="Crisis keywords detected - immediate intervention"
            )
        
        polarity, polarity_conf, polarity_scores = self.detect_polarity(text)
        polarity_certain = self.is_confident(polarity_scores, text)
        
        intent, intent_conf = self.classify_intent(text)
        
        if intent == 'educational' and intent_conf and intent_conf >= 0.55:
            return ClassificationResult(
                polarity='NEUTRAL',
                polarity_confidence=polarity_conf,
                emotion=None,
                emotion_confidence=None,
                intent='educational',
                intent_confidence=intent_conf,
                final_category='educational',
                is_certain=True,
                reasoning="Educational intent overrides emotional polarity"
            )
        
        if polarity == 'POSITIVE':
            return ClassificationResult(
                polarity='POSITIVE',
                polarity_confidence=polarity_conf,
                emotion=None,
                emotion_confidence=None,
                intent=intent,
                intent_confidence=intent_conf,
                final_category='positive',
                is_certain=polarity_certain,
                reasoning=f"Positive polarity detected ({polarity_conf:.3f})"
            )
        
        elif polarity == 'NEUTRAL':
            if intent:
                final_cat = intent
                reasoning = f"Neutral polarity, intent: {intent}"
            else:
                final_cat = 'neutral'
                reasoning = "Neutral polarity, no strong intent"
            
            return ClassificationResult(
                polarity='NEUTRAL',
                polarity_confidence=polarity_conf,
                emotion=None,
                emotion_confidence=None,
                intent=intent,
                intent_confidence=intent_conf,
                final_category=final_cat,
                is_certain=polarity_certain,
                reasoning=reasoning
            )
        
        else:
            emotion, emotion_conf, emotion_scores = self.classify_emotion(text)
            
            if emotion == 'crisis' and not re.search(r"\b(i|me|my|myself)\b", text.lower()):
                emotion = max(
                    {k: v for k, v in emotion_scores.items() if k != 'crisis'},
                    key=emotion_scores.get
                )
                emotion_conf = emotion_scores[emotion]
            
            emotion_certain = self.is_confident(emotion_scores, text)
            
            if emotion_conf < self.emotion_threshold:
                return ClassificationResult(
                    polarity='NEGATIVE',
                    polarity_confidence=polarity_conf,
                    emotion=emotion,
                    emotion_confidence=emotion_conf,
                    intent=intent,
                    intent_confidence=intent_conf,
                    final_category='sadness',
                    is_certain=False,
                    reasoning=f"Negative polarity but uncertain emotion ({emotion_conf:.3f})"
                )
            
            if intent == 'coping' and intent_conf and intent_conf > emotion_conf:
                final_cat = 'coping'
                reasoning = f"Coping intent overrides emotion"
            else:
                final_cat = emotion
                reasoning = f"Negative polarity -> {emotion} emotion"
            
            return ClassificationResult(
                polarity='NEGATIVE',
                polarity_confidence=polarity_conf,
                emotion=emotion,
                emotion_confidence=emotion_conf,
                intent=intent,
                intent_confidence=intent_conf,
                final_category=final_cat,
                is_certain=polarity_certain and emotion_certain,
                reasoning=reasoning
            )
    
    def classify_with_details(self, text: str) -> Dict:
        """Full classification with all scores for analysis"""
        result = self.classify(text)
        
        _, _, polarity_scores = self.detect_polarity(text)
        
        emotion_scores = {}
        if result.polarity == 'NEGATIVE':
            _, _, emotion_scores = self.classify_emotion(text)
        
        return {
            'result': result,
            'polarity_scores': polarity_scores,
            'emotion_scores': emotion_scores,
            'text': text
        }


if __name__ == "__main__":
    classifier = HierarchicalEmotionClassifier()
    
    test_cases = [
        "I'm feeling so blissful!",
        "I'm feeling ecstatic!",
        "Life is wonderful right now",
        "I'm on top of the world",
        "I feel so sad and alone",
        "I'm anxious about everything",
        "I'm so angry right now",
        "I want to end my life",
        "Everyone would be better off without me",
        "What is depression?",
        "How can I calm down?",
        "Hello, how are you?",
        "You're not worth it",
        "I'm feeling overwhelmed"
    ]
    
    for text in test_cases:
        print(f"\nInput: \"{text}\"")
        result = classifier.classify(text)
        
        print(f"Polarity: {result.polarity} ({result.polarity_confidence:.3f})")
        if result.emotion:
            print(f"Emotion: {result.emotion} ({result.emotion_confidence:.3f})")
        if result.intent:
            print(f"Intent: {result.intent} ({result.intent_confidence:.3f})")
        print(f"Final Category: {result.final_category}")
        print(f"Certain: {'Yes' if result.is_certain else 'No (Uncertain)'}")
        print(f"Reasoning: {result.reasoning}")
        print("-" * 70)