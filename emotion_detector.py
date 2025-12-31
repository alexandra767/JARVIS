"""
Alexandra AI - Emotion Detection
Analyzes text sentiment to select appropriate avatar expressions
"""

import re

class EmotionDetector:
    """Detect emotions from text to select avatar expressions"""

    def __init__(self):
        # Emotion keywords
        self.positive_words = {
            'happy', 'great', 'wonderful', 'excellent', 'amazing', 'love',
            'fantastic', 'awesome', 'good', 'nice', 'beautiful', 'joy',
            'excited', 'glad', 'pleased', 'delighted', 'thankful', 'grateful'
        }

        self.negative_words = {
            'sad', 'sorry', 'unfortunately', 'bad', 'terrible', 'awful',
            'horrible', 'disappointed', 'upset', 'worried', 'concerned',
            'difficult', 'problem', 'issue', 'wrong', 'error', 'fail'
        }

        self.question_indicators = {'?', 'what', 'how', 'why', 'when', 'where', 'who', 'which'}

        self.excitement_words = {
            'wow', 'amazing', 'incredible', 'awesome', 'fantastic',
            'excited', 'thrilling', 'extraordinary', '!'
        }

        self.thinking_words = {
            'think', 'consider', 'perhaps', 'maybe', 'possibly',
            'wonder', 'interesting', 'hmm', 'let me see'
        }

    def detect(self, text):
        """
        Analyze text and return detected emotion

        Returns: dict with 'emotion', 'confidence', 'avatar_suggestion'
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        # Count matches
        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)
        excitement_count = len(words & self.excitement_words)
        thinking_count = len(words & self.thinking_words)

        # Check for questions
        is_question = '?' in text or any(q in text_lower for q in ['what ', 'how ', 'why ', 'when ', 'where ', 'who ', 'which '])

        # Check for exclamation (excitement)
        exclamation_count = text.count('!')

        # Determine emotion
        scores = {
            'positive': positive_count + (exclamation_count * 0.5 if positive_count > 0 else 0),
            'negative': negative_count,
            'excited': excitement_count + exclamation_count,
            'thinking': thinking_count + (1 if is_question else 0),
            'neutral': 1  # Base score
        }

        # Get highest scoring emotion
        emotion = max(scores, key=scores.get)
        confidence = min(scores[emotion] / 5.0, 1.0)  # Normalize to 0-1

        # Map to avatar
        avatar_map = {
            'positive': 'happy',
            'negative': 'concerned',
            'excited': 'happy',
            'thinking': 'thinking',
            'neutral': 'default'
        }

        return {
            'emotion': emotion,
            'confidence': confidence,
            'avatar_suggestion': avatar_map.get(emotion, 'default'),
            'scores': scores
        }

    def get_voice_modifier(self, emotion):
        """
        Get voice modification parameters based on emotion
        (For future use with more advanced TTS)
        """
        modifiers = {
            'positive': {'pitch': 1.05, 'speed': 1.02, 'energy': 1.1},
            'negative': {'pitch': 0.98, 'speed': 0.95, 'energy': 0.9},
            'excited': {'pitch': 1.1, 'speed': 1.1, 'energy': 1.2},
            'thinking': {'pitch': 1.0, 'speed': 0.9, 'energy': 0.95},
            'neutral': {'pitch': 1.0, 'speed': 1.0, 'energy': 1.0}
        }
        return modifiers.get(emotion, modifiers['neutral'])


# Advanced emotion detection using VADER (if available)
class AdvancedEmotionDetector(EmotionDetector):
    """Uses VADER sentiment analysis for more accurate detection"""

    def __init__(self):
        super().__init__()
        self.vader = None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except ImportError:
            print("[EmotionDetector] VADER not available, using basic detection")

    def detect(self, text):
        """Enhanced detection with VADER"""
        if not self.vader:
            return super().detect(text)

        # Get VADER scores
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']

        # Determine emotion from compound score
        if compound >= 0.5:
            emotion = 'excited' if compound >= 0.7 else 'positive'
        elif compound <= -0.5:
            emotion = 'negative'
        elif '?' in text:
            emotion = 'thinking'
        else:
            emotion = 'neutral'

        # Map to avatar
        avatar_map = {
            'positive': 'happy',
            'negative': 'concerned',
            'excited': 'happy',
            'thinking': 'thinking',
            'neutral': 'default'
        }

        return {
            'emotion': emotion,
            'confidence': abs(compound),
            'avatar_suggestion': avatar_map.get(emotion, 'default'),
            'vader_scores': scores
        }


# Singleton instance
_detector = None

def get_detector():
    """Get or create emotion detector singleton"""
    global _detector
    if _detector is None:
        _detector = AdvancedEmotionDetector()
    return _detector

def detect_emotion(text):
    """Convenience function to detect emotion"""
    return get_detector().detect(text)


if __name__ == "__main__":
    # Test
    detector = AdvancedEmotionDetector()

    test_texts = [
        "I'm so happy to see you!",
        "This is terrible news.",
        "What do you think about that?",
        "WOW! That's amazing!!!",
        "Let me think about this for a moment.",
        "The weather is nice today.",
    ]

    for text in test_texts:
        result = detector.detect(text)
        print(f"Text: {text}")
        print(f"  Emotion: {result['emotion']}, Avatar: {result['avatar_suggestion']}")
        print()
