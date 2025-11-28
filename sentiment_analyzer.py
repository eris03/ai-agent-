from transformers import pipeline
from typing import Dict, List
import re

class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment & intent models"""
        self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                          model="distilbert-base-uncased-finetuned-sst-2-english")
        self.zero_shot = pipeline("zero-shot-classification")
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment (positive/negative/neutral)"""
        result = self.sentiment_pipeline(text[:512])
        
        sentiment = "NEGATIVE" if result['label'] == "NEGATIVE" else "POSITIVE"
        confidence = result['score']
        
        # Detect frustration level
        frustration_keywords = ["frustrated", "angry", "terrible", "worst", "useless", "terrible", "stupid"]
        frustration = "HIGH" if any(kw in text.lower() for kw in frustration_keywords) else \
                      "MEDIUM" if any(word in text.lower() for word in ["problem", "issue", "help", "please"]) else \
                      "LOW"
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "frustration_level": frustration
        }
    
    def detect_intents(self, text: str) -> List[str]:
        """Detect multiple intents in query"""
        candidate_labels = [
            "password reset",
            "payment update",
            "return/refund",
            "shipping/tracking",
            "billing question",
            "account issue",
            "technical support",
            "general question"
        ]
        
        result = self.zero_shot(text, candidate_labels, multi_class=True)
        
        intents = []
        for label, score in zip(result['labels'], result['scores']):
            if score > 0.3:  # Threshold
                intents.append(label)
        
        return intents if intents else ["general question"]
    
    def classify_category(self, text: str) -> str:
        """Classify into department"""
        categories = ["Account", "Billing", "Technical", "Returns", "Shipping", "General"]
        
        result = self.zero_shot(text, categories)
        return result['labels']
