import json
import os
from datetime import datetime
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from retrieval import HybridRetriever
from sentiment_analyzer import SentimentAnalyzer
from dotenv import load_dotenv

load_dotenv()

class SupportAgent:
    def __init__(self):
        """Initialize the support agent"""
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("TEMPERATURE", 0.3)),
            max_tokens=int(os.getenv("MAX_TOKENS", 1000))
        )
        
        self.retriever = HybridRetriever("faq_database.json")
        self.sentiment_analyzer = SentimentAnalyzer()
        self.conversation_history = []
        
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """Load the master system prompt"""
        return """You are an ELITE AI Support Assistant designed for maximum accuracy and user satisfaction.

YOUR PRIMARY GOALS:
1. Provide ACCURATE, HELPFUL answers
2. Maintain TRANSPARENCY about confidence
3. Gracefully ESCALATE when needed
4. LEARN from feedback
5. Track conversation CONTEXT

RESPONSE RULES:
- Always return structured JSON
- Include confidence scores (0-1)
- Detect multiple intents
- Ask clarifying questions if needed
- Acknowledge frustration empathetically
- Never guess - escalate if unsure

RESPONSE FORMAT:
{
  "response_type": "faq_match|llm_generated|clarification|escalation",
  "message": "[Main answer]",
  "confidence_score": 0.92,
  "confidence_level": "HIGH|MEDIUM|LOW",
  "intent": ["intent1"],
  "sentiment": "POSITIVE|NEGATIVE|NEUTRAL",
  "frustration_level": "LOW|MEDIUM|HIGH",
  "category": "[Category]",
  "sources": [...],
  "escalation_needed": false,
  "next_actions": []
}"""
    
    def process_query(self, user_query: str) -> Dict:
        """Process user query through the agent"""
        
        # Step 1: Sentiment & Intent Analysis
        sentiment_data = self.sentiment_analyzer.analyze_sentiment(user_query)
        intents = self.sentiment_analyzer.detect_intents(user_query)
        category = self.sentiment_analyzer.classify_category(user_query)
        
        # Step 2: Hybrid Retrieval
        retrieved_faqs = self.retriever.retrieve(user_query, top_k=3)
        
        # Step 3: Determine confidence threshold
        best_confidence = retrieved_faqs["confidence"] if retrieved_faqs else 0
        
        # Step 4: Route to appropriate handler
        if best_confidence >= 0.85:
            response = self._handle_faq_match(user_query, retrieved_faqs, sentiment_data, intents, category)
        elif best_confidence >= 0.70:
            response = self._handle_medium_confidence(user_query, retrieved_faqs, sentiment_data, intents, category)
        elif sentiment_data["frustration_level"] == "HIGH":
            response = self._handle_escalation(user_query, "HIGH_FRUSTRATION", sentiment_data, intents, category)
        else:
            response = self._handle_llm_fallback(user_query, retrieved_faqs, sentiment_data, intents, category)
        
        # Step 5: Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "agent_response": response
        })
        
        return response
    
    def _handle_faq_match(self, query: str, faqs: List, sentiment: Dict, intents: List, category: str) -> Dict:
        """Handle high-confidence FAQ matches"""
        top_faq = faqs
        
        return {
            "response_type": "faq_match",
            "message": f"I found the perfect answer for you:\n\n{top_faq['content']}",
            "confidence_score": round(top_faq["confidence"], 2),
            "confidence_level": "HIGH",
            "intent": intents,
            "sentiment": sentiment["sentiment"],
            "frustration_level": sentiment["frustration_level"],
            "category": category,
            "sources": [
                {
                    "id": f"FAQ #{top_faq['faq_id']}",
                    "title": top_faq['title'],
                    "relevance_score": round(top_faq["relevance_score"], 2)
                }
            ],
            "related_faqs": [
                {
                    "id": f"FAQ #{faq['faq_id']}",
                    "title": faq['title']
                } for faq in faqs[1:3]
            ],
            "escalation_needed": False,
            "next_actions": ["Did this help? Let me know!"],
            "feedback_requested": True
        }
    
    def _handle_medium_confidence(self, query: str, faqs: List, sentiment: Dict, intents: List, category: str) -> Dict:
        """Handle medium-confidence matches"""
        top_faq = faqs
        
        return {
            "response_type": "faq_match",
            "message": f"Here's what I found (this might be what you're looking for):\n\n{top_faq['content']}\n\nIf this isn't quite right, let me know!",
            "confidence_score": round(top_faq["confidence"], 2),
            "confidence_level": "MEDIUM",
            "intent": intents,
            "sentiment": sentiment["sentiment"],
            "frustration_level": sentiment["frustration_level"],
            "category": category,
            "sources": [{"id": f"FAQ #{top_faq['faq_id']}", "title": top_faq['title']}],
            "escalation_needed": False,
            "warning": "Please verify with FAQ if available",
            "feedback_requested": True
        }
    
    def _handle_escalation(self, query: str, reason: str, sentiment: Dict, intents: List, category: str) -> Dict:
        """Handle escalation cases"""
        
        conversation_summary = "\n".join([
            f"Q: {h['user_query']}" for h in self.conversation_history[-3:]
        ])
        
        return {
            "response_type": "escalation",
            "message": f"I understand this is important to you. I'm connecting you with a specialist who can help right away. Thank you for your patience!",
            "confidence_score": 0.0,
            "confidence_level": "LOW",
            "intent": intents,
            "sentiment": sentiment["sentiment"],
            "frustration_level": sentiment["frustration_level"],
            "category": category,
            "escalation_needed": True,
            "escalation_reason": reason,
            "escalation_priority": "URGENT" if sentiment["frustration_level"] == "HIGH" else "HIGH",
            "conversation_summary": conversation_summary,
            "human_agent_notes": f"Customer is {sentiment['frustration_level'].lower()} frustration. Issue: {query}",
            "next_actions": ["Connecting to {category} specialist..."]
        }
    
    def _handle_llm_fallback(self, query: str, faqs: List, sentiment: Dict, intents: List, category: str) -> Dict:
        """Handle LLM-generated responses"""
        
        context = "\n".join([f"- {faq['title']}: {faq['content']}" for faq in faqs]) if faqs else "No direct FAQ match found."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", f"Context: {context}\n\nUser Query: {query}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        
        return {
            "response_type": "llm_generated",
            "message": response.content,
            "confidence_score": 0.68,
            "confidence_level": "MEDIUM",
            "intent": intents,
            "sentiment": sentiment["sentiment"],
            "frustration_level": sentiment["frustration_level"],
            "category": category,
            "sources": [],
            "warning": "This answer was AI-generated. Please verify if needed.",
            "escalation_available": True,
            "feedback_requested": True
        }
    
    def get_conversation_history(self) -> List:
        """Get full conversation history"""
        return self.conversation_history
    
    def get_analytics(self) -> Dict:
        """Get agent analytics"""
        if not self.conversation_history:
            return {}
        
        total_queries = len(self.conversation_history)
        faq_matches = sum(1 for h in self.conversation_history 
                         if h['agent_response']['response_type'] == 'faq_match')
        escalations = sum(1 for h in self.conversation_history 
                         if h['agent_response']['escalation_needed'])
        
        return {
            "total_queries": total_queries,
            "faq_resolution_rate": round((faq_matches / total_queries) * 100, 1) if total_queries > 0 else 0,
            "escalation_rate": round((escalations / total_queries) * 100, 1) if total_queries > 0 else 0,
            "llm_fallback_rate": round(((total_queries - faq_matches - escalations) / total_queries) * 100, 1) if total_queries > 0 else 0
        }
