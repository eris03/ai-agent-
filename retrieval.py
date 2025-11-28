import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import os

class HybridRetriever:
    def __init__(self, faq_path: str = "faq_database.json"):
        """Initialize hybrid retriever with semantic + keyword search"""
        
        # Load FAQs
        with open(faq_path, 'r') as f:
            self.faqs = json.load(f)['faqs']
        
        # Initialize semantic search model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings for all FAQs
        self.faq_embeddings = []
        self.faq_texts = []
        for faq in self.faqs:
            combined_text = f"{faq['title']} {faq['content']}"
            self.faq_texts.append(combined_text)
            embedding = self.embedding_model.encode(combined_text, convert_to_tensor=False)
            self.faq_embeddings.append(embedding)
        
        # Initialize BM25 for keyword search
        tokenized_faqs = [text.lower().split() for text in self.faq_texts]
        self.bm25 = BM25Okapi(tokenized_faqs)
        
        print(f"✅ Loaded {len(self.faqs)} FAQs and created hybrid index")
    
    def query_expansion(self, query: str, num_expansions: int = 3) -> List[str]:
        """Expand query with alternative phrasings"""
        # Simple expansion: add synonyms and variations
        expansions = [query]
        
        # Add variations
        if "password" in query.lower():
            expansions.extend(["account access", "login issues", "authentication"])
        if "return" in query.lower():
            expansions.extend(["refund", "send back", "exchange"])
        if "shipping" in query.lower():
            expansions.extend(["delivery", "tracking", "order"])
        
        return expansions[:num_expansions + 1]
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Semantic search using embeddings"""
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        
        scores = []
        for i, faq_embedding in enumerate(self.faq_embeddings):
            # Cosine similarity
            similarity = np.dot(query_embedding, faq_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(faq_embedding) + 1e-8
            )
            scores.append((i, similarity))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x, reverse=True)
        return scores[:top_k]
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Keyword search using BM25"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x, reverse=True)
        return indexed_scores[:top_k]
    
    def reciprocal_rank_fusion(self, semantic_results, keyword_results, k: int = 60) -> Dict:
        """Merge semantic and keyword results using RRF"""
        rrf_scores = {}
        
        # RRF formula: 1 / (k + rank)
        for rank, (idx, score) in enumerate(semantic_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        for rank, (idx, score) in enumerate(keyword_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        # Sort by combined score
        merged = sorted(rrf_scores.items(), key=lambda x: x, reverse=True)
        return merged
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Main retrieval function: hybrid search"""
        
        # Expand query
        expanded_queries = self.query_expansion(query)
        
        all_semantic = []
        all_keyword = []
        
        # Search with all expanded queries
        for exp_query in expanded_queries:
            semantic = self.semantic_search(exp_query, top_k=5)
            keyword = self.keyword_search(exp_query, top_k=5)
            all_semantic.extend(semantic)
            all_keyword.extend(keyword)
        
        # Merge results
        merged = self.reciprocal_rank_fusion(all_semantic, all_keyword)
        
        # Extract top results
        results = []
        for idx, score in merged[:top_k]:
            faq = self.faqs[idx]
            results.append({
                "faq_id": faq['id'],
                "title": faq['title'],
                "content": faq['content'],
                "category": faq['category'],
                "relevance_score": float(score),
                "confidence": min(score * 1.2, 1.0)  # Normalize to 0-1
            })
        
        return results
