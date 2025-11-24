from typing import List, Dict, Set
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class RAGEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def precision_at_k(self, retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Calcule Precision@K
        
        Args:
            retrieved_ids: Liste des IDs récupérés (déjà triés par pertinence)
            relevant_ids: Set des IDs pertinents (ground truth)
            k: Nombre de résultats à considérer
        
        Returns:
            Precision@K (0.0 à 1.0)
        """
        if k == 0 or not retrieved_ids:
            return 0.0
        
        top_k = retrieved_ids[:k]
        relevant_retrieved = len([id for id in top_k if id in relevant_ids])
        
        return relevant_retrieved / min(k, len(top_k))
    
    def recall_at_k(self, retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Calcule Recall@K
        
        Args:
            retrieved_ids: Liste des IDs récupérés
            relevant_ids: Set des IDs pertinents (ground truth)
            k: Nombre de résultats à considérer
        
        Returns:
            Recall@K (0.0 à 1.0)
        """
        if not relevant_ids:
            return 1.0 if not retrieved_ids else 0.0
        
        top_k = retrieved_ids[:k]
        relevant_retrieved = len([id for id in top_k if id in relevant_ids])
        
        return relevant_retrieved / len(relevant_ids)

    
    def faithfulness(self, answer: str, context: str) -> float:
        """
        Évalue la fidélité de la réponse aux sources (0.0 à 1.0)
        Utilise un LLM pour vérifier si la réponse est basée sur le contexte
        
        Args:
            answer: Réponse générée
            context: Contexte utilisé pour générer la réponse
        
        Returns:
            Score de fidélité (0.0 à 1.0)
        """
        try:
            prompt = f"""Évalue si la réponse suivante est fidèle au contexte fourni.
Réponds uniquement par un nombre entre 0.0 et 1.0 où:
- 1.0 = La réponse est entièrement basée sur le contexte
- 0.5 = La réponse est partiellement basée sur le contexte
- 0.0 = La réponse n'est pas basée sur le contexte ou contient des informations inventées

Contexte:
{context}

Réponse:
{answer}

Score (nombre uniquement):"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            
            return max(0.0, min(1.0, score))
        
        except Exception as e:
            print(f"Erreur calcul faithfulness: {e}")
            return 0.5
    
    
    def evaluate_retrieval(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Évalue la qualité du retrieval avec plusieurs métriques
        
        Returns:
            Dict avec Precision@K, Recall@K
        """
        results = {}
        
        for k in k_values:
            results[f"precision@{k}"] = self.precision_at_k(retrieved_ids, relevant_ids, k)
            results[f"recall@{k}"] = self.recall_at_k(retrieved_ids, relevant_ids, k)
        
        return results
    
    def evaluate_generation(
        self,
        answer: str,
        context: str
    ) -> Dict[str, float]:
        """
        Évalue la qualité de la génération
        
        Returns:
            Dict avec faithfulness et answer_relevance
        """
        return {
            "faithfulness": self.faithfulness(answer, context)
        }


