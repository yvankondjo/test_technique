from typing import Dict, Optional, List
from datetime import datetime
import json
from pathlib import Path
from src.config import Config
import tiktoken

class RAGMonitor:
    def __init__(self, log_file: str = Config.LOG_FILE, database=None):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-5-mini")
        except KeyError:
            try:
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Compte les tokens dans un texte"""
        return len(self.encoding.encode(text))
    
    def log_rag_call(
        self,
        query: str,
        answer: str,
        sources: List[str],
        context: str,
        retrieval_time: float,
        generation_time: float,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None
    ):
        """
        Log une requête RAG avec toutes les métriques
        
        Args:
            query: Question de l'utilisateur
            answer: Réponse générée
            sources: Liste des IDs des chunks utilisés
            context: Contexte envoyé au LLM
            retrieval_time: Temps de retrieval en secondes
            generation_time: Temps de génération en secondes
            conversation_id: ID de la conversation (optionnel)
            message_id: ID du message (optionnel)
            input_tokens: Tokens d'entrée depuis l'API (optionnel, calculé si non fourni)
            output_tokens: Tokens de sortie depuis l'API (optionnel, calculé si non fourni)
        """
        context_length = len(context)
        if input_tokens is None:
            input_tokens = self.count_tokens(context + query)
        if output_tokens is None:
            output_tokens = self.count_tokens(answer)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "message_id": message_id,
            "query": query,
            "answer": answer[:500],
            "sources": sources,
            "context_length": context_length,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "total_time": round(retrieval_time + generation_time, 3)
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        if self.database:
            from src.db.schema import RAGLog
            sources_str = json.dumps(sources) if sources else None
            rag_log = RAGLog(
                timestamp=datetime.now(),
                conversation_id=conversation_id,
                message_id=message_id,
                query=query,
                answer=answer[:500],
                sources=sources_str,
                context_length=context_length,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                retrieval_time=int(retrieval_time * 1000),
                generation_time=int(generation_time * 1000),
                total_time=int((retrieval_time + generation_time) * 1000)
            )
            self.database.session.add(rag_log)
            self.database.session.commit()
    
    def get_logs(self, limit: Optional[int] = None) -> List[Dict]:
        """Récupère les logs (optionnellement limités)"""
        if not self.log_file.exists():
            return []
        
        logs = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        
        if limit:
            return logs[-limit:]
        return logs
    
    def get_stats(self) -> Dict:
        """Calcule des statistiques sur les logs"""
        logs = self.get_logs()
        if not logs:
            return {}
        
        total_calls = len(logs)
        avg_retrieval_time = sum(log['retrieval_time'] for log in logs) / total_calls
        avg_generation_time = sum(log['generation_time'] for log in logs) / total_calls
        avg_total_time = sum(log['total_time'] for log in logs) / total_calls
        avg_input_tokens = sum(log['input_tokens'] for log in logs) / total_calls
        avg_output_tokens = sum(log['output_tokens'] for log in logs) / total_calls
        avg_context_length = sum(log['context_length'] for log in logs) / total_calls
        
        return {
            "total_calls": total_calls,
            "avg_retrieval_time": round(avg_retrieval_time, 3),
            "avg_generation_time": round(avg_generation_time, 3),
            "avg_total_time": round(avg_total_time, 3),
            "avg_input_tokens": round(avg_input_tokens, 1),
            "avg_output_tokens": round(avg_output_tokens, 1),
            "avg_context_length": round(avg_context_length, 1),
            "total_tokens": sum(log['total_tokens'] for log in logs)
        }

