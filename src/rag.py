from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import os
import time
import uuid
import tiktoken
import logging  
from src.vector_store import VectorStore
from src.monitoring import RAGMonitor
from src.db.database import Database
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(
        self, 
        vector_store: VectorStore,
        database: Database,
        model: str = "gpt-5-mini",
        monitor: Optional[RAGMonitor] = None,
        max_context_tokens: int = 128000,
        context_threshold: float = 0.95
    ):
        self.vector_store = vector_store
        self.database = database
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model
        self.monitor = monitor or RAGMonitor(database=database)
        self.max_context_tokens = max_context_tokens
        self.context_threshold = context_threshold
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            try:
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        self.system_prompt = """Tu es un assistant juridique interne.
Tu DOIS répondre uniquement à partir des sources fournies.
Si la réponse n'est pas dans le corpus, dis : "Les documents internes ne permettent pas de répondre avec certitude."
Interdiction d'inventer."""
    
    def count_tokens(self, text: str) -> int:
        """Compte les tokens dans un texte"""
        return len(self.encoding.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict]) -> int:
        """Compte le nombre total de tokens dans une liste de messages"""
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.get('content', ''))
        return total
    
    def summarize_history(self, messages: List[Dict], keep_recent: int = 4) -> Tuple[str, List[Dict]]:
        """
        Résume l'historique de conversation en gardant les messages récents intacts
        
        Args:
            messages: Tous les messages de la conversation
            keep_recent: Nombre de messages récents à garder intacts
        
        Returns:
            (summary, recent_messages) - Résumé des anciens + messages récents
        """
        if not messages:
            return "", []
        
        if len(messages) <= keep_recent:
            return "", messages
        
        old_messages = messages[:-keep_recent]
        recent_messages = messages[-keep_recent:]
        
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in old_messages
        ])
        
        summarize_prompt = f"""Résume cette conversation en conservant les informations importantes et le contexte juridique. 
Le résumé doit être concis mais complet pour permettre la continuité de la conversation.

Conversation à résumer:
{history_text}

Résumé:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": summarize_prompt}],
                max_tokens=1000
            )
            summary = response.choices[0].message.content.strip()
            return summary, recent_messages
        except Exception as e:
            return f"[Résumé non disponible - erreur: {type(e).__name__}]", recent_messages
    
    def manage_context(
        self, 
        conversation_history: List[Dict],
        query: str,
        context: str
    ) -> Tuple[List[Dict], bool, Optional[str]]:
        """
        Gère le contexte : vérifie la taille et summarise si nécessaire
        
        Returns:
            (historique_géré, was_summarized, summary)
        """
        test_messages = [{"role": "system", "content": self.system_prompt}]
        if conversation_history:
            test_messages.extend(conversation_history)
        test_messages.append({
            "role": "user",
            "content": f"Contexte:\n{context}\n\nQuestion: {query}"
        })
        
        total_tokens = self.count_messages_tokens(test_messages)
        threshold_tokens = int(self.max_context_tokens * self.context_threshold)
        
        if total_tokens > threshold_tokens and conversation_history:
            summary, recent_messages = self.summarize_history(conversation_history, keep_recent=4)
            managed_history = [
                {"role": "system", "content": f"Résumé de la conversation précédente: {summary}"}
            ]
            managed_history.extend(recent_messages)
            return managed_history, True, summary
        
        return (conversation_history if conversation_history else []), False, None
    
    def validate_query_size(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Valide la taille de la requête
        
        Returns:
            (is_valid, error_message)
        """
        query_tokens = self.count_tokens(query)
        max_query_tokens = int(self.max_context_tokens * self.context_threshold)
        
        if query_tokens > max_query_tokens:
            return False, f"La requête est trop longue ({query_tokens} tokens). Maximum autorisé: {max_query_tokens} tokens."
        
        return True, None
    
    def retrieve(self, query: str, k: int = 5) -> Dict:
        """
        Retrieve les chunks les plus pertinents
        
        Returns:
            Dict avec 'context', 'sources', 'chunks' (liste de dicts avec 'text', 'id', 'metadata')
        """
        try:
            results = self.vector_store.query(query, k=k)
            
            context = results.get('context', '')
            sources = results.get('sources', [])
            
            retrieved_chunks = []
            if results.get('ids') and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                documents = results.get('documents', [[]])[0] if results.get('documents') else []
                metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
                ids = results['ids'][0]
                
                for i in range(len(ids)):
                    logger.info(f"Retrieved chunk ID: {ids[i]} \n distance: {results['distances'][0][i]}, m")
                    retrieved_chunks.append({
                        'id': ids[i],
                        'text': documents[i] if i < len(documents) else '',
                        'metadata': metadatas[i] if i < len(metadatas) else {}
                    })
            
            return {
                'context': context,
                'sources': sources,
                'chunks': retrieved_chunks
            }
        except Exception as e:
            return {'context': '', 'sources': [], 'chunks': []}
    
    def build_prompt(
        self, 
        query: str, 
        context: str, 
        conversation_history: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Construit le prompt système + contexte + historique"""
        # Utiliser le system prompt défini lors de l'initialisation (évite la duplication)
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        user_message = f"""Contexte des documents internes :
{context}

Question : {query}"""
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def chat(
        self,
        message: str,
        conversation_id: Optional[int] = None,
        k: int = 5
    ) -> Dict:
        """
        Fonction simple qui fait tout : prend un message et conversation_id (ou None pour nouveau)
        
        Args:
            message: Message de l'utilisateur
            conversation_id: ID de la conversation (None pour créer une nouvelle)
            k: Nombre de chunks à récupérer
        
        Returns:
            Dict avec 'answer', 'conversation_id', 'sources', 'retrieval_time', 'generation_time'
        """
        is_valid, error_msg = self.validate_query_size(message)
        if not is_valid:
            return {
                "answer": f"Erreur: {error_msg}",
                "conversation_id": conversation_id,
                "sources": [],
                "retrieval_time": 0.0,
                "generation_time": 0.0,
                "error": error_msg
            }
        
        if conversation_id is None:
            conversation_id = self.database.create_conversation()
        
        conversation_history = self.database.get_conversation_history(conversation_id)
    
        retrieval_start = time.time()
        retrieved_data = self.retrieve(message, k=k)
        retrieval_time = time.time() - retrieval_start
        # Vérifier les résultats : on considère absence de résultat si pas de chunks ET pas de contexte
        context = retrieved_data.get('context', '')
        chunks = retrieved_data.get('chunks', [])
        if (not chunks or len(chunks) == 0) and (not context or len(context.strip()) == 0):
            # Appeler get_all_documents() seulement si nécessaire (pour savoir si la collection est vide)
            vector_stats = self.vector_store.get_all_documents()
            has_documents = len(vector_stats.get("document_ids", [])) > 0

            if not has_documents:
                answer = "Aucun document n'est actuellement indexé dans la base. Veuillez d'abord uploader et indexer des documents via la page 'Documents'."
            else:
                answer = "Les documents internes ne permettent pas de répondre avec certitude à cette question. Aucun document pertinent trouvé dans la base."

            self.database.add_message(conversation_id, "user", message)
            self.database.add_message(conversation_id, "assistant", answer, [])

            return {
                "answer": answer,
                "conversation_id": conversation_id,
                "sources": [],
                "chunks": [],
                "retrieval_time": retrieval_time,
                "generation_time": 0.0,
                "message_id": str(uuid.uuid4())
            }
        
        # context et chunks déjà extraits plus haut
        context = context
        
        managed_history, was_summarized, summary = self.manage_context(conversation_history, message, context)
        
        if was_summarized and summary:
            all_messages = self.database.get_conversation_messages(conversation_id)
            summary_at_id = all_messages[-4].id if len(all_messages) >= 4 else None
            self.database.save_summary(conversation_id, summary, summary_at_id)
        
        messages = self.build_prompt(message, context, managed_history)
        
        generation_start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            generation_time = time.time() - generation_start
            answer = response.choices[0].message.content
            if not answer:
                answer = "Erreur: réponse vide de l'API OpenAI"
            usage = response.usage
        except Exception as e:
            generation_time = time.time() - generation_start
            error_type = type(e).__name__
            if "rate_limit" in str(e).lower() or "quota" in str(e).lower():
                answer = "Limite de taux ou quota dépassé. Veuillez réessayer dans quelques instants."
            elif "authentication" in str(e).lower() or "api_key" in str(e).lower():
                answer = "Erreur d'authentification avec l'API OpenAI. Vérifiez votre clé API."
            elif "timeout" in str(e).lower():
                answer = "Timeout lors de l'appel à l'API. Veuillez réessayer."
            else:
                answer = f"Erreur lors de la génération de la réponse ({error_type}): {str(e)[:200]}"
            usage = type('Usage', (), {'prompt_tokens': 0, 'completion_tokens': 0})()
    
        sources = retrieved_data['sources']
        chunks = retrieved_data.get('chunks', [])
        message_id = str(uuid.uuid4())
        
        self.database.add_message(conversation_id, "user", message)
        self.database.add_message(conversation_id, "assistant", answer, sources)
        
        self.monitor.log_rag_call(
            query=message,
            answer=answer,
            sources=sources,
            context=context,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            conversation_id=str(conversation_id),
            message_id=message_id
        )
        
        return {
            "answer": answer,
            "conversation_id": conversation_id,
            "sources": sources,
            "chunks": chunks,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "message_id": message_id
        }
