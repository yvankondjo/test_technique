from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from typing import List, Optional
from src.config import Config
import json

from src.db.schema import Base, Conversation, Message, Document

class Database:
    def __init__(self, db_path: str = Config.PERSIST_DATABASE):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)
        
        SessionLocal = sessionmaker(bind=self.engine)
        self.SessionLocal = SessionLocal
        self.session = SessionLocal()
    
    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()
    
    def create_conversation(self, title: Optional[str] = None) -> int:
        """Crée une nouvelle conversation et retourne son ID"""
        conversation = Conversation(title=title)
        self.session.add(conversation)
        self.session.commit()
        return conversation.id
    
    def add_message(
        self, 
        conversation_id: int, 
        role: str, 
        content: str,
        sources: Optional[List[str]] = None
    ) -> int:
        """Ajoute un message à une conversation"""
        sources_str = json.dumps(sources) if sources else None
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources_str
        )
        self.session.add(message)
        self.session.commit()
        return message.id
    
    def get_conversation_messages(self, conversation_id: int) -> List[Message]:
        """Récupère tous les messages d'une conversation"""
        return self.session.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at).all()
    
    def get_conversation_history(self, conversation_id: int, for_llm: bool = True) -> List[dict]:
        """
        Récupère l'historique formaté pour le LLM ou complet
        
        Args:
            conversation_id: ID de la conversation
            for_llm: Si True, retourne résumé + messages récents. Si False, retourne tout.
        """
        conversation = self.session.query(Conversation).filter(Conversation.id == conversation_id).first()
        all_messages = self.get_conversation_messages(conversation_id)
        
        if not for_llm or not conversation.summary:
            return [
                {"role": msg.role, "content": msg.content}
                for msg in all_messages
            ]
        
        summary_message = {
            "role": "system",
            "content": f"Résumé de la conversation précédente: {conversation.summary}"
        }
        
        if conversation.summary_at_message_id:
            recent_messages = [
                msg for msg in all_messages 
                if msg.id > conversation.summary_at_message_id
            ]
        else:
            recent_messages = all_messages[-4:]
        
        result = [summary_message]
        result.extend([
            {"role": msg.role, "content": msg.content}
            for msg in recent_messages
        ])
        
        return result
    
    def save_summary(self, conversation_id: int, summary: str, summary_at_message_id: Optional[int] = None):
        """
        Sauvegarde un résumé pour une conversation (sans supprimer les messages)
        Tous les messages restent dans la DB, on marque juste où commence le résumé
        """
        conversation = self.session.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            conversation.summary = summary
            conversation.summary_at_message_id = summary_at_message_id
            self.session.commit()
    
    def save_document(
        self,
        document_id: str,
        file_path: str,
        file_name: str,
        file_type: str,
        chunk_count: int
    ):
        """Sauvegarde les métadonnées d'un document"""
        doc = Document(
            document_id=document_id,
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            chunk_count=chunk_count
        )
        self.session.merge(doc)
        self.session.commit()
    
    def get_all_conversations(self) -> List[Conversation]:
        """Récupère toutes les conversations"""
        return self.session.query(Conversation).order_by(
            Conversation.created_at.desc()
        ).all()
    
    def get_all_documents(self) -> List[Document]:
        """Récupère tous les documents"""
        return self.session.query(Document).order_by(
            Document.created_at.desc()
        ).all()
    
    def update_document_chunk_count(self, document_id: str, chunk_count: int):
        """Met à jour le nombre de chunks d'un document"""
        doc = self.session.query(Document).filter(Document.document_id == document_id).first()
        if doc:
            doc.chunk_count = chunk_count
            self.session.commit()
    
    def delete_conversation(self, conversation_id: int):
        """Supprime une conversation et tous ses messages"""
        self.session.query(Message).filter(
            Message.conversation_id == conversation_id
        ).delete()
        self.session.query(Conversation).filter(
            Conversation.id == conversation_id
        ).delete()
        self.session.commit()
