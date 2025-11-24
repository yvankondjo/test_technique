from pathlib import Path
from typing import List, Dict
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from src.config import Config
class VectorStore:
    def __init__(
        self, 
        collection_name: str = Config.VECTOR_STORE_COLLECTION,
        persist_directory: str = Config.PERSIST_DIRECTORY,
        embedding_model: str = Config.EMBEDDING_MODEL,
        threshold: float = 0.75
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embedding_model
        )
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        self.threshold = threshold

    def add_documents(self, doc_metadata: Dict, chunks: List[str]):
        """
        Ajoute des chunks d'un document à la collection (batch insertion)
        
        Args:
            doc_metadata: Métadonnées du document (doit contenir 'document_id')
            chunks: Les chunks du document
        """
        if not chunks:
            return
        
        import uuid
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [doc_metadata.copy() for _ in chunks]
        
        for i, metadata in enumerate(metadatas):
            metadata['chunk_index'] = i
        
        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )

    def query(self, query: str, k: int = 5):
        """
        Recherche les chunks les plus similaires
        
        Args:
            query: Texte de la requête
            k: Nombre de résultats
        
        Returns:
            Résultats de la recherche
        """
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            return {"context": "", "sources": [], "ids": [], "documents": [], "metadatas": []}
        if len(query) == 0:
            return {"context": "", "sources": [], "ids": [], "documents": [], "metadatas": []}
        
        if len(query) > 8000:
            query = query[:8000]
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
        except Exception as e:
            return {"context": "", "sources": [], "ids": [], "documents": [], "metadatas": [],"distances": []}
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return {"context": "", "sources": [], "ids": [], "documents": [], "metadatas": [],"distances": []}
        
        context_parts = []
        sources = []
        documents = results['documents'][0]
        distances = results.get('distances', [[]])[0] if results.get('distances') else []
        ids = results.get('ids', [[]])[0] if results.get('ids') else []
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        for i, chunk in enumerate(documents):
            if i < len(distances) and distances[i] < self.threshold:
                context_parts.append(f"[Source {i+1}]\n{chunk}\n")
                if i < len(ids):
                    sources.append(ids[i])
        
        if not context_parts:
            return {"context": "", "sources": [], "ids": [ids], "documents": [documents], "metadatas": [metadatas],"distances": [distances]}
        
        return {
            "context": "\n".join(context_parts),
            "sources": sources,
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances]
        }
    
    def delete_document(self, document_id: str) -> bool:
        """
        Supprime tous les chunks d'un document
        
        Args:
            document_id: L'identifiant du document à supprimer
        
        Returns:
            True si des chunks ont été supprimés, False sinon
        """
        try:
            results = self.collection.delete(where={"document_id": document_id})
            return True
        except Exception as e:
            print(f"Erreur lors de la suppression du document {document_id}: {e}")
            return False
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """
        Récupère les chunks et leurs métadonnées à partir de leurs IDs
        
        Args:
            chunk_ids: Liste des IDs des chunks à récupérer
            
        Returns:
            Liste de dicts avec 'id', 'text', 'metadata'
        """
        if not chunk_ids:
            return []
        
        try:
            results = self.collection.get(ids=chunk_ids)
            chunks = []
            
            if results.get('ids'):
                ids = results['ids']
                documents = results.get('documents', [])
                metadatas = results.get('metadatas', [])
                
                for i, chunk_id in enumerate(ids):
                    chunks.append({
                        'id': chunk_id,
                        'text': documents[i] if i < len(documents) else '',
                        'metadata': metadatas[i] if i < len(metadatas) else {}
                    })
            
            return chunks
        except Exception as e:
            return []
    def get_all_documents(self) -> Dict[str, List[str]]:
        """Retourne la liste de tous les document_id, file_path, chunk_id et chunks dans la collection
        Returns:
            Dict avec les listes de 'document_ids', 'document_paths', 'chunks_ids', 'chunks'
        """
        results = self.collection.get()

        if not results or not results.get("metadatas"):
            return {
                "document_ids": [],
                "document_paths": [],
                "chunks_ids": [],
                "chunks": []
            }

        ids_list = results.get("ids", [])
        metadatas_list = results.get("metadatas", [])
        documents_list = results.get("documents", [])

        document_ids = set()
        document_paths = set()
        chunks_ids = []
        chunks = []

        for i in range(len(ids_list)):
            chunk_id = ids_list[i]
            metadata = metadatas_list[i] if i < len(metadatas_list) else {}
            document = documents_list[i] if i < len(documents_list) else ""

            chunks_ids.append(chunk_id)
            chunks.append({
                "id": chunk_id,
                "metadata": metadata,
                "document": document
            })

            if isinstance(metadata, dict):
                if "document_id" in metadata:
                    document_ids.add(metadata["document_id"])
                if "file_path" in metadata:
                    document_paths.add(metadata["file_path"])

        return {
            "document_ids": sorted(list(document_ids)),
            "document_paths": sorted(list(document_paths)),
            "chunks_ids": sorted(chunks_ids),
            "chunks": chunks
        }
