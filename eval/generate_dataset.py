from pathlib import Path
from typing import List, Dict
import json
import os
import random
import sys

from dotenv import load_dotenv
from openai import OpenAI

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

load_dotenv()

from src.vector_store import VectorStore

def generate_qa_pairs_from_documents(
    output_file: str = "./eval/dataset.json",
) -> List[Dict]:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    vector_store = VectorStore()
    documents_in_vector = vector_store.get_all_documents()
    if not documents_in_vector["document_ids"]:
        print("⚠️ Aucun document indexé. Veuillez d'abord indexer des documents.")
        return []
    dataset = []

    # Sélectionner jusqu'à 10 chunks au hasard pour générer des questions
    chunks = documents_in_vector["chunks"]
    chunks_ids = documents_in_vector["chunks_ids"]
    total_chunks = len(chunks)
    max_samples = min(10, total_chunks)
    if max_samples == 0:
        print("⚠️ Aucun chunk disponible pour la génération de questions.")
        return []
    sampled_indices = random.sample(range(total_chunks), k=max_samples)

    for i in sampled_indices:
        chunk_id = chunks_ids[i]
        chunk = chunks[i]

        print(f"\n📄 Traitement du document ID: {chunk_id}")
        
        
        try:
            
                prompt = f"""À partir du texte suivant extrait d'un document juridique, génère:
1. Une question précise qu'un utilisateur pourrait poser dans un search engine 
2. La réponse attendue basée uniquement sur ce texte

Texte:
{chunk}

Format de réponse (JSON):
{{
"question": "question précise",
"answer": "réponse basée uniquement sur le texte",
"context": "le texte source utilisé"
}}"""
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                qa_data = json.loads(response.choices[0].message.content)
                
                dataset.append({
                    "question": qa_data.get("question", ""),
                    "ground_truth_answer": qa_data.get("answer", ""),
                    "context": qa_data.get("context", chunk),
                    "chunk_index": chunk_id
                })
                print(f"  ✅ Question générée: {qa_data.get('question', '')}")
        except Exception as e:
            print(f"  ❌ Erreur: {str(e)}")
            continue
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset généré: {len(dataset)} paires QA sauvegardées dans {output_file}")
    
    return dataset

if __name__ == "__main__":
    generate_qa_pairs_from_documents()

