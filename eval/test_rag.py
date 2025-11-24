import json
from pathlib import Path
from typing import List, Dict, Set
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()
import sys
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.vector_store import VectorStore
from src.rag import RAGEngine
from src.db.database import Database
from eval.metrics import RAGEvaluator

def load_dataset(dataset_path: str = "./eval/dataset.json") -> List[Dict]:
    """Charge le dataset d'évaluation"""
    if not Path(dataset_path).exists():
        print(f"❌ Dataset non trouvé: {dataset_path}")
        print("   Exécutez d'abord: python eval/generate_dataset.py")
        return []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_relevant_chunks(
    vector_store: VectorStore,
    question: str,
    context: str,
    k: int = 20
) -> Set[str]:
    """
    Trouve les chunks pertinents pour une question donnée
    Utilise une recherche sémantique pour identifier les chunks qui contiennent le contexte
    """

    results = vector_store.query(question, k=k)

    relevant_ids = set()
    ids_nested = results.get('ids')
    if not ids_nested or len(ids_nested) == 0 or len(ids_nested[0]) == 0:
        return relevant_ids

    retrieved_ids = ids_nested[0]
    distances = results.get('distances', [[]])[0] if results.get('distances') else []

    if k == 1:
        if distances:
            min_idx = int(np.argmin(distances))
            relevant_ids.add(retrieved_ids[min_idx])
        else:
            relevant_ids.add(retrieved_ids[0])
    else:

        relevant_ids.update(retrieved_ids)

    return relevant_ids

def evaluate_rag_system(
    dataset_path: str = "./eval/dataset.json",
    k_values: List[int] = [1, 3, 5]
):
    """
    Évalue le système RAG complet
    """
    print("=" * 80)
    print("ÉVALUATION DU SYSTÈME RAG")
    print("=" * 80)
    
    dataset = load_dataset(dataset_path)
    
    if not dataset:
        return
    
    print(f"\n📊 Dataset chargé: {len(dataset)} exemples")
    
    vector_store = VectorStore()
    database = Database()
    rag_engine = RAGEngine(vector_store, database)
    evaluator = RAGEvaluator()
    
    all_retrieval_metrics = []
    all_generation_metrics = []
    detailed_examples: List[Dict] = []
    
    print("\n" + "=" * 80)
    print("EXÉCUTION DES TESTS")
    print("=" * 80)
    
    for i, example in enumerate(dataset, 1):
        print(f"\n[{i}/{len(dataset)}] Question: {example['question'][:80]}...")
        
        question = example['question']
        context = example['context']
        
        try:
            start_time = time.time()
            # Utilise la méthode `chat` de RAGEngine qui fait retrieval + génération
            result = rag_engine.chat(question, k=5)
            elapsed_time = time.time() - start_time
            
            answer = result.get('answer', '')
            retrieved_sources = result.get('sources', [])
            
            print(f"  ⏱️  Temps: {elapsed_time:.2f}s")
            print(f"  📎 Sources récupérées: {len(retrieved_sources)}")
            
            chunk_index = example.get('chunk_index')
            if isinstance(chunk_index, list):
                relevant_ids = {cid for cid in chunk_index if isinstance(cid, str)}
            elif isinstance(chunk_index, str):
                relevant_ids = {chunk_index}
            else:
                relevant_ids = set()

            retrieved_ids = [
                chunk.get('id')
                for chunk in result.get('chunks', [])
                if isinstance(chunk, dict) and isinstance(chunk.get('id'), str)
            ]
            if not retrieved_ids:
                retrieved_ids = [
                    chunk_id for chunk_id in retrieved_sources
                    if isinstance(chunk_id, str)
                ]
            
            retrieval_metrics = None
            
            if relevant_ids and retrieved_ids:
                retrieval_metrics = evaluator.evaluate_retrieval(
                    retrieved_ids,
                    relevant_ids,
                    k_values=k_values
                )
                all_retrieval_metrics.append(retrieval_metrics)
                
                print(f"  📊 Precision@5: {retrieval_metrics['precision@5']:.3f}")
                print(f"  📊 Recall@5: {retrieval_metrics['recall@5']:.3f}")
            
            generation_metrics = None

            if answer:
                generation_metrics = evaluator.evaluate_generation(
                    answer,
                    context
                )
                all_generation_metrics.append(generation_metrics)

                print(f"  ✅ Faithfulness: {generation_metrics.get('faithfulness', 0.0):.3f}")
            
            detailed_examples.append({
                "question": question,
                "expected_answer": example.get('ground_truth_answer', ''),
                "generated_answer": answer,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": sorted(list(relevant_ids)),
                "retrieval_metrics": retrieval_metrics or {},
                "generation_metrics": generation_metrics or {}
            })
            
        except Exception as e:
            print(f"  ❌ Erreur: {str(e)}")
            continue
    
    print("\n" + "=" * 80)
    print("RÉSULTATS FINAUX")
    print("=" * 80)
    
    if all_retrieval_metrics:
        print("\n📊 MÉTRIQUES DE RETRIEVAL:")
        print("-" * 80)
        
        for k in k_values:
            avg_precision = np.mean([m[f"precision@{k}"] for m in all_retrieval_metrics])
            avg_recall = np.mean([m[f"recall@{k}"] for m in all_retrieval_metrics])
            
            print(f"  Precision@{k}: {avg_precision:.3f}")
            print(f"  Recall@{k}:    {avg_recall:.3f}")
            print()
    
    if all_generation_metrics:
        print("\n✅ MÉTRIQUES DE GÉNÉRATION:")
        print("-" * 80)

        avg_faithfulness = np.mean([m.get('faithfulness', 0.0) for m in all_generation_metrics])

        print(f"  Faithfulness:      {avg_faithfulness:.3f}")
        print()

        overall_score = avg_faithfulness
        print(f"  Score Global:      {overall_score:.3f}")
    
    print("\n" + "=" * 80)
    
    results_summary = {
        "retrieval": ({
            k: {
                "precision": float(np.mean([m.get(f"precision@{k}", 0.0) for m in all_retrieval_metrics])),
                "recall": float(np.mean([m.get(f"recall@{k}", 0.0) for m in all_retrieval_metrics]))
            } for k in k_values
        } if all_retrieval_metrics else {}),
        "generation": ({
            "faithfulness": float(np.mean([m.get('faithfulness', 0.0) for m in all_generation_metrics]))
        } if all_generation_metrics else {}),
        "examples": detailed_examples
    }
    
    results_file = Path("./eval/results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Résultats sauvegardés dans: {results_file}")

if __name__ == "__main__":
    evaluate_rag_system()

