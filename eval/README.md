# Évaluation du Système RAG

Ce dossier contient les outils pour évaluer les performances du système RAG.

## Métriques utilisées

### Métriques de Retrieval
1. **Precision@K** : Proportion de documents pertinents parmi les K premiers résultats
2. **Recall@K** : Proportion de documents pertinents réellement retrouvés
3. **F1@K** : Moyenne harmonique de Precision@K et Recall@K

### Métriques de Génération
4. **Faithfulness** : Fidélité de la réponse aux sources (0.0 à 1.0)
5. **Answer Relevance** : Pertinence de la réponse à la question (0.0 à 1.0)

## Utilisation

### Étape 1 : Générer le dataset d'évaluation

```bash
python eval/generate_dataset.py
```

Ce script :
- Lit tous les documents indexés dans la base
- Génère automatiquement des paires question-réponse pour chaque document
- Sauvegarde le dataset dans `eval/dataset.json`

**Note** : Assurez-vous d'avoir des documents indexés avant d'exécuter ce script.

### Étape 2 : Exécuter l'évaluation

```bash
python eval/test_rag.py
```

Ce script :
- Charge le dataset d'évaluation
- Teste chaque question avec le système RAG
- Calcule toutes les métriques
- Affiche les résultats et les sauvegarde dans `eval/results.json`

## Structure des fichiers

- `generate_dataset.py` : Génère le dataset QA depuis les documents indexés
- `metrics.py` : Implémente les métriques d'évaluation
- `test_rag.py` : Script principal d'évaluation
- `dataset.json` : Dataset généré (créé après étape 1)
- `results.json` : Résultats de l'évaluation (créé après étape 2)

## Format du dataset

```json
[
  {
    "question": "Quelle est la question?",
    "ground_truth_answer": "La réponse attendue",
    "context": "Contexte source utilisé",
    "document_id": "id_du_document",
    "document_name": "nom_fichier.txt",
    "chunk_index": 0
  }
]
```

## Interprétation des résultats

- **Precision@K** : Plus élevé = moins de résultats non pertinents
- **Recall@K** : Plus élevé = moins de résultats pertinents manqués
- **F1@K** : Score équilibré entre précision et rappel
- **Faithfulness** : Plus élevé = réponse plus fidèle aux sources
- **Answer Relevance** : Plus élevé = réponse plus pertinente à la question

## Amélioration continue

Pour améliorer les performances :
1. Ajustez les paramètres de chunking (`chunk_size`, `chunk_overlap`)
2. Testez différents modèles d'embedding
3. Optimisez le nombre de chunks récupérés (`k`)
4. Améliorez les prompts système


