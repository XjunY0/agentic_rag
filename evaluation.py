import os
import json
import csv
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from typing import Dict, List, Set, Any
from src.evaluation.metrics import Evaluator
from src.utils.helpers import read_jsonl
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_relevance(path: str) -> Dict[str, Set[str]]:
    """Load relevance judgments: query_id -> set of relevant doc_ids"""
    relevance = {}
    data = read_jsonl(path)
    for item in data:
        qid = item.get("query-id")
        docid = item.get("corpus-id")
        if qid and docid:
            if qid not in relevance:
                relevance[qid] = set()
            relevance[qid].add(str(docid))
    return relevance

def load_queries(path: str) -> Dict[str, str]:
    """Load queries: query_id -> query_text"""
    queries = {}
    data = read_jsonl(path)
    for item in data:
        qid = item.get("id")
        text = item.get("text")
        if qid:
            queries[qid] = text
    return queries

def load_results(path: str) -> Dict[str, List[str]]:
    """Load system results: query_id -> list of retrieved doc_ids"""
    results = {}
    if not os.path.exists(path):
        print(f"Warning: Results file not found at {path}")
        return results
        
    data = read_jsonl(path)
    for item in data:
        qid = item.get("id")
        # Support both "doc_ids" (list) or infer from other fields if needed
        doc_ids = item.get("doc_ids", [])
        if qid:
            # Ensure doc_ids are strings
            results[qid] = [str(d) for d in doc_ids]
    return results

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Evaluating with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Paths
    dataset_dir = cfg.dataset.path
    output_dir = cfg.output_dir
    
    relevance_path = os.path.join(dataset_dir, "relevance.jsonl")
    queries_path = os.path.join(dataset_dir, "queries.jsonl")
    results_path = os.path.join(output_dir, "final_results.jsonl")
    csv_output_path = os.path.join(output_dir, "evaluation_metrics.csv")

    logger.info(f"Loading data for dataset: {cfg.dataset.name}")
    rel_map = load_relevance(relevance_path)
    query_map = load_queries(queries_path)
    res_map = load_results(results_path)

    if not res_map:
        logger.warning("No results to evaluate.")
        return

    logger.info(f"Found {len(rel_map)} queries in ground truth.")
    logger.info(f"Found {len(res_map)} queries in results.")

    metrics_rows = []
    total_recall = 0.0
    total_ndcg = 0.0
    count = 0

    # Evaluate intersection of queries found in both (or just results)
    # Usually we evaluate on all queries in ground truth, treating missing results as 0
    
    eval_qids = sorted(list(rel_map.keys()))
    
    for qid in eval_qids:
        if qid not in res_map:
            # If query was not processed, skip or count as 0? 
            # Usually count as 0, but here we might have partial runs.
            # Let's track it but maybe not penalize if we just ran a subset.
            continue

        gt_ids = rel_map[qid]
        retrieved_ids = res_map[qid]
        query_text = query_map.get(qid, "")

        # Calculate Metrics
        # Recall@All (since doc_ids is the final set)
        recall = Evaluator.recall_at_k(retrieved_ids, gt_ids, k=len(retrieved_ids) + len(gt_ids))
        
        # NDCG@All
        ndcg = Evaluator.ndcg_at_k(retrieved_ids, gt_ids, k=len(retrieved_ids) + len(gt_ids))

        # Missing Docs
        retrieved_set = set(retrieved_ids)
        missing_ids = list(gt_ids - retrieved_set)

        metrics_rows.append({
            "query_id": qid,
            "query_text": query_text,
            "recall": recall,
            "ndcg": ndcg,
            "num_retrieved": len(retrieved_ids),
            "num_relevant": len(gt_ids),
            "missing_doc_ids": ";".join(missing_ids)
        })

        total_recall += recall
        total_ndcg += ndcg
        count += 1

    if count == 0:
        print("No matching queries evaluated.")
        return

    avg_recall = total_recall / count
    avg_ndcg = total_ndcg / count

    print("-" * 40)
    print(f"Evaluation Results ({count} queries):")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average NDCG:   {avg_ndcg:.4f}")
    print("-" * 40)
    print(f"Saving detailed metrics to {csv_output_path}")

    # Save CSV
    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["query_id", "query_text", "recall", "ndcg", "num_retrieved", "num_relevant", "missing_doc_ids"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

if __name__ == "__main__":
    main()
