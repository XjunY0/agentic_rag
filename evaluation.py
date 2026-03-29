import os
import csv
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Set, Any
from src.utils.helpers import read_jsonl
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

KS = [1, 5, 10, 20]

def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    return len(retrieved_k.intersection(ground_truth_ids)) / len(ground_truth_ids)

def precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k:
        return 0.0
    rel_hits = sum(1 for doc_id in retrieved_k if doc_id in ground_truth_ids)
    return rel_hits / len(retrieved_k)

def f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def r_precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    Truncated R-Precision at k:
    precision over the first min(R, k) results, where R = number of relevant docs.
    This keeps the metric comparable across fixed cutoffs and @all.
    """
    R = len(ground_truth_ids)
    if R == 0:
        return 0.0
    cutoff = min(R, k)
    if cutoff <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:cutoff]
    if not retrieved_k:
        return 0.0
    rel_hits = sum(1 for doc_id in retrieved_k if doc_id in ground_truth_ids)
    return rel_hits / cutoff

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
    totals = {
        "recall_all": 0.0,
        "r_precision_all": 0.0,
        "f1_all": 0.0,
    }
    for k in KS:
        totals[f"recall_at_{k}"] = 0.0
        totals[f"r_precision_at_{k}"] = 0.0
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

        all_k = max(len(retrieved_ids), len(gt_ids))
        recall_all = recall_at_k(retrieved_ids, gt_ids, k=all_k)
        precision_all = precision_at_k(retrieved_ids, gt_ids, k=len(retrieved_ids))
        f1_all = f1_from_precision_recall(precision_all, recall_all)
        r_precision_all = r_precision_at_k(retrieved_ids, gt_ids, k=all_k)

        row = {
            "query_id": qid,
            "query_text": query_text,
            "num_retrieved": len(retrieved_ids),
            "num_relevant": len(gt_ids),
        }

        for k in KS:
            row[f"recall_at_{k}"] = recall_at_k(retrieved_ids, gt_ids, k=k)
            row[f"r_precision_at_{k}"] = r_precision_at_k(retrieved_ids, gt_ids, k=k)

        row["recall_all"] = recall_all
        row["r_precision_all"] = r_precision_all
        row["f1_all"] = f1_all

        # Missing Docs
        retrieved_set = set(retrieved_ids)
        missing_ids = list(gt_ids - retrieved_set)
        row["missing_doc_ids"] = ";".join(missing_ids)
        metrics_rows.append(row)

        for k in KS:
            totals[f"recall_at_{k}"] += row[f"recall_at_{k}"]
            totals[f"r_precision_at_{k}"] += row[f"r_precision_at_{k}"]
        totals["recall_all"] += recall_all
        totals["r_precision_all"] += r_precision_all
        totals["f1_all"] += f1_all
        count += 1

    if count == 0:
        print("No matching queries evaluated.")
        return

    avg_metrics = {name: value / count for name, value in totals.items()}

    print("-" * 40)
    print(f"Evaluation Results ({count} queries):")
    for k in KS:
        print(f"Average Recall@{k}:       {avg_metrics[f'recall_at_{k}']:.4f}")
    print(f"Average Recall@All:     {avg_metrics['recall_all']:.4f}")
    for k in KS:
        print(f"Average R-Precision@{k}: {avg_metrics[f'r_precision_at_{k}']:.4f}")
    print(f"Average R-Precision@All:{avg_metrics['r_precision_all']:.4f}")
    print(f"Average F1@All:         {avg_metrics['f1_all']:.4f}")
    print("-" * 40)
    print(f"Saving detailed metrics to {csv_output_path}")

    # Save CSV
    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_id",
            "query_text",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "recall_at_20",
            "recall_all",
            "r_precision_at_1",
            "r_precision_at_5",
            "r_precision_at_10",
            "r_precision_at_20",
            "r_precision_all",
            "f1_all",
            "num_retrieved",
            "num_relevant",
            "missing_doc_ids",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

if __name__ == "__main__":
    main()
