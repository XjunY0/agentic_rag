import argparse
import asyncio
import csv
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.core.engine import OmniSearch
from src.io.data_loader import DataLoader
from src.utils.helpers import load_json, read_jsonl, save_json
from src.utils.logger import setup_logger


logger = setup_logger(__name__)

KS = [1, 5, 10, 20]
ABLATION_COMPONENTS = ("bm25", "vector", "entity", "ontology")
DEFAULT_EXPERIMENTS = ["full", "no_bm25", "no_vector", "no_entity", "no_ontology"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent ablation experiments for BM25 / vector / entity / ontology "
            "by overriding search.weights in the ranker."
        )
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent),
        help="Project root that contains the config/ directory. Defaults to the repository root.",
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Root config file name inside config/, without .yaml suffix.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset config name override, for example toy_dataset100_vague_rawquery.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model config name override, for example gpt or qwen.",
    )
    parser.add_argument(
        "--search",
        default=None,
        help="Optional search config name override, for example default.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help=(
            "Experiment presets or custom weight specs. "
            "Built-ins: full, no_bm25, no_vector, no_entity, no_ontology, "
            "only_bm25, only_vector, only_entity, only_ontology. "
            "Custom format: name=bm25:0.5,vector:0.8,entity:0,ontology:1."
        ),
    )
    parser.add_argument(
        "--query-ids",
        nargs="+",
        default=None,
        help="Explicit query ids to run. Example: --query-ids 1 5 42",
    )
    parser.add_argument(
        "--query-ids-file",
        default=None,
        help=(
            "Optional file that defines the query subset. Supports .txt (one id per line), "
            ".json (array or object with query_ids), and .jsonl (each item with id/query_id)."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Directory used to store all experiment outputs. "
            "Defaults to outputs/ablation/<dataset_name>/<timestamp>."
        ),
    )
    parser.add_argument(
        "--query-concurrency",
        type=int,
        default=None,
        help="Override query concurrency. Defaults to search.query_concurrency or 8.",
    )
    parser.add_argument(
        "--reuse-indices-only",
        action="store_true",
        help=(
            "Fail fast if an index is missing instead of building it. "
            "Useful when you only want to rerun ranker ablations on existing indices."
        ),
    )
    parser.add_argument(
        "--disable-ontology-when-zero",
        action="store_true",
        help=(
            "If set, experiments with ontology weight 0 will also set search.ontology_enabled=false "
            "to skip ontology retrieval cost."
        ),
    )
    return parser.parse_args()


def register_hydra_runtime_resolver(project_root: Path) -> None:
    runtime = {"runtime": {"cwd": str(project_root)}}

    def hydra_resolver(path: str) -> Any:
        value: Any = runtime
        for part in str(path).split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                raise KeyError(f"Unsupported hydra resolver path: {path}")
        return value

    OmegaConf.clear_resolver("hydra")
    OmegaConf.register_new_resolver("hydra", hydra_resolver)


def load_repo_config(
    project_root: Path,
    config_name: str,
    dataset_override: Optional[str] = None,
    model_override: Optional[str] = None,
    search_override: Optional[str] = None,
) -> DictConfig:
    register_hydra_runtime_resolver(project_root)

    config_dir = project_root / "config"
    root_cfg_path = config_dir / f"{config_name}.yaml"
    if not root_cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {root_cfg_path}")

    root_cfg = OmegaConf.load(root_cfg_path)
    defaults = list(root_cfg.get("defaults", []))

    chosen_groups: Dict[str, str] = {}
    for item in defaults:
        if isinstance(item, (dict, DictConfig)):
            for group_name, group_value in item.items_ex(resolve=False):
                chosen_groups[group_name] = group_value

    if dataset_override:
        chosen_groups["dataset"] = dataset_override
    if model_override:
        chosen_groups["model"] = model_override
    if search_override:
        chosen_groups["search"] = search_override

    merged = OmegaConf.create(
        {
            "hydra": {
                "runtime": {
                    "cwd": str(project_root),
                }
            }
        }
    )

    for group_name, group_value in chosen_groups.items():
        if group_name == "_self_":
            continue
        group_cfg_path = config_dir / group_name / f"{group_value}.yaml"
        if not group_cfg_path.exists():
            raise FileNotFoundError(f"Config group file not found: {group_cfg_path}")
        merged = OmegaConf.merge(merged, OmegaConf.create({group_name: OmegaConf.load(group_cfg_path)}))

    root_without_defaults = OmegaConf.create(
        OmegaConf.to_container(root_cfg, resolve=False)
    )
    if "defaults" in root_without_defaults:
        del root_without_defaults["defaults"]

    merged = OmegaConf.merge(merged, root_without_defaults)
    OmegaConf.resolve(merged)
    return merged


def load_query_id_subset(args: argparse.Namespace) -> Optional[Set[str]]:
    ids: Set[str] = set()

    if args.query_ids:
        ids.update(str(qid) for qid in args.query_ids)

    if args.query_ids_file:
        ids.update(load_query_ids_from_file(Path(args.query_ids_file)))

    return ids or None


def load_query_ids_from_file(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Query id file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    if suffix == ".json":
        payload = load_json(str(path))
        if isinstance(payload, list):
            return {str(item["id"] if isinstance(item, dict) and "id" in item else item) for item in payload}
        if isinstance(payload, dict):
            if "query_ids" in payload and isinstance(payload["query_ids"], list):
                return {str(item) for item in payload["query_ids"]}
            if "ids" in payload and isinstance(payload["ids"], list):
                return {str(item) for item in payload["ids"]}
        raise ValueError(f"Unsupported JSON payload in {path}. Expected an array or an object with query_ids/ids.")

    if suffix == ".jsonl":
        ids = set()
        for item in read_jsonl(str(path)):
            if "id" in item:
                ids.add(str(item["id"]))
            elif "query_id" in item:
                ids.add(str(item["query_id"]))
            else:
                raise ValueError(f"Unsupported JSONL row in {path}: {item}")
        return ids

    raise ValueError(f"Unsupported query id file format: {path}")


def select_queries(all_queries: Sequence[Dict[str, Any]], query_id_subset: Optional[Set[str]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not query_id_subset:
        return list(all_queries), []

    query_map = {str(item.get("id")): item for item in all_queries if item.get("id") is not None}
    selected_ids = [qid for qid in query_map.keys() if qid in query_id_subset]
    missing_ids = sorted(query_id_subset - set(selected_ids))
    selected_queries = [query_map[qid] for qid in selected_ids]
    return selected_queries, missing_ids


def build_experiment_specs(base_weights: Dict[str, float], experiment_names: Iterable[str]) -> List[Dict[str, Any]]:
    specs = []
    for raw_name in experiment_names:
        spec = parse_experiment_spec(raw_name, base_weights)
        specs.append(spec)
    return specs


def parse_experiment_spec(raw_name: str, base_weights: Dict[str, float]) -> Dict[str, Any]:
    if "=" in raw_name:
        name, weights_expr = raw_name.split("=", 1)
        weights = deepcopy(base_weights)
        for item in weights_expr.split(","):
            item = item.strip()
            if not item:
                continue
            component, value = item.split(":", 1)
            component = component.strip()
            if component not in ABLATION_COMPONENTS:
                raise ValueError(f"Unknown component '{component}' in custom experiment '{raw_name}'")
            weights[component] = float(value)
        return {"name": name.strip(), "weights": normalize_weights(weights), "source": raw_name}

    weights = deepcopy(base_weights)

    if raw_name == "full":
        return {"name": raw_name, "weights": normalize_weights(weights), "source": raw_name}

    if raw_name.startswith("no_"):
        component = raw_name.replace("no_", "", 1)
        if component not in ABLATION_COMPONENTS:
            raise ValueError(f"Unsupported experiment preset: {raw_name}")
        weights[component] = 0.0
        return {"name": raw_name, "weights": normalize_weights(weights), "source": raw_name}

    if raw_name.startswith("only_"):
        component = raw_name.replace("only_", "", 1)
        if component not in ABLATION_COMPONENTS:
            raise ValueError(f"Unsupported experiment preset: {raw_name}")
        weights = {name: 0.0 for name in ABLATION_COMPONENTS}
        weights[component] = float(base_weights.get(component, 0.0) or 1.0)
        return {"name": raw_name, "weights": normalize_weights(weights), "source": raw_name}

    raise ValueError(f"Unsupported experiment preset: {raw_name}")


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    normalized = {}
    for component in ABLATION_COMPONENTS:
        normalized[component] = float(weights.get(component, 0.0))
    return normalized


def ensure_indices_ready(storage_path: str) -> None:
    required_paths = [
        Path(storage_path) / "bm25",
        Path(storage_path) / "vector",
        Path(storage_path) / "vector.ids.json",
        Path(storage_path) / "entity",
        Path(storage_path) / "entity.appc.json",
        Path(storage_path) / "ontology.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prebuilt indices required by --reuse-indices-only:\n" + "\n".join(missing)
        )


async def run_queries(
    engine: OmniSearch,
    queries: Sequence[Dict[str, Any]],
    query_concurrency: int,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(int(query_concurrency))

    async def run_one(q_item: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            qid = str(q_item.get("id"))
            q_text = q_item.get("text") or q_item.get("query") or ""
            try:
                res = await engine.search(q_text, request_id=qid)
                if isinstance(res, dict):
                    res.setdefault("id", qid)
                else:
                    res = {"result": res, "id": qid}
                return res
            except Exception as exc:
                logger.exception("Search failed for query %s", qid)
                return {"query": q_text, "error": str(exc), "id": qid, "doc_ids": []}

    tasks = [asyncio.create_task(run_one(query)) for query in queries]
    results: List[Dict[str, Any]] = []
    with tqdm(total=len(tasks), desc="Running queries") as pbar:
        for future in asyncio.as_completed(tasks):
            results.append(await future)
            pbar.update(1)

    results.sort(key=lambda item: str(item.get("id", "")))
    return results


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
    relevant_count = len(ground_truth_ids)
    if relevant_count == 0:
        return 0.0
    cutoff = min(relevant_count, k)
    if cutoff <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:cutoff]
    if not retrieved_k:
        return 0.0
    rel_hits = sum(1 for doc_id in retrieved_k if doc_id in ground_truth_ids)
    return rel_hits / cutoff


def load_relevance(path: str) -> Dict[str, Set[str]]:
    relevance: Dict[str, Set[str]] = {}
    for item in read_jsonl(path):
        query_id = item.get("query-id")
        corpus_id = item.get("corpus-id")
        if query_id and corpus_id:
            relevance.setdefault(str(query_id), set()).add(str(corpus_id))
    return relevance


def load_query_map(path: str) -> Dict[str, str]:
    query_map: Dict[str, str] = {}
    for item in read_jsonl(path):
        query_id = item.get("id")
        text = item.get("text")
        if query_id is not None:
            query_map[str(query_id)] = text
    return query_map


def evaluate_results(
    dataset_dir: str,
    results: Sequence[Dict[str, Any]],
    query_ids: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    relevance_path = os.path.join(dataset_dir, "relevance.jsonl")
    queries_path = os.path.join(dataset_dir, "queries.jsonl")

    rel_map = load_relevance(relevance_path)
    query_map = load_query_map(queries_path)
    res_map = {str(item.get("id")): [str(doc_id) for doc_id in item.get("doc_ids", [])] for item in results}

    metrics_rows: List[Dict[str, Any]] = []
    totals: Dict[str, float] = {
        "recall_all": 0.0,
        "r_precision_all": 0.0,
        "f1_all": 0.0,
    }
    for k in KS:
        totals[f"recall_at_{k}"] = 0.0
        totals[f"r_precision_at_{k}"] = 0.0

    evaluated = 0
    requested_qids = [str(qid) for qid in query_ids]
    missing_from_relevance: List[str] = []
    missing_from_results: List[str] = []

    for query_id in requested_qids:
        if query_id not in rel_map:
            missing_from_relevance.append(query_id)
            continue
        if query_id not in res_map:
            missing_from_results.append(query_id)
            continue

        gt_ids = rel_map[query_id]
        retrieved_ids = res_map[query_id]
        query_text = query_map.get(query_id, "")

        all_k = max(len(retrieved_ids), len(gt_ids))
        recall_all = recall_at_k(retrieved_ids, gt_ids, k=all_k)
        precision_all = precision_at_k(retrieved_ids, gt_ids, k=len(retrieved_ids))
        f1_all = f1_from_precision_recall(precision_all, recall_all)
        r_precision_all = r_precision_at_k(retrieved_ids, gt_ids, k=all_k)

        row: Dict[str, Any] = {
            "query_id": query_id,
            "query_text": query_text,
            "num_retrieved": len(retrieved_ids),
            "num_relevant": len(gt_ids),
            "recall_all": recall_all,
            "r_precision_all": r_precision_all,
            "f1_all": f1_all,
            "missing_doc_ids": ";".join(sorted(gt_ids - set(retrieved_ids))),
        }

        for k in KS:
            row[f"recall_at_{k}"] = recall_at_k(retrieved_ids, gt_ids, k=k)
            row[f"r_precision_at_{k}"] = r_precision_at_k(retrieved_ids, gt_ids, k=k)
            totals[f"recall_at_{k}"] += row[f"recall_at_{k}"]
            totals[f"r_precision_at_{k}"] += row[f"r_precision_at_{k}"]

        totals["recall_all"] += recall_all
        totals["r_precision_all"] += r_precision_all
        totals["f1_all"] += f1_all
        metrics_rows.append(row)
        evaluated += 1

    summary: Dict[str, Any] = {
        "requested_queries": len(requested_qids),
        "evaluated_queries": evaluated,
        "missing_from_relevance": missing_from_relevance,
        "missing_from_results": missing_from_results,
    }

    if evaluated == 0:
        for k in KS:
            summary[f"recall_at_{k}"] = 0.0
            summary[f"r_precision_at_{k}"] = 0.0
        summary["recall_all"] = 0.0
        summary["r_precision_all"] = 0.0
        summary["f1_all"] = 0.0
        return summary, metrics_rows

    for key, value in totals.items():
        summary[key] = value / evaluated

    return summary, metrics_rows


def write_metrics_csv(path: Path, metrics_rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def write_summary_csv(path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "requested_queries",
        "evaluated_queries",
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
        "bm25_weight",
        "vector_weight",
        "entity_weight",
        "ontology_weight",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def build_output_root(project_root: Path, dataset_name: str, output_root_arg: Optional[str]) -> Path:
    if output_root_arg:
        return Path(output_root_arg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "outputs" / "ablation" / dataset_name / timestamp


def make_experiment_cfg(
    base_cfg: DictConfig,
    weights: Dict[str, float],
    disable_ontology_when_zero: bool,
) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.search.weights = OmegaConf.create(weights)
    if disable_ontology_when_zero and float(weights.get("ontology", 0.0)) == 0.0:
        cfg.search.ontology_enabled = False
    return cfg


def summarize_row(experiment_name: str, weights: Dict[str, float], summary: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "experiment": experiment_name,
        "requested_queries": summary.get("requested_queries", 0),
        "evaluated_queries": summary.get("evaluated_queries", 0),
        "recall_at_1": summary.get("recall_at_1", 0.0),
        "recall_at_5": summary.get("recall_at_5", 0.0),
        "recall_at_10": summary.get("recall_at_10", 0.0),
        "recall_at_20": summary.get("recall_at_20", 0.0),
        "recall_all": summary.get("recall_all", 0.0),
        "r_precision_at_1": summary.get("r_precision_at_1", 0.0),
        "r_precision_at_5": summary.get("r_precision_at_5", 0.0),
        "r_precision_at_10": summary.get("r_precision_at_10", 0.0),
        "r_precision_at_20": summary.get("r_precision_at_20", 0.0),
        "r_precision_all": summary.get("r_precision_all", 0.0),
        "f1_all": summary.get("f1_all", 0.0),
        "bm25_weight": weights["bm25"],
        "vector_weight": weights["vector"],
        "entity_weight": weights["entity"],
        "ontology_weight": weights["ontology"],
    }
    return row


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    cfg = load_repo_config(
        project_root=project_root,
        config_name=args.config_name,
        dataset_override=args.dataset,
        model_override=args.model,
        search_override=args.search,
    )
    base_weights = normalize_weights(OmegaConf.to_container(cfg.search.weights, resolve=True))
    experiment_specs = build_experiment_specs(base_weights, args.experiments)

    dataset_name = str(cfg.dataset.name)
    dataset_dir = str(cfg.dataset.path)
    output_root = build_output_root(project_root, dataset_name, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.reuse_indices_only:
        ensure_indices_ready(str(cfg.storage.path))

    logger.info("Dataset: %s", dataset_name)
    logger.info("Dataset path: %s", dataset_dir)
    logger.info("Storage path: %s", cfg.storage.path)
    logger.info("Experiments: %s", ", ".join(spec["name"] for spec in experiment_specs))

    data_loader = DataLoader(dataset_dir)
    corpus = data_loader.load_corpus()
    all_queries = data_loader.load_queries()
    query_id_subset = load_query_id_subset(args)
    selected_queries, missing_query_ids = select_queries(all_queries, query_id_subset)

    if not selected_queries:
        raise ValueError("No queries selected for the ablation experiment.")

    if missing_query_ids:
        logger.warning("The following query ids were not found and will be skipped: %s", missing_query_ids)

    selected_query_ids = [str(item["id"]) for item in selected_queries]
    query_concurrency = int(args.query_concurrency or cfg.search.get("query_concurrency", 8))

    run_manifest = {
        "dataset": dataset_name,
        "dataset_path": dataset_dir,
        "storage_path": str(cfg.storage.path),
        "selected_query_count": len(selected_queries),
        "selected_query_ids": selected_query_ids,
        "missing_query_ids": missing_query_ids,
        "experiments": experiment_specs,
        "query_concurrency": query_concurrency,
        "disable_ontology_when_zero": bool(args.disable_ontology_when_zero),
        "reuse_indices_only": bool(args.reuse_indices_only),
    }
    save_json(str(output_root / "run_manifest.json"), run_manifest)

    summary_rows: List[Dict[str, Any]] = []

    for index, spec in enumerate(experiment_specs, start=1):
        experiment_name = spec["name"]
        experiment_weights = spec["weights"]
        experiment_dir = output_root / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[%d/%d] Running experiment '%s' with weights: %s",
            index,
            len(experiment_specs),
            experiment_name,
            experiment_weights,
        )

        experiment_cfg = make_experiment_cfg(
            base_cfg=cfg,
            weights=experiment_weights,
            disable_ontology_when_zero=args.disable_ontology_when_zero,
        )
        save_json(
            str(experiment_dir / "experiment_manifest.json"),
            {
                "experiment": experiment_name,
                "weights": experiment_weights,
                "source": spec.get("source"),
                "ontology_enabled": bool(experiment_cfg.search.get("ontology_enabled", False)),
                "query_ids": selected_query_ids,
            },
        )

        engine = OmniSearch(OmegaConf.to_container(experiment_cfg, resolve=True))

        async def _run_experiment() -> List[Dict[str, Any]]:
            await engine.build_indices(corpus)
            return await run_queries(engine, selected_queries, query_concurrency)

        results = asyncio.run(_run_experiment())
        data_loader.save_results(results, str(experiment_dir / "final_results.jsonl"))

        summary, metrics_rows = evaluate_results(
            dataset_dir=dataset_dir,
            results=results,
            query_ids=selected_query_ids,
        )
        write_metrics_csv(experiment_dir / "evaluation_metrics.csv", metrics_rows)
        save_json(str(experiment_dir / "evaluation_summary.json"), summary)

        summary_row = summarize_row(experiment_name, experiment_weights, summary)
        summary_rows.append(summary_row)

        logger.info(
            "Experiment '%s' finished. Recall@10=%.4f, Recall@All=%.4f, F1@All=%.4f",
            experiment_name,
            summary_row["recall_at_10"],
            summary_row["recall_all"],
            summary_row["f1_all"],
        )

    write_summary_csv(output_root / "ablation_summary.csv", summary_rows)
    save_json(str(output_root / "ablation_summary.json"), summary_rows)

    logger.info("Ablation experiments completed. Summary saved to %s", output_root / "ablation_summary.csv")


if __name__ == "__main__":
    main()
