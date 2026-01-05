import os
from typing import List, Dict, Any
from ..utils.helpers import read_jsonl, write_jsonl, load_json, save_json

class DataLoader:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    def load_corpus(self) -> List[Dict[str, Any]]:
        path = os.path.join(self.dataset_dir, "corpus.jsonl")
        return read_jsonl(path)

    def load_queries(self) -> List[Dict[str, Any]]:
        path = os.path.join(self.dataset_dir, "queries.jsonl")
        return read_jsonl(path)

    def load_relevance(self) -> List[Dict[str, Any]]:
        path = os.path.join(self.dataset_dir, "relevance.jsonl")
        return read_jsonl(path)

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        write_jsonl(output_path, results)
