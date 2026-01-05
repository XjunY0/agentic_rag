import numpy as np
import asyncio
import json
import faiss
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from .ontology_index import TreeNode
from ..models.base import BaseLLMClient, BaseEmbeddingModel
from ..prompts.manager import PromptManager
from ..utils.helpers import parse_llm_json
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class KMeans:
    def __init__(self, k: int, niter: int = 5):
        self.k = k
        self.niter = niter

    def _normalize(self, X):
        norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norm

    def run(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n, d = x.shape
        x = x.astype('float32')
        x = self._normalize(x)
        
        # Try GPU first
        try:
            res = faiss.StandardGpuResources()
            kmeans = faiss.Kmeans(d, self.k, niter=self.niter, verbose=False, gpu=True)
            kmeans.train(x)
            D, I = kmeans.index.search(x, 1)
            return I.reshape(-1), kmeans.centroids
        except Exception:
            # Fallback to CPU
            kmeans = faiss.Kmeans(d, self.k, niter=self.niter, verbose=False, gpu=False)
            kmeans.train(x)
            D, I = kmeans.index.search(x, 1)
            return I.reshape(-1), kmeans.centroids

class Selector:
    @staticmethod
    def medoids(x: np.ndarray, labels: np.ndarray, top_k: int = 5) -> Dict[int, List[int]]:
        """Find indices of documents closest to cluster centroids."""
        unique_labels = np.unique(labels)
        res = {}
        for label in unique_labels:
            if label == -1: continue
            idx = np.where(labels == label)[0]
            if len(idx) == 0: continue
            
            cluster_points = x[idx]
            centroid = np.mean(cluster_points, axis=0)
            
            # Euclidean distance to centroid
            dists = np.sum((cluster_points - centroid) ** 2, axis=1)
            rel_idx = np.argsort(dists)[:top_k]
            res[int(label)] = idx[rel_idx].tolist()
        return res

class Router:
    @staticmethod
    def route(doc_embs: np.ndarray, cat_embs: np.ndarray, threshold: float = 0.45) -> Tuple[np.ndarray, np.ndarray]:
        # Cosine similarity
        sims = doc_embs @ cat_embs.T
        
        assignment = []
        max_scores = []

        for i in range(len(sims)):
            cid = np.argmax(sims[i])
            score = sims[i][cid]
            max_scores.append(score)

            if score < threshold:
                assignment.append(-1)
            else:
                assignment.append(cid)

        return np.array(assignment), np.array(max_scores)

class TreeBuilder:
    def __init__(
        self,
        llm: BaseLLMClient,
        embedder: BaseEmbeddingModel,
        max_depth: int = 4,
        min_cluster_k: int = 10,
        max_cluster_k: int = 60,
        min_docs_to_split: int = 100,
        concurrency: int = 16
    ):
        self.llm = llm
        self.embedder = embedder
        self.max_depth = max_depth
        self.min_cluster_k = min_cluster_k
        self.max_cluster_k = max_cluster_k
        self.min_docs_to_split = min_docs_to_split
        # Semaphore created lazily per event loop to avoid binding to wrong loop
        self._sem = None
        self._concurrency = concurrency

    def _ensure_sem(self) -> asyncio.Semaphore:
        """Ensure a semaphore exists for the current running event loop."""
        loop = asyncio.get_running_loop()
        sem = getattr(self, "_sem", None)
        sem_loop = getattr(self, "_sem_loop", None)
        # create/recreate semaphore for the current loop if needed
        if sem is None or sem_loop is not loop:
            sem = asyncio.Semaphore(self._concurrency)
            self._sem = sem
            self._sem_loop = loop
        return sem

    def choose_raw_k(self, N: int) -> int:
        k = int(np.sqrt(N / 2))
        k = max(self.min_cluster_k, k)
        k = min(self.max_cluster_k, k)
        if N <= 50:
            k = 4
        return k

    async def build_layer(self, docs: List[str], ids: List[str], X: np.ndarray, depth: int, parent_chain: List[str]) -> TreeNode:
        current_name = parent_chain[-1]
        node = TreeNode(name=current_name, level=depth + 1)
        
        # Leaf condition
        if depth >= self.max_depth or len(docs) < self.min_docs_to_split:
            # logger.info(f"Depth {depth}: Leaf reached for {current_name} ({len(docs)} docs)")
            node.doc_ids = ids
            return node

        N = len(docs)
        raw_k = self.choose_raw_k(N)
        raw_k = min(raw_k, N - 1)
        if raw_k <= 1:
            # logger.info(f"Depth {depth}: Too few clusters for {current_name}")
            node.doc_ids = ids
            return node

        # 1. Clustering
        labels, _ = KMeans(raw_k).run(X)
        reps = Selector.medoids(X, labels, top_k=5)

        cluster_samples = {
            cid: [docs[i] for i in idx_list]
            for cid, idx_list in reps.items()
        }

        # 2. Generate raw topics (Sequential to match SelectRAG exactly)
        parent_path = " > ".join(parent_chain)
        raw_topics = []
        for cid in tqdm(sorted(cluster_samples.keys()), desc=f"Generating topics depth={depth}"):
            samples = cluster_samples[cid]
            sample_text = "\n\n".join([s[:500] for s in samples])
            prompt = PromptManager.get_prompt('TREE_GEN_TOPIC').format(
                parent_path=parent_path,
                sample_text=sample_text
            )
            sem = self._ensure_sem()
            async with sem:
                res = await self.llm.chat([{"role": "user", "content": prompt}], response_format={"type": "json_object"})
            
            try:
                data = parse_llm_json(res) or {}
                name = data.get("name", f"Subtopic {cid}")
                desc = data.get("description", "")
                raw_topics.append({"name": name, "description": desc, "cid": cid})
            except Exception as e:
                logger.warning(f"Failed to parse topic for cluster {cid}: {e}")
                raw_topics.append({"name": f"Subtopic {cid}", "description": "", "cid": cid})

        # logger.info(f"Depth {depth}: Generated {len(raw_topics)} raw topics for {parent_path}")

        # 3. Merge raw topics into broader categories
        prompt_merge = PromptManager.get_prompt('TREE_MERGE_CATEGORIES').format(
            depth=depth + 1,
            parent_path=parent_path,
            raw_topics_json=json.dumps(raw_topics, indent=2)
        )
        sem = self._ensure_sem()
        async with sem:
            res_merge = await self.llm.chat([{"role": "user", "content": prompt_merge}], response_format={"type": "json_object"})
        try:
            merged = parse_llm_json(res_merge)
            if not isinstance(merged, list):
                if isinstance(merged, dict) and "categories" in merged:
                    merged = merged["categories"]
                else:
                    raise ValueError("Merged result is not a list")
        except Exception:
            # Fallback: treat each raw topic as a separate merged group
            merged = [{"name": t["name"], "description": t["description"], "member_names": [t["name"]]} for t in raw_topics]

        merged_names = [m["name"] for m in merged]
        merged_descs = [m["description"] for m in merged]
        
        # logger.info(f"Depth {depth}: Merged into {len(merged_names)} categories for {parent_path}")

        # 4. Embedding merged categories (Sequential to match SelectRAG)
        cat_embs_list = []
        for i in tqdm(range(len(merged_names)), desc="Embedding merged categories"):
            text = f"{merged_names[i]} — {merged_descs[i]}"
            emb = await self.embedder.encode_async([text])
            cat_embs_list.append(emb[0])
        cat_embs = np.vstack(cat_embs_list)
        faiss.normalize_L2(cat_embs)

        # 5. Map raw topics to merged categories
        raw_names = [rt["name"] for rt in raw_topics]
        raw_embs_list = []
        for name in tqdm(raw_names, desc="Embedding raw topics"):
            emb = await self.embedder.encode_async([name])
            raw_embs_list.append(emb[0])
        raw_embs = np.vstack(raw_embs_list)
        faiss.normalize_L2(raw_embs)

        raw_assign, raw_scores = Router.route(raw_embs, cat_embs, threshold=0.45)
        # logger.info(f"Depth {depth}: Raw topic mapping scores: min={np.min(raw_scores):.4f}, avg={np.mean(raw_scores):.4f}")

        # 6. Map each doc to merged category
        assignment = np.full(N, -1, dtype=int)
        for raw_cid, merged_cid in enumerate(raw_assign):
            if merged_cid == -1: continue
            doc_idx = np.where(labels == raw_cid)[0]
            assignment[doc_idx] = merged_cid

        # 7. Summarize node description
        child_block = "\n".join([f"- {d}" for d in merged_descs if d])
        if child_block:
            prompt_summ = PromptManager.get_prompt('TREE_SUMMARIZE_NODE').format(
                parent_path=parent_path,
                child_block=child_block
            )
            sem = self._ensure_sem()
            async with sem:
                node.desc = await self.llm.chat([{"role": "user", "content": prompt_summ}])
        
        # 8. Recursive build
        for cid in range(len(merged_names)):
            idx = np.where(assignment == cid)[0]
            if len(idx) == 0:
                continue

            sub_docs = [docs[i] for i in idx]
            sub_ids = [ids[i] for i in idx]
            sub_X = X[idx]

            child_chain = parent_chain + [merged_names[cid]]
            child_node = await self.build_layer(sub_docs, sub_ids, sub_X, depth + 1, child_chain)
            child_node.desc = merged_descs[cid]
            node.add_child(child_node)

        # Documents not assigned to any child stay at this node
        unassigned_idx = np.where(assignment == -1)[0]
        for i in unassigned_idx:
            node.add_doc(ids[i])

        return node

    async def build(self, docs: List[str], ids: List[str], X: np.ndarray) -> TreeNode:
        logger.info(f"Starting hierarchical tree building for {len(docs)} documents...")
        return await self.build_layer(docs, ids, X, depth=0, parent_chain=["ROOT"])
