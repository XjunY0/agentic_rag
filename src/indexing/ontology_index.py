import os
import json
import uuid
import re
import hashlib
import asyncio
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Set, Tuple
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from ..utils.logger import setup_logger
from ..models.base import BaseLLMClient, BaseEmbeddingModel
from ..prompts.manager import PromptManager
from ..utils.helpers import parse_llm_json
from ..utils.visualize_ontology import generate_html

logger = setup_logger(__name__)

def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return " ".join(name.split())

class TreeNode:
    def __init__(self, name: str, level: int, desc: str = "", node_id: str = None):
        self.node_id = node_id or uuid.uuid4().hex
        self.name = name
        self.level = level
        self.desc = desc
        self.parent: Optional["TreeNode"] = None
        self.children: List["TreeNode"] = []
        self.doc_ids: List[str] = []

    def add_child(self, child: "TreeNode"):
        child.parent = self
        self.children.append(child)

    def add_doc(self, doc_id: str):
        if doc_id not in self.doc_ids:
            self.doc_ids.append(doc_id)

    def clear_docs(self):
        self.doc_ids = []
        for child in self.children:
            child.clear_docs()

    def concept_path(self) -> str:
        names = []
        n = self
        while n:
            names.append(n.name)
            n = n.parent
        return " → ".join(reversed(names))

    def embedding_text(self) -> str:
        path = self.concept_path()
        parts = [
            PromptManager.NODE_EMBED_PROMPT,
            "Concept hierarchy path:\n",
            path,
        ]
        if self.desc:
            parts.extend([
                "\n\nCore semantic definition of this node:\n",
                self.desc
            ])
        return "".join(parts)

    def path_string(self):
        # Keep for backward compatibility or internal use if needed, 
        # but embedding_text is preferred for indexing.
        return self.concept_path()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "desc": self.desc,
            "level": self.level,
            "doc_ids": self.doc_ids,
            "children": [c.to_dict() for c in self.children],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any], parent: Optional["TreeNode"] = None) -> "TreeNode":
        node = TreeNode(
            name=data.get("name", "Unnamed"),
            level=data.get("level", 1),
            desc=data.get("desc", ""),
            node_id=data.get("node_id"),
        )
        node.doc_ids = data.get("doc_ids", [])
        node.parent = parent
        for child_data in data.get("children", []):
            node.add_child(TreeNode.from_dict(child_data, node))
        return node


def format_node_for_prompt(
    node: TreeNode,
    include_desc: bool = False,
    node_similarity: Optional[float] = None,
) -> str:
    lines = [
        f"node_id: {node.node_id}",
        f"concept_path: {node.concept_path()}",
        f"attached_doc_count: {len(node.doc_ids)}",
    ]
    if node_similarity is not None:
        lines.append(f"node_similarity: {node_similarity:.4f}")
    if include_desc and node.desc:
        lines.append(f"semantic_definition: {node.desc}")
    return "\n   ".join(lines)

class FaissNodeIndex:
    def __init__(self, dim: int, M: int = 32):
        self.dim = dim
        self.M = M
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200
        self.nodes: List[TreeNode] = []

    def add_nodes_with_vectors(self, nodes: List[TreeNode], vectors: np.ndarray):
        vectors = np.ascontiguousarray(vectors)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.nodes.extend(nodes)

    def search(self, vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        vectors = np.ascontiguousarray(vectors)
        faiss.normalize_L2(vectors)
        return self.index.search(vectors, top_k)

    def save(self, index_path: str):
        faiss.write_index(self.index, index_path)

class VectorClassifier:
    """Vector-based document assignment with dynamic splitting and index updates."""
    def __init__(
        self, 
        root: TreeNode, 
        embedder: BaseEmbeddingModel, 
        llm: BaseLLMClient,
        config: Dict[str, Any]
    ):
        self.root = root
        self.embedder = embedder
        self.llm = llm
        self.config = config
        
        self.min_level = config.get('min_level_for_mount', 2)
        self.top_k_recall = config.get('top_k_recall', 10)
        self.top_k_attach = config.get('top_k_mount', 2)
        self.split_threshold = config.get('split_threshold', 60)
        self.max_split_children = config.get('max_k_children', 3)
        self.max_level = config.get('max_depth', 5)
        self.llm_max_concurrency = config.get('concurrency', 16)
        self.batch_size = config.get('batch_size', 64)
        
        self.index: Optional[FaissNodeIndex] = None
        # lazy-create index lock per event loop to avoid loop binding issues
        self._index_lock = None
        self._node_locks: Dict[str, asyncio.Lock] = {}
        self._split_tasks: Dict[str, asyncio.Task] = {}

    def _get_node_lock(self, node_id: str) -> asyncio.Lock:
        if node_id not in self._node_locks:
            self._node_locks[node_id] = asyncio.Lock()
        return self._node_locks[node_id]

    @staticmethod
    def _is_ancestor(ancestor: TreeNode, descendant: TreeNode) -> bool:
        cur = descendant.parent
        while cur:
            if cur is ancestor:
                return True
            cur = cur.parent
        return False

    def collect_nodes(self) -> List[TreeNode]:
        nodes = []
        def dfs(n: TreeNode):
            if n.level >= self.min_level:
                nodes.append(n)
            for c in n.children:
                dfs(c)
        dfs(self.root)
        return nodes

    async def build_index_initial(self):
        nodes = self.collect_nodes()
        if not nodes:
            # If no nodes at min_level, at least index the root or children of root
            nodes = self.root.children if self.root.children else [self.root]

        texts = [n.embedding_text() for n in nodes]
        # Use async-aware encode
        encode_fn = getattr(self.embedder, "encode_async", None)
        if encode_fn is None:
            vecs = await asyncio.to_thread(self.embedder.encode, texts)
        else:
            vecs = await encode_fn(texts)

        dim = vecs.shape[1]
        self.index = FaissNodeIndex(dim=dim)
        self.index.add_nodes_with_vectors(nodes, vecs)

    async def _faiss_search_one(self, doc_vec: np.ndarray) -> List[TreeNode]:
        # Perform FAISS search in a thread to avoid blocking the event loop.
        # We avoid holding the global index lock for reads so multiple searches can run concurrently.
        if self.index is None:
            return []

        # ensure contiguous and correct dtype/shape
        q = np.ascontiguousarray(doc_vec.astype('float32').reshape(1, -1))

        def _search():
            return self.index.search(q, self.top_k_recall)

        try:
            scores, indices = await asyncio.to_thread(_search)
        except Exception as e:
            logger.error(f"Faiss search failed: {e}")
            return []

        idxs = indices[0].tolist()
        # snapshot nodes list to avoid race with concurrent writes
        nodes_snapshot = list(self.index.nodes)
        candidates = [nodes_snapshot[i] for i in idxs if i >= 0 and i < len(nodes_snapshot)]
        return candidates

    async def _faiss_add_new_nodes(self, new_nodes: List[TreeNode]):
        texts = [n.embedding_text() for n in new_nodes]
        encode_fn = getattr(self.embedder, "encode_async", None)
        if encode_fn is None:
            vecs = await asyncio.to_thread(self.embedder.encode, texts)
        else:
            vecs = await encode_fn(texts)
        # ensure lock exists for current loop
        lock = getattr(self, "_index_lock", None)
        if lock is None:
            self._index_lock = asyncio.Lock()
            lock = self._index_lock
        async with lock:
            self.index.add_nodes_with_vectors(new_nodes, vecs)

    async def _llm_filter_nodes(
        self,
        doc_text: str,
        candidates: List[TreeNode],
        top_k: Optional[int] = None,
    ) -> List[str]:
        include_desc = bool(self.config.get("include_desc_in_mount_prompt", True))
        limit = int(top_k if top_k is not None else self.top_k_attach)
        nodes_text = "\n".join(
            "- " + format_node_for_prompt(n, include_desc=include_desc)
            for n in candidates
        )
        prompt = PromptManager.get_prompt('PROMPT_FILTER_NODES').format(
            doc_text=doc_text[:2000], # Truncate for LLM
            nodes=nodes_text,
            top_k=limit,
        )
        try:
            res_text = await self.llm.chat([{"role": "user", "content": prompt}], response_format={"type": "json_object"})
            data = parse_llm_json(res_text)
            if not data:
                return []
            keep = data.get("keep", [])
            if not isinstance(keep, list):
                return []
            cand_ids = {n.node_id for n in candidates}
            kept_ids = [x for x in keep if x in cand_ids][:limit]
            if not self.config.get("prefer_deepest_mount_only", False):
                return kept_ids

            id_to_node = {n.node_id: n for n in candidates}
            pruned = []
            for nid in kept_ids:
                node = id_to_node.get(nid)
                if node is None:
                    continue
                shadowed = False
                for other_id in kept_ids:
                    if other_id == nid:
                        continue
                    other = id_to_node.get(other_id)
                    if other is not None and self._is_ancestor(node, other):
                        shadowed = True
                        break
                if not shadowed:
                    pruned.append(nid)
            return pruned[:limit]
        except Exception:
            return []

    async def _wait_for_pending_splits(self):
        pending = [t for t in self._split_tasks.values() if t and not t.done()]
        if pending:
            await asyncio.gather(*pending)

    def collect_attached_doc_ids(self) -> Set[str]:
        attached: Set[str] = set()

        def dfs(node: TreeNode):
            for doc_id in node.doc_ids:
                attached.add(doc_id)
            for child in node.children:
                dfs(child)

        dfs(self.root)
        return attached

    async def attach_doc_to_node(self, node: TreeNode, doc_id: str, corpus_map: Dict[str, str]):
        lk = self._get_node_lock(node.node_id)
        async with lk:
            node.add_doc(doc_id)
            if len(node.doc_ids) >= self.split_threshold and node.level < self.max_level:
                t = self._split_tasks.get(node.node_id)
                if t is None or t.done():
                    self._split_tasks[node.node_id] = asyncio.create_task(
                        self._split_node_locked(node, corpus_map)
                    )

    async def _split_node_locked(self, node: TreeNode, corpus_map: Dict[str, str]):
        lk = self._get_node_lock(node.node_id)
        async with lk:
            if len(node.doc_ids) < self.split_threshold:
                return
            
            docs_payload = []
            for d in node.doc_ids:
                text = corpus_map.get(d, "")
                if text:
                    docs_payload.append(f'- doc_id: "{d}"\n  summary: "{text[:500]}"')
            
            docs_text = "\n".join(docs_payload)
            children_names = ", ".join(c.name for c in node.children) if node.children else "(none)"
            
            prompt = PromptManager.get_prompt('PROMPT_SPLIT_NODE').format(
                node_chain=node.concept_path(),
                children=children_names,
                docs=docs_text if docs_text else "(none)",
            )
        try:
            res_text = await self.llm.chat([{"role": "user", "content": prompt}], response_format={"type": "json_object"})
            data = parse_llm_json(res_text)
            if not data:
                return

            children_specs = data.get("children", [])
            remain = data.get("remain_doc_ids", [])

            if not isinstance(children_specs, list):
                children_specs = []
            if not isinstance(remain, list):
                remain = []

            children_specs = children_specs[:self.max_split_children]
            allowed = set(node.doc_ids)
            new_children_nodes = []
            assigned_to_children = set()

            for spec in children_specs:
                name = str(spec.get("name", "")).strip()
                desc = str(spec.get("desc", "")).strip()
                doc_ids = spec.get("doc_ids", [])
                if not name or not isinstance(doc_ids, list):
                    continue

                child = TreeNode(name=name, desc=desc, level=node.level + 1)
                for d in doc_ids:
                    if d in allowed:
                        child.add_doc(d)
                        assigned_to_children.add(d)
                node.add_child(child)
                new_children_nodes.append(child)

            remain_valid = [d for d in remain if d in allowed and d not in assigned_to_children]
            mentioned = assigned_to_children.union(set(remain_valid))
            fallback = [d for d in node.doc_ids if d not in mentioned]
            node.doc_ids = remain_valid + fallback

            if new_children_nodes:
                await self._faiss_add_new_nodes(new_children_nodes)
        except Exception as e:
            logger.error(f"Split failed for node {node.node_id}: {e}")

    async def process_docs_streaming(self, corpus: List[Dict[str, Any]], doc_vecs: np.ndarray):
        sem = asyncio.Semaphore(self.llm_max_concurrency)
        corpus_map = {str(d["id"]): d["text"] for d in corpus}
        
        async def handle_one(i: int):
            doc = corpus[i]
            doc_id = str(doc["id"])
            vec = doc_vecs[i]
            
            # FAISS search is fast, do it outside the LLM semaphore
            candidates = await self._faiss_search_one(vec)
            if not candidates: return
            
            async with sem: # Only LLM filtering is rate-limited
                keep_ids = await self._llm_filter_nodes(doc["text"], candidates)
                if not keep_ids: return
                
                id_to_node = {n.node_id: n for n in candidates}
                for nid in keep_ids:
                    n = id_to_node.get(nid)
                    if n:
                        await self.attach_doc_to_node(n, doc_id, corpus_map)

        tasks = [handle_one(i) for i in range(len(corpus))]
        
        # Use as_completed for better resource management during streaming
        for fut in tqdm_asyncio.as_completed(tasks, desc="Processing docs for ontology"):
            try:
                await fut
            except Exception as e:
                logger.error(f"Error processing doc: {e}")

        await self._wait_for_pending_splits()

    async def ensure_full_mount(self, corpus: List[Dict[str, Any]], doc_vecs: np.ndarray) -> Dict[str, int]:
        stats = {
            "unmounted_before": 0,
            "llm_attached": 0,
            "vector_fallback_attached": 0,
            "still_unmounted": 0,
        }
        if not self.config.get("ensure_full_mount", True):
            return stats

        mounted = self.collect_attached_doc_ids()
        unmounted_indices = [
            i for i, doc in enumerate(corpus)
            if str(doc["id"]) not in mounted
        ]
        stats["unmounted_before"] = len(unmounted_indices)
        if not unmounted_indices:
            return stats

        sem = asyncio.Semaphore(self.llm_max_concurrency)
        corpus_map = {str(d["id"]): d["text"] for d in corpus}

        async def handle_one(i: int) -> str:
            doc = corpus[i]
            doc_id = str(doc["id"])
            vec = doc_vecs[i]
            candidates = await self._faiss_search_one(vec)
            if not candidates:
                return "still_unmounted"

            async with sem:
                keep_ids = await self._llm_filter_nodes(doc["text"], candidates)
            id_to_node = {n.node_id: n for n in candidates}
            if keep_ids:
                for nid in keep_ids:
                    node = id_to_node.get(nid)
                    if node is not None:
                        await self.attach_doc_to_node(node, doc_id, corpus_map)
                return "llm_attached"

            if self.config.get("force_mount_fallback_to_best_candidate", True):
                await self.attach_doc_to_node(candidates[0], doc_id, corpus_map)
                return "vector_fallback_attached"

            return "still_unmounted"

        tasks = [handle_one(i) for i in unmounted_indices]
        for fut in tqdm_asyncio.as_completed(tasks, desc="Repairing unmounted docs"):
            try:
                outcome = await fut
                stats[outcome] = stats.get(outcome, 0) + 1
            except Exception as e:
                logger.error(f"Failed to repair unmounted doc: {e}")
                stats["still_unmounted"] += 1

        await self._wait_for_pending_splits()

        mounted_after = self.collect_attached_doc_ids()
        stats["still_unmounted"] = sum(
            1 for doc in corpus if str(doc["id"]) not in mounted_after
        )
        return stats

    async def push_docs_to_children(self, corpus: List[Dict[str, Any]], doc_vecs: np.ndarray) -> Dict[str, int]:
        stats = {
            "docs_considered": 0,
            "docs_pushed": 0,
            "parent_docs_retained": 0,
        }
        if not self.config.get("pushdown_after_mount", True):
            return stats

        corpus_map = {str(d["id"]): d["text"] for d in corpus}
        doc_id_to_idx = {str(d["id"]): i for i, d in enumerate(corpus)}
        sem = asyncio.Semaphore(self.llm_max_concurrency)
        child_vec_cache: Dict[Tuple[str, ...], np.ndarray] = {}
        candidate_children = int(self.config.get("pushdown_candidate_children", 3))
        top_k_children = int(self.config.get("pushdown_top_k_children", 1))

        async def get_child_vecs(children: List[TreeNode]) -> np.ndarray:
            cache_key = tuple(child.node_id for child in children)
            if cache_key in child_vec_cache:
                return child_vec_cache[cache_key]

            texts = [child.embedding_text() for child in children]
            encode_fn = getattr(self.embedder, "encode_async", None)
            if encode_fn is None:
                vecs = await asyncio.to_thread(self.embedder.encode, texts)
            else:
                vecs = await encode_fn(texts)

            vecs = np.ascontiguousarray(vecs)
            faiss.normalize_L2(vecs)
            child_vec_cache[cache_key] = vecs
            return vecs

        async def recurse(node: TreeNode):
            children = [child for child in node.children if child is not None]
            if children and node.doc_ids:
                child_vecs = await get_child_vecs(children)
                remaining_doc_ids = []

                for doc_id in list(node.doc_ids):
                    stats["docs_considered"] += 1
                    doc_idx = doc_id_to_idx.get(doc_id)
                    doc_text = corpus_map.get(doc_id, "")
                    if doc_idx is None or not doc_text:
                        remaining_doc_ids.append(doc_id)
                        stats["parent_docs_retained"] += 1
                        continue

                    sims = child_vecs @ doc_vecs[doc_idx]
                    shortlist_idx = np.argsort(-sims)[: min(len(children), candidate_children)]
                    candidates = [children[i] for i in shortlist_idx]

                    async with sem:
                        keep_ids = await self._llm_filter_nodes(
                            doc_text,
                            candidates,
                            top_k=top_k_children,
                        )

                    if keep_ids:
                        id_to_node = {candidate.node_id: candidate for candidate in candidates}
                        moved = False
                        for nid in keep_ids[:top_k_children]:
                            target = id_to_node.get(nid)
                            if target is not None:
                                await self.attach_doc_to_node(target, doc_id, corpus_map)
                                moved = True
                        if moved:
                            stats["docs_pushed"] += 1
                        else:
                            remaining_doc_ids.append(doc_id)
                            stats["parent_docs_retained"] += 1
                    else:
                        remaining_doc_ids.append(doc_id)
                        stats["parent_docs_retained"] += 1

                node.doc_ids = remaining_doc_ids

            for child in children:
                await recurse(child)

        await recurse(self.root)
        await self._wait_for_pending_splits()
        return stats

class OntologyIndex:
    def __init__(self, index_path: str, config: Dict[str, Any] = None):
        self.index_path = index_path
        self.config = config or {}
        self.root = TreeNode("Root", 1)
        self.search_index = None
        self.search_node_ids = []
        self.doc_ids: List[str] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        self.doc_vecs: Optional[np.ndarray] = None
        self.corpus_doc_ids: List[str] = []
        self.build_stats: Dict[str, Any] = {}

    def _cache_runtime_doc_data(self, corpus: List[Dict[str, Any]], doc_vecs: Optional[Any]):
        self.doc_ids = [str(doc["id"]) for doc in corpus]
        self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        if doc_vecs is None:
            self.doc_vecs = None
            return

        vecs = np.ascontiguousarray(np.asarray(doc_vecs, dtype="float32"))
        if vecs.ndim != 2 or vecs.shape[0] != len(self.doc_ids):
            logger.warning(
                "Skipping ontology runtime doc-vector cache due to shape mismatch: "
                f"vecs={getattr(vecs, 'shape', None)} docs={len(self.doc_ids)}"
            )
            self.doc_vecs = None
            return

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        if np.any((norms > 0) & (np.abs(norms - 1.0) > 1e-3)):
            vecs = vecs / np.clip(norms, 1e-12, None)
        self.doc_vecs = vecs

    def _score_docs_for_node(
        self,
        query_vec: np.ndarray,
        node: TreeNode,
        node_relevance: float,
        node_similarity: float,
        max_docs_per_node: int,
        doc_similarity_weight: float,
        node_similarity_weight: float,
    ) -> List[Dict[str, Any]]:
        if not node.doc_ids:
            return []

        doc_sims: Dict[str, float] = {}
        if self.doc_vecs is not None and self.doc_id_to_idx:
            valid_pairs = [(did, self.doc_id_to_idx.get(did)) for did in node.doc_ids]
            valid_pairs = [(did, idx) for did, idx in valid_pairs if idx is not None]
            if valid_pairs:
                indices = [idx for _, idx in valid_pairs]
                sims = np.dot(self.doc_vecs[indices], query_vec[0])
                doc_sims = {
                    did: float(sim)
                    for (did, _), sim in zip(valid_pairs, sims)
                }

        scored_docs = []
        for did in node.doc_ids:
            doc_sim = max(0.0, doc_sims.get(did, 0.0))
            score = (
                float(node_relevance)
                + doc_similarity_weight * doc_sim
                + node_similarity_weight * max(0.0, float(node_similarity))
            )
            scored_docs.append({
                "id": did,
                "score": float(score),
                "node_id": node.node_id,
                "concept_path": node.concept_path(),
                "node_relevance": float(node_relevance),
                "node_similarity": float(node_similarity),
                "doc_similarity": float(doc_sim),
            })

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:max_docs_per_node]

    def build_report(self) -> Dict[str, Any]:
        nodes = self.get_all_nodes()
        leaves = [n for n in nodes if not n.children]
        internal_nodes = [n for n in nodes if n.children]
        level_counts: Dict[int, int] = {}
        docs_to_nodes: Dict[str, List[TreeNode]] = {}
        sibling_name_duplicates = []

        for node in nodes:
            level_counts[node.level] = level_counts.get(node.level, 0) + 1
            for did in node.doc_ids:
                docs_to_nodes.setdefault(did, []).append(node)
            if node.children:
                seen_names: Dict[str, int] = {}
                for child in node.children:
                    norm = normalize_name(child.name)
                    seen_names[norm] = seen_names.get(norm, 0) + 1
                dup_names = [name for name, count in seen_names.items() if name and count > 1]
                if dup_names:
                    sibling_name_duplicates.append({
                        "concept_path": node.concept_path(),
                        "duplicate_names": dup_names,
                    })

        def _stats(counts: List[int]) -> Dict[str, float]:
            if not counts:
                return {"min": 0, "mean": 0.0, "median": 0.0, "max": 0}
            arr = np.asarray(counts)
            return {
                "min": int(arr.min()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "max": int(arr.max()),
            }

        top_n = int(
            self.config.get(
                "build_report_top_n",
                self.config.get("audit_top_large_nodes", 20),
            )
        )
        overloaded_nodes = sorted(
            [
                {
                    "level": node.level,
                    "name": node.name,
                    "concept_path": node.concept_path(),
                    "doc_count": len(node.doc_ids),
                    "child_count": len(node.children),
                    "has_desc": bool(node.desc),
                }
                for node in nodes
            ],
            key=lambda x: (-x["doc_count"], x["level"], x["concept_path"]),
        )[:top_n]

        corpus_doc_ids = set(self.corpus_doc_ids)
        mounted_doc_ids = set(docs_to_nodes.keys())
        unmounted_doc_ids = sorted(corpus_doc_ids - mounted_doc_ids) if corpus_doc_ids else []

        return {
            "corpus_doc_count": len(corpus_doc_ids),
            "mounted_doc_count": len(mounted_doc_ids),
            "unmounted_doc_count": len(unmounted_doc_ids),
            "unmounted_doc_ids_preview": unmounted_doc_ids[:50],
            "total_nodes": len(nodes),
            "leaf_nodes": len(leaves),
            "internal_nodes": len(internal_nodes),
            "max_depth": max((node.level for node in nodes), default=0),
            "level_counts": level_counts,
            "leaf_doc_stats": _stats([len(node.doc_ids) for node in leaves]),
            "internal_doc_stats": _stats([len(node.doc_ids) for node in internal_nodes]),
            "nodes_without_desc": sum(1 for node in nodes if not (node.desc or "").strip()),
            "internal_nodes_with_docs": sum(1 for node in internal_nodes if node.doc_ids),
            "docs_attached_to_multiple_nodes": sum(1 for attached in docs_to_nodes.values() if len(attached) > 1),
            "max_doc_membership": max((len(attached) for attached in docs_to_nodes.values()), default=0),
            "sibling_name_duplicates": sibling_name_duplicates[:top_n],
            "overloaded_nodes": overloaded_nodes,
            "build_stats": self.build_stats,
        }

    async def build(self, corpus: List[Dict[str, Any]], llm: BaseLLMClient, embedder: BaseEmbeddingModel, doc_vecs: Optional[Any] = None, force: bool = False):
        self.corpus_doc_ids = [str(doc["id"]) for doc in corpus]
        self.build_stats = {}
        if os.path.exists(self.index_path) and not force:
            logger.info(f"Ontology index already exists at {self.index_path}. Skipping build.")
            self.load()
            self._cache_runtime_doc_data(corpus, doc_vecs)
            return

        logger.info(f"Building Ontology index at {self.index_path}...")
        
        # 1. Hierarchical Clustering to build initial tree (if root is empty)
        if not self.root.children and not self.root.doc_ids:
            from .tree_builder import TreeBuilder
            
            texts = [d["text"] for d in corpus]
            ids = [str(d["id"]) for d in corpus]

            # Use provided precomputed doc_vecs if available to avoid recomputing embeddings
            if doc_vecs is None:
                batch_size = self.config.get("batch_size", 64)
                vecs = []
                for i in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus"):
                    batch = texts[i : i + batch_size]
                    # Use async encode if available
                    encode_fn = getattr(embedder, "encode_async", None)
                    if encode_fn:
                        b_vecs = await encode_fn(batch)
                    else:
                        b_vecs = await asyncio.to_thread(embedder.encode, batch)
                    vecs.append(np.ascontiguousarray(b_vecs))
                doc_vecs = np.vstack(vecs)

            # Normalize for cosine similarity
            doc_vecs = np.ascontiguousarray(doc_vecs)
            faiss.normalize_L2(doc_vecs)
            
            builder = TreeBuilder(
                llm=llm,
                embedder=embedder,
                max_depth=self.config.get("max_depth", 4),
                min_cluster_k=self.config.get("min_cluster_k", 10),
                max_cluster_k=self.config.get("max_cluster_k", 60),
                min_docs_to_split=self.config.get("min_docs_to_split", 100),
                concurrency=self.config.get("concurrency", 16),
                sample_docs_for_split=self.config.get("sample_docs_for_split", 10),
            )
            
            self.root = await builder.build(texts, ids, doc_vecs)
            
            # Clear doc_ids from initial clustering to allow VectorClassifier to re-assign them properly
            self.root.clear_docs()
        else:
            logger.info("Using existing tree structure for document mounting.")
            # If we already have a tree (e.g. loaded from file), we still need doc_vecs for mounting
            if doc_vecs is None:
                texts = [d["text"] for d in corpus]
                batch_size = self.config.get("batch_size", 64)
                vecs = []
                for i in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus for mounting"):
                    batch = texts[i : i + batch_size]
                    encode_fn = getattr(embedder, "encode_async", None)
                    if encode_fn:
                        b_vecs = await encode_fn(batch)
                    else:
                        b_vecs = await asyncio.to_thread(embedder.encode, batch)
                    vecs.append(np.ascontiguousarray(b_vecs))
                doc_vecs = np.vstack(vecs)
                doc_vecs = np.ascontiguousarray(doc_vecs)
                faiss.normalize_L2(doc_vecs)

        # 2. Initial node index build for future dynamic updates
        classifier = VectorClassifier(self.root, embedder, llm, self.config)
        await classifier.build_index_initial()

        # 3. Streaming assign with split-on-the-fly
        await classifier.process_docs_streaming(corpus, doc_vecs)
        self.build_stats = {
            "mount_mode": "vector_classifier",
        }

        # 4. Clean tree
        self.clean_tree()
        
        # 5. Build Search Index (Static for now)
        logger.info("Building static search index for ontology...")
        nodes = self.get_all_nodes()
        # Filter nodes that are useful for search (e.g. have docs attached)
        self.search_node_ids = [n.node_id for n in nodes if n.doc_ids and n.level > 1]
        
        if self.search_node_ids:
            nodes_map = {n.node_id: n for n in nodes}
            target_nodes = [nodes_map[nid] for nid in self.search_node_ids]
            texts = [n.embedding_text() for n in target_nodes]
            
            encode_fn = getattr(embedder, "encode_async", None)
            if encode_fn:
                vecs = await encode_fn(texts)
            else:
                vecs = await asyncio.to_thread(embedder.encode, texts)
            
            vecs = np.ascontiguousarray(vecs)
            faiss.normalize_L2(vecs)
            self.search_index = faiss.IndexFlatIP(vecs.shape[1])
            self.search_index.add(vecs)

        self._cache_runtime_doc_data(corpus, doc_vecs)
        
        self.save()
        if classifier.index:
            classifier.index.save(self.index_path + ".faiss")
        logger.info("Ontology index built successfully.")

    def save(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.root.to_dict(), f, ensure_ascii=False, indent=2)
            
        # Save search index meta
        if self.search_node_ids:
            with open(self.index_path + ".meta", "w", encoding="utf-8") as f:
                json.dump(self.search_node_ids, f)
            
            if self.search_index:
                faiss.write_index(self.search_index, self.index_path + ".search.faiss")
                
        # Generate visualization
        html_path = os.path.splitext(self.index_path)[0] + ".html"
        try:
            generate_html(self.index_path, html_path)
            logger.info(f"Ontology visualization saved to {html_path}")
        except Exception as e:
            logger.error(f"Failed to generate ontology visualization: {e}")

        if self.config.get("build_report_enabled") or self.config.get("audit_enabled"):
            report_path = os.path.splitext(self.index_path)[0] + ".report.json"
            audit_path = os.path.splitext(self.index_path)[0] + ".audit.json"
            try:
                report = self.build_report()
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                with open(audit_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                logger.info(f"Ontology build report saved to {report_path}")
            except Exception as e:
                logger.error(f"Failed to save ontology build report: {e}")

    def load(self):
        with open(self.index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.root = TreeNode.from_dict(data)
            
        if os.path.exists(self.index_path + ".meta"):
            with open(self.index_path + ".meta", "r") as f:
                self.search_node_ids = json.load(f)
        
        if os.path.exists(self.index_path + ".search.faiss"):
            self.search_index = faiss.read_index(self.index_path + ".search.faiss")

    def clean_tree(self, node: Optional[TreeNode] = None):
        if node is None:
            node = self.root
        kept = []
        for c in node.children:
            self.clean_tree(c)
            if c.doc_ids or c.children:
                kept.append(c)
        node.children = kept

    def get_all_nodes(self) -> List[TreeNode]:
        nodes = []
        def dfs(node):
            nodes.append(node)
            for child in node.children:
                dfs(child)
        dfs(self.root)
        return nodes

    async def search(self, query: str, embedder: BaseEmbeddingModel, llm: BaseLLMClient, top_k_nodes: int = 20) -> Dict[str, Any]:
        """Search for relevant documents via ontology nodes."""
        if not self.search_index or not self.search_node_ids:
            return {"results": [], "trace": None}
        
        encode_fn = getattr(embedder, "encode_async", None)
        if encode_fn is None:
            query_vec = await asyncio.to_thread(embedder.encode, [query])
        else:
            query_vec = await encode_fn([query])
        query_vec = np.ascontiguousarray(query_vec)
        faiss.normalize_L2(query_vec)
        
        scores, indices = self.search_index.search(query_vec, min(top_k_nodes, len(self.search_node_ids)))
        
        all_nodes = {n.node_id: n for n in self.get_all_nodes()}
        candidate_entries = []
        for rank, (node_score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx != -1 and idx < len(self.search_node_ids):
                nid = self.search_node_ids[idx]
                if nid in all_nodes:
                    candidate_entries.append({
                        "rank": rank,
                        "node": all_nodes[nid],
                        "node_similarity": float(node_score),
                    })
        
        if not candidate_entries:
            return {"results": [], "trace": None}
        
        include_desc = bool(self.config.get("include_desc_in_search_prompt", True))
        concept_lines = [
            f"{entry['rank']}. {format_node_for_prompt(entry['node'], include_desc=include_desc, node_similarity=entry['node_similarity'])}"
            for entry in candidate_entries
        ]
        concepts_text = "\n".join(concept_lines)
        
        prompt = PromptManager.get_prompt('ONTOLOGY_RELEVANCE_USER').format(query=query, concepts_text=concepts_text)
        messages = [
            {"role": "system", "content": PromptManager.get_prompt('ONTOLOGY_RELEVANCE_SYSTEM')},
            {"role": "user", "content": prompt}
        ]
        try:
            res_text = await llm.chat(messages, response_format={"type": "json_object"})
            relevance_map = parse_llm_json(res_text)
            if not relevance_map:
                return {"results": [], "trace": None}

            relevance2score = {"strong": 2, "weak": 1, "none": 0}
            max_docs_per_node = int(self.config.get("search_max_docs_per_node", 8))
            doc_similarity_weight = float(self.config.get("search_doc_similarity_weight", 1.0))
            node_similarity_weight = float(self.config.get("search_node_similarity_weight", 0.15))

            doc_scores: Dict[str, Dict[str, Any]] = {}
            node_map = {entry["node"].node_id: entry for entry in candidate_entries}
            selected_nodes = []
            for nid, rel in relevance_map.items():
                score = relevance2score.get(rel.lower().strip(), 0)
                if score > 0 and nid in node_map:
                    entry = node_map[nid]
                    node = entry["node"]
                    selected_nodes.append({
                        "node_id": node.node_id,
                        "concept_path": node.concept_path(),
                        "node_similarity": float(entry["node_similarity"]),
                        "llm_relevance": rel.lower().strip(),
                        "doc_count": len(node.doc_ids),
                    })
                    for doc_res in self._score_docs_for_node(
                        query_vec=query_vec,
                        node=node,
                        node_relevance=float(score),
                        node_similarity=float(entry["node_similarity"]),
                        max_docs_per_node=max_docs_per_node,
                        doc_similarity_weight=doc_similarity_weight,
                        node_similarity_weight=node_similarity_weight,
                    ):
                        did = doc_res["id"]
                        prev = doc_scores.get(did)
                        if prev is None or doc_res["score"] > prev["score"]:
                            doc_scores[did] = doc_res

            ranked_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
            results = [
                {
                    "id": item["id"],
                    "score": float(item["score"]),
                    "node_id": item["node_id"],
                    "concept_path": item["concept_path"],
                    "node_relevance": item["node_relevance"],
                    "node_similarity": item["node_similarity"],
                    "doc_similarity": item["doc_similarity"],
                }
                for item in ranked_docs
            ]

            trace = None
            if self.config.get("search_trace_enabled", True):
                trace = {
                    "query_preview": query[:200],
                    "top_k_nodes": int(top_k_nodes),
                    "candidate_nodes": [
                        {
                            "rank": entry["rank"],
                            "node_id": entry["node"].node_id,
                            "concept_path": entry["node"].concept_path(),
                            "desc": entry["node"].desc,
                            "doc_count": len(entry["node"].doc_ids),
                            "node_similarity": float(entry["node_similarity"]),
                            "llm_relevance": relevance_map.get(entry["node"].node_id, "none"),
                        }
                        for entry in candidate_entries
                    ],
                    "selected_nodes": selected_nodes,
                    "returned_docs": [
                        {
                            "id": item["id"],
                            "score": float(item["score"]),
                            "node_id": item["node_id"],
                            "doc_similarity": float(item["doc_similarity"]),
                        }
                        for item in ranked_docs[: min(len(ranked_docs), 20)]
                    ],
                }

            return {"results": results, "trace": trace}
        except Exception as e:
            logger.error(f"Ontology search failed: {e}")
            return {"results": [], "trace": None}
