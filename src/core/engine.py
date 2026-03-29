import asyncio
import os
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm_asyncio

from ..models.remote_model import RemoteLLMClient
from ..models.local_model import QwenEmbeddingModel
from ..planner.query_planner import QueryPlanner
from ..indexing.bm25_index import BM25Index
from ..indexing.vector_index import VectorIndex
from ..indexing.entity_index import EntityIndex
from ..indexing.ontology_index import OntologyIndex
from ..search.retriever import MultiModalRetriever
from ..search.ranker import HybridRanker
from ..verifier.verifier import Verifier
from ..verifier.reflector import Reflector
from ..utils.logger import setup_logger
from ..io.data_loader import DataLoader
from .resources import ResourceManager

logger = setup_logger(__name__)

class OmniSearch:
    """
    OmniSearch: An Agentic RAG Engine with Multi-modal Retrieval and Concept Tree.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize Resource Manager
        ResourceManager.setup(config.get('resources', {}))
        
        self.corpus = {} # Map of doc_id -> text
        self.storage_dir = config['storage']['path']
        
        # Initialize Models
        # concurrency settings
        llm_conc = config['search'].get('llm_concurrency', 4) if 'search' in config else 4
        emb_conc = config['search'].get('embedding_concurrency', 1) if 'search' in config else 1
        self.subquery_concurrency = config['search'].get('subquery_concurrency', 4) if 'search' in config else 4

        self.llm_client = RemoteLLMClient(
            api_key=config['model']['api_key'],
            base_url=config['model']['base_url'],
            model=config['model']['llm_name'],
            max_concurrency=llm_conc
        )
        self.embedding_model = QwenEmbeddingModel(
            model_name=config['model']['embedding_name'],
            max_concurrency=emb_conc,
            target_dim=config['model'].get('embedding_dim')
        )
        
        # Initialize Indices
        storage_dir = config['storage']['path']
        os.makedirs(storage_dir, exist_ok=True)
        self.bm25_index = BM25Index(os.path.join(storage_dir, "bm25"))
        self.vector_index = VectorIndex(
            os.path.join(storage_dir, "vector"), 
            dimension=config['model']['embedding_dim'],
            hnsw_m=config['search'].get('faiss', {}).get('hnsw_m', 32),
            ef_construction=config['search'].get('faiss', {}).get('ef_construction', 200)
        )
        self.entity_index = EntityIndex(
            os.path.join(storage_dir, "entity"),
            spacy_model=config['search'].get('spacy', {}).get('model', 'en_core_web_lg')
        )
        self.ontology_index = OntologyIndex(
            os.path.join(storage_dir, "ontology.json"),
            config=config['search'].get('ontology', {})
        )
        # Whether to enable ontology-based search fusion (default: False)
        self.ontology_enabled = bool(config['search'].get('ontology_enabled', False)) if 'search' in config else False
        
        # Initialize Components
        self.planner = QueryPlanner(self.llm_client)
        self.retriever = MultiModalRetriever(
            self.bm25_index, self.vector_index, self.entity_index, self.ontology_index,
            self.embedding_model, self.llm_client
        )
        self.ranker = HybridRanker(
            weights=config['search'].get('weights'), 
            concept_bonus=config['search'].get('concept_bonus', 1.5),
            rrf_k=config['search'].get('rrf_k', 60)
        )
        self.verifier = Verifier(self.llm_client)
        self.reflector = Reflector(self.llm_client)
        
        self.max_turns = config['search'].get('max_turns', 3)

    async def build_indices(self, corpus: List[Dict[str, Any]]):
        """Build all indices if they don't exist."""
        self.corpus = {str(doc['id']): doc['text'] for doc in corpus}
        # Build indices in threadpool where operations are blocking
        await asyncio.to_thread(self.bm25_index.build, corpus)
        await asyncio.to_thread(self.vector_index.build, corpus, self.embedding_model)
        await asyncio.to_thread(self.entity_index.build, corpus)

        # Try to reuse precomputed embeddings from vector index to avoid recomputing
        doc_vecs = self.vector_index.load_embeddings()
        if doc_vecs is None:
            # fallback: compute via embedding model
            encode_fn = getattr(self.embedding_model, "encode_async", None)
            texts = [doc["text"] for doc in corpus]
            if encode_fn is None:
                doc_vecs = await asyncio.to_thread(self.embedding_model.encode, texts)
            else:
                doc_vecs = await encode_fn(texts)

        force_rebuild_ontology = bool(self.config.get('storage', {}).get('force_rebuild_ontology', False))
        await self.ontology_index.build(
            corpus,
            self.llm_client,
            self.embedding_model,
            doc_vecs=doc_vecs,
            force=force_rebuild_ontology,
        )

    async def search(self, query: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform agentic multi-turn search."""
        current_query = query
        turn = 0
        all_evidences = []
        all_doc_ids = set()
        trace_queries: List[str] = []
        ontology_traces: List[Dict[str, Any]] = []
        
        while turn < self.max_turns:
            logger.info(f"Turn {turn + 1}: Planning for query: {current_query}")
            plan = await self.planner.plan(current_query)
            sub_queries = plan.get("sub_queries", [current_query])
            # record the query used in this turn
            trace_queries.append(current_query)
            # Process sub-queries with controlled concurrency
            sem = asyncio.Semaphore(self.subquery_concurrency)

            last_reflection = None

            async def _process_subq(sub_q: str):
                # limit number of parallel retrieve calls
                async with sem:
                    multi_results = await self.retriever.retrieve(sub_q, use_ontology=self.ontology_enabled)

                ranked_docs = self.ranker.rank(multi_results)

                # Fetch full text for verification
                docs_for_verify = []
                for doc_res in ranked_docs:
                    doc_id = str(doc_res['id'])
                    if doc_id in self.corpus:
                        docs_for_verify.append({"id": doc_id, "text": self.corpus[doc_id]})

                verification = await self.verifier.verify(sub_q, docs_for_verify)
                verification["sub_query"] = sub_q
                if multi_results.get("ontology_trace"):
                    verification["ontology_trace"] = multi_results["ontology_trace"]
                return verification

            tasks = [_process_subq(sq) for sq in sub_queries]
            turn_results = await asyncio.gather(*tasks)

            for verification in turn_results:
                all_evidences.extend(verification.get("evidences_chain", []))
                all_doc_ids.update(verification.get("keep_ids", []))
                ontology_trace = verification.get("ontology_trace")
                if ontology_trace:
                    ontology_traces.append({
                        "turn": turn + 1,
                        "sub_query": verification.get("sub_query"),
                        "trace": ontology_trace,
                    })
            
            # Reflect on the current turn's results using the query that was planned/executed this turn
            # Provide original query, current (rewritten) query, verifier results and accumulated evidences
            reflection = await self.reflector.reflect(query, current_query, turn_results, all_evidences)
            last_reflection = reflection
            if reflection.get("answered"):
                logger.info("Query fully answered.")
                # persist trace for answered requests as well
                try:
                    import json
                    os.makedirs(self.storage_dir, exist_ok=True)
                    trace_path = os.path.join(self.storage_dir, "query_traces.jsonl")
                    with open(trace_path, "a", encoding="utf-8") as tf:
                        rid = str(request_id) if request_id is not None else "unknown"
                        ordered = {"id": rid, "queries": trace_queries}
                        tf.write(json.dumps(ordered, ensure_ascii=False) + "\n")
                except Exception:
                    logger.warning("Failed to write query trace for answered request")

                return {
                    "query": query,
                    "answer": reflection.get("final_answer"),
                    "evidences": all_evidences,
                    "thought": reflection.get("thought"),
                    "doc_ids": list(all_doc_ids),
                    "turns": turn + 1,
                    "ontology_traces": ontology_traces,
                }
            # Update current_query to the new query suggested by reflector (defaults to previous current_query)
            current_query = reflection.get("new_query", current_query)
            turn += 1

        final_ans = (last_reflection.get("final_answer") if last_reflection else "Unknown")
        # persist trace: one line per request with id first
        try:
            import json
            os.makedirs(self.storage_dir, exist_ok=True)
            trace_path = os.path.join(self.storage_dir, "query_traces.jsonl")
            with open(trace_path, "a", encoding="utf-8") as tf:
                rid = str(request_id) if request_id is not None else "unknown"
                # write JSON object with id first, then queries list
                ordered = {"id": rid, "queries": trace_queries}
                tf.write(json.dumps(ordered, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("Failed to write query trace")
        return {
            "query": query,
            "answer": "Max turns reached. Partial answer: " + (final_ans or "Unknown"),
            "evidences": all_evidences,
            "thought": reflection.get("thought"),
            "doc_ids": list(all_doc_ids),
            "turns": turn,
            "ontology_traces": ontology_traces,
        }
