import torch
import asyncio
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Any
from ..utils.logger import setup_logger
from .base import BaseEmbeddingModel

logger = setup_logger(__name__)

class QwenEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: Optional[str] = None, max_length: int = 8192, max_concurrency: int = 1, target_dim: Optional[int] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        
        # Check if flash_attention_2 is available
        attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "eager"
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)
        
        self.max_length = max_length
        self.default_instruction = "Represent the following text for retrieval: "
        # Limit concurrent encode calls (GPU-sensitive)
        # Semaphores are created lazily per-running-event-loop to avoid "bound to a different event loop" errors
        self._sem = None
        self._sem_loop = None
        self._max_concurrency = max_concurrency
        # target embedding dimension to enforce (None means use model output dim)
        self.target_dim = int(target_dim) if target_dim is not None else None

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def format_query(self, query: str, instruction: Optional[str] = None) -> str:
        instr = instruction if instruction is not None else self.default_instruction
        return f"Instruction: {instr}\nQuery: {query}"

    def encode(self, texts: List[str], is_query: bool = False, instruction: Optional[str] = None) -> Any:
        if isinstance(texts, str):
            texts = [texts]
        
        if is_query:
            texts = [self.format_query(t, instruction) for t in texts]

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        arr = embeddings.cpu().numpy().astype("float32")

        # enforce target_dim if configured
        if self.target_dim is not None:
            cur_dim = arr.shape[1]
            if cur_dim < self.target_dim:
                # Dimension is smaller than expected -> raise error to alert configuration mismatch
                raise ValueError(f"Embedding dimension mismatch: model returned dim={cur_dim} but target_dim={self.target_dim}")
            if cur_dim > self.target_dim:
                # logger.info(f"Truncating embeddings from {cur_dim} to target_dim={self.target_dim}")
                arr = arr[:, : self.target_dim]

        return arr

    async def encode_async(self, texts: List[str], is_query: bool = False, instruction: Optional[str] = None) -> Any:
        """Async wrapper for encode that runs in threadpool and limits concurrency."""
        # ensure semaphore is created for current running loop
        loop = asyncio.get_running_loop()
        sem = getattr(self, "_sem", None)
        sem_loop = getattr(self, "_sem_loop", None)
        # If semaphore was created in a different loop, recreate it for the current loop
        if sem is None or sem_loop is not loop:
            self._sem = asyncio.Semaphore(self._max_concurrency)
            self._sem_loop = loop
            sem = self._sem
        async with sem:
            return await asyncio.to_thread(self.encode, texts, is_query, instruction)
