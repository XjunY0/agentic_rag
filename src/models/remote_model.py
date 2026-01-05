import asyncio
import random
import time
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from .base import BaseLLMClient
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class RemoteLLMClient(BaseLLMClient):
    def __init__(self, api_key: str, base_url: str, model: str, **kwargs):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        # concurrency control (lazy per event loop)
        self._sem = None
        self._sem_loop = None
        # allow override from kwargs
        self._max_concurrency = int(kwargs.get("max_concurrency", 16))

    async def chat(self, messages: List[Dict[str, str]], max_retries: int = 3, **kwargs) -> str:
        # ensure semaphore bound to current running loop
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        sem = getattr(self, "_sem", None)
        sem_loop = getattr(self, "_sem_loop", None)
        if sem is None or sem_loop is not loop:
            self._sem = asyncio.Semaphore(self._max_concurrency)
            self._sem_loop = loop
            sem = self._sem

        last_err = None
        for attempt in range(max_retries):
            try:
                async with sem:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **kwargs
                    )
                return response.choices[0].message.content
            except Exception as e:
                last_err = e
                wait_time = (2 ** attempt) + random.random()
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
        
        logger.error(f"API call failed after {max_retries} attempts: {last_err}")
        raise last_err
