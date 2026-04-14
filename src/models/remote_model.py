import asyncio
import random
import time
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, NotFoundError
from .base import BaseLLMClient
from ..utils.logger import setup_logger
from ..core.resources import ResourceManager

logger = setup_logger(__name__)

class RemoteLLMClient(BaseLLMClient):
    def __init__(self, api_key: str, base_url: str, model: str, **kwargs):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_url = base_url
        self.api_mode = kwargs.get("api_mode", "auto")
        # concurrency control handled by ResourceManager

    async def _chat_completions(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    def _prepare_responses_request(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        req_kwargs = dict(kwargs)
        response_format = req_kwargs.pop("response_format", None)

        prepared_messages = [dict(msg) for msg in messages]
        if response_format == {"type": "json_object"}:
            # The responses API in this environment does not accept chat-style
            # response_format, so reinforce the instruction in-band instead.
            json_hint = (
                "\n\nReturn valid JSON only. Do not wrap it in markdown fences "
                "and do not include any extra commentary."
            )
            if prepared_messages and prepared_messages[0].get("role") == "system":
                prepared_messages[0]["content"] = prepared_messages[0]["content"] + json_hint
            else:
                prepared_messages.insert(
                    0,
                    {"role": "system", "content": "Return valid JSON only."},
                )

        return prepared_messages, req_kwargs

    async def _responses_api(self, messages: List[Dict[str, str]], **kwargs) -> str:
        messages, kwargs = self._prepare_responses_request(messages, **kwargs)
        input_items = []
        for msg in messages:
            input_items.append(
                {
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "input_text",
                            "text": msg["content"],
                        }
                    ],
                }
            )

        if "max_tokens" in kwargs and "max_output_tokens" not in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        response = await self.client.responses.create(
            model=self.model,
            input=input_items,
            **kwargs
        )
        return response.output_text

    async def _create_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if self.api_mode == "chat":
            return await self._chat_completions(messages, **kwargs)
        if self.api_mode == "responses":
            return await self._responses_api(messages, **kwargs)

        try:
            return await self._chat_completions(messages, **kwargs)
        except NotFoundError as err:
            logger.warning(
                "chat.completions returned 404 for model=%s base_url=%s. "
                "Falling back to responses API.",
                self.model,
                self.base_url,
            )
            try:
                return await self._responses_api(messages, **kwargs)
            except Exception:
                raise err

    async def chat(self, messages: List[Dict[str, str]], max_retries: int = 3, **kwargs) -> str:
        sem = ResourceManager.get_instance().llm_semaphore

        last_err = None
        for attempt in range(max_retries):
            try:
                async with sem:
                    return await self._create_response(messages, **kwargs)
            except Exception as e:
                last_err = e
                wait_time = (2 ** attempt) + random.random()
                logger.warning(
                    "API call failed (attempt %s/%s) for model=%s base_url=%s api_mode=%s: %s. "
                    "Retrying in %.2fs...",
                    attempt + 1,
                    max_retries,
                    self.model,
                    self.base_url,
                    self.api_mode,
                    e,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
        
        logger.error(
            "API call failed after %s attempts for model=%s base_url=%s api_mode=%s: %s",
            max_retries,
            self.model,
            self.base_url,
            self.api_mode,
            last_err,
        )
        raise last_err
