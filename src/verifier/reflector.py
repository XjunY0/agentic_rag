import json
from typing import List, Dict, Any
from ..models.base import BaseLLMClient
from ..prompts.manager import PromptManager
from ..utils.helpers import parse_llm_json
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class Reflector:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    async def reflect(
        self,
        original_query: str,
        current_query: str,
        verifier_results: List[Dict[str, Any]],
        accumulated_evidences: List[str]
    ) -> Dict[str, Any]:
        """Reflector now receives both the original user query and the current (possibly rewritten) query,
        along with verifier results and a global list of accumulated evidences. This allows global reasoning
        and proposing a new_query that better targets remaining gaps.
        """
        system_prompt = PromptManager.get_prompt('REFLECTOR_SYSTEM')

        payload = {
            "original_query": original_query,
            "current_query": current_query,
            "accumulated_evidences": accumulated_evidences,
            "sub_queries_status": [
                {
                    "q": res["sub_query"],
                    "facts": res.get("evidences_chain", [])
                }
                for res in verifier_results
            ]
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
        
        try:
            response_text = await self.llm_client.chat(messages, response_format={"type": "json_object"})
            data = parse_llm_json(response_text)
            if not data:
                return {"answered": False, "new_query": current_query}

            return data
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return {"answered": False, "new_query": current_query}
