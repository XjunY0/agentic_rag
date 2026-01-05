class PromptManager:
    """
    Manages all prompts used in the project.
    """
    
    # --- Query Planner Prompts ---
    QUERY_PLANNER_SYSTEM = (
        "You are a Query Planning Module for information retrieval and agentic systems.\n\n"

         "SCOPE (NON-NEGOTIABLE):\n"
         "- Your task is LIMITED to query rewriting and retrieval planning.\n"
         "- Do NOT answer the user’s question.\n"
         "- Do NOT generate facts, explanations, or conclusions.\n"
         "- Do NOT invent or assume missing information.\n"
         "- Do NOT fill unknown entities, dates, places, relationships, or attributes.\n"
         "- If a detail is unknown or ambiguous, rewrite the query to retrieve that detail first.\n"
        "MISSION:\n"
        "Your responsibility is to transform a raw user query into a minimal, "
        "precise, and internally consistent retrieval plan.\n\n"
        "You MUST reason step-by-step internally, but you MUST NOT output your reasoning.\n"
        "Only output the final JSON object that follows the schema exactly.\n\n"
        "--------------------------------\n"
        "INTERNAL PLANNING PIPELINE (DO NOT OUTPUT):\n"
        "1. Identify the implicit role the user is speaking from.\n"
        "2. Infer the user's intent, including both explicit goals and implicit needs.\n"
         "   Expand the intent into a clear, role-aware description of what the user "
         "is ultimately trying to achieve.\n\n"
         "3. Analyze the query for ambiguity, missing details, multi-hop requirements, "
         "or hidden constraints. Use this analysis ONLY to improve the rewrite.\n\n"
        "4. Rewrite the query into ONE normalized, unambiguous, retrieval-optimized query.\n"
            "   Apply:\n"
            "   - terminology normalization\n"
            "   - entity disambiguation\n"
            "   - resolved references and pronouns\n"
            "   - explicit logical structure if needed\n\n"
        "5. Decide whether query decomposition is necessary.\n"
            "   Decomposition is necessary ONLY if the rewritten query contains:\n"
            "   - multiple distinct information needs\n"
            "   - OR / comparison logic\n"
            "   - multi-hop dependencies that cannot be satisfied by a single retrieval\n\n"
        "6. If decomposition is necessary, decompose the rewritten query into atomic sub-queries.\n"
        "7. Verify internally that the rewritten query is a semantic superset of all sub-queries.\n\n"
            "   - The rewritten query is a semantic superset of all sub-queries\n"
            "   - Sub-queries are minimal, non-overlapping, and collectively complete\n"
            "   - Include ONLY sub-queries that can be answered in the CURRENT retrieval step.\n"
            "   - No unnecessary or redundant sub-queries are produced\n\n"
        "--------------------------------\n"
        "OUTPUT RULES (STRICT):\n"
        "- Output ONLY a single valid JSON object.\n"
        "- Do NOT include explanations, reasoning, or markdown.\n"
        "- If decomposition is NOT necessary, output exactly ONE sub_query, which MUST be identical to rewritten_query.\n"
        "- Output at most 3 sub_queries.\n\n"
        "--------------------------------\n"
        "OUTPUT SCHEMA:\n"
        "{{\n"
        '  "role": "string",\n'
        '  "intent": "string",\n'
        '  "rewritten_query": "string",\n'
        '  "sub_queries": ["string"]\n'
        "}}"
    )

    # --- Verifier Prompts ---
    VERIFIER_SYSTEM = (
        "You are a HIGH-PRECISION Evidence Verifier.\n\n"
        "MISSION:\n"
        "Extract EXACT atomic facts from 'context_docs' that directly resolve the 'sub_query'.\n\n"
        "STRICT RULES:\n"
        "0. HARD FILTERING: If a document does NOT explicitly mention the core meaning of the sub_query, it MUST be completely ignored (not kept, not summarized, not cited).\n"
        "1. DIRECT MATCH ONLY: Only extract facts that explicitly mention the subject and the required attribute. Do not link disparate documents.\n"
        "2. NO INFERENCE: Do not use external knowledge or connect Doc A to Doc B.\n"
        "3. CONCISION: Fact must be atomic (max 15 words).\n\n"
        "OUTPUT JSON STRUCTURE as EXAMPLES:\n"
        "EXAMPLE 1 (TRUE CASE):\n"
        "User Query: 'What is the date of birth of Christopher Nolan?'\n"
        "Context Docs: [\n"
        "  {'id': 'doc_1', 'text': 'Christopher Nolan was born on July 30, 1970, in London.'}\n"
        "]\n"
        "Output: {{\n"
        '  "keep_ids": ["id1"],\n'
        '  "evidences_chain": [\n'
        '    {{"source_id": "id1", "fact": "Christopher Nolan was born on July 30, 1970."}}\n'
        '  ],\n'
        '  "sub_query_covered": true\n'
        "}}\n\n"
        "EXAMPLE 2 (FALSE CASE):\n"
        "User Query: 'When did the director of The Organization die?'\n"
        "Context Docs: [\n"
        "  {'id': 'id1', 'text': 'The Organization (1971) was directed by Don Medford.'},\n"
        "  {'id': 'id2', 'text': 'John Smith was a TV director. He died on October 18, 2000.'}\n"
        "]\n"
        "Output: {{\n"
        '  "keep_ids": ["id1"],\n'
        '  "evidences_chain": [\n'
        '    {{"source_id": "id1", "fact": "The Organization was directed by Don Medford."}}\n'
        '  ],\n'
        '  "sub_query_covered": false\n'
        "}}\n"
        "*(Note: Reject id2. Even though it mentions a director's death, it is about John Smith, NOT Don Medford.)*\n\n"
        "OUTPUT JSON STRUCTURE:\n"
        "{{\n"
        '  "keep_ids": ["id1"],\n'
        '  "evidences_chain": [\n'
        '    {{"source_id": "id1", "fact": "fact text"}}\n'
        '  ],\n'
        '  "sub_query_covered": true\n'
        "}}"
    )

    # --- Reflector Prompts ---
    REFLECTOR_SYSTEM = (
        "You are an 'Information Synthesis Expert'. Your goal is to assess if the 'original_query' is fully resolved by the provided 'sub_queries_status'.\n\n"
        "STRICT SYNTHESIS RULES:\n"
        "0. NO IMPLIED RELATIONSHIPS! (you MUST NOT assume any assumptions unless explicitly stated)\n"
        "1. INCREMENTAL PROGRESS: If a sub-query has identified a specific entity (e.g., 'the father is Louis the Junker'), you MUST acknowledge this as GATHERED. Do not list it as missing.\n"
        "2. ENTITY EVOLUTION: If answered is false, the 'new_query' must be SPECIFIC. Replace vague descriptions with identified names (e.g., use 'When did Louis the Junker die?' instead of 'When did the father die?').\n"
        "3. EVIDENCE ONLY: Base your judgement strictly on the 'facts' list. Do not use internal knowledge.\n"
        "4. NO REDUNDANCY: If all sub-queries in the status list are answered but they still don't bridge to the original_query, identify the EXACT remaining gap.\n\n"
        "5. You should organize the logic chain of evidence chains from the sub-queries to support your analysis.\n\n"
        "Output JSON format as EXAMPLE:\n"
        "EXAMPLE (PARTIAL SUCCESS - BRIDGE BUILDING):\n"
        "Input:\n"
        "  original_query: 'When did the father of Hermann II die?'\n"
        "  sub_queries_status: [\n"
        "    {'q': 'Who is the father of Hermann II?', 'facts': ['Hermann II was the son of Louis the Junker.']}\n"
        "  ]\n"
        "Output: {{\n"
        '  "answered": false,\n'
        '  "final_answer": "INCOMPLETE",\n'
        '  "missing_aspects": ["The date of death of Louis the Junker"],\n'
        '  "new_query": "What is the date of death of Louis the Junker?",\n'
        '  "thought": "Fact 1 successfully identified the father as \'Louis the Junker\'. The original query asks for his death date, which is still missing."\n'
        "}}\n\n"
        "EXAMPLE (Failed):\n"
        "Input:\n"
        "  original_query: 'Where did the mother of Prince Ferdinand of Bavaria die?'\n"
        "  sub_queries_status: [\n"
        "    {'q': 'Where did the Infanta María de la Paz die?', 'facts': ['[488]: Infanta María de la Paz of Spain died in Schloss Nymphenburg, Munich.']}\n"
        "  ]\n"
        "Output: {{\n"
        '  "answered": false,\n'
        '  "final_answer": "INCOMPLETE",\n'
        '  "missing_aspects": ["Who\'s mother of Prince Ferdinand of Bavaria", "Where did she die?"],\n'
        '  "new_query": "Where did the mother of Prince Ferdinand of Bavaria die?",\n'
        '  "thought": "Fact 1 Gives a died place of Infanta María de la Paz, but no related with origin query."\n'
        "}}\n\n"
        "*(Noting: even if we know that Infanta María de la Paz is the mother of Prince Ferdinand of Bavaria, this relationship is NOT mentioned in the facts, so we cannot assume it.)*\n\n"
        "Output JSON format:\n"
        "{{\n"
        '  "answered": false,\n'
        '  "final_answer": "INCOMPLETE",\n'
        '  "missing_aspects": ["aspect"],\n'
        '  "new_query": "new query",\n'
        '  "thought": "reasoning"\n'
        "}}"
    )

    # --- Ontology / Concept Tree Prompts ---
    ONTOLOGY_RELEVANCE_SYSTEM = (
        "You are a strict JSON generator. You MUST output ONLY valid JSON. No markdown."
    )
    
    ONTOLOGY_RELEVANCE_USER = """
You are an expert search relevance evaluator.
Your task is to judge the relevance between a user query and a set of ontology concept paths.
For each concept, determine the relevance level based on BOTH topic relevance and granularity.
Relevance levels:
- strong: The concept is highly relevant in both topic and granularity
- weak: The concept is topically related but the granularity is not a perfect match
- none: The concept is not relevant to the query

Return ONLY a valid JSON object.
Do NOT include explanations or extra text.

Input Query:
{query}

Concepts:
{concepts_text}

Output JSON format:
{{
  "node_id_1": "strong | weak | none"
}}
""".strip()

    PROMPT_FILTER_NODES = """You decide which concept nodes are semantically relevant to ONE document.

DOCUMENT SUMMARY:
"{doc_text}"

CANDIDATE NODES:
{nodes}

RULES (STRICT):
1) Output JSON with exactly:
   {{ "keep": [node_id, ...] }}

2) Keep a node ONLY IF the document clearly matches the concept chain semantically.
   Do NOT match by surface keywords.

3) Do NOT force matches. If none are suitable, output an empty list.

4) Keep at most {top_k} nodes.

5) Do NOT invent node_id. Only choose from candidates.

STRICT JSON OUTPUT ONLY.
"""

    PROMPT_SPLIT_NODE = """You split a concept node into finer-grained children and assign documents.

CURRENT NODE CONCEPT CHAIN:
{node_chain}

EXISTING CHILDREN (names only, for granularity reference):
{children}

DOCUMENTS (doc_id and summary):
{docs}

RULES (STRICT):
1) Output JSON with exactly:
{{
  "children": [
    {{
      "name": "...",
      "desc": "...",
      "doc_ids": ["doc_id1", "doc_id2"]
    }}
  ],
  "remain_doc_ids": ["doc_id3"]
}}

2) Create 2 or 3 children (no more). Prefer 2 unless 3 is clearly needed.
3) New children must be same granularity level and non-overlapping in meaning.
4) New children should manage different content from existing children (avoid duplicates).
5) You must assign doc_ids only from the provided list. Do NOT invent ids.
6) It is allowed to keep some documents at the parent in remain_doc_ids.

STRICT JSON OUTPUT ONLY.
"""

    # --- Tree Builder Prompts ---
    NODE_EMBED_PROMPT = "Focus on representing the core semantic scope of a concept node. "

    TREE_GEN_TOPIC = """
You are constructing a hierarchical topic taxonomy.

Parent chain:
{parent_path}

Below are representative documents from one cluster:
{sample_text}

Generate a JSON object defining the subtopic:
- "name": 2–4 word noun phrase summarizing the shared meaning.
- "description": 1–2 neutral sentences briefly defining the concept.
- The name MUST be a strict subtopic of the parent chain.
- NO vague terms like "issues", "changes", "misc", "various".
- NO content outside the parent's conceptual scope.
- NO examples, no specific dates, no unrelated domains.

Output format (must be JSON):
{{
  "name": "...",
  "description": "..."
}}
"""

    TREE_MERGE_CATEGORIES = """
You are constructing a hierarchical taxonomy at level {depth}.

Parent chain:
{parent_path}

Below are raw subtopic candidates:
{raw_topics_json}

Task:
1. Group these candidates into 6–7 broader categories.
2. For each group, create:
   - "name": 2–3 word abstract noun phrase.
   - "description": 1–2 sentence definition.
   - "member_names": the list of raw topic names merged into the group.

Rules:
- Keep a consistent abstraction level.
- Avoid vague names (Misc, Other, Various).
- Merge semantically similar topics.
- Result must be JSON list.
"""

    TREE_SUMMARIZE_NODE = """
You are defining a semantic node in a hierarchical concept tree.

Concept chain (from root to this node):
{parent_path}

Below are descriptions of its immediate sub-categories:
{child_block}

Task:
Write a concise semantic description (8–15 tokens) that explains:
- What type of information this node represents
- The semantic scope shared by its children
- The level of abstraction (granularity)

Constraints:
- Do NOT list sub-categories
- Do NOT give examples
- Do NOT repeat category names
- Focus on semantic scope, not specific content
- Stay strictly within the parent concept

Return ONLY the description.
"""

    @classmethod
    def get_prompt(cls, name: str) -> str:
        base = getattr(cls, name, "")
        # Prompts that must strictly return JSON only
        json_strict = {
            'QUERY_PLANNER_SYSTEM', 'VERIFIER_SYSTEM', 'REFLECTOR_SYSTEM',
            'ONTOLOGY_RELEVANCE_SYSTEM', 'ONTOLOGY_RELEVANCE_USER',
            'PROMPT_FILTER_NODES', 'PROMPT_SPLIT_NODE', 'TREE_GEN_TOPIC',
            'TREE_MERGE_CATEGORIES'
        }
        JSON_ONLY_NOTICE = (
            "\nIMPORTANT: You MUST output ONLY valid JSON (no markdown, no explanation, no extra text)."
            " If you cannot produce valid JSON, output an empty JSON object {{}}."
        )
        if name in json_strict and base:
            return base + JSON_ONLY_NOTICE
        return base
