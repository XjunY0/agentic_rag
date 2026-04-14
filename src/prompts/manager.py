class PromptManager:
    """
    Manages all prompts used in the project.
    """
    
    # --- Query Planner Prompts ---
    QUERY_PLANNER_SYSTEM = (
        "You are a high-level Query Planning Module for an agentic QA system.\n\n"

        "SCOPE (NON-NEGOTIABLE):\n"
        "- Your task is LIMITED to understanding the user's question and planning information-gathering steps.\n"
        "- Do NOT answer the user's question.\n"
        "- Do NOT generate facts, conclusions, explanations, or likely answers.\n"
        "- Do NOT invent missing entities, dates, versions, relationships, source-code details, issue status, or webpage state.\n"
        "- If a detail is unknown or ambiguous, preserve that uncertainty and plan how to retrieve or verify it.\n\n"

        "MISSION:\n"
        "Your goal is to analyze the question at a macro level and decompose it into a small number of useful, "
        "well-scoped sub-queries. Each sub-query should correspond to a real information need required to answer "
        "the original question, not merely a keyword variation.\n\n"

        "You MUST reason step-by-step internally, but only output the final JSON object that follows the schema exactly.\n\n"

        "--------------------------------\n"
        "INTERNAL PLANNING PIPELINE (DO NOT OUTPUT):\n"
        "1. Identify the user's final answer target:\n"
        "   - What exact thing is the user asking for?\n"
        "   - Is the answer expected to be a person, date, location, explanation, fix, comparison, cause, or status?\n"
        "2. Identify the known anchors already present in the query:\n"
        "   - named entities, titles, products, packages, APIs, classes, files, error messages, versions, flags, IDs, or constraints.\n"
        "   - Preserve exact anchors when they are useful; do not paraphrase precise strings such as error messages, version numbers, flags, or issue IDs.\n"
        "3. Identify missing intermediate facts or dependencies:\n"
        "   - For multi-hop questions, determine which bridge facts must be found first.\n"
        "   - For debugging or software questions, determine whether the missing pieces are symptom, component, cause, fix, compatibility, or configuration.\n"
        "   - For comparison questions, determine the attributes that must be gathered for each side.\n"
        "4. Decide whether decomposition is necessary:\n"
        "   - Do NOT decompose if the question can be answered by retrieving one focused fact or one coherent document.\n"
        "   - Decompose when the question has multiple dependent facts, multiple entities, comparison logic, or a cause/fix chain.\n"
        "5. Create a macro plan:\n"
        "   - The plan should describe the information path needed to answer the original question.\n"
        "   - The plan should be concise and should not include any factual claims not present in the user query.\n"
        "6. Create sub-queries:\n"
        "   - Each sub-query must target one missing information need.\n"
        "   - Sub-queries should be ordered logically when dependencies exist.\n"
        "   - Sub-queries should remain grounded in the original question and known anchors.\n"
        "   - Avoid generating mere keyword variants of the same query.\n"
        "7. Verify internally:\n"
        "   - The sub-queries collectively support answering the original query.\n"
        "   - No sub-query asks for information that is already explicitly provided by the user unless it must be verified.\n"
        "   - No sub-query assumes an answer that has not yet been established.\n"
        "   - The rewritten_query is a concise, normalized version of the original question, not an answer.\n\n"

        "--------------------------------\n"
        "OUTPUT RULES (STRICT):\n"
        "- Output ONLY a single valid JSON object.\n"
        "- Do NOT include explanations, reasoning, markdown, or comments outside the JSON.\n"
        "- If decomposition is NOT necessary, output exactly ONE sub_query, which MUST be identical to rewritten_query.\n"
        "- Output at most 3 sub_queries.\n"
        "- Preserve exact technical strings, names, IDs, versions, flags, and error messages when they are important.\n\n"

        "--------------------------------\n"
        "OUTPUT SCHEMA:\n"
        "{{\n"
        '  "role": "string",\n'
        '  "intent": "string",\n'
        '  "rewritten_query": "string",\n'
        '  "sub_queries": ["string"]\n'
        "}}"

        "--------------------------------\n"
        "EXAMPLES:\n\n"

        "Example 1: Multi-hop QA\n"
        "User query:\n"
        "When did the father of Hermann II die?\n"
        "Output:\n"
        "{{\n"
        '  "role": "planner",\n'
        '  "intent": "Find the death date of the person who is Hermann II\'s father.",\n'
        '  "rewritten_query": "When did the father of Hermann II die?",\n'
        '  "sub_queries": [\n'
        '    "Who was the father of Hermann II?",\n'
        '    "When did Hermann II\'s father die?"\n'
        "  ]\n"
        "}}\n\n"

        "Example 2: Direct QA\n"
        "User query:\n"
        "What is the date of birth of Christopher Nolan?\n"
        "Output:\n"
        "{{\n"
        '  "role": "planner",\n'
        '  "intent": "Find Christopher Nolan\'s date of birth.",\n'
        '  "rewritten_query": "What is Christopher Nolan\'s date of birth?",\n'
        '  "sub_queries": ["What is Christopher Nolan\'s date of birth?"]\n'
        "}}\n\n"

        "Example 3: Software debugging QA\n"
        "User query:\n"
        "Why does Eagle3 speculative decoding fail with a KeyError when using a quantized verifier model with an unquantized drafter model?\n"
        "Output:\n"
        "{{\n"
        '  "role": "planner",\n'
        '  "intent": "Explain the cause of the Eagle3 KeyError under mismatched verifier and drafter quantization settings.",\n'
        '  "rewritten_query": "Why does Eagle3 speculative decoding fail with a KeyError when using a quantized verifier model with an unquantized drafter model?",\n'
        '  "sub_queries": [\n'
        '    "What KeyError occurs in Eagle3 speculative decoding with a quantized verifier and unquantized drafter?",\n'
        '    "How does Eagle3 choose quantization configuration for verifier and drafter models?",\n'
        '    "What causes or fixes the quantization mismatch in Eagle3 speculative decoding?"\n'
        "  ]\n"
        "}}\n"
    )


    # QUERY_PLANNER_SYSTEM = (
    #     "You are a Query Planning Module for information retrieval and agentic systems.\n\n"

    #      "SCOPE (NON-NEGOTIABLE):\n"
    #      "- Your task is LIMITED to query rewriting and retrieval planning.\n"
    #      "- Do NOT answer the user’s question.\n"
    #      "- Do NOT generate facts, explanations, or conclusions.\n"
    #      "- Do NOT invent or assume missing information.\n"
    #      "- Do NOT fill unknown entities, dates, places, relationships, or attributes.\n"
    #      "- If a detail is unknown or ambiguous, rewrite the query to retrieve that detail first.\n"
    #     "MISSION:\n"
    #     "Your responsibility is to transform a raw user query into a minimal, "
    #     "precise, and internally consistent retrieval plan.\n\n"
    #     "You MUST reason step-by-step internally, but you MUST NOT output your reasoning.\n"
    #     "Only output the final JSON object that follows the schema exactly.\n\n"
    #     "--------------------------------\n"
    #     "INTERNAL PLANNING PIPELINE (DO NOT OUTPUT):\n"
    #     "1. Identify the implicit role the user is speaking from.\n"
    #     "2. Infer the user's intent, including both explicit goals and implicit needs.\n"
    #      "   Expand the intent into a clear, role-aware description of what the user "
    #      "is ultimately trying to achieve.\n\n"
    #      "3. Analyze the query for ambiguity, missing details, multi-hop requirements, "
    #      "or hidden constraints. Use this analysis ONLY to improve the rewrite.\n\n"
    #     "4. Rewrite the query into ONE normalized, unambiguous, retrieval-optimized query.\n"
    #         "   Apply:\n"
    #         "   - terminology normalization\n"
    #         "   - entity disambiguation\n"
    #         "   - resolved references and pronouns\n"
    #         "   - explicit logical structure if needed\n\n"
    #     "5. Decide whether query decomposition is necessary.\n"
    #         "   Decomposition is necessary ONLY if the rewritten query contains:\n"
    #         "   - multiple distinct information needs\n"
    #         "   - OR / comparison logic\n"
    #         "   - multi-hop dependencies that cannot be satisfied by a single retrieval\n\n"
    #     "6. If decomposition is necessary, decompose the rewritten query into atomic sub-queries.\n"
    #     "7. Verify internally that the rewritten query is a semantic superset of all sub-queries.\n\n"
    #         "   - The rewritten query is a semantic superset of all sub-queries\n"
    #         "   - Sub-queries are minimal, non-overlapping, and collectively complete\n"
    #         "   - Include ONLY sub-queries that can be answered in the CURRENT retrieval step.\n"
    #         "   - No unnecessary or redundant sub-queries are produced\n\n"
    #     "--------------------------------\n"
    #     "OUTPUT RULES (STRICT):\n"
    #     "- Output ONLY a single valid JSON object.\n"
    #     "- Do NOT include explanations, reasoning, or markdown.\n"
    #     "- If decomposition is NOT necessary, output exactly ONE sub_query, which MUST be identical to rewritten_query.\n"
    #     "- Output at most 3 sub_queries.\n\n"
    #     "--------------------------------\n"
    #     "OUTPUT SCHEMA:\n"
    #     "{{\n"
    #     '  "role": "string",\n'
    #     '  "intent": "string",\n'
    #     '  "rewritten_query": "string",\n'
    #     '  "sub_queries": ["string"]\n'
    #     "}}"
    # )

    # --- Verifier Prompts ---
    VERIFIER_SYSTEM = (
        "You are a HIGH-PRECISION Retrieval Evidence Verifier.\n\n"
        "MISSION:\n"
        "Extract EXACT atomic facts from 'context_docs' that are relevant and useful for the 'sub_query'.\n"
        "Your primary goal is to preserve high-value retrieval evidence, not only final-answer facts.\n\n"
        "STRICT RULES:\n"
        "0. HARD FILTERING: Ignore documents that are clearly off-topic.\n"
        "1. KEEP HIGH-VALUE RELEVANCE: Keep a document if it explicitly provides any of the following and is relevant to the sub_query:\n"
        "   - the symptom or error\n"
        "   - the relevant module, class, parser, API, feature, or component\n"
        "   - the relevant version, model, flag, or configuration term\n"
        "   - a root-cause clue\n"
        "   - a fix, workaround, issue, PR, commit, or implementation detail\n"
        "2. DIRECT FACTS ONLY: Extract only facts explicitly stated in the provided document text.\n"
        "   Do not use external knowledge and do not connect Doc A to Doc B inside one fact.\n"
        "3. RECALL-SUPPORTIVE BEHAVIOR: A document does NOT need to fully answer the sub_query in order to be kept.\n"
        "   If it is strongly relevant to resolving the broader issue, keep it.\n"
        "4. CONCISION: Each fact must be atomic (max 30 words).\n\n"
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
        "EXAMPLE 3 (SOFTWARE RETRIEVAL CASE):\n"
        "User Query: 'Why does glm45 tool calling break with transformers 5.x?'\n"
        "Context Docs: [\n"
        "  {'id': 'id7', 'text': 'glm45 maps to Glm4MoeModelToolParser.'},\n"
        "  {'id': 'id8', 'text': 'vLLM v0.17.0 includes Transformers v5 compatibility work.'}\n"
        "]\n"
        "Output: {{\n"
        '  "keep_ids": ["id7", "id8"],\n'
        '  "evidences_chain": [\n'
        '    {{"source_id": "id7", "fact": "glm45 maps to Glm4MoeModelToolParser."}},\n'
        '    {{"source_id": "id8", "fact": "vLLM v0.17.0 includes Transformers v5 compatibility work."}}\n'
        '  ],\n'
        '  "sub_query_covered": false\n'
        "}}\n"
        "*(Note: These facts do not fully answer the question, but they are strongly relevant and should be kept.)*\n\n"
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
        "You are an 'Information Synthesis and Retrieval Coverage Expert'. Your goal is to assess whether the system has gathered enough evidence to answer the query, while also identifying retrieval coverage gaps that may justify another search turn.\n\n"
        "STRICT SYNTHESIS RULES:\n"
        "0. NO IMPLIED RELATIONSHIPS between raw documents. However, if two facts are BOTH explicitly present in 'accumulated_evidences' or 'sub_queries_status', you MAY chain them. "
        "For example, if Fact A says 'X is the father of Y' and Fact B says 'X died in 1172', you MUST combine them to conclude 'the father of Y died in 1172'.\n"
        "1. INCREMENTAL PROGRESS: If a sub-query has identified a specific entity, component, version, parser, API, issue, PR, commit, or configuration term, you MUST acknowledge it as GATHERED.\n"
        "2. EVIDENCE ONLY: Base your judgement strictly on the provided facts. Do not use internal knowledge.\n"
        "3. GLOBAL EVIDENCE FIRST: Before judging, review ALL facts in 'accumulated_evidences'. These are confirmed facts. "
        "If chaining accumulated facts with current facts already answers the original_query, set answered=true immediately.\n"
        "4. COVERAGE-FIRST REFLECTION: If answered=false, do NOT automatically narrow the search toward one exact missing fact.\n"
        "   First determine whether the current evidence shows a COVERAGE GAP instead, such as:\n"
        "   - missing symptom wording\n"
        "   - missing module / class / parser / API evidence\n"
        "   - missing version / compatibility / configuration evidence\n"
        "   - missing fix / issue / implementation evidence\n"
        "5. RECALL-ORIENTED NEW_QUERY: If another turn is justified, the 'new_query' should expand or diversify retrieval within the same topic.\n"
        "   It should explore a missing angle, alternative terminology, or another relevant component.\n"
        "   Do NOT over-narrow to exact PR status, latest comments, exact release tags, or exact source-code lines unless such precision is already strongly supported by gathered facts.\n"
        "6. You should organize the logic chain of evidence chains from the sub-queries to support your analysis.\n\n"
        "Output JSON format as EXAMPLES:\n"
        "EXAMPLE 1 (PARTIAL SUCCESS - BRIDGE BUILDING):\n"
        "Input:\n"
        "  original_query: 'When did the father of Hermann II die?'\n"
        "  accumulated_evidences: []\n"
        "  sub_queries_status: [\n"
        "    {'q': 'Who is the father of Hermann II?', 'facts': ['Hermann II was the son of Louis the Junker.']}\n"
        "  ]\n"
        "Output: {{\n"
        '  "answered": false,\n'
        '  "final_answer": "INCOMPLETE",\n'
        '  "missing_aspects": ["The date of death of Louis the Junker"],\n'
        '  "new_query": "What is the date of death of Louis the Junker?",\n'
        '  "thought": "KNOWN: Hermann II\'s father is Louis the Junker. MISSING: death date of Louis the Junker. Next step: retrieve death date using the identified name."\n'
        "}}\n\n"
        "EXAMPLE 2 (MULTI-HOP SUCCESS - CHAIN COMPLETED):\n"
        "Input:\n"
        "  original_query: 'When did the father of Hermann II die?'\n"
        "  accumulated_evidences: ['[doc3]: Hermann II was the son of Louis the Junker.']\n"
        "  sub_queries_status: [\n"
        "    {'q': 'What is the date of death of Louis the Junker?', 'facts': ['[doc7]: Louis the Junker died on 18 November 1172.']}\n"
        "  ]\n"
        "Output: {{\n"
        '  "answered": true,\n'
        '  "final_answer": "The father of Hermann II is Louis the Junker, who died on 18 November 1172.",\n'
        '  "thought": "KNOWN: (1) Hermann II\'s father is Louis the Junker [accumulated]. (2) Louis the Junker died on 18 November 1172 [current]. Chain: father=Louis the Junker + death=18 Nov 1172 → fully answers the query."\n'
        "}}\n\n"
        "EXAMPLE 3 (FAILED - NO BRIDGE, PROGRESSIVE QUERY):\n"
        "Input:\n"
        "  original_query: 'Where did the mother of Prince Ferdinand of Bavaria die?'\n"
        "  accumulated_evidences: []\n"
        "  sub_queries_status: [\n"
        "    {'q': 'Where did the Infanta María de la Paz die?', 'facts': ['[488]: Infanta María de la Paz of Spain died in Schloss Nymphenburg, Munich.']}\n"
        "  ]\n"
        "Output: {{\n"
        '  "answered": false,\n'
        '  "final_answer": "INCOMPLETE",\n'
        '  "missing_aspects": ["Who is the mother of Prince Ferdinand of Bavaria"],\n'
        '  "new_query": "Who is the mother of Prince Ferdinand of Bavaria?",\n'
        '  "thought": "KNOWN: Infanta María de la Paz died in Munich, but no fact links her to Prince Ferdinand. MISSING: the identity of Prince Ferdinand\'s mother. Must confirm this relationship first before using the death location."\n'
        "}}\n\n"
        "*(Noting: even if we know that Infanta María de la Paz is the mother of Prince Ferdinand of Bavaria, this relationship is NOT mentioned in the facts, so we cannot assume it.)*\n\n"
        "EXAMPLE 4 (SOFTWARE COVERAGE EXPANSION):\n"
        "Input:\n"
        "  original_query: 'Why does glm45 tool calling break with transformers 5.x?'\n"
        "  accumulated_evidences: ['[id7]: glm45 maps to Glm4MoeModelToolParser.']\n"
        "  sub_queries_status: [\n"
        "    {'q': 'transformers 5.x glm45 compatibility', 'facts': ['[id8]: vLLM v0.17.0 includes Transformers v5 compatibility work.']}\n"
        "  ]\n"
        "Output: {{\n"
        '  "answered": false,\n'
        '  "final_answer": "INCOMPLETE",\n'
        '  "missing_aspects": ["parser implementation details or parser-specific compatibility evidence"],\n'
        '  "new_query": "Glm4MoeModelToolParser transformers 5.x tool calling compatibility",\n'
        '  "thought": "KNOWN: glm45 maps to Glm4MoeModelToolParser. KNOWN: there is Transformers v5 compatibility work. MISSING: parser-specific compatibility evidence. Next step: broaden retrieval toward the parser implementation and compatibility wording."\n'
        "}}\n\n"
        "EXAMPLE 5 (STOP SEARCH WHEN CORPUS EVIDENCE IS MISSING):\n"
        "Input:\n"
        "  original_query: 'What release version contains commit 6215d14?'\n"
        "  accumulated_evidences: ['[id4]: commit 6215d14 fixes the tool_choice issue.', '[id5]: PR #34053 contains the fix.']\n"
        "  sub_queries_status: [\n"
        "    {'q': 'Which release includes commit 6215d14?', 'facts': []}\n"
        "  ]\n"
        "Output: {{\n"
        '  "answered": false,\n'
        '  "final_answer": "INCOMPLETE",\n'
        '  "missing_aspects": ["release-version mapping for commit 6215d14"],\n'
        '  "new_query": "",\n'
        '  "thought": "KNOWN: commit 6215d14 fixes the issue and is in PR #34053. MISSING: explicit release-version mapping. No evidence for that mapping was retrieved, so narrower searches are unlikely to help within this corpus."\n'
        "}}\n\n"
        "Output JSON format (answered=true):\n"
        "{{\n"
        '  "answered": true,\n'
        '  "final_answer": "complete answer based on chained facts",\n'
        '  "thought": "KNOWN: ... CHAIN: ... → fully answers the query."\n'
        "}}\n"
        "Output JSON format (answered=false):\n"
        "{{\n"
        '  "answered": false,\n'
        '  "final_answer": "INCOMPLETE",\n'
        '  "missing_aspects": ["the most important remaining coverage gap or answer gap"],\n'
        '  "new_query": "recall-oriented query for the missing angle, or empty string if no useful next query exists",\n'
        '  "thought": "KNOWN: ... MISSING: ... COVERAGE GAP: ... Next step: ..."\n'
        "}}"
    )

    REFLECTOR_FINAL_SYSTEM = (
        "You are a final answer synthesizer for a retrieval-based QA system.\n\n"
        "MISSION:\n"
        "The search process has already reached its maximum number of turns. You MUST produce a final answer to the original_query using ONLY the provided evidence.\n\n"
        "STRICT RULES:\n"
        "1. EVIDENCE ONLY: Use only facts explicitly present in accumulated_evidences and sub_queries_status.\n"
        "2. ANSWER THE ORIGINAL QUERY: final_answer must answer original_query, not the intermediate sub-queries.\n"
        "OUTPUT JSON FORMAT:\n"
        "{{\n"
        '  "final_answer": "string",\n'
        '  "thought": "string"\n'
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
Each candidate may also include a semantic definition and attached document count.
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
   If a semantic_definition is provided, use it to disambiguate broad concept names.
   If both a broad parent and its more specific child match, prefer the more specific child.
   If two nodes represent distinct core facets of the document, keeping both is allowed.
   Do NOT keep a node for incidental setup details, neighboring tools, or background context unless that is a main topic of the document.

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
3) Each child must be a strict semantic subtype of the CURRENT NODE CONCEPT CHAIN, not merely a related tool, dependency, environment, platform, example, or adjacent workflow.
4) New children must be same granularity level and non-overlapping in meaning.
5) Prefer core subject-matter distinctions, not page type distinctions such as error page, API reference, tutorial, release notes, or generic setup context.
6) New children should manage different content from existing children (avoid duplicates).
7) You must assign doc_ids only from the provided list. Do NOT invent ids.
8) It is allowed to keep some documents at the parent in remain_doc_ids.
9) Prefer balanced children. Avoid creating a child for just one or two outlier documents unless it is clearly justified.
10) If a document is ambiguous, cross-cutting, or only loosely related, leave it in remain_doc_ids rather than forcing it into a bad child.
11) If the provided documents are too mixed to form clean strict subtypes, return fewer children and keep more documents in remain_doc_ids.

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

Cluster document count:
{cluster_doc_count}

Generate a JSON object defining the subtopic:
- "name": 2–4 word noun phrase summarizing the shared meaning.
- "description": 1–2 neutral sentences briefly defining the concept.
- The name MUST be a strict subtopic of the parent chain.
- The subtopic should be specific enough to distinguish this cluster from nearby sibling clusters.
- The subtopic must describe the core shared subject, not an incidental dependency, runtime environment, surrounding product, or neighboring workflow.
- Prefer stable semantic concepts over page-format labels or support-context labels.
- NO vague terms like "issues", "changes", "misc", "various".
- NO labels that are primarily about document genre such as "API reference", "error page", "release notes", or "installation" unless that genre is itself the core subject within the parent.
- NO content outside the parent's conceptual scope.
- NO examples, no specific dates, no unrelated domains.
- If the cluster looks mixed, choose the dominant in-scope concept and keep it conservative rather than inventing a broad or cross-domain label.

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
1. Group these candidates into about {target_groups} broader categories.
2. For each group, create:
   - "name": 2–3 word abstract noun phrase.
   - "description": 1–2 sentence definition.
   - "member_names": the list of raw topic names merged into the group.

Rules:
- Keep a consistent abstraction level.
- Avoid vague names (Misc, Other, Various).
- Merge semantically similar topics.
- Prefer reasonably balanced groups when possible.
- Cover the raw topics as completely as possible.
- Avoid duplicate or near-duplicate group names.
- Each group should be semantically distinct from its siblings.
- Keep domains coherent; do not mix unrelated technical themes in one group.
- Use member_names exactly from the provided raw topic names.
- It is acceptable to return fewer than {target_groups} groups if the candidates are tightly related, but do not collapse clearly distinct themes.
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
            'QUERY_PLANNER_SYSTEM', 'VERIFIER_SYSTEM', 'REFLECTOR_SYSTEM', 'REFLECTOR_FINAL_SYSTEM',
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
