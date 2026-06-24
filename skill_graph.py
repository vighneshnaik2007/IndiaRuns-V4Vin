# ============================================================
# skill_graph.py — Related Skills Knowledge Graph
# Feature #2: Skill Graph Matching
# Team V4Vin | India Runs Hackathon 2026
# ============================================================

# This is a knowledge graph that maps related skills together.
# If JD says "Pinecone" and candidate has "Qdrant" — both are
# vector databases, so our system gives partial credit instead
# of completely missing the match like keyword filters do.

SKILL_GRAPH = {
    # ── Vector Databases ────────────────────────────────────
    "pinecone": ["weaviate", "qdrant", "milvus", "chroma", "faiss",
                 "vector database", "vector db", "pgvector", "vespa"],
    "weaviate": ["pinecone", "qdrant", "milvus", "chroma", "faiss",
                 "vector database", "vector search"],
    "qdrant": ["pinecone", "weaviate", "milvus", "chroma", "faiss",
               "vector database", "vector search"],
    "milvus": ["pinecone", "weaviate", "qdrant", "chroma", "faiss",
               "vector database"],
    "faiss": ["pinecone", "weaviate", "qdrant", "milvus",
              "vector search", "ann search", "approximate nearest neighbor"],
    "chroma": ["pinecone", "weaviate", "qdrant", "faiss",
               "vector database", "embeddings store"],

    # ── ML Frameworks ────────────────────────────────────────
    "pytorch": ["tensorflow", "keras", "jax", "deep learning",
                "neural network", "torch"],
    "tensorflow": ["pytorch", "keras", "jax", "deep learning",
                   "neural network", "tf"],
    "keras": ["pytorch", "tensorflow", "deep learning", "neural network"],
    "jax": ["pytorch", "tensorflow", "numerical computing", "flax"],

    # ── LLM / NLP ────────────────────────────────────────────
    "langchain": ["llamaindex", "llm", "rag",
                  "retrieval augmented generation", "agents", "llm orchestration"],
    "llamaindex": ["langchain", "llm", "rag",
                   "retrieval augmented generation", "knowledge base"],
    "huggingface": ["transformers", "bert", "gpt", "llm", "nlp",
                    "fine-tuning", "sentence transformers"],
    "transformers": ["huggingface", "bert", "gpt", "nlp",
                     "attention", "fine-tuning"],
    "bert": ["transformers", "huggingface", "nlp",
             "embeddings", "sentence transformers", "roberta"],
    "gpt": ["openai", "llm", "language model", "transformers", "chatgpt"],
    "rag": ["retrieval augmented generation", "langchain", "llamaindex",
            "vector search", "knowledge retrieval"],
    "llm": ["gpt", "bert", "huggingface", "language model",
            "foundation model", "generative ai"],

    # ── Embeddings & Search ──────────────────────────────────
    "embeddings": ["vector search", "semantic search",
                   "sentence transformers", "dense retrieval", "representations"],
    "semantic search": ["embeddings", "vector search",
                        "dense retrieval", "faiss", "neural search"],
    "hybrid search": ["bm25", "semantic search", "elasticsearch",
                      "dense retrieval", "sparse retrieval"],
    "bm25": ["elasticsearch", "sparse retrieval", "hybrid search",
             "information retrieval", "tf-idf"],
    "elasticsearch": ["opensearch", "solr", "bm25",
                      "full text search", "hybrid search"],

    # ── Python Ecosystem ─────────────────────────────────────
    "python": ["fastapi", "flask", "django", "pandas",
               "numpy", "scikit-learn", "pydantic"],
    "fastapi": ["python", "rest api", "backend",
                "microservices", "async", "pydantic"],
    "flask": ["python", "rest api", "backend", "web framework"],
    "django": ["python", "rest api", "backend",
               "web framework", "orm"],

    # ── MLOps / Deployment ───────────────────────────────────
    "mlflow": ["mlops", "model tracking",
               "experiment tracking", "wandb", "model registry"],
    "wandb": ["mlflow", "experiment tracking",
              "model monitoring", "weights and biases"],
    "docker": ["kubernetes", "containerization",
               "devops", "deployment", "containers"],
    "kubernetes": ["docker", "orchestration",
                   "devops", "deployment", "k8s"],
    "ray": ["distributed computing", "mlops",
            "parallel processing", "ray tune"],

    # ── Ranking / Recommendation ─────────────────────────────
    "recommendation system": ["ranking", "collaborative filtering",
                              "matrix factorization", "learning to rank",
                              "content based filtering"],
    "learning to rank": ["ndcg", "mrr", "map",
                         "ranking", "recommendation system", "listwise", "pairwise"],
    "ndcg": ["ranking metrics", "information retrieval",
             "learning to rank", "mrr", "map"],
    "information retrieval": ["search", "ranking", "bm25",
                              "ndcg", "precision recall"],

    # ── Cloud ────────────────────────────────────────────────
    "aws": ["gcp", "azure", "cloud", "s3",
            "ec2", "sagemaker", "lambda"],
    "gcp": ["aws", "azure", "cloud",
            "bigquery", "vertex ai", "google cloud"],
    "azure": ["aws", "gcp", "cloud",
              "azure ml", "microsoft azure"],

    # ── Data Engineering ─────────────────────────────────────
    "spark": ["hadoop", "pyspark", "big data",
              "data engineering", "apache spark"],
    "airflow": ["data pipeline", "etl",
                "workflow orchestration", "data engineering", "prefect"],

    # ── Fine-tuning ──────────────────────────────────────────
    "fine-tuning": ["lora", "qlora", "peft", "rlhf",
                    "instruction tuning", "transfer learning", "huggingface"],
    "lora": ["fine-tuning", "qlora", "peft",
             "parameter efficient", "adapters"],
    "rlhf": ["fine-tuning", "reinforcement learning",
             "human feedback", "alignment", "ppo"],
}

# JD Skills extracted from Senior AI Engineer role
JD_SKILLS = [
    "python", "embeddings", "vector database", "semantic search",
    "hybrid search", "ranking", "learning to rank", "ndcg", "mrr",
    "recommendation system", "llm", "fine-tuning", "huggingface",
    "pytorch", "faiss", "qdrant", "weaviate", "pinecone",
    "information retrieval", "elasticsearch", "langchain", "rag"
]


def get_related_skills(skill):
    """
    Get all related skills for a given skill using the knowledge graph.
    Example: get_related_skills("pinecone") returns weaviate, qdrant, faiss etc.
    """
    skill_lower = skill.lower().strip()
    related = set()
    for key, values in SKILL_GRAPH.items():
        if key in skill_lower or skill_lower in key:
            related.add(key)
            related.update(values)
        for v in values:
            if v in skill_lower or skill_lower in v:
                related.add(key)
                related.update(values)
    return related


def skill_graph_score(candidate_skills, jd_skills=None):
    """
    Score candidate skills against JD skills using knowledge graph.

    Scoring:
    - Direct match = 1.0 (full credit)
    - Related skill match = 0.7 (partial credit)
    - No match = 0.0

    Returns normalized score between 0 and 1.
    """
    if jd_skills is None:
        jd_skills = JD_SKILLS

    if not jd_skills or not candidate_skills:
        return 0.0

    candidate_skills_lower = set(s.lower().strip() for s in candidate_skills)
    jd_skills_lower = set(s.lower().strip() for s in jd_skills)

    matched = 0.0
    for jd_skill in jd_skills_lower:
        # Direct exact match
        if jd_skill in candidate_skills_lower:
            matched += 1.0
            continue
        # Check partial match (skill is substring of candidate skill or vice versa)
        partial = any(jd_skill in cs or cs in jd_skill
                      for cs in candidate_skills_lower)
        if partial:
            matched += 0.8
            continue
        # Knowledge graph related match
        related = get_related_skills(jd_skill)
        if related & candidate_skills_lower:
            matched += 0.7

    return min(matched / max(len(jd_skills_lower), 1), 1.0)


def get_skill_matches(candidate_skills, jd_skills=None):
    """
    Returns detailed breakdown of skill matches for explainability.
    Used by explainer.py to generate human-readable reasoning.
    """
    if jd_skills is None:
        jd_skills = JD_SKILLS

    candidate_skills_lower = set(s.lower().strip() for s in candidate_skills)
    jd_skills_lower = set(s.lower().strip() for s in jd_skills)

    direct_matches = []
    related_matches = []
    missing_skills = []

    for jd_skill in jd_skills_lower:
        if jd_skill in candidate_skills_lower:
            direct_matches.append(jd_skill)
        else:
            related = get_related_skills(jd_skill)
            matched_related = related & candidate_skills_lower
            if matched_related:
                related_matches.append(
                    f"{jd_skill} (via {list(matched_related)[0]})")
            else:
                missing_skills.append(jd_skill)

    return {
        "direct_matches": direct_matches,
        "related_matches": related_matches,
        "missing_skills": missing_skills,
        "match_rate": len(direct_matches) / max(len(jd_skills_lower), 1)
    }