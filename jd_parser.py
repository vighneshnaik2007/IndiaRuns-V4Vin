# ============================================================
# jd_parser.py — Dynamic Job Description Parser
# Feature #9: Dynamic JD Parser
# Team V4Vin | India Runs Hackathon 2026
# ============================================================

# Instead of hardcoding JD text, this module automatically
# extracts key requirements from ANY job description:
# - Required skills
# - Experience range
# - Location preferences
# - Company type preferences
# - Seniority level
# - Domain/industry

import re

# ── Skill Keywords to detect in JD ──────────────────────────
SKILL_KEYWORDS = [
    # AI/ML
    "python", "pytorch", "tensorflow", "keras", "jax",
    "machine learning", "deep learning", "nlp", "computer vision",
    "llm", "large language model", "generative ai", "fine-tuning",
    "rlhf", "lora", "qlora", "peft",

    # Search & Retrieval
    "embeddings", "vector database", "semantic search", "hybrid search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
    "elasticsearch", "opensearch", "bm25", "information retrieval",

    # Ranking
    "learning to rank", "ndcg", "mrr", "map", "ranking",
    "recommendation system", "collaborative filtering",

    # LLM Ecosystem
    "langchain", "llamaindex", "rag", "retrieval augmented generation",
    "huggingface", "transformers", "bert", "gpt", "openai",

    # MLOps
    "mlflow", "wandb", "docker", "kubernetes", "ray",
    "model deployment", "model serving", "mlops",

    # Cloud
    "aws", "gcp", "azure", "sagemaker", "vertex ai",

    # Data
    "spark", "airflow", "sql", "pandas", "numpy",

    # APIs & Backend
    "fastapi", "flask", "rest api", "grpc", "microservices",
]

# ── Location Keywords ────────────────────────────────────────
INDIA_CITIES = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "ncr",
    "bangalore", "bengaluru", "chennai", "kolkata", "india",
    "remote", "hybrid"
]

# ── Seniority Keywords ───────────────────────────────────────
SENIORITY_MAP = {
    "junior": 1, "entry": 1, "associate": 1,
    "mid": 2, "intermediate": 2,
    "senior": 3, "lead": 4, "staff": 4,
    "principal": 5, "architect": 5,
    "director": 6, "head": 6, "vp": 7
}

# ── Negative Keywords (things JD says NOT to have) ──────────
NEGATIVE_KEYWORDS = [
    "not consulting", "no consulting", "not tcs", "not infosys",
    "not wipro", "product company only", "no services",
    "not pure research", "not academic"
]

# ── Default Senior AI Engineer JD ───────────────────────────
DEFAULT_JD = """
Senior AI Engineer at Redrob AI. 5-9 years experience.
Production experience with embeddings-based retrieval systems,
vector databases, hybrid search. Strong Python.
Hands-on evaluation frameworks for ranking systems (NDCG, MRR, MAP).
Shipped end-to-end ranking, search, or recommendation system
to real users at scale. Product company experience preferred.
NOT consulting firms like TCS, Infosys, Wipro, Accenture.
Location: Pune or Noida India preferred.
Open to Hyderabad, Mumbai, Delhi NCR.
NOT pure researchers. Must write production code.
Scrappy product-engineering attitude.
LLM fine-tuning, learning-to-rank experience a plus.
Active on platform, willing to relocate,
short notice period preferred.
"""


def extract_experience_range(jd_text):
    """
    Extract required experience range from JD text.
    Examples: "5-9 years", "3+ years", "minimum 5 years"
    Returns (min_years, max_years)
    """
    jd_lower = jd_text.lower()

    # Pattern: "5-9 years" or "5 to 9 years"
    range_pattern = r'(\d+)\s*[-to]+\s*(\d+)\s*years?'
    match = re.search(range_pattern, jd_lower)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Pattern: "5+ years" or "minimum 5 years"
    min_pattern = r'(\d+)\+\s*years?|minimum\s+(\d+)\s*years?|at\s+least\s+(\d+)\s*years?'
    match = re.search(min_pattern, jd_lower)
    if match:
        min_val = int(next(g for g in match.groups() if g))
        return min_val, min_val + 6  # assume max is min + 6

    # Pattern: just "X years"
    simple_pattern = r'(\d+)\s*years?\s+(?:of\s+)?experience'
    match = re.search(simple_pattern, jd_lower)
    if match:
        val = int(match.group(1))
        return max(0, val - 2), val + 4

    return 3, 12  # default range if nothing found


def extract_skills(jd_text):
    """
    Extract required skills from JD text.
    Returns list of detected skills.
    """
    jd_lower = jd_text.lower()
    detected = []

    for skill in SKILL_KEYWORDS:
        if skill.lower() in jd_lower:
            detected.append(skill)

    return detected


def extract_locations(jd_text):
    """
    Extract preferred locations from JD text.
    Returns list of preferred cities/regions.
    """
    jd_lower = jd_text.lower()
    detected = []

    for city in INDIA_CITIES:
        if city in jd_lower:
            detected.append(city)

    return detected if detected else ["india"]


def extract_seniority(jd_text):
    """
    Extract seniority level from JD text.
    Returns (level_name, level_number)
    """
    jd_lower = jd_text.lower()

    for keyword, level in sorted(
        SENIORITY_MAP.items(), key=lambda x: x[1], reverse=True
    ):
        if keyword in jd_lower:
            return keyword, level

    return "mid", 2  # default


def extract_negative_signals(jd_text):
    """
    Detect what the JD explicitly says NOT to have.
    Returns list of negative requirements.
    """
    jd_lower = jd_text.lower()
    negatives = []

    # Check for consulting rejection
    consulting_firms = [
        "tcs", "infosys", "wipro", "accenture",
        "cognizant", "capgemini"
    ]
    for firm in consulting_firms:
        patterns = [
            f"not {firm}", f"no {firm}",
            f"except {firm}", f"excluding {firm}"
        ]
        if any(p in jd_lower for p in patterns):
            negatives.append(f"no_{firm}")

    # Generic consulting rejection
    if any(kw in jd_lower for kw in [
        "not consulting", "no consulting",
        "product company", "not services"
    ]):
        negatives.append("no_consulting")

    # Research rejection
    if any(kw in jd_lower for kw in [
        "not pure research", "not academic",
        "production code", "must ship"
    ]):
        negatives.append("no_pure_research")

    return negatives


def parse_jd(jd_text=None):
    """
    Main function — parse a job description and return
    structured requirements dictionary.

    This is what our ranker uses to score candidates.
    """
    if not jd_text or jd_text.strip() == "":
        jd_text = DEFAULT_JD

    exp_min, exp_max = extract_experience_range(jd_text)
    skills = extract_skills(jd_text)
    locations = extract_locations(jd_text)
    seniority_name, seniority_level = extract_seniority(jd_text)
    negatives = extract_negative_signals(jd_text)

    # Build JD profile
    jd_profile = {
        "raw_text": jd_text,
        "experience": {
            "min": exp_min,
            "max": exp_max,
            "ideal": (exp_min + exp_max) / 2
        },
        "required_skills": skills,
        "preferred_locations": locations,
        "seniority": {
            "name": seniority_name,
            "level": seniority_level
        },
        "negative_signals": negatives,
        "is_ai_role": any(kw in jd_text.lower() for kw in [
            "ai", "ml", "machine learning", "nlp",
            "deep learning", "llm", "embeddings"
        ]),
        "requires_product_exp": "product" in jd_text.lower(),
        "requires_india": any(
            city in jd_text.lower()
            for city in ["india", "pune", "noida",
                         "bangalore", "hyderabad", "mumbai"]
        )
    }

    return jd_profile


def get_jd_summary(jd_profile):
    """
    Generate human readable summary of parsed JD.
    Shown in the Gradio UI so users can verify parsing.
    """
    exp = jd_profile["experience"]
    summary = f"""
📋 **Parsed JD Requirements:**
- **Experience:** {exp['min']}–{exp['max']} years (ideal: {exp['ideal']:.0f} years)
- **Seniority:** {jd_profile['seniority']['name'].title()}
- **Required Skills ({len(jd_profile['required_skills'])}):** {', '.join(jd_profile['required_skills'][:8])}{'...' if len(jd_profile['required_skills']) > 8 else ''}
- **Preferred Locations:** {', '.join(jd_profile['preferred_locations'])}
- **AI Role:** {'Yes ✅' if jd_profile['is_ai_role'] else 'No'}
- **Product Exp Required:** {'Yes ✅' if jd_profile['requires_product_exp'] else 'No'}
- **Negative Signals:** {', '.join(jd_profile['negative_signals']) if jd_profile['negative_signals'] else 'None detected'}
    """.strip()
    return summary