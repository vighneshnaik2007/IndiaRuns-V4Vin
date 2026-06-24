# ============================================================
# nlq.py — Natural Language Querying (fully local, no LLM API)
# Parses plain-English recruiter queries into structured filters
# using regex + keyword matching against our own local skill graph
# and location lists. No external API calls.
# ============================================================

import re
from skill_graph import SKILL_GRAPH, JD_SKILLS

INDIA_CITIES = ["bengaluru", "bangalore", "pune", "noida", "hyderabad",
                 "mumbai", "delhi", "chennai", "kolkata", "gurgaon",
                 "gurugram", "ahmedabad", "kochi", "coimbatore", "indore",
                 "chandigarh", "trivandrum", "vizag", "bhubaneswar"]

SENIORITY_WORDS = {
    "junior": (0, 3), "fresher": (0, 2), "entry": (0, 2), "entry-level": (0, 2),
    "mid": (3, 6), "mid-level": (3, 6),
    "senior": (5, 12), "lead": (7, 15), "staff": (8, 16), "principal": (10, 20),
}

COMPANY_TYPE_WORDS = {
    "startup": "startup", "product company": "product", "product-based": "product",
    "consulting": "consulting", "service-based": "consulting",
}


def extract_location(query):
    q = query.lower()
    for city in INDIA_CITIES:
        if city in q:
            return city.title()
    return None


def extract_experience(query):
    q = query.lower()
    for word, (lo, hi) in SENIORITY_WORDS.items():
        if word in q:
            return lo, hi
    m = re.search(r'(\d+)\+?\s*(?:years|yrs|year)', q)
    if m:
        val = int(m.group(1))
        return val, val + 4
    return None


def extract_skills(query):
    q = query.lower()
    found = []
    for skill in JD_SKILLS:
        if skill in q:
            found.append(skill)
    for key in SKILL_GRAPH.keys():
        if key in q and key not in found:
            found.append(key)
    return found


def extract_company_type(query):
    q = query.lower()
    for phrase, kind in COMPANY_TYPE_WORDS.items():
        if phrase in q:
            return kind
    return None


def extract_role_keywords(query):
    """Pull out likely job-title words like 'python developer', 'ml engineer'."""
    q = query.lower()
    role_patterns = [
        r'(senior|junior|lead|staff|principal)?\s*([a-z]+(?:\s[a-z]+)?\s(?:developer|engineer|scientist|analyst|architect|manager))'
    ]
    roles = []
    for pat in role_patterns:
        for m in re.finditer(pat, q):
            role = m.group(2).strip()
            if role:
                roles.append(role)
    return list(set(roles))


def parse_nlq(query):
    """
    Main entry point. Converts a plain English query like:
    "Find me a senior Python developer in Bengaluru who has worked at a startup"
    into a structured filter dict — all done locally with regex/keyword
    matching against our own data. No API call involved.
    """
    if not query or not query.strip():
        return {
            "raw_query": "",
            "location": None,
            "experience_range": None,
            "skills": [],
            "company_type": None,
            "roles": [],
            "summary": "No query entered."
        }

    location = extract_location(query)
    exp_range = extract_experience(query)
    skills = extract_skills(query)
    company_type = extract_company_type(query)
    roles = extract_role_keywords(query)

    summary_parts = []
    if roles:
        summary_parts.append(f"role: {', '.join(roles)}")
    if exp_range:
        summary_parts.append(f"experience: {exp_range[0]}-{exp_range[1]}yrs")
    if location:
        summary_parts.append(f"location: {location}")
    if skills:
        summary_parts.append(f"skills: {', '.join(skills[:5])}")
    if company_type:
        summary_parts.append(f"company type: {company_type}")

    summary = "Interpreted as → " + "; ".join(summary_parts) if summary_parts else \
        "Could not extract specific filters — showing all results."

    return {
        "raw_query": query,
        "location": location,
        "experience_range": exp_range,
        "skills": skills,
        "company_type": company_type,
        "roles": roles,
        "summary": summary
    }


def candidate_matches_nlq(candidate, parsed_query):
    """Returns True/False whether a candidate matches the parsed NLQ filters."""
    p = candidate.get("profile", {})

    if parsed_query["location"]:
        loc = (p.get("location", "") + " " + p.get("country", "")).lower()
        if parsed_query["location"].lower() not in loc:
            return False

    if parsed_query["experience_range"]:
        lo, hi = parsed_query["experience_range"]
        yoe = p.get("years_of_experience", 0)
        if not (lo - 1 <= yoe <= hi + 2):
            return False

    if parsed_query["skills"]:
        candidate_skills = set(s["name"].lower() for s in candidate.get("skills", []))
        if not any(sk in " ".join(candidate_skills) for sk in parsed_query["skills"]):
            return False

    return True
