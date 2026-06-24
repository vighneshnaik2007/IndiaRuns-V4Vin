# ============================================================
# domain.py — Company Tier & Domain Alignment
# Weighs candidate's background against hiring company's domain
# ============================================================

DOMAIN_COMPANIES = {
    "fintech": ["razorpay", "phonepe", "paytm", "cred", "groww", "zerodha",
                "upstox", "policybazaar", "lendingkart", "slice", "jupiter",
                "stripe", "square", "plaid", "revolut", "chime"],
    "ecommerce": ["flipkart", "amazon", "meesho", "myntra", "nykaa", "ajio",
                  "shopify", "etsy", "alibaba"],
    "foodtech": ["zomato", "swiggy", "doordash", "ubereats", "zepto", "blinkit"],
    "edtech": ["byju", "unacademy", "vedantu", "upgrad", "coursera", "udemy"],
    "healthtech": ["practo", "pharmeasy", "1mg", "curefit", "cure.fit"],
    "saas_b2b": ["freshworks", "zoho", "browserstack", "postman", "chargebee",
                 "atlassian", "salesforce", "hubspot"],
    "ai_ml": ["openai", "anthropic", "redrob", "deepmind", "huggingface",
              "cohere", "stability ai"],
    "banking": ["hdfc", "icici", "axis bank", "sbi", "kotak", "jpmorgan",
                "goldman sachs", "morgan stanley", "citibank"],
    "ride_hailing": ["uber", "ola", "lyft", "rapido"],
    "gaming": ["dream11", "mpl", "nazara", "epic games", "zynga"],
}

TIER_1_PRODUCT = ["google", "microsoft", "amazon", "meta", "apple", "netflix",
                   "openai", "anthropic", "stripe", "uber", "airbnb"]
TIER_2_PRODUCT = ["flipkart", "razorpay", "zomato", "swiggy", "phonepe",
                   "freshworks", "zoho", "cred", "groww"]


def detect_jd_domain(jd_text):
    """Guess which domain the hiring company belongs to, from JD text."""
    jd_lower = jd_text.lower()
    scores = {}
    for domain, companies in DOMAIN_COMPANIES.items():
        hits = sum(1 for c in companies if c in jd_lower)
        if domain in jd_lower.replace("_", " "):
            hits += 2
        if hits:
            scores[domain] = hits
    if not scores:
        return None
    return max(scores, key=scores.get)


def get_company_domain(company_name):
    """Return which domain(s) a company belongs to."""
    if not company_name:
        return []
    name = company_name.lower()
    matches = []
    for domain, companies in DOMAIN_COMPANIES.items():
        if any(c in name for c in companies):
            matches.append(domain)
    return matches


def get_company_tier(company_name):
    """Return 1, 2, or 0 (unknown/other) tier for a company."""
    if not company_name:
        return 0
    name = company_name.lower()
    if any(c in name for c in TIER_1_PRODUCT):
        return 1
    if any(c in name for c in TIER_2_PRODUCT):
        return 2
    return 0


def domain_alignment_score(candidate, jd_domain):
    """
    Score 0.0 - 1.0 based on how well candidate's company history
    aligns with the hiring company's domain.
    """
    if not jd_domain:
        return 0.0

    career = candidate.get("career_history", [])
    current_company = candidate.get("profile", {}).get("current_company", "")
    companies = [current_company] + [j.get("company", "") for j in career]

    score = 0.0
    for i, co in enumerate(companies):
        domains = get_company_domain(co)
        if jd_domain in domains:
            # current company matters most, decay for older roles
            weight = 1.0 if i == 0 else max(0.3, 1.0 - i * 0.2)
            score = max(score, weight)

    return score


def domain_alignment_label(score):
    if score >= 0.8:
        return "🎯 Direct domain match"
    elif score >= 0.4:
        return "🔶 Prior domain exposure"
    else:
        return "⚪ No domain history found"
