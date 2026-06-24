# ============================================================
# bias_audit.py — Algorithmic Bias Auditing + Blind Hiring Mode
# Monitors whether the ranking disproportionately favors/penalizes
# specific demographics. Pure local statistics, no internet needed.
# ============================================================

from collections import Counter


def audit_distribution(ranked_candidates, field_extractor, field_name):
    """
    Generic distribution auditor.
    field_extractor: function(candidate) -> value (e.g. location, gender, institution_tier)
    Returns counts + percentage breakdown.
    """
    values = [field_extractor(c) for _, _, c, _ in ranked_candidates]
    values = [v for v in values if v]
    counts = Counter(values)
    total = max(len(values), 1)
    breakdown = {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in counts.items()}
    return {"field": field_name, "total": total, "breakdown": breakdown}


def get_location_extractor():
    def extractor(candidate):
        p = candidate.get("profile", {})
        return p.get("location", "Unknown")
    return extractor


def get_company_type_extractor():
    from career import is_consulting_company, is_product_company
    def extractor(candidate):
        p = candidate.get("profile", {})
        co = p.get("current_company", "")
        if is_consulting_company(co):
            return "Consulting"
        elif is_product_company(co):
            return "Product"
        return "Other"
    return extractor


def get_institution_tier_extractor():
    from credentials import classify_education
    def extractor(candidate):
        edu = candidate.get("education", [])
        return classify_education(edu)["type"]
    return extractor


def run_full_audit(ranked_candidates):
    """
    Run all available bias audits on the final ranked shortlist.
    Returns a dict of audit results — surfaced in the dashboard.
    """
    if not ranked_candidates:
        return {}

    results = {
        "location": audit_distribution(ranked_candidates, get_location_extractor(), "Location"),
        "company_type": audit_distribution(ranked_candidates, get_company_type_extractor(), "Company Type"),
        "education_tier": audit_distribution(ranked_candidates, get_institution_tier_extractor(), "Education Tier"),
    }

    flags = []
    for key, result in results.items():
        for label, stats in result["breakdown"].items():
            if stats["pct"] > 60:
                flags.append(
                    f"⚠️ {result['field']} '{label}' makes up {stats['pct']}% of shortlist — "
                    f"consider reviewing for unintended skew."
                )
    results["flags"] = flags if flags else ["✅ No major distribution skew detected across audited fields."]
    return results


# ── Blind Hiring Mode ────────────────────────────────────────
MASKABLE_FIELDS = ["anonymized_name", "photo_url", "gender", "graduation_year", "institution_name"]


def apply_blind_mode(candidate_display_dict):
    """
    Given a dict prepared for UI display, mask sensitive fields.
    Returns a new dict — does not mutate ranking logic, only display.
    """
    masked = dict(candidate_display_dict)
    if "name" in masked:
        masked["name"] = "Candidate " + masked.get("candidate_id", "")[-4:]
    if "photo_url" in masked:
        masked["photo_url"] = None
    if "gender" in masked:
        masked["gender"] = "Hidden"
    if "graduation_year" in masked:
        masked["graduation_year"] = "Hidden"
    if "institution_name" in masked:
        masked["institution_name"] = "Hidden (Blind Mode)"
    return masked
