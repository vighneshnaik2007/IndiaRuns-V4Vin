# ============================================================
# outreach.py — Batch Actioning & Outreach (LOCAL ONLY)
# Generates personalized email DRAFTS locally. Nothing is sent.
# HR opens their own mail client (mailto:) or copies the draft
# once they have internet access. No email API, no network call.
# ============================================================

import urllib.parse


def generate_outreach_email(candidate, jd_title="Senior AI Engineer", company_name="our team"):
    """
    Generate a personalized outreach email draft for one candidate.
    Pure string templating — no API call.
    """
    p = candidate.get("profile", {})
    name = p.get("anonymized_name", "Candidate")
    title = p.get("current_title", "")
    company = p.get("current_company", "")

    subject = f"Opportunity: {jd_title} role"
    body = f"""Hi {name},

I came across your profile and was impressed by your background as a {title} at {company}.

We have an open {jd_title} position that I think could be a strong fit for your experience. Would you be open to a quick call this week to discuss?

Looking forward to hearing from you.

Best regards,
Hiring Team, {company_name}"""

    return {"to_candidate_id": candidate.get("candidate_id"), "subject": subject, "body": body}


def generate_mailto_link(email_draft, to_email="candidate@example.com"):
    """
    Build a mailto: link. Clicking it opens the user's own local email
    client (Gmail/Outlook desktop app) with subject+body pre-filled.
    No network call happens on our end — the actual send is fully
    in the recruiter's control once they have internet.
    """
    subject = urllib.parse.quote(email_draft["subject"])
    body = urllib.parse.quote(email_draft["body"])
    return f"mailto:{to_email}?subject={subject}&body={body}"


def generate_batch_drafts(top_candidates_rows, jd_title="Senior AI Engineer", company_name="our team"):
    """
    Given the top N ranked candidate rows (dicts with candidate + profile info),
    generate a draft + mailto link for each. Returns a list of dicts ready
    to render in the UI or save to a local .txt bundle.
    """
    drafts = []
    for row in top_candidates_rows:
        draft = generate_outreach_email(row, jd_title, company_name)
        draft["mailto_link"] = generate_mailto_link(draft)
        drafts.append(draft)
    return drafts


def export_drafts_as_text(drafts):
    """Combine all drafts into a single downloadable .txt bundle."""
    out = []
    for i, d in enumerate(drafts, 1):
        out.append(f"{'='*60}\nDRAFT #{i} — Candidate: {d['to_candidate_id']}\n{'='*60}")
        out.append(f"Subject: {d['subject']}\n")
        out.append(d["body"])
        out.append("\n")
    return "\n".join(out)


# ============================================================
# ATS Export — local CSV formatting for manual bulk-import.
# No push to any external ATS API — just generates the exact
# column structure most ATS platforms accept for CSV import.
# ============================================================

import pandas as pd
import io


def export_for_ats(df, candidates_by_id, ats_format="generic"):
    """
    df: our ranked submission dataframe (candidate_id, rank, score, reasoning)
    candidates_by_id: dict of candidate_id -> full candidate dict (profile, skills, etc.)
    ats_format: 'generic' | 'greenhouse' | 'lever'

    Builds a CSV using the actual structured fields most ATS platforms expect
    for bulk/manual CSV import — not our internal scoring text.
    """
    rows = []
    for _, r in df.iterrows():
        cand = candidates_by_id.get(r["candidate_id"], {})
        p = cand.get("profile", {})
        skills = ", ".join(s.get("name", "") for s in cand.get("skills", [])[:8])
        name = p.get("anonymized_name") or f"Candidate {r['candidate_id'][-4:]}"
        email = p.get("email", "")
        phone = p.get("phone", "")
        location = f"{p.get('location','')}, {p.get('country','')}".strip(", ")
        title = p.get("current_title", "")
        company = p.get("current_company", "")
        yoe = p.get("years_of_experience", "")

        if ats_format == "greenhouse":
            rows.append({
                "First Name": name.split()[0] if name else "",
                "Last Name": " ".join(name.split()[1:]) if len(name.split()) > 1 else "",
                "Email": email,
                "Phone": phone,
                "Current Company": company,
                "Current Title": title,
                "Location": location,
                "Source": "RankSense AI Shortlist",
                "Stage": "Sourced",
                "Match Score": r["score"],
                "Rank": r["rank"],
            })
        elif ats_format == "lever":
            rows.append({
                "Name": name,
                "Email": email,
                "Phone": phone,
                "Headline": f"{title} at {company}".strip(" at "),
                "Location": location,
                "Origin": "Sourced - RankSense AI",
                "Stage": "New Lead",
                "Tags": "ai-ranked",
                "Score": r["score"],
            })
        else:  # generic
            rows.append({
                "Candidate ID": r["candidate_id"],
                "Name": name,
                "Email": email,
                "Phone": phone,
                "Current Title": title,
                "Current Company": company,
                "Years Experience": yoe,
                "Location": location,
                "Top Skills": skills,
                "Rank": r["rank"],
                "Match Score": r["score"],
            })

    out = pd.DataFrame(rows)
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return buf.getvalue()


# ============================================================
# Continuous Feedback Loop (RLHF-lite) — local only
# Stores recruiter accept/reject decisions in a local JSON file
# and nudges scoring weights for the remaining pipeline.
# No external training service, no API — just local file + math.
# ============================================================

import json
import os

FEEDBACK_FILE = "recruiter_feedback.json"

REJECTION_REASON_WEIGHTS = {
    "Too junior": {"rules": -0.05},
    "Wrong tech stack": {"skill": -0.08},
    "Overqualified": {"rules": -0.03},
    "Location mismatch": {"rules": -0.05},
    "Inactive candidate": {"signals": -0.05},
    "Weak career trajectory": {"career": -0.05},
}


def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return {"accepted": [], "rejected": []}


def save_feedback(feedback):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback, f, indent=2)


def record_rejection(candidate_id, reason):
    feedback = load_feedback()
    feedback["rejected"].append({"candidate_id": candidate_id, "reason": reason})
    save_feedback(feedback)
    return feedback


def record_acceptance(candidate_id):
    feedback = load_feedback()
    feedback["accepted"].append({"candidate_id": candidate_id})
    save_feedback(feedback)
    return feedback


def compute_weight_nudges():
    """
    Look at accumulated rejection reasons and compute small weight
    nudges to apply to the remaining (not-yet-reviewed) candidates.
    This is a lightweight, local approximation of RLHF — no model
    training, just a frequency-weighted adjustment.
    """
    feedback = load_feedback()
    rejections = feedback.get("rejected", [])
    if not rejections:
        return {}

    nudge = {"semantic": 0.0, "skill": 0.0, "signals": 0.0, "rules": 0.0, "career": 0.0}
    for r in rejections:
        reason = r.get("reason")
        adj = REJECTION_REASON_WEIGHTS.get(reason, {})
        for k, v in adj.items():
            nudge[k] = nudge.get(k, 0.0) + v

    # Cap nudges so they can't swing scoring wildly
    for k in nudge:
        nudge[k] = max(-0.15, min(0.15, nudge[k]))

    return nudge