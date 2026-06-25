# ============================================================
# app_v4.py — RankSense: AI Candidate Intelligence Platform
# Three-column workspace · Glass-box controls · Tech-Enterprise UI
# 100% local inference — no external API calls during ranking.
# ============================================================

import gradio as gr
import json
import pandas as pd
import tempfile
import os
import zipfile
import urllib.parse
from datetime import date as _date

from ranker_v4 import RankSenseRanker, DEFAULT_WEIGHTS
from honeypot import get_honeypot_report
from jd_parser import parse_jd, get_jd_summary
from nlq import parse_nlq, candidate_matches_nlq
from lookalike import build_lookalike_text, find_lookalikes, lookalike_explanation
from bias_audit import run_full_audit, apply_blind_mode
from outreach import generate_batch_drafts, export_drafts_as_text, export_for_ats
from outreach import record_rejection, record_acceptance, compute_weight_nudges, load_feedback

# ── HuggingFace Demo Mode ─────────────────────────────────────
HF_DEMO_MODE   = os.path.exists("sample_candidates.json")
SAMPLE_FILE    = "sample_candidates.json"
HF_DEMO_BANNER = """
<div style="background:linear-gradient(135deg,#4F46E5,#0D9488);border-radius:12px;
            padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;gap:14px;">
    <span style="font-size:26px;">⚡</span>
    <div>
        <div style="font-weight:800;color:#fff;font-size:14px;">Live Demo — Pre-loaded with 50 sample candidates</div>
        <div style="color:rgba(255,255,255,0.85);font-size:12px;margin-top:2px;">
            Click <strong>🚀 Run Ranking</strong> immediately to see RankSense in action.
            Upload your own .jsonl to rank a full dataset.
        </div>
    </div>
</div>
"""

# ============================================================
# THEME / CSS
# ============================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
html, body, .gradio-container { overflow-x: hidden !important; }

:root {
    --bg-main: #F8FAFC;
    --bg-panel: #FFFFFF;
    --bg-subtle: #F1F5F9;
    --border-color: #E2E8F0;
    --text-primary: #1E293B;
    --text-secondary: #64748B;
    --text-muted: #94A3B8;
    --accent: #4F46E5;
    --accent-light: #EEF2FF;
    --accent-teal: #0D9488;
    --success: #059669;
    --warning: #D97706;
    --danger: #DC2626;
}

/* Apply gradient to the ENTIRE browser window so there are no grey edges on widescreen monitors */
.dark, .dark body {
    --bg-main: #0d0221;
    --bg-panel: #1a0533;
    --bg-subtle: #150a30;
    --border-color: rgba(167, 139, 250, 0.25);
    --text-primary: #e2e8f0;
    --text-secondary: #a78bfa;
    --text-muted: #7c6a9c;
    --accent: #a855f7;
    --accent-light: rgba(139,92,246,0.15);
    --accent-teal: #2DD4BF;
    
    background: radial-gradient(ellipse at top left, #0d0221 0%, #0a0a1a 40%, #000510 100%) !important;
    background-attachment: fixed !important;
}

/* Make the container transparent so the body gradient seamlessly shines through */
.dark .gradio-container {
    background: transparent !important;
}

.dark .rs-header, .dark .rs-panel, .dark .cand-card, .dark .download-box, .dark .filter-bar {
    background: var(--bg-panel) !important;
    border-color: var(--border-color) !important;
}
.dark .match-score, .dark .rs-logo { color: #c4b5fd !important; }

/* ── Base layout ─────────────────────────────────────────── */
.gradio-container {
    max-width: 100% !important; /* Stretches to fit the full screen width */
    margin: 0 auto !important; /* Centers the content properly */
    padding: 16px 24px !important;
    background: var(--bg-main) !important;
    min-height: 100vh;
}
.row, .tabs, .form { gap: 8px !important; margin-bottom: 0px !important; }
.block { padding: 8px !important; margin: 0px !important; border-radius: 8px !important; }
.form > *, .block > * { margin-bottom: 8px !important; }

/* Header */
.rs-header {
    background: var(--bg-panel);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.rs-logo {
    font-size: 20px; font-weight: 900; color: var(--text-primary);
    display: flex; align-items: center; gap: 10px;
}
.rs-logo-badge {
    background: linear-gradient(135deg, #4F46E5, #0D9488);
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 18px;
}
.rs-tagline { color: var(--text-secondary); font-size: 13px; margin-top: 2px; }

/* Panels */
.rs-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 8px;
}
.rs-panel-title {
    font-size: 13px; font-weight: 700; color: var(--text-primary);
    margin-bottom: 8px; display: flex; align-items: center; gap: 6px;
}

/* Buttons */
button.lg.primary {
    background: linear-gradient(135deg, #4F46E5, #6366F1) !important;
    border: none !important; border-radius: 10px !important;
    color: white !important; font-weight: 700 !important; font-size: 14px !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.3) !important;
}
button.lg.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(79,70,229,0.4) !important;
}
button.lg.secondary {
    background: var(--accent-light) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 10px !important; color: var(--accent) !important; font-weight: 600 !important;
}

/* Sliders */
input[type=range] { accent-color: var(--accent) !important; }

/* Tabs */
.tab-nav { border-bottom: 1px solid var(--border-color) !important; }
.tab-nav button {
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-secondary) !important; font-weight: 600 !important;
    padding: 10px 16px !important; border-radius: 0 !important;
}
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Candidate card */
.cand-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.cand-card:hover {
    border-color: var(--accent);
    box-shadow: 0 4px 14px rgba(79,70,229,0.12);
}
.match-score { font-size: 18px; font-weight: 900; color: var(--accent); }
.cand-meta { color: var(--text-muted); font-size: 12px; }
.strength-chip {
    display: inline-block; background: var(--accent-light); color: var(--accent);
    border-radius: 14px; padding: 2px 10px; font-size: 11px; font-weight: 600;
    margin: 2px 3px 0 0;
}

/* Skeleton / spinner */
.skeleton {
    background: linear-gradient(90deg, var(--bg-subtle) 25%, var(--border-color) 50%, var(--bg-subtle) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 8px;
}
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-subtle); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }

/* Weight badge / step label */
.weight-badge {
    background: var(--accent-light); color: var(--accent);
    border-radius: 8px; padding: 4px 10px; font-size: 11px; font-weight: 700;
}
.step-label {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    background: var(--accent); color: white;
    font-size: 11px; font-weight: 800; margin-right: 6px;
}
.schema-note {
    background: var(--bg-subtle); border: 1px solid var(--border-color);
    border-radius: 10px; padding: 8px 10px;
    color: var(--text-secondary); font-size: 11px; line-height: 1.5;
}
.weight-total {
    border: 1px solid; border-radius: 10px;
    background: var(--bg-subtle); padding: 7px 9px; margin: 6px 0 8px;
}

/* Empty / loading states */
.empty-state, .loading-state {
    background: var(--bg-panel); border: 1px solid var(--border-color);
    border-radius: 12px; padding: 16px; min-height: 120px;
    display: flex; gap: 22px; align-items: center; color: var(--text-secondary);
}
.empty-visual { flex: 1; min-width: 180px; }
.empty-card, .empty-profile {
    border: 1px solid var(--border-color); border-radius: 10px;
    background: var(--bg-subtle); padding: 9px; margin-bottom: 6px;
    display: flex; gap: 10px; align-items: center;
}
.empty-card.primary { border-color: var(--accent); background: var(--accent-light); }
.empty-rank {
    width: 34px; height: 34px; border-radius: 8px;
    background: var(--bg-panel); color: var(--accent);
    display: flex; align-items: center; justify-content: center; font-weight: 900;
}
.empty-card b, .empty-profile b { display: block; color: var(--text-primary); font-size: 13px; }
.empty-card span, .empty-profile span { display: block; color: var(--text-secondary); font-size: 11px; margin-top: 3px; }
.empty-copy { flex: 1; }
.empty-copy b { display:block; color: var(--text-primary); font-size: 16px; margin-bottom: 6px; }
.empty-side { min-height: 200px; display: flex; align-items: center; justify-content: center; }
.empty-profile.large { max-width: 320px; }
.loading-state { flex-direction: column; justify-content: center; text-align: center; }
.spinner {
    width: 42px; height: 42px; border: 4px solid var(--border-color);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin 0.85s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.progress-bar {
    width: min(360px, 100%); height: 8px;
    background: var(--bg-subtle); border-radius: 999px; overflow: hidden;
}
.progress-bar div {
    width: 45%; height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-teal));
    border-radius: 999px; animation: loading-slide 1.2s ease-in-out infinite;
}
@keyframes loading-slide {
    0% { transform: translateX(-120%); }
    100% { transform: translateX(240%); }
}
.status-pill { border-radius: 10px; padding: 10px 12px; font-size: 12px; font-weight: 700; }
.status-pill.running { background: var(--accent-light); color: var(--accent); }

/* Setup view (Centered configuration) */
.setup-view {
    width: 100% !important;
    max-width: 760px !important;
    margin: 0 auto !important;
    align-self: center !important;
    background: var(--bg-panel); border: 1px solid var(--border-color);
    border-radius: 12px; padding: 14px !important;
}

/* Results header row */
.results-nav { align-items: center !important; margin-bottom: 8px !important; }
.results-header {
    background: var(--bg-panel); border: 1px solid var(--border-color);
    border-radius: 10px; padding: 8px !important; align-items: center !important;
}

/* ── WORKSPACE: side-by-side panels ──────────────────────── */
.workspace-outer {
    display: flex !important;
    gap: 12px !important;
    align-items: flex-start !important;
    width: 100% !important;
    margin-top: 8px !important;
    overflow-x: hidden !important;
}

/* Each half */
.ws-left, .ws-right {
    flex: 0 0 calc(50% - 6px) !important;
    max-width: calc(50% - 6px) !important;
    min-width: 0 !important;
    height: 580px !important;
    max-height: 580px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    background: var(--bg-panel) !important;
    padding: 12px !important;
    box-sizing: border-box !important;
}

/* Gradio column pair used for the workspace */
.workspace-col-left {
    flex: 0 0 calc(50% - 6px) !important;
    max-width: calc(50% - 6px) !important;
    min-width: 0 !important;
    width: calc(50% - 6px) !important;
}
.workspace-col-right {
    flex: 0 0 calc(50% - 6px) !important;
    max-width: calc(50% - 6px) !important;
    min-width: 0 !important;
    width: calc(50% - 6px) !important;
}

/* The scrollable box inside each Gradio column */
.panel-scroll-box {
    height: 580px !important;
    max-height: 580px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    background: var(--bg-panel) !important;
    padding: 12px !important;
    box-sizing: border-box !important;
    width: 100% !important;
}

/* Make HTML output components fill their column */
.workspace-col-left .block,
.workspace-col-right .block {
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    width: 100% !important;
}

.workspace-col-left .prose,
.workspace-col-right .prose,
.workspace-col-left .wrap,
.workspace-col-right .wrap {
    width: 100% !important;
    min-width: 0 !important;
}

/* Metric strip */
.metric-strip { display: flex; gap: 8px; margin-bottom: 8px; }
.metric-strip > div {
    flex: 1; background: var(--bg-panel); border: 1px solid var(--border-color);
    border-radius: 12px; padding: 14px; text-align: center;
}

/* ── FIXED 3-COLUMN FOOTER CARDS ── */
.footer-actions-row { margin-top: 8px !important; gap: 12px !important; display: flex; }
.footer-card {
    height: 380px !important;
    max-height: 380px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 16px !important;
    flex: 1;
    display: block !important;
}
.footer-desc {
    font-size: 12px; color: var(--text-secondary);
    line-height: 1.5; margin-bottom: 16px; margin-top: 4px;
}

/* Misc */
.sidebar-compact textarea { min-height: 120px !important; }
.nlq-row {
    background: var(--bg-panel); border: 1px solid var(--border-color);
    border-radius: 10px; padding: 8px; margin-bottom: 8px;
}
.scroll-col { max-height: 550px; overflow-y: auto; padding-right: 4px; }

/* Perfect alignment for the Search Button against the Textbox */
.search-row { align-items: stretch !important; }
.align-search-btn { display: flex !important; margin: 0 !important; }
.align-search-btn button { 
    height: 100% !important; 
    min-height: 100% !important;
    background: var(--bg-subtle) !important; 
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
}
.align-search-btn button:hover {
    background: var(--border-color) !important;
}

/* Reject chips */
.reject-chip button {
    background: #FEF2F2 !important; border: 1px solid #FCA5A5 !important;
    color: #DC2626 !important; border-radius: 20px !important;
    font-size: 12px !important; padding: 6px 14px !important; font-weight: 600 !important;
}

/* Hidden link for click events */
.rs-hidden-cand-id {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
    z-index: -100 !important;
    height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
}
"""

JD_TEMPLATE = """Senior AI Engineer. 5-9 years experience.
Production experience with embeddings-based retrieval systems, vector databases, hybrid search.
Strong Python. Hands-on evaluation frameworks for ranking systems (NDCG, MRR, MAP).
Shipped end-to-end ranking, search, or recommendation system to real users at scale.
Product company experience preferred. NOT consulting firms like TCS, Infosys, Wipro, Accenture.
Location: Pune or Noida India preferred. Open to Hyderabad, Mumbai, Delhi NCR.
NOT pure researchers. Must write production code.
LLM fine-tuning, learning-to-rank experience a plus."""

LABEL_COLORS = {
    "Ideal Fit": "#059669", "Rising Star": "#0D9488", "Strong Match": "#4F46E5",
    "Risky Pick": "#DC2626", "Potential": "#64748B",
}

REJECTION_REASONS = [
    "Missing Core Skill",
    "Location Mismatch",
    "Experience Level",
    "Wrong Tech Stack",
    "Overqualified",
    "Inactive Candidate",
    "Weak Career Trajectory",
]

WEIGHT_PRESETS = {
    "Balanced": DEFAULT_WEIGHTS,
    "Technical Heavy": {
        "semantic": 0.24, "skill": 0.28, "signals": 0.10, "rules": 0.10,
        "career": 0.10, "domain": 0.13, "credentials": 0.05,
    },
    "Culture First": {
        "semantic": 0.20, "skill": 0.14, "signals": 0.26, "rules": 0.10,
        "career": 0.15, "domain": 0.08, "credentials": 0.07,
    },
}


def get_label_color(label_title):
    for key, color in LABEL_COLORS.items():
        if key in label_title:
            return color
    return "#64748B"


def parse_reasoning(reasoning):
    parts = [p.strip() for p in reasoning.split(';')]
    label_title = parts[0] if parts else ""
    skills_part = next((p for p in parts if 'skills:' in p), "")
    skills = [s.strip() for s in skills_part.replace('skills:', '').split(',')] if skills_part else []
    career = next((p for p in parts if 'career:' in p), "").replace('career:', '').strip()
    domain = next((p for p in parts if 'domain:' in p and 'scores' not in p), "").replace('domain:', '').strip()
    scores_part = next((p for p in parts if 'scores:' in p), "")
    scores = {}
    if scores_part:
        try:
            for item in scores_part.replace('scores:', '').split(','):
                k, v = item.strip().split('=')
                scores[k.strip()] = float(v.strip())
        except Exception:
            pass
    location = ""
    if '(' in label_title and ')' in label_title:
        location = label_title[label_title.rfind('(')+1:label_title.rfind(')')]
    label_name = "Potential"
    for key in LABEL_COLORS:
        if key in label_title:
            label_name = key
            break
    return {
        "label_title": label_title, "skills": skills, "career": career,
        "domain": domain, "scores": scores, "location": location, "label_name": label_name
    }


# ── Initialize ranker ────────────────────────────────────────
print("[*] Initializing RankSense Engine...")
ranker = RankSenseRanker()
ranker.load_models()
print("[OK] RankSense ready!")

STATE = {"df": None, "candidates": None, "embeddings": None, "selected_id": None}


def candidate_card_html(row, idx, selected_id=None):
    rank = int(row['rank'])
    # Convert 0-1 score to a whole integer percentage scale
    display_score = int(round(float(row['score']) * 100))
    meta = parse_reasoning(row['reasoning'])
    label_color = get_label_color(meta["label_title"])
    is_selected = (row['candidate_id'] == selected_id)
    border = "2px solid #4F46E5" if is_selected else "1px solid var(--border-color)"
    bg = "var(--accent-light)" if is_selected else "var(--bg-panel)"

    strengths = []
    if meta["scores"].get("sem", 0) >= 0.55:    strengths.append("Strong JD Match")
    if meta["scores"].get("skill", 0) >= 0.4:   strengths.append("Skilled")
    if meta["scores"].get("signals", 0) >= 0.7: strengths.append("Highly Active")
    if meta["scores"].get("career", 0) >= 0.15: strengths.append("Strong Growth")
    if meta["scores"].get("domain", 0) >= 0.5:  strengths.append("Domain Match")
    if not strengths:
        strengths = meta["skills"][:3]

    chips = "".join(f'<span class="strength-chip">{s}</span>' for s in strengths[:3])

    # Safe name extraction
    if '|' in meta['label_title']:
        display_name = meta['label_title'].split('|')[1].strip()[:42]
    else:
        display_name = meta['label_title'][:42]

    # Javascript trigger to update the hidden text box when clicked
    cand_id = str(row['candidate_id']).replace("'", "\\'") 
    js_code = f"let els = document.querySelectorAll('.rs-hidden-cand-id textarea, .rs-hidden-cand-id input'); if(els.length > 0) {{ els[0].value = '{cand_id}'; els[0].dispatchEvent(new Event('input', {{bubbles: true}})); }}"

    return f"""
<div class="cand-card" onclick="{js_code}" style="border:{border};background:{bg};">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
            <div style="font-weight:700;color:var(--text-primary);font-size:14px;">#{rank} &nbsp;{display_name}</div>
            <div class="cand-meta" style="margin-top:3px;">{meta['location']}</div>
        </div>
        <div class="match-score">{display_score}%</div>
    </div>
    <div style="margin-top:8px;">{chips}</div>
    <div style="font-size:10px;color:{label_color};font-weight:700;margin-top:6px;">{meta['label_name']}</div>
</div>"""


def deep_dive_html(row):
    meta = parse_reasoning(row['reasoning'])
    label_color = get_label_color(meta["label_title"])
    skill_tags = "".join(
        f'<span class="strength-chip">{s}</span>'
        for s in meta["skills"][:8] if s.strip()
    )

    if '|' in meta['label_title']:
        display_name = meta['label_title'].split('|')[1].strip()[:50]
    else:
        display_name = meta['label_title']

    # Convert overall score to 0-100 percentage scale
    overall_pct = int(round(float(row['score']) * 100))

    def bar(label, key, max_val, color):
        val = meta["scores"].get(key, 0)
        # Convert internal values into 0-100 format for the breakdown progress sliders
        pct = min(abs(val) / max_val * 100, 100) if max_val > 0 else 0
        return f"""<div style="margin:8px 0;">
            <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;">
                <span style="color:var(--text-secondary);font-weight:600;">{label}</span>
                <span style="color:{color};font-weight:700;">{pct:.0f}%</span>
            </div>
            <div style="background:var(--bg-subtle);border-radius:5px;height:7px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{color};border-radius:5px;"></div>
            </div></div>"""

    return f"""
<div class="rs-panel" style="margin-bottom:0;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;">
        <div>
            <div style="font-size:18px;font-weight:800;color:var(--text-primary);">{display_name}</div>
            <div style="color:var(--text-secondary);font-size:13px;margin-top:4px;">📍 {meta['location']}</div>
        </div>
        <div style="text-align:center;background:var(--accent-light);border-radius:12px;padding:10px 16px;">
            <div style="font-size:26px;font-weight:900;color:var(--accent);">{overall_pct}%</div>
            <div style="font-size:10px;color:{label_color};font-weight:700;">{meta['label_name'].upper()}</div>
        </div>
    </div>
    <div style="margin-bottom:14px;">{skill_tags}</div>
    <div style="background:var(--bg-subtle);border-radius:10px;padding:6px 12px;margin-bottom:14px;font-size:12px;color:var(--text-secondary);">
        📈 {meta['career']} &nbsp;·&nbsp; 🎯 {meta['domain']}
    </div>
    <div style="font-size:12px;font-weight:700;color:var(--text-primary);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">AI Justification — Match Breakdown</div>
    {bar("🧠 Semantic Fit", "sem", 1.0, "#4F46E5")}
    {bar("🕸️ Skill Match", "skill", 1.0, "#0D9488")}
    {bar("📡 Behavioral Signals", "signals", 1.0, "#059669")}
    {bar("📋 Rule-Based Fit", "rules", 0.5, "#D97706")}
    {bar("📈 Career Trajectory", "career", 0.4, "#DC2626")}
    {bar("🏢 Domain Alignment", "domain", 1.0, "#7C3AED")}
    {bar("🎓 Alt. Credentials", "credentials", 0.15, "#0891B2")}
</div>"""


def stats_strip(total, honeypots, clean, top_score):
    top_score_pct = int(round(float(top_score) * 100)) if top_score > 0 else 0
    def card(value, label, color):
        return (
            f'<div style="flex:1;background:var(--bg-panel);border:1px solid var(--border-color);'
            f'border-radius:12px;padding:14px;text-align:center;">'
            f'<div style="font-size:22px;font-weight:900;color:{color};">{value}</div>'
            f'<div style="font-size:11px;color:var(--text-secondary);margin-top:2px;">{label}</div></div>'
        )
    return (
        f'<div class="metric-strip">'
        f'{card(f"{total:,}", "Processed", "var(--text-primary)")}'
        f'{card(honeypots, "Honeypots Removed", "#DC2626")}'
        f'{card(f"{clean:,}", "Clean Candidates", "#059669")}'
        f'{card(f"{top_score_pct}%", "Top Score", "var(--accent)")}'
        f'</div>'
    )


def weights_total_html(w_semantic, w_skill, w_signals, w_rules, w_career, w_domain, w_cred):
    total = sum(float(v or 0) for v in [w_semantic, w_skill, w_signals, w_rules, w_career, w_domain, w_cred])
    delta = abs(total - 1.0)
    color = "#059669" if delta <= 0.005 else "#D97706" if delta <= 0.05 else "#DC2626"
    note = "Ready" if delta <= 0.005 else "Will auto-normalize when ranking runs"
    return (
        f'<div class="weight-total" style="border-color:{color};">'
        f'<div><strong style="color:{color};">Total Weight: {total:.2f}/1.00</strong>'
        f'<span style="color:var(--text-secondary);font-size:11px;margin-left:8px;">{note}</span>'
        f'</div></div>'
    )


def normalize_weights(w_semantic, w_skill, w_signals, w_rules, w_career, w_domain, w_cred):
    weights = [float(v or 0) for v in [w_semantic, w_skill, w_signals, w_rules, w_career, w_domain, w_cred]]
    total = sum(weights)
    if total <= 0:
        weights = [1 / 7] * 7
    else:
        weights = [round(v / total, 2) for v in weights]
        drift = round(1.0 - sum(weights), 2)
        weights[0] = round(weights[0] + drift, 2)
    return (*weights, weights_total_html(*weights))


def preset_weights(name):
    preset = WEIGHT_PRESETS[name]
    weights = [
        preset["semantic"], preset["skill"], preset["signals"], preset["rules"],
        preset["career"], preset["domain"], preset["credentials"],
    ]
    return (*weights, weights_total_html(*weights))


def empty_workspace_html():
    return """
<div class="empty-state">
    <div class="empty-visual">
        <div class="empty-card primary">
            <div class="empty-rank">#1</div>
            <div><b>Senior AI Engineer</b><span>82% match</span></div>
        </div>
        <div class="empty-card">
            <div class="empty-rank">#2</div>
            <div><b>Search Platform Lead</b><span>76% match</span></div>
        </div>
        <div class="empty-profile">
            <b>Ranked shortlist appears here</b>
            <span>Upload candidates and run ranking to populate.</span>
        </div>
    </div>
    <div class="empty-copy">
        <b>Start with a candidate file.</b>
        <span>Upload JSON or JSONL, confirm the job requirements, then run ranking.</span>
    </div>
</div>"""


def empty_deep_dive_html():
    return """
<div class="rs-panel empty-side">
    <div class="empty-profile large">
        <b>No candidate selected</b>
        <span>Select a ranked candidate to see the AI justification, component scores, and action controls.</span>
    </div>
</div>"""


def loading_workspace_html():
    return """
<div class="loading-state">
    <div class="spinner"></div>
    <strong>Ranking candidates...</strong>
    <span>Running local embeddings, rules, signals, and bias audit. Large files can take a moment.</span>
    <div class="progress-bar"><div></div></div>
</div>"""


def show_ranking_loading():
    return (
        gr.update(visible=False),   # setup_container
        gr.update(visible=True),    # results_container
        "<div class='status-pill running'>Ranking in progress...</div>",  # status_text
        stats_strip(0, 0, 0, 0),   # stats_output
        loading_workspace_html(),   # list_output
        empty_deep_dive_html(),     # deep_dive_output
        gr.update(visible=False),   # candidate_selector
        gr.update(visible=False),   # quick_actions_panel
    )


def back_to_configuration():
    return gr.update(visible=True), gr.update(visible=False), ""


# ============================================================
# Core actions
# ============================================================

def run_ranking(file, jd_text, top_n, w_semantic, w_skill, w_signals, w_rules, w_career, w_domain, w_cred, blind_mode):
    # HF Demo Mode: if no file uploaded, auto-use sample_candidates.json
    if file is None and HF_DEMO_MODE:
        class _FakeFile:
            name = SAMPLE_FILE
        file = _FakeFile()
    if file is None:
        return (
            gr.update(visible=True), gr.update(visible=False),
            stats_strip(0, 0, 0, 0), "", empty_deep_dive_html(),
            None, "", "<p style='color:#DC2626;padding:8px;'>Please upload a candidates file.</p>",
            gr.update(visible=False), gr.update(visible=False)
        )

    if not jd_text or not jd_text.strip():
        jd_text = JD_TEMPLATE

    try:
        ranker.jd_profile = parse_jd(jd_text)
        from domain import detect_jd_domain
        ranker.jd_domain = detect_jd_domain(jd_text)

        candidates = []
        with open(file.name, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                candidates = json.loads(content)
            else:
                for line in content.split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            candidates.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        if not candidates:
            return (
                gr.update(visible=True), gr.update(visible=False),
                stats_strip(0, 0, 0, 0), "", empty_deep_dive_html(),
                None, "", "<p style='color:#DC2626;padding:8px;'>No candidates found in file.</p>",
                gr.update(visible=False), gr.update(visible=False)
            )

        weight_overrides = {
            "semantic": w_semantic, "skill": w_skill, "signals": w_signals,
            "rules": w_rules, "career": w_career, "domain": w_domain, "credentials": w_cred
        }
        try:
            nudge = compute_weight_nudges()
        except Exception:
            nudge = {}  # HF filesystem fallback

        df, honeypots, clean_candidates, embeddings = ranker.rank(
            candidates=candidates, top_n=int(top_n), save_embeddings=False,  # HF filesystem is ephemeral
            weight_overrides=weight_overrides, feedback_nudge=nudge
        )

        STATE["df"] = df
        STATE["candidates"] = clean_candidates
        STATE["embeddings"] = embeddings
        STATE["selected_id"] = df.iloc[0]["candidate_id"] if len(df) else None

        total = len(candidates)
        clean_n = total - len(honeypots)
        top_score = float(df['score'].max()) if len(df) else 0
        stats = stats_strip(total, len(honeypots), clean_n, top_score)

        # Build ranked list HTML
        list_html = '<div style="width:100%;">'
        for idx, row in df.iterrows():
            list_html += candidate_card_html(row, idx, STATE["selected_id"])
        list_html += "</div>"

        # Dropdown choices
        dropdown_choices = []
        for _, row in df.iterrows():
            meta = parse_reasoning(row['reasoning'])
            if '|' in row['reasoning']:
                name_part = meta['label_title'].split('|')[1].strip()[:40]
            else:
                name_part = row['candidate_id']
            pct_val = int(round(float(row['score']) * 100))
            label = f"#{int(row['rank'])} · {name_part} · {pct_val}%"
            dropdown_choices.append((label, row['candidate_id']))

        # Deep dive for top candidate
        deep_dive = deep_dive_html(df.iloc[0]) if len(df) else empty_deep_dive_html()

        # Create properly named output directory and file
        out_dir = tempfile.mkdtemp()
        csv_path = os.path.join(out_dir, "submission.csv")
        df.to_csv(csv_path, index=False)

        # Bias audit
        audit_input = [
            (r['candidate_id'], r['score'], c, {})
            for (_, r), c in zip(df.head(50).iterrows(), clean_candidates[:50])
        ]
        audit = run_full_audit(audit_input)
        audit_html = "<div class='rs-panel'><div class='rs-panel-title'>⚖️ Bias Audit</div>"
        for flag in audit.get("flags", []):
            audit_html += f"<div style='font-size:12px;color:var(--text-secondary);margin:4px 0;'>{flag}</div>"
        audit_html += "</div>"

        # **FIXED: Keep the unneeded dropdown hidden by setting visible=False**
        selector_update = gr.update(choices=dropdown_choices, value=STATE["selected_id"], visible=False)

        print(f"[OK] Ranking complete: {len(df)} candidates, top score: {top_score:.4f}")
        print(f"[OK] list_html length: {len(list_html)} chars")

        return (
            gr.update(visible=False), gr.update(visible=True),
            stats, list_html, deep_dive,
            csv_path, audit_html, "Ranking complete ✅",
            selector_update, gr.update(visible=True)
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] run_ranking failed:\n{tb}")
        return (
            gr.update(visible=True), gr.update(visible=False),
            stats_strip(0, 0, 0, 0), "",
            f"<pre style='font-size:11px;color:var(--text-secondary);white-space:pre-wrap;'>{tb}</pre>",
            None, "", f"<p style='color:#DC2626;padding:8px;'>Failed: {str(e)}</p>",
            gr.update(visible=False), gr.update(visible=False)
        )


def run_nlq_search(query):
    df = STATE["df"]
    if df is None:
        return "<p style='color:var(--text-secondary);padding:16px;'>Run a ranking first.</p>"
    parsed = parse_nlq(query)
    summary = (
        f"<div style='font-size:12px;color:var(--accent);background:var(--accent-light);"
        f"border-radius:8px;padding:8px 12px;margin-bottom:10px;'>{parsed['summary']}</div>"
    )

    if not STATE["candidates"]:
        return summary

    matches = []
    cand_by_id = {c["candidate_id"]: c for c in STATE["candidates"]}
    for _, row in df.iterrows():
        cand = cand_by_id.get(row["candidate_id"])
        if cand and candidate_matches_nlq(cand, parsed):
            matches.append(row)

    if not matches:
        return summary + "<p style='color:var(--text-secondary);font-size:13px;'>No candidates match this query.</p>"

    html = summary + '<div style="width:100%;">'
    for idx, row in enumerate(matches):
        html += candidate_card_html(row, idx, STATE["selected_id"])
    html += "</div>"
    return html


def select_candidate(candidate_id):
    df = STATE["df"]
    if df is None or not candidate_id:
        return "", empty_deep_dive_html(), gr.update(visible=False)
    
    match = df[df["candidate_id"] == candidate_id]
    if match.empty:
        # Fallback if somehow not found
        list_html = '<div style="width:100%;">'
        for idx, r in df.iterrows():
            list_html += candidate_card_html(r, idx, STATE["selected_id"])
        list_html += "</div>"
        return list_html, empty_deep_dive_html(), gr.update(visible=False)

    STATE["selected_id"] = candidate_id
    row = match.iloc[0]

    # Rebuild list with new selection highlighted
    list_html = '<div style="width:100%;">'
    for idx, r in df.iterrows():
        list_html += candidate_card_html(r, idx, STATE["selected_id"])
    list_html += "</div>"

    return list_html, deep_dive_html(row), gr.update(visible=True)


def reject_candidate(reason_tag, reason_note):
    reason = reason_tag or ""
    if reason_note and reason_note.strip():
        reason = f"{reason}: {reason_note.strip()}" if reason else reason_note.strip()
    if STATE["selected_id"] and reason:
        try:
            record_rejection(STATE["selected_id"], reason)
            fb = load_feedback()
            count = len(fb["rejected"])
        except Exception:
            count = 1  # HF read-only filesystem — feedback not persisted
        return (
            f"<div style='font-size:12px;color:#DC2626;padding:8px;background:#FEF2F2;border-radius:8px;'>"
            f"Noted: '{reason}' — feedback recorded ({count} total). "
            f"Future rankings will adjust.</div>"
        )
    return "<div style='font-size:12px;color:var(--text-secondary);padding:8px;'>Select a candidate and choose a reason before submitting.</div>"


def extract_jd_title(jd_text):
    if not jd_text or not jd_text.strip():
        return "this role"
    first_line = jd_text.strip().split("\n")[0]
    first_sentence = first_line.split(".")[0].strip()
    return first_sentence[:60] if first_sentence else "this role"


def generate_outreach(jd_text):
    df = STATE["df"]
    if df is None or not STATE["candidates"]:
        return None, "<p style='color:var(--text-secondary);'>Run a ranking first.</p>"

    cand_by_id = {c["candidate_id"]: c for c in STATE["candidates"]}
    jd_title = extract_jd_title(jd_text)

    top10_candidates = []
    for _, row in df.head(10).iterrows():
        cand = cand_by_id.get(row["candidate_id"])
        if cand:
            top10_candidates.append(cand)

    drafts = generate_batch_drafts(top10_candidates, jd_title=jd_title, company_name="our team")
    text_content = export_drafts_as_text(drafts)
    
    # Create properly named output directory and zip file
    out_dir = tempfile.mkdtemp()
    zip_path = os.path.join(out_dir, "draftmail.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # 1. Add text file with all drafts
        zf.writestr("outreach_drafts/drafts.txt", text_content)
        
        # 2. Add an HTML file that acts as a local click-to-send hub
        html_links = "<h2>Offline Outreach Drafts</h2><p>Click below to open in your default email client (Gmail/Outlook):</p><ul>"
        for i, draft in enumerate(drafts):
            # Attempt to safely extract basic details for the mailto link
            name = draft.get("candidate_name", f"Candidate {i+1}")
            subject = urllib.parse.quote(draft.get("subject", "Opportunity with our team"))
            body = urllib.parse.quote(draft.get("body", "Hello!"))
            email = draft.get("email", "") 
            
            html_links += f"<li><a href='mailto:{email}?subject={subject}&body={body}' target='_blank'>Send email to {name}</a></li>"
        html_links += "</ul>"
        
        zf.writestr("outreach_drafts/send_emails.html", html_links)

    preview = (
        f"<div style='font-size:12px;color:var(--text-secondary);margin-top:8px;'>"
        f"✅ Generated {len(drafts)} drafts for \"{jd_title}\". Download the ZIP above to access the .txt drafts and your local HTML email client launcher."
        f"</div>"
    )
    return zip_path, preview


def export_ats(format_choice):
    df = STATE["df"]
    if df is None or not STATE["candidates"]:
        return None
    cand_by_id = {c["candidate_id"]: c for c in STATE["candidates"]}
    csv_text = export_for_ats(df, cand_by_id, ats_format=format_choice.lower())
    
    # Create properly named output directory and file
    out_dir = tempfile.mkdtemp()
    ats_path = os.path.join(out_dir, "ats.csv")
    with open(ats_path, "w", encoding="utf-8") as f:
        f.write(csv_text)
        
    return ats_path


def preview_jd(jd_text):
    if not jd_text:
        return ""
    profile = parse_jd(jd_text)
    summary = get_jd_summary(profile)
    return (
        f"<div style='background:var(--accent-light);border-radius:10px;padding:12px;"
        f"font-size:12px;color:var(--text-secondary);margin-top:8px;'>"
        f"{summary.replace(chr(10), '<br>')}</div>"
    )

# ============================================================
# HTML/JS Injection to Force Dark Theme immediately
# ============================================================

DARK_MODE_HEAD = """
<script>
    // Force dark mode on first visit before Gradio paints the UI
    if (localStorage.getItem('theme') === null) {
        localStorage.setItem('theme', 'dark');
        document.documentElement.classList.add('dark');
    }
</script>
"""

# ============================================================
# UI Layout
# ============================================================

# We pass the DARK_MODE_HEAD script directly into the HTML head so it runs first
with gr.Blocks(title="RankSense — AI Candidate Intelligence", head=DARK_MODE_HEAD) as demo:

    gr.HTML("""
    <div class="rs-header">
        <div>
            <div class="rs-logo">
                <div class="rs-logo-badge">⚡</div> RankSense
            </div>
            <div class="rs-tagline">AI-native candidate intelligence — explainable, local, recruiter-controlled</div>
        </div>
        <div style="font-size:11px;color:var(--text-muted);">100% local inference · No data leaves your machine</div>
    </div>
    """)

    if HF_DEMO_MODE:
        gr.HTML(HF_DEMO_BANNER)

    with gr.Tabs():

        # ── WORKSPACE TAB ────────────────────────────────────
        with gr.TabItem("🔍 Workspace"):

            # ── SETUP VIEW ───────────────────────────────────
            with gr.Column(visible=True, elem_classes=["setup-view"]) as setup_container:

                gr.HTML('<div class="rs-panel-title"><span class="step-label">1</span>Upload Candidates</div>')
                gr.HTML("""
                <div class="schema-note">
                    Accepts .json arrays or newline-delimited .jsonl. Required top-level keys:
                    candidate_id, profile, career_history, education, skills, redrob_signals.
                    Use sample_candidates.json or candidate_schema.json as the local template.
                </div>""")

                with gr.Row():
                    file_input = gr.File(
                        label="Upload Candidates",
                        file_types=[".jsonl", ".json"],
                        scale=2
                    )
                    sample_template = gr.File(
                        value="sample_candidates.json",
                        label="Download template",
                        interactive=False,
                        scale=1
                    )

                top_n_slider = gr.Slider(10, 100, value=100, step=10, label="Shortlist size")

                with gr.Accordion("Step 2: Define Job Requirements", open=False):
                    jd_input = gr.Textbox(value=JD_TEMPLATE, lines=7, label="")
                    jd_parse_btn = gr.Button("Preview Parsing", size="sm")
                    jd_preview = gr.HTML("")

                with gr.Accordion("Step 3: Adjust Ranking Weights", open=False):
                    with gr.Row():
                        preset_balanced  = gr.Button("Balanced", size="sm")
                        preset_technical = gr.Button("Technical Heavy", size="sm")
                        preset_culture   = gr.Button("Culture First", size="sm")
                    normalize_btn = gr.Button("Normalize to 1.00", size="sm")
                    weight_total = gr.HTML(weights_total_html(
                        DEFAULT_WEIGHTS["semantic"], DEFAULT_WEIGHTS["skill"],
                        DEFAULT_WEIGHTS["signals"], DEFAULT_WEIGHTS["rules"],
                        DEFAULT_WEIGHTS["career"],  DEFAULT_WEIGHTS["domain"],
                        DEFAULT_WEIGHTS["credentials"]
                    ))
                    w_semantic = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["semantic"], step=0.01, label="Semantic Fit")
                    w_skill    = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["skill"],    step=0.01, label="Skill Relevance")
                    w_signals  = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["signals"],  step=0.01, label="Behavioral Signals")
                    w_rules    = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["rules"],    step=0.01, label="Location / Rules")
                    w_career   = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["career"],   step=0.01, label="Career Trajectory")
                    w_domain   = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["domain"],   step=0.01, label="Domain Alignment")
                    w_cred     = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["credentials"], step=0.01, label="Alt. Credentials")

                blind_mode_toggle = gr.Checkbox(label="🕶️ Blind Hiring Mode", value=False)
                rank_btn   = gr.Button("🚀 Run Ranking", variant="primary")
                status_text = gr.HTML("")

            # ── RESULTS VIEW ─────────────────────────────────
            with gr.Column(visible=False) as results_container:

                # Back button + status
                with gr.Row(elem_classes=["results-nav"]):
                    back_to_setup_btn = gr.Button("⬅️ Back to Configuration", size="sm")

                # 1. Natural Language Search Bar (Full Width)
                with gr.Row(elem_classes=["results-header"]):
                    with gr.Column(scale=1, min_width=0):
                        gr.HTML('<div class="rs-panel-title">Natural Language Search</div>')
                        with gr.Row(elem_classes=["search-row"]):
                            nlq_box = gr.Textbox(
                                placeholder='e.g. "senior python developer in bengaluru who worked at a startup"',
                                label="Textbox", scale=4
                            )
                            nlq_btn = gr.Button("Search", size="md", elem_classes=["align-search-btn"], scale=2)
                            
                # 2. Stats Strip (Full Width below the search bar)
                with gr.Row():
                    with gr.Column(scale=1, min_width=0):
                        stats_output = gr.HTML(stats_strip(0, 0, 0, 0))

                # ── MAIN WORKSPACE: two columns side by side ──
                with gr.Row():
                    # LEFT: Ranked Shortlist
                    with gr.Column(scale=1, min_width=0, elem_classes=["workspace-col-left"]):
                        
                        candidate_selector = gr.Dropdown(
                            choices=[], label="Jump to candidate", visible=False
                        )
                        
                        # The invisible textbox that receives the JS click event
                        selected_cand_state = gr.Textbox(elem_classes=["rs-hidden-cand-id"], container=False)
                        
                        with gr.Column(elem_classes=["panel-scroll-box"]):
                            gr.HTML(
                                '<div class="rs-panel-title">📋 Ranked Shortlist</div>'
                            )
                            list_output = gr.HTML(empty_workspace_html())

                    # RIGHT: Deep-Dive Profile
                    with gr.Column(scale=1, min_width=0, elem_classes=["workspace-col-right"]):
                        with gr.Column(elem_classes=["panel-scroll-box"]):
                            gr.HTML('<div class="rs-panel-title">🔬 Deep-Dive Profile</div>')
                            deep_dive_output = gr.HTML(empty_deep_dive_html())

                            # Quick Actions (reject)
                            with gr.Column(visible=False) as quick_actions_panel:
                                gr.HTML('<div class="rs-panel"><div class="rs-panel-title">⚡ Quick Actions — Reject with Reason</div>')
                                reject_reason_dropdown = gr.Dropdown(
                                    choices=REJECTION_REASONS, label="Reason tag"
                                )
                                reject_reason_note = gr.Textbox(
                                    lines=2, label="Optional note",
                                    placeholder="Add context for the feedback loop"
                                )
                                reject_submit_btn = gr.Button("Submit Feedback", size="sm")
                                reject_feedback   = gr.HTML("")
                                gr.HTML('</div>')

                            # Bias Audit
                            audit_output = gr.HTML("")

                # ── FOOTER: Downloads, ATS, and Outreach ──
                # Using 3 equal width columns for the footer tools
                with gr.Row(elem_classes=["footer-actions-row"]):
                    
                    # 1. Download Card
                    with gr.Column(scale=1, min_width=0, elem_classes=["footer-card"]):
                        gr.Markdown("**📥 Download Ranked Results**")
                        gr.HTML("<div class='footer-desc'>Export the complete scored and ranked shortlist as a standard CSV file, including all AI reasoning and component scores for internal review.</div>")
                        download_output = gr.File(label="Full Submission CSV")

                    # 2. ATS Export Card
                    with gr.Column(scale=1, min_width=0, elem_classes=["footer-card"]):
                        gr.Markdown("**📤 Export for ATS**")
                        gr.HTML("<div class='footer-desc'>Generate a formatted CSV optimized for seamless import into popular Applicant Tracking Systems like Greenhouse and Lever.</div>")
                        ats_format = gr.Dropdown(
                            ["Generic", "Greenhouse", "Lever"],
                            value="Generic", label="Select ATS Platform"
                        )
                        ats_btn  = gr.Button("Export", size="sm")
                        ats_file = gr.File(label="ATS CSV")

                    # 3. Offline Batch Outreach Card
                    with gr.Column(scale=1, min_width=0, elem_classes=["footer-card"], visible=True) as outreach_panel:
                        gr.Markdown("**✉️ Offline Batch Outreach**")
                        # Using the exact shortened description you requested
                        gr.HTML("<div class='footer-desc'>Generate personalized email content locally using our existing candidate data (name, role, why they matched).</div>")
                        outreach_btn     = gr.Button("Generate Top-10 Drafts", size="sm", variant="primary")
                        outreach_file    = gr.File(label="Drafts ZIP")
                        outreach_preview = gr.HTML("")

        # ── HOW IT WORKS TAB ─────────────────────────────────
        with gr.TabItem("ℹ️ How It Works"):
            gr.HTML("""
            <div style="max-width:780px;margin:0 auto;">
                <div class="rs-panel">
                    <h2 style="color:var(--text-primary);">RankSense Scoring Architecture</h2>
                    <p style="color:var(--text-secondary);">7-component hybrid pipeline, fully recruiter-tunable via the weight sliders.</p>
                    <ul style="color:var(--text-secondary);line-height:1.9;">
                        <li><strong style="color:var(--accent);">Semantic Fit</strong> — dual local embedding models (MiniLM + MPNet), no API</li>
                        <li><strong style="color:var(--accent);">Skill Relevance</strong> — 50+ technology relationship graph</li>
                        <li><strong style="color:var(--accent);">Behavioral Signals</strong> — 23 platform engagement signals</li>
                        <li><strong style="color:var(--accent);">Location / Rules</strong> — JD-derived hard constraints</li>
                        <li><strong style="color:var(--accent);">Career Trajectory</strong> — 6-dimension growth analysis</li>
                        <li><strong style="color:var(--accent);">Domain Alignment</strong> — company-domain matching (FinTech, EdTech, etc.)</li>
                        <li><strong style="color:var(--accent);">Alt. Credentials</strong> — bootcamps & certifications valued alongside degrees</li>
                    </ul>
                    <p style="color:var(--text-secondary);">
                        Plus: Honeypot detection, diversity-aware adjustment, blind hiring mode,
                        bias auditing, natural-language search, lookalike search, and a local
                        feedback loop that nudges weights based on your rejections — all without
                        a single network call during ranking.
                    </p>
                </div>
            </div>""")

        # ── LOOKALIKE SEARCH TAB ──────────────────────────────
        with gr.TabItem("👤 Lookalike Search"):
            gr.HTML('<div class="rs-panel"><div class="rs-panel-title">Find candidates similar to an existing top performer</div>')
            with gr.Row():
                seed_title   = gr.Textbox(label="Title",   placeholder="Senior AI Engineer")
                seed_company = gr.Textbox(label="Company", placeholder="Razorpay")
                seed_years   = gr.Number(label="Years Experience", value=6)
            seed_skills  = gr.Textbox(label="Key Skills (comma separated)", placeholder="python, pytorch, faiss, vector search")
            seed_summary = gr.Textbox(label="Summary (optional)", lines=3)
            lookalike_btn    = gr.Button("Find Lookalikes", variant="primary")
            lookalike_output = gr.HTML("")
            gr.HTML('</div>')

    # ============================================================
    # Events
    # ============================================================
    weight_inputs  = [w_semantic, w_skill, w_signals, w_rules, w_career, w_domain, w_cred]
    weight_outputs = [w_semantic, w_skill, w_signals, w_rules, w_career, w_domain, w_cred, weight_total]

    for wi in weight_inputs:
        wi.change(fn=weights_total_html, inputs=weight_inputs, outputs=[weight_total])

    normalize_btn.click(fn=normalize_weights,                    inputs=weight_inputs, outputs=weight_outputs)
    preset_balanced.click( fn=lambda: preset_weights("Balanced"),        inputs=[], outputs=weight_outputs)
    preset_technical.click(fn=lambda: preset_weights("Technical Heavy"), inputs=[], outputs=weight_outputs)
    preset_culture.click(  fn=lambda: preset_weights("Culture First"),   inputs=[], outputs=weight_outputs)

    back_to_setup_btn.click(
        fn=back_to_configuration,
        inputs=[],
        outputs=[setup_container, results_container, status_text]
    )

    rank_btn.click(
        fn=show_ranking_loading,
        inputs=[],
        outputs=[
            setup_container, results_container, status_text,
            stats_output, list_output, deep_dive_output,
            candidate_selector, quick_actions_panel
        ]
    ).then(
        fn=run_ranking,
        inputs=[
            file_input, jd_input, top_n_slider,
            w_semantic, w_skill, w_signals, w_rules, w_career, w_domain, w_cred,
            blind_mode_toggle
        ],
        outputs=[
            setup_container, results_container,
            stats_output, list_output, deep_dive_output,
            download_output, audit_output, status_text,
            candidate_selector, quick_actions_panel
        ]
    )

    # Triggered by the Dropdown
    candidate_selector.change(
        fn=select_candidate,
        inputs=[candidate_selector],
        outputs=[list_output, deep_dive_output, quick_actions_panel]
    )
    
    # Triggered by clicking a Candidate Card
    selected_cand_state.change(
        fn=select_candidate,
        inputs=[selected_cand_state],
        outputs=[list_output, deep_dive_output, quick_actions_panel]
    )

    jd_parse_btn.click(fn=preview_jd,        inputs=[jd_input],                                   outputs=[jd_preview])
    nlq_btn.click(     fn=run_nlq_search,    inputs=[nlq_box],                                    outputs=[list_output])
    reject_submit_btn.click(
        fn=reject_candidate,
        inputs=[reject_reason_dropdown, reject_reason_note],
        outputs=[reject_feedback]
    )
    outreach_btn.click(fn=generate_outreach, inputs=[jd_input],                                   outputs=[outreach_file, outreach_preview])
    ats_btn.click(     fn=export_ats,        inputs=[ats_format],                                 outputs=[ats_file])

    def run_lookalike(title, company, years, skills_str, summary):
        if STATE["embeddings"] is None or not STATE["candidates"]:
            return "<p style='color:var(--text-secondary);'>Run a main ranking first to build the candidate index.</p>"
        seed_profile = {
            "title": title, "company": company, "years_experience": years,
            "skills": [s.strip() for s in skills_str.split(",") if s.strip()],
            "summary": summary
        }
        seed_text = build_lookalike_text(seed_profile)
        results = find_lookalikes(
            seed_text, ranker.models[0], STATE["embeddings"], STATE["candidates"], top_n=15
        )
        html = '<div class="scroll-col">'
        for cand, sim in results:
            p = cand.get("profile", {})
            reason = lookalike_explanation(seed_profile, cand, sim)
            html += (
                f'<div class="cand-card">'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<div><strong>{p.get("current_title","")}</strong> at {p.get("current_company","")}</div>'
                f'<div class="match-score" style="font-size:16px;">{sim*100:.0f}%</div>'
                f'</div>'
                f'<div class="cand-meta">{reason}</div>'
                f'</div>'
            )
        html += "</div>"
        return html

    lookalike_btn.click(
        fn=run_lookalike,
        inputs=[seed_title, seed_company, seed_years, seed_skills, seed_summary],
        outputs=[lookalike_output]
    )

demo.launch(
    server_name="0.0.0.0",   # required for HuggingFace Spaces
    footer_links=["gradio", "settings"],
    css=CUSTOM_CSS,
    theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate")
)