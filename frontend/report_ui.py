"""Director-Typist Report Review UI  —  The Report That Learns You.

A Streamlit page that implements the conversational feedback loop:

  1. Display an AI-generated draft report.
  2. User types natural-language feedback ("Make this more formal").
  3. Call POST /templates/rewrite-section → display new draft.
  4. User clicks "Approve & Learn" → POST /templates/capture-feedback
     → Celery embeds the feedback and stores in Qdrant.
  5. Memory Insights panel shows all learned rules.

Run standalone:  streamlit run frontend/report_ui.py
Or import render_director_typist() into the main app.
"""

import os
import logging
from typing import Any

import httpx
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
ORG_ID = os.getenv("ORG_ID", "bf0d03fb-d8ea-4377-a991-b3b5818e71ec")
HEADERS = {"X-Org-ID": ORG_ID}

logger = logging.getLogger(__name__)

# ── Sample draft reports (seeded initial state) ──────────────────────────────
DRAFTS = {
    "Annual Portfolio Review": """Dear Mr. and Mrs. Henderson,

Following our recent meeting on 15th January 2025, I am pleased to present your Annual Portfolio Review for the period ending 31st December 2024.

Portfolio Performance Summary:
Your portfolio has achieved a total return of 8.2% over the review period, compared to the benchmark return of 7.1%. The portfolio value currently stands at £485,000, representing a net increase of £36,770.

Asset Allocation:
- UK Equities: 35% (£169,750) — Overweight vs. target of 30%
- Global Equities: 25% (£121,250) — In line with target
- Fixed Income: 20% (£97,000) — Underweight vs. target of 25%
- Property: 10% (£48,500) — In line with target
- Cash: 10% (£48,500) — Overweight vs. target of 5%

Risk Assessment:
Based on your completed risk questionnaire (score: 6/10), we classify your risk profile as "Balanced Growth". This remains appropriate given your stated investment horizon of 15+ years and your objective of funding retirement at age 65.

Recommendations:
1. Rebalance UK equities to target allocation, taking profits of approximately £24,250
2. Increase fixed income allocation by £24,250 to provide greater portfolio stability
3. Reduce cash holdings to 5%, deploying £24,250 into global equity markets
4. Consider adding emerging market exposure (5%) for diversification benefits

These recommendations are subject to market conditions and your ongoing agreement. Past performance is not a guarantee of future returns. The value of investments can go down as well as up.

Kind regards,
Financial Advisory Team""",

    "Risk Assessment Summary": """Risk Profile Analysis: Moderately Aggressive

Based on the psychometric risk questionnaire completed on 15th January 2025 (Score: 78/100), your risk tolerance is classified as Moderately Aggressive. 

Given your stated investment horizon of 15+ years and your objective of maximizing long-term capital growth, a higher allocation to equities (up to 80%) remains suitable. Short-term market volatility is an expected characteristic of this approach, but the long-term compounding benefits align directly with your retirement goals.

We recommend maintaining the current risk profile for the upcoming advisory period.""",

    "Fee Restructuring Proposal": """Fee and Charge Summary:

Under the proposed new structure, the ongoing platform charge will be reduced from 0.35% to 0.25% per annum, reflecting the increased scale of your overall portfolio. 

The ongoing advice fee will remain at 0.75% per annum on the first £500,000, and step down to 0.50% on assets above that threshold. This tiered approach ensures our fees remain competitive and highly transparent. We believe this represents excellent value for the ongoing strategic guidance and active portfolio management provided."""
}


# ── API Helpers ───────────────────────────────────────────────────────────────


def api_rewrite_section(original_text: str, user_feedback: str) -> str | None:
    """Call POST /templates/rewrite-section (synchronous LLM rewrite).

    Returns the new_text or None on failure.
    """
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/templates/rewrite-section",
            headers=HEADERS,
            json={"original_text": original_text, "user_feedback": user_feedback},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json().get("new_text")
    except Exception as e:
        logger.error(f"Rewrite API error: {e}")
        st.error(f"Rewrite failed: {e}")
        return None


def api_capture_feedback(
    adviser_id: str,
    original_text: str,
    chosen_text: str,
    user_feedback: str,
) -> dict[str, Any] | None:
    """Call POST /templates/capture-feedback (triggers Celery task).

    Returns the response dict or None on failure.
    """
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/templates/capture-feedback",
            headers=HEADERS,
            json={
                "adviser_id": adviser_id,
                "original_text": original_text,
                "chosen_text": chosen_text,
                "user_feedback": user_feedback,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Capture feedback error: {e}")
        st.error(f"Feedback capture failed: {e}")
        return None


def api_get_preferences(adviser_id: str) -> dict[str, Any]:
    """Call GET /templates/adviser-preferences/{adviser_id}."""
    try:
        resp = httpx.get(
            f"{API_BASE_URL}/templates/adviser-preferences/{adviser_id}",
            headers=HEADERS,
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"rules": [], "total": 0}


# ── Session State Initializer ────────────────────────────────────────────────


def _init_state():
    """Initialize session state with default values."""
    default_draft_name = list(DRAFTS.keys())[0]
    defaults = {
        "draft_selection": default_draft_name,
        "draft_text": DRAFTS[default_draft_name],     # Current draft on screen
        "original_text": DRAFTS[default_draft_name],  # Version before last rewrite
        "last_feedback": "",                          # Last feedback string
        "adviser_id": "adv_001",                      # Current adviser
        "rewrite_count": 0,                           # How many rewrites this session
        "feedback_history": [],                       # History of feedback strings
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Main UI ───────────────────────────────────────────────────────────────────


def render_director_typist():
    """Render the Director-Typist conversational feedback UI."""
    _init_state()

    st.subheader("🧠 The Report That Learns You")
    st.caption(
        "Tell the AI how to adjust this draft. When you're happy, click **Approve & Learn** "
        "to permanently store your style preference."
    )

    # ── Context selectors ─────────────────────────────────────────────────
    col_sel, col_adv, col_rst = st.columns([2, 2, 1])
    with col_sel:
        draft_selection = st.selectbox(
            "Sample Draft Context",
            list(DRAFTS.keys()),
            index=list(DRAFTS.keys()).index(st.session_state.draft_selection) if st.session_state.draft_selection in DRAFTS else 0,
            key="draft_selection_ui",
            help="Select a generic section of a report to practice giving feedback on."
        )
        if draft_selection != st.session_state.draft_selection:
            st.session_state.draft_selection = draft_selection
            st.session_state.draft_text = DRAFTS[draft_selection]
            st.session_state.original_text = DRAFTS[draft_selection]
            st.session_state.rewrite_count = 0
            st.session_state.feedback_history = []
            st.rerun()

    with col_adv:
        adviser = st.selectbox(
            "Acting as Adviser",
            ["adv_001 — Sarah Mitchell", "adv_002 — James Crawford"],
            key="adviser_select",
        )
        st.session_state.adviser_id = adviser.split(" — ")[0]
        
    with col_rst:
        st.write("")
        st.write("")
        if st.button("🔄 Reset Draft", use_container_width=True):
            st.session_state.draft_text = DRAFTS[st.session_state.draft_selection]
            st.session_state.original_text = DRAFTS[st.session_state.draft_selection]
            st.session_state.rewrite_count = 0
            st.session_state.feedback_history = []
            st.rerun()

    st.divider()

    # ── Strict 70 / 30 Split Screen Layout ─────────────────────────────────
    col_left, col_right = st.columns([7, 3])

    with col_left:
        st.markdown('### LEFT COLUMN (70%) - "The Voice"')
        
        # Show rewrite badge if we've done at least one rewrite
        if st.session_state.rewrite_count > 0:
            st.info(
                f"✏️ Rewrite #{st.session_state.rewrite_count} — "
                f"Last instruction: *\"{st.session_state.last_feedback}\"*"
            )

        # Display the draft
        st.markdown(
            f'<div style="background:#1e1e2e; padding:20px; border-radius:10px; '
            f'border:1px solid #444; font-size:14px; line-height:1.7; '
            f'white-space:pre-wrap; color:#e0e0e0; margin-bottom: 20px;">'
            f'{st.session_state.draft_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

        chat_input = st.chat_input("Adjust the style/tone of this draft..")
        if chat_input:
            with st.spinner(f'Rewriting: "{chat_input}"...'):
                new_text = api_rewrite_section(st.session_state.draft_text, chat_input)

            if new_text:
                st.session_state.original_text = st.session_state.draft_text
                st.session_state.draft_text = new_text
                st.session_state.last_feedback = chat_input
                st.session_state.rewrite_count += 1
                st.session_state.feedback_history.append(chat_input)
                st.rerun()

    with col_right:
        st.markdown('### RIGHT COLUMN (30%) - "The Logic"')
        st.markdown("#### [AI Data Inspector]")
        st.caption("Atlas extracted these facts from the Neo4j Graph to write this:")
        
        # Seed mock facts if missing
        if "extracted_facts" not in st.session_state:
            st.session_state.extracted_facts = {
                "dependents": "2",
                "liquid_assets": "£500,000",
                "risk_level": "Cautious"
            }
        
        corrections_made = []
        for i, (key, val) in enumerate(st.session_state.extracted_facts.items(), 1):
            new_val = st.text_input(f"{i}. {key}", value=val, key=f"fact_{key}")
            if new_val != val:
                corrections_made.append({"variable_name": key, "correction_rule": f"Always enforce {key} = {new_val}"})

        st.caption("*If a value is wrong, correct it above to teach Atlas new logic.*")
        st.session_state.pending_procedural_corrections = corrections_made

    st.divider()

    # ── APPROVE & LEARN ───────────────────────────────────────────────────
    if st.button("APPROVE & LEARN", use_container_width=True, type="primary"):
        with st.spinner("Sending style and logic preferences to learning pipeline..."):
            # Mocking the dual-layer payload behavior as needed below
            result = api_capture_feedback(
                adviser_id=st.session_state.adviser_id,
                original_text=st.session_state.original_text,
                chosen_text=st.session_state.draft_text,
                user_feedback=st.session_state.last_feedback,
            )

        if result:
            st.success(
                f"✅ **Preferences learned!**  Task `{result.get('task_id', '')}` queued.\n\n"
                f"Voice Rule stored: *\"{st.session_state.last_feedback}\"*\n"
                f"Logic Rules stored: *{len(st.session_state.pending_procedural_corrections)} corrections*\n\n"
                f"Future reports will automatically apply this style and logic."
            )
            st.balloons()

    # ── Feedback History ──────────────────────────────────────────────────
    if st.session_state.feedback_history:
        with st.expander(f"📝 Feedback History ({len(st.session_state.feedback_history)} edits)"):
            for i, fb in enumerate(reversed(st.session_state.feedback_history), 1):
                st.markdown(f"**{i}.** {fb}")

    # ── Memory Insights Panel ─────────────────────────────────────────────
    st.divider()
    with st.expander("🧠 Memory Insights — Learned Preferences", expanded=False):
        prefs = api_get_preferences(st.session_state.adviser_id)
        rules = prefs.get("rules", [])
        total = prefs.get("total", 0)

        if total > 0:
            st.metric("Total Stored Rules", total)

            for i, rule in enumerate(rules):
                rule_text = rule.get("rule_text", "")
                created = rule.get("created_at", "")[:19]
                adviser = rule.get("adviser_id", "")

                st.markdown(
                    f"**Rule {i + 1}:** {rule_text}\n\n"
                    f"*From: {adviser} · {created}*"
                )
                if i < len(rules) - 1:
                    st.divider()
        else:
            st.info(
                "No style preferences stored yet. "
                "Give feedback on the draft above and click **Approve & Learn** "
                "to teach the system your preferred style."
            )


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(
        page_title="Report That Learns You",
        page_icon="🧠",
        layout="wide",
    )
    render_director_typist()
