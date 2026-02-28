import re

with open("/Users/ledjke/Desktop/Challenge/frontend/app.py", "r") as f:
    content = f.read()

# Replace the dispatch in render_template_engine
old_dispatch = '''    elif mode == "🧠 Review & Learn":
        from report_ui import render_director_typist
        render_director_typist(client)'''

new_dispatch = '''    elif mode == "🧠 Review & Learn":
        render_director_typist_ui(client)'''

content = content.replace(old_dispatch, new_dispatch)

# Add our new render_director_typist_ui function above render_template_engine
new_ui_func = '''def render_director_typist_ui(client: TemplateAPIClient) -> None:
    """Render the Report Review & Learn interface with Director-Typist UI."""
    st.subheader("🧠 Report Review & Style Learning (Director-Typist)")
    
    # Session state init
    if "dt_report_text" not in st.session_state:
        st.session_state.dt_report_text = """Dear Mr. and Mrs. Henderson,

Following our recent meeting on 15th January 2025, I am pleased to present your Annual Portfolio Review for the period ending 31st December 2024.

Portfolio Performance Summary:
Your portfolio has achieved a total return of 8.2% over the review period, compared to the benchmark return of 7.1%. The portfolio value currently stands at £485,000, representing a net increase of £36,770.

Risk Assessment:
Based on your completed risk questionnaire (score: 6/10), we classify your risk profile as "Balanced Growth". This remains appropriate given your stated investment horizon of 15+ years and your objective of funding retirement at age 65.
"""
    if "dt_ai_variables" not in st.session_state:
        st.session_state.dt_ai_variables = {
            "Dependents": "0",
            "Total_Assets": "£485,000",
            "Risk_Score": "6/10",
            "Horizon": "15+ years"
        }
    if "dt_user_variables" not in st.session_state:
        st.session_state.dt_user_variables = st.session_state.dt_ai_variables.copy()
    if "dt_stylistic_feedback" not in st.session_state:
        st.session_state.dt_stylistic_feedback = ""

    col_left, col_right = st.columns([7, 3])

    with col_left:
        st.markdown("### 📄 The Report")
        st.markdown(
            f'<div style="background:#1e1e2e; padding:20px; border-radius:10px; '
            f'border:1px solid #444; font-size:14px; line-height:1.7; '
            f'white-space:pre-wrap; color:#e0e0e0; margin-bottom:20px;">'
            f'{st.session_state.dt_report_text}'
            f'</div>',
            unsafe_allow_html=True,
        )
        
        feedback = st.chat_input("Adjust the style/tone of this draft...")
        if feedback:
            st.session_state.dt_stylistic_feedback = feedback
            st.rerun()
            
        if st.session_state.dt_stylistic_feedback:
            st.info(f"**Instructions:** {st.session_state.dt_stylistic_feedback}")
            
    with col_right:
        st.markdown("### 🔍 Data Inspector")
        st.caption("Extracted from Atlas/Neo4j")
        
        procedural_corrections = {}
        for key, ai_val in st.session_state.dt_ai_variables.items():
            user_val = st.text_input(
                key, 
                value=st.session_state.dt_user_variables.get(key, ai_val), 
                key=f"inspector_{key}"
            )
            if user_val != st.session_state.dt_user_variables.get(key):
                st.session_state.dt_user_variables[key] = user_val
            
            if st.session_state.dt_user_variables[key] != ai_val:
                procedural_corrections[key] = {
                    "original": ai_val, 
                    "corrected": st.session_state.dt_user_variables[key]
                }
                st.warning(f"Modified: {ai_val} ➡️ {st.session_state.dt_user_variables[key]}")
        
        st.divider()
        
        if st.button("🧠 Approve & Learn", type="primary", use_container_width=True):
            payload = {
                "stylistic_feedback": st.session_state.dt_stylistic_feedback,
                "procedural_corrections": procedural_corrections
            }
            
            try:
                # Wrap inside a spinner
                with st.spinner("Updating Atlas..."):
                    response = httpx.post(
                        f"{client.base_url}/api/feedback/capture",
                        json=payload,
                        timeout=10.0,
                        headers=client.headers
                    )
                    response.raise_for_status()
                st.success("Atlas has updated its memory.")
                st.toast("Atlas has updated its memory.")
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Feedback capture error: {e}")
                st.error("Engine offline. Could not connect to Atlas.")

def render_template_engine(client: TemplateAPIClient) -> None:'''

content = content.replace("def render_template_engine(client: TemplateAPIClient) -> None:", new_ui_func)

with open("/Users/ledjke/Desktop/Challenge/frontend/app.py", "w") as f:
    f.write(content)
