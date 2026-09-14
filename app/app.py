import os
import time
import requests
import urllib.parse
import streamlit as st

# --------------------------------------------------------------------
# Global Layout & Platform Configuration
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal DAM Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")

def parse_url_state():
    """Parses state from browser URL parameters to persist state on refresh."""
    params = st.query_params
    if "token" in params:
        st.session_state["token"] = params["token"]
        st.session_state["username"] = params.get("user", "Guest")
        st.session_state["role"] = params.get("role", "viewer")
        st.session_state["authenticated"] = True

def sync_url_state():
    """Flushes active session parameters to browser query parameters."""
    if st.session_state.get("authenticated"):
        st.query_params["token"] = st.session_state["token"]
        st.query_params["user"] = st.session_state["username"]
        st.query_params["role"] = st.session_state["role"]
    else:
        st.query_params.clear()

# Initialize session parameters
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    parse_url_state()

# --------------------------------------------------------------------
# Authentication View Gateway Panel
# --------------------------------------------------------------------
if not st.session_state["authenticated"]:
    st.title("🔒 Enterprise Multimodal DAM Portal")
    st.markdown("Please authenticate with your credentials to access the asset management space.")

    with st.form("Login Form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Authenticate")

        if submitted:
            try:
                res = requests.post(
                    f"{API_GATEWAY_URL}/api/token",
                    json={"username": username, "password": password},
                    timeout=10
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["token"] = data["access_token"]
                    st.session_state["role"] = data["role"]
                    st.session_state["username"] = username
                    st.session_state["authenticated"] = True
                    sync_url_state()
                    st.rerun()
                else:
                    st.error(f"Authentication rejected: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Gateway interface unreachable: {str(e)}")
    st.stop()

# Authorization Header Mapping
auth_headers = {"Authorization": f"Bearer {st.session_state.get('token')}"}
token_query_str = f"?token={urllib.parse.quote(st.session_state.get('token', ''))}"


# --------------------------------------------------------------------
# Sidebar Observability Engine & System Health Monitoring
# --------------------------------------------------------------------

with st.sidebar:
    st.title("🖥️ System Monitor")
    st.markdown(f"**Connected As:** `{st.session_state['username']}`")
    st.markdown(f"**Clearance Profile:** `{st.session_state['role'].upper()}`")
    st.divider()

    st.subheader("Infrastructure Health")
    try:
        health_check = requests.get(f"{API_GATEWAY_URL}/metrics", timeout=3)
        if health_check.status_code == 200:
            st.success("API Gateway Layer: ONLINE")
        else:
            st.warning("API Gateway Layer: UNSTABLE")
    except Exception:
        st.error("API Gateway Layer: UNREACHABLE")
    st.divider()

    if st.button("Terminate Session Profile", use_container_width=True):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()

    # Administrative Scope Sweeper
    if st.session_state["role"] == "admin":
        st.divider()
        st.subheader("🛠️ Administrative Controls")
        if st.button("Wipe & Reset Storage Spaces", type="primary", use_container_width=True):
            with st.spinner("Scrubbing namespaces across core storage nodes..."):
                res = requests.post(f"{API_GATEWAY_URL}/api/init-db", headers=auth_headers, timeout=60)
                if res.status_code == 200:
                    st.success("All data namespaces successfully reset.")
                    st.balloons()
                else:
                    st.error("Failed to run execution database reset step.")


# --------------------------------------------------------------------
# Dashboard Layout Interface View Tabs
# --------------------------------------------------------------------
st.title("📂 Multimodal Asset Management Space")
tab_upload, tab_search, tab_rag_summary = st.tabs([
    "📥 Streaming Ingestion",
    "🔍 Cross-Modal Search",
    "🧠 RAG & Summarization Reports"
])

# --- Tab 1: Streaming Ingestion Layout Layer ---
with tab_upload:
    st.header("Upload Multimedia Assets")
    st.caption("Supported formats: PDFs, Images, Audio, and Video files.")

    uploaded_file = st.file_uploader(
        "Select asset file",
        type=["pdf", "png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv", "mp3", "wav"]
    )

    if uploaded_file is not None:
        if st.button("Deploy Pipeline Processing Worker", use_container_width=True):
            try:
                with st.spinner("Streaming assets to infrastructure node boundaries..."):
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    res = requests.post(
                        f"{API_GATEWAY_URL}/api/upload",
                        files=files,
                        headers=auth_headers,
                        timeout=30
                    )

                if res.status_code == 202:
                    task_data = res.json()
                    task_id = task_data["task_id"]
                    st.info(f"Asset accepted. Pipeline worker assigned Task ID: `{task_id}`")

                    # Live Task Status Tracking Poller Loop
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    while True:
                        response = requests.get(f"{API_GATEWAY_URL}/api/tasks/{task_id}", headers=auth_headers, timeout=5)
                        if response.status_code == 200:
                            task_res = response.json()
                            state = task_res.get("state")
                            progress = task_res.get("progress") or 0

                            progress_bar.progress(int(progress))
                            status_text.text(f"Current Worker Ingestion Pipeline Status: {state} ({progress}%)")

                            if state == "SUCCESS":
                                st.success("Background ingestion asset parsing loops executed perfectly.")
                                st.balloons()
                                break
                            elif state == "FAILURE":
                                error_msg = task_res.get("result") or task_res.get("error") or "Internal Ingestion Exception"
                                st.error(f"Pipeline crashed during asset analysis: {error_msg}")
                                break
                        else:
                            st.error(f"Task manager endpoint returned error code: {response.status_code}")
                            break

                        time.sleep(2)
                else:
                    st.error(f"Asset rejected by target gateway: {res.text}")
            except Exception as e:
                st.error(f"Ingestion scheduling loop failure: {str(e)}")


# --- Tab 2: Parallel Dual-Vector Semantic Search Interface ---
with tab_search:
    st.header("Cross-Modal Semantic Discovery Hub")
    search_query = st.text_input("Enter natural concept or keyword description")
    limit_count = st.slider("Max return limits", min_value=1, max_value=20, value=5)

    if search_query:
        try:
            res = requests.get(
                f"{API_GATEWAY_URL}/api/search",
                params={"query": search_query, "limit": limit_count},
                headers=auth_headers,
                timeout=15,
            )
            if res.status_code == 200:
                data = res.json()
                text_matches = data.get("text_matches", [])
                visual_matches = data.get("visual_matches", [])
                col_text, col_visual = st.columns(2)

                with col_text:
                    st.subheader("📝 Context Blocks & Document Segments")
                    if not text_matches:
                        st.caption("No matching text segments found.")
                    for match in text_matches:
                        with st.container(border=True):
                            st.markdown(f"**Source File:** `{match['filename']}`")
                            st.markdown(f"**Page Index / Timestamp:** `{match['timestamp']}`")
                            st.info(f"\"{match['text']}\"")
                            st.caption(f"Relevance Score: {round(match['score'], 4)}")

                with col_visual:
                    st.subheader("🖼️ Extracted Visual Matrices & Keyframes")
                    if not visual_matches:
                        st.caption("No matching visual assets found.")
                    for match in visual_matches:
                        with st.container(border=True):
                            st.markdown(f"**Visual Asset File:** `{match['filename']}`")
                            st.markdown(f"**Timestamp Offset:** {match['timestamp']}s")

                            media_link_res = requests.get(
                                f"{API_GATEWAY_URL}/api/media/{match['filename']}",
                                headers=auth_headers,
                                timeout=5,
                            )
                            if media_link_res.status_code == 200:
                                url = media_link_res.json().get("url")
                                if match["filename"].lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                                    st.video(url, start_time=int(match["timestamp"]))
                                else:
                                    st.image(url, use_column_width=True)
                                st.caption(f"Visual Relevance Score: {round(match['score'], 4)}")
            else:
                st.error("Failed to query vector database models cleanly.")
        except Exception as e:
            st.error(f"Search endpoint failed: {str(e)}")

# --- Tab 3: Grounded Agentic RAG & Automation Analytics Report Generator ---
with tab_rag_summary:
    st.header("Generative RAG Analytics & Summarization Engines")
    st.subheader("1. Grounded Media Cross-Examination (RAG)")
    rag_query = st.text_input("Ask a question across all ingested asset content boundaries")

    if rag_query:
        with st.spinner("Assembling context and generating responses..."):
            try:
                res = requests.get(
                    f"{API_GATEWAY_URL}/api/rag",
                    params={"query": rag_query},
                    headers=auth_headers,
                    timeout=120,
                )
                if res.status_code == 200:
                    data = res.json()
                    st.markdown("### Answer Insight Summary")
                    st.write(data.get("answer"))
                    with st.expander("Grounding Context Traces"):
                        for block in data.get("grounding_context", []):
                            st.caption(block)
                else:
                    st.error("RAG pipeline failed to parse context layers.")
            except Exception as e:
                st.error(f"LLM service execution timed out or failed: {str(e)}")

    st.divider()
    st.subheader("2. Automated Narrative Executive Document Summarizer")
    try:
        assets_res = requests.get(f"{API_GATEWAY_URL}/api/assets", headers=auth_headers, timeout=5)
        if assets_res.status_code == 200:
            asset_options = [a["filename"] for a in assets_res.json().get("assets", [])]
            selected_asset = st.selectbox(
                "Choose an asset to summarize",
                options=asset_options,
                key="summary_asset_selector_widget",
            )

            if "saved_markdown_report" not in st.session_state:
                st.session_state["saved_markdown_report"] = None
            if "tracked_filename" not in st.session_state:
                st.session_state["tracked_filename"] = selected_asset

            if selected_asset != st.session_state["tracked_filename"]:
                st.session_state["saved_markdown_report"] = None
                st.session_state["tracked_filename"] = selected_asset

            if st.button("Compile Executive Summary Report", use_container_width=True):
                with st.spinner("Aggregating chunks and computing layouts via inference engine..."):
                    sum_res = requests.get(
                        f"{API_GATEWAY_URL}/api/summarize",
                        params={"filename": selected_asset},
                        headers=auth_headers,
                        timeout=120,
                    )
                    if sum_res.status_code == 200:
                        st.session_state["saved_markdown_report"] = sum_res.json().get("summary")
                    else:
                        error_detail = sum_res.json().get('detail', 'Unknown error infrastructure message')
                        st.error(f"⚠️ Unable to summarize asset: {error_detail}")

            if st.session_state["saved_markdown_report"] is not None:
                st.markdown("### Generated Summary Document")
                st.markdown(st.session_state["saved_markdown_report"])
                st.download_button(
                    label="Export Report to Markdown (.md)",
                    data=st.session_state["saved_markdown_report"],
                    file_name=f"Executive_Summary_{selected_asset}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="permanent_md_export_trigger",
                )
        else:
            st.caption("No assets available for summary compilation.")
    except Exception as e:
        st.error(f"Asset lookup execution failed: {str(e)}")
