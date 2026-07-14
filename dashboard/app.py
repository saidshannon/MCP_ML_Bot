"""
Streamlit observability dashboard for the Autonomous ML Operations Agent Platform.
Reads from: runs, agent_calls tables in PostgreSQL.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Ops Agent Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
    }

    /* Hide default streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Page header */
    .page-header {
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #1e2530;
        margin-bottom: 2rem;
    }

    .page-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .page-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin: 0.25rem 0 0 0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Metric cards */
    .metric-card {
        background: #111318;
        border: 1px solid #1e2530;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }

    .metric-delta {
        font-size: 0.75rem;
        color: #22c55e;
        margin-top: 0.4rem;
    }

    .metric-delta.negative {
        color: #ef4444;
    }

    /* Section headers */
    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e2530;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
    }

    .badge-success {
        background: #052e16;
        color: #22c55e;
        border: 1px solid #166534;
    }

    .badge-failed {
        background: #2d0a0a;
        color: #ef4444;
        border: 1px solid #7f1d1d;
    }

    .badge-started {
        background: #0c1a2e;
        color: #60a5fa;
        border: 1px solid #1e3a5f;
    }

    /* Run row */
    .run-row {
        background: #111318;
        border: 1px solid #1e2530;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.15s;
    }

    .run-row:hover {
        border-color: #334155;
    }

    .run-task {
        font-size: 0.9rem;
        font-weight: 500;
        color: #e2e8f0;
        margin-bottom: 0.4rem;
    }

    .run-meta {
        font-size: 0.75rem;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Agent call row */
    .agent-row {
        background: #0d0f12;
        border-left: 3px solid #1e2530;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 6px 6px 0;
    }

    .agent-row.success {
        border-left-color: #22c55e;
    }

    .agent-row.failed {
        border-left-color: #ef4444;
    }

    .agent-name {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 0.25rem;
    }

    .agent-detail {
        font-size: 0.75rem;
        color: #475569;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #334155;
    }

    .empty-state-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }

    .empty-state-text {
        font-size: 0.85rem;
        color: #475569;
    }

    /* Report viewer */
    .report-container {
        background: #111318;
        border: 1px solid #1e2530;
        border-radius: 8px;
        padding: 1.5rem;
        font-size: 0.85rem;
        line-height: 1.7;
        color: #cbd5e1;
        max-height: 400px;
        overflow-y: auto;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid #1e2530;
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.6rem 1rem;
        border-radius: 0;
    }

    .stTabs [aria-selected="true"] {
        background: transparent;
        color: #f1f5f9;
        border-bottom: 2px solid #818cf8;
    }

    /* Dataframe overrides */
    .stDataFrame {
        border: 1px solid #1e2530;
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.4rem 1rem;
        transition: all 0.15s;
    }

    .stButton > button:hover {
        background: #334155;
        border-color: #475569;
        color: #f1f5f9;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #0d0f12; }
    ::-webkit-scrollbar-thumb { background: #1e2530; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Database helpers ───────────────────────────────────────────────────────────

def get_connection():
    """Get a psycopg connection using environment DSN."""
    import psycopg
    dsn = os.getenv("DATABASE_URL") or os.getenv("PGDSN")
    if not dsn:
        raise ValueError("DATABASE_URL environment variable not set.")
    return psycopg.connect(dsn)


@st.cache_data(ttl=10)
def fetch_runs() -> pd.DataFrame:
    """Fetch all pipeline runs ordered by most recent first."""
    try:
        with get_connection() as conn:
            df = pd.read_sql("""
                SELECT
                    run_id,
                    task_description,
                    status,
                    started_at,
                    finished_at,
                    EXTRACT(EPOCH FROM (finished_at - started_at))::int AS duration_seconds
                FROM runs
                ORDER BY started_at DESC
                LIMIT 100
            """, conn)
        return df
    except Exception as e:
        st.error(f"Could not load runs: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=10)
def fetch_agent_calls(run_id: Optional[int] = None) -> pd.DataFrame:
    """Fetch agent calls, optionally filtered by run."""
    try:
        with get_connection() as conn:
            query = """
                SELECT
                    call_id,
                    run_id,
                    agent_name,
                    duration_ms,
                    success,
                    error_message,
                    timestamp,
                    input,
                    output
                FROM agent_calls
                {where}
                ORDER BY timestamp ASC
            """.format(
                where=f"WHERE run_id = {run_id}" if run_id else ""
            )
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Could not load agent calls: {e}")
        return pd.DataFrame()


# ── Helpers ────────────────────────────────────────────────────────────────────

def status_badge(status: str) -> str:
    cls = {
        "completed": "badge-success",
        "failed": "badge-failed",
        "started": "badge-started",
    }.get(status, "badge-started")
    return f'<span class="badge {cls}">{status.upper()}</span>'


def format_duration(seconds) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"


def format_ts(ts) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return ts.strftime("%d %b %Y  %H:%M:%S")


def safe_json(val) -> dict:
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except Exception:
        return {"raw": str(val)}


# ── Summary metrics ────────────────────────────────────────────────────────────

def render_metrics(runs: pd.DataFrame, agent_calls: pd.DataFrame):
    total = len(runs)
    completed = len(runs[runs["status"] == "completed"]) if total else 0
    failed = len(runs[runs["status"] == "failed"]) if total else 0
    success_rate = round((completed / total) * 100) if total else 0

    avg_duration = runs["duration_seconds"].dropna().mean() if total else None

    total_agent_calls = len(agent_calls)
    agent_success_rate = (
        round((agent_calls["success"].sum() / total_agent_calls) * 100)
        if total_agent_calls else 0
    )

    cols = st.columns(4)

    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Runs</div>
            <div class="metric-value">{total}</div>
            <div class="metric-delta">{completed} completed · {failed} failed</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        delta_class = "negative" if success_rate < 70 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pipeline Success Rate</div>
            <div class="metric-value">{success_rate}%</div>
            <div class="metric-delta {delta_class}">across all runs</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Run Duration</div>
            <div class="metric-value">{format_duration(avg_duration)}</div>
            <div class="metric-delta">end-to-end pipeline</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        delta_class = "negative" if agent_success_rate < 80 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Agent Call Success</div>
            <div class="metric-value">{agent_success_rate}%</div>
            <div class="metric-delta {delta_class}">{total_agent_calls} total calls</div>
        </div>
        """, unsafe_allow_html=True)


# ── Run history ────────────────────────────────────────────────────────────────

def render_run_history(runs: pd.DataFrame) -> Optional[int]:
    """Renders run list. Returns selected run_id if user clicks one."""
    st.markdown('<div class="section-header">Run History</div>', unsafe_allow_html=True)

    if runs.empty:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">⚡</div>
            <div class="empty-state-text">No runs yet. Start the pipeline to see results here.</div>
        </div>
        """, unsafe_allow_html=True)
        return None

    selected_run_id = None

    for _, row in runs.iterrows():
        duration_str = format_duration(row.get("duration_seconds"))
        ts_str = format_ts(row.get("started_at"))
        badge = status_badge(row["status"])
        task = row["task_description"][:80] + ("..." if len(str(row["task_description"])) > 80 else "")

        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"""
            <div class="run-row">
                <div class="run-task">{task}</div>
                <div class="run-meta">
                    run #{row['run_id']} &nbsp;·&nbsp; {ts_str} &nbsp;·&nbsp; {duration_str}
                    &nbsp;&nbsp; {badge}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("Inspect", key=f"inspect_{row['run_id']}"):
                selected_run_id = int(row["run_id"])

    return selected_run_id


# ── Run detail ─────────────────────────────────────────────────────────────────

def render_run_detail(run_id: int, runs: pd.DataFrame):
    run = runs[runs["run_id"] == run_id]
    if run.empty:
        st.warning("Run not found.")
        return

    row = run.iloc[0]
    agent_calls = fetch_agent_calls(run_id)

    st.markdown(f'<div class="section-header">Run #{run_id} — Detail</div>', unsafe_allow_html=True)

    # Run summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", row["status"].upper())
    c2.metric("Duration", format_duration(row.get("duration_seconds")))
    c3.metric("Agent Calls", len(agent_calls))
    c4.metric("Started", format_ts(row.get("started_at")))

    tabs = st.tabs(["Agent Calls", "Report"])

    # ── Tab 1: Agent calls ──
    with tabs[0]:
        if agent_calls.empty:
            st.markdown('<div class="empty-state-text">No agent calls recorded for this run.</div>',
                        unsafe_allow_html=True)
        else:
            for _, call in agent_calls.iterrows():
                success = call["success"]
                row_class = "success" if success else "failed"
                icon = "✓" if success else "✗"
                duration = f"{call['duration_ms']}ms" if call["duration_ms"] else "—"
                error = f" · {call['error_message']}" if not success and call.get("error_message") else ""

                st.markdown(f"""
                <div class="agent-row {row_class}">
                    <div class="agent-name">{icon} &nbsp; {call['agent_name']}</div>
                    <div class="agent-detail">{format_ts(call['timestamp'])} · {duration}{error}</div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"Input / Output — {call['agent_name']}", expanded=False):
                    col_in, col_out = st.columns(2)
                    with col_in:
                        st.markdown("**Input**")
                        st.json(safe_json(call.get("input")))
                    with col_out:
                        st.markdown("**Output**")
                        st.json(safe_json(call.get("output")))

    # ── Tab 2: Report ──
    with tabs[1]:
        # Try to get report from agent_calls output of report_writer
        report_text = None
        if not agent_calls.empty:
            report_rows = agent_calls[agent_calls["agent_name"] == "report_writer"]
            if not report_rows.empty:
                output = safe_json(report_rows.iloc[0].get("output"))
                report_text = output.get("report") or output.get("raw_response")

        if report_text:
            st.markdown(report_text)
        else:
            st.markdown('<div class="empty-state-text">No report generated for this run.</div>',
                        unsafe_allow_html=True)


# ── Agent performance chart ────────────────────────────────────────────────────

def render_agent_performance(agent_calls: pd.DataFrame):
    st.markdown('<div class="section-header">Agent Performance</div>', unsafe_allow_html=True)

    if agent_calls.empty:
        st.markdown('<div class="empty-state-text">No agent calls recorded yet.</div>',
                    unsafe_allow_html=True)
        return

    # Success rate per agent
    summary = agent_calls.groupby("agent_name").agg(
        total=("call_id", "count"),
        successes=("success", "sum"),
        avg_duration_ms=("duration_ms", "mean"),
    ).reset_index()
    summary["success_rate"] = (summary["successes"] / summary["total"] * 100).round(1)
    summary["avg_duration_ms"] = summary["avg_duration_ms"].round(0).astype("Int64")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Success rate by agent**")
        chart_data = summary.set_index("agent_name")["success_rate"]
        st.bar_chart(chart_data, color="#818cf8")

    with col2:
        st.markdown("**Avg latency by agent (ms)**")
        latency_data = summary.set_index("agent_name")["avg_duration_ms"].dropna()
        st.bar_chart(latency_data, color="#22c55e")

    # Failure breakdown
    failures = agent_calls[agent_calls["success"] == False]
    if not failures.empty:
        st.markdown("**Failure breakdown**")
        failure_counts = failures.groupby("agent_name").size().reset_index(name="failures")
        st.dataframe(
            failure_counts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "agent_name": st.column_config.TextColumn("Agent"),
                "failures": st.column_config.NumberColumn("Failed Calls"),
            }
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:

    # Header
    st.markdown("""
    <div class="page-header">
        <p class="page-title">⚡ ML Ops Agent Platform</p>
        <p class="page-subtitle">autonomous · multi-agent · observable</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    runs = fetch_runs()
    all_agent_calls = fetch_agent_calls()

    # Top metrics
    render_metrics(runs, all_agent_calls)

    # Layout: run list on left, detail on right
    st.markdown('<div class="section-header">Pipeline Runs</div>', unsafe_allow_html=True)

    # Session state for selected run
    if "selected_run_id" not in st.session_state:
        st.session_state.selected_run_id = None

    col_list, col_detail = st.columns([2, 3])

    with col_list:
        # Refresh button
        if st.button("↻  Refresh"):
            st.cache_data.clear()
            st.rerun()

        selected = render_run_history(runs)
        if selected:
            st.session_state.selected_run_id = selected

    with col_detail:
        if st.session_state.selected_run_id:
            render_run_detail(st.session_state.selected_run_id, runs)
        else:
            st.markdown("""
            <div class="empty-state" style="margin-top: 4rem;">
                <div class="empty-state-icon">←</div>
                <div class="empty-state-text">Select a run to inspect its agent calls and report.</div>
            </div>
            """, unsafe_allow_html=True)

    # Agent performance section below
    st.markdown("---")
    render_agent_performance(all_agent_calls)


if __name__ == "__main__":
    main()