import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Palo Alto Networks",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] label {
    color: #c9d1d9 !important;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #388bfd; }
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    line-height: 1.1;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #58a6ff;
    border-left: 3px solid #1f6feb;
    padding-left: 12px;
    margin: 24px 0 12px 0;
}

/* Page title */
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 0.85rem;
    color: #8b949e;
    margin-bottom: 24px;
}

/* Alert boxes */
.insight-box {
    background: #0d2137;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #79c0ff;
    margin: 8px 0;
}

/* Divider */
hr {
    border-color: #21262d !important;
    margin: 20px 0;
}

/* Streamlit native elements styling */
[data-testid="stMetricValue"] { color: #58a6ff !important; }
.stSelectbox label, .stMultiSelect label, .stSlider label { color: #c9d1d9 !important; }
.stDataFrame { background: #161b22 !important; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING & FEATURE ENGINEERING ─────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Palo Alto Networks.csv")

    # Add EmployeeID
    df.insert(0, "EmployeeID", range(1001, 1001 + len(df)))

    # Attrition label
    df["Attrition Status"] = df["Attrition"].map({1: "Left", 0: "Active"})

    # Core ratios (clip tenure at 1 to avoid div by zero)
    tenure = df["YearsAtCompany"].clip(lower=1)
    df["Promotion Gap Ratio"]    = (df["YearsSinceLastPromotion"] / tenure).round(3)
    df["Role Stagnation Index"]  = (df["YearsInCurrentRole"]      / tenure).round(3)
    df["Training Intensity"]     = (df["TrainingTimesLastYear"]    / tenure).round(3)

    # Promotion Gap Score
    def gap_score(r):
        if r >= 0.5:  return "High"
        if r >= 0.25: return "Medium"
        return "Low"
    df["Promotion Gap Score"] = df["Promotion Gap Ratio"].apply(gap_score)

    # Training Need
    df["Training Need"] = df["Training Intensity"].apply(
        lambda x: "Needs Training" if x < 0.5 else "Adequate"
    )

    # Manager Stability
    def mgr_stab(row):
        if row["YearsAtCompany"] <= 1: return "New Joiner"
        ratio = row["YearsWithCurrManager"] / max(row["YearsAtCompany"], 1)
        return "Stable" if ratio >= 0.75 else "Frequent Change"
    df["Manager Stability"] = df.apply(mgr_stab, axis=1)

    # Career Cluster
    def career_cluster(row):
        if row["YearsAtCompany"] <= 1:
            return "Early-Career Explorer"
        if row["Promotion Gap Ratio"] >= 0.5 and (
            row["JobSatisfaction"] <= 2 or row["EnvironmentSatisfaction"] <= 2
        ):
            return "High-Risk Stagnation"
        if row["Promotion Gap Ratio"] >= 0.5:
            return "Promotion-Stalled"
        if row["Promotion Gap Ratio"] < 0.25 and row["PerformanceRating"] >= 3:
            return "Fast-Track Performer"
        return "Stable Long-Term Contributor"
    df["Career Cluster"] = df.apply(career_cluster, axis=1)

    # Retention Opportunity
    df["Retention Opportunity"] = (
        (df["Attrition"] == 0) &
        (df["Promotion Gap Score"] == "High") &
        (df["JobSatisfaction"] >= 3)
    ).map({True: "Yes", False: "No"})

    # Suggested Action
    def suggest(row):
        if row["Retention Opportunity"] != "Yes":
            return "No Action Needed"
        if row["Training Need"] == "Needs Training":
            return "Skill Development / Training"
        if row["Role Stagnation Index"] >= 0.8:
            return "Lateral Rotation"
        return "Promotion Review"
    df["Suggested Action"] = df.apply(suggest, axis=1)

    # Age Band
    def age_band(a):
        if a < 30: return "Under 30"
        if a < 40: return "30–39"
        if a < 50: return "40–49"
        return "50+"
    df["Age Band"] = df["Age"].apply(age_band)

    # Tenure Band
    def tenure_band(t):
        if t < 2:  return "0–1 yrs"
        if t < 5:  return "2–4 yrs"
        if t < 10: return "5–9 yrs"
        return "10+ yrs"
    df["Tenure Band"] = df["YearsAtCompany"].apply(tenure_band)

    return df

df = load_data()

# ─── PLOTLY THEME ────────────────────────────────────────────────────────────
PALETTE = ["#58a6ff","#3fb950","#d29922","#f85149","#bc8cff","#39d353"]
CLUSTER_COLORS = {
    "Fast-Track Performer":        "#3fb950",
    "High-Risk Stagnation":        "#f85149",
    "Promotion-Stalled":           "#d29922",
    "Early-Career Explorer":       "#58a6ff",
    "Stable Long-Term Contributor":"#bc8cff",
}
BG   = "#0d1117"
CARD = "#161b22"
GRID = "#21262d"
TEXT = "#e6edf3"
MUTED= "#8b949e"

def dark_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=14, family="Inter"), x=0),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter", size=12),
        height=height,
        margin=dict(l=16, r=16, t=40 if title else 16, b=16),
        legend=dict(bgcolor=CARD, bordercolor=GRID, font=dict(color=TEXT)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=MUTED)),
    )
    return fig

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔵 Palo Alto Networks")
    st.markdown("**HR Career Intelligence Dashboard**")
    st.markdown("---")

    st.markdown("#### 🔽 Filters")

    all_depts   = ["All"] + sorted(df["Department"].unique().tolist())
    all_roles   = ["All"] + sorted(df["JobRole"].unique().tolist())
    all_clusters= ["All"] + sorted(df["Career Cluster"].unique().tolist())
    all_stages  = ["All"] + sorted(df["Tenure Band"].unique().tolist())

    sel_dept    = st.selectbox("Department", all_depts)
    sel_role    = st.selectbox("Job Role", all_roles)
    sel_cluster = st.selectbox("🔵 Cluster Explorer", all_clusters)
    sel_stage   = st.selectbox("Career Stage", all_stages)

    gap_threshold = st.slider(
        "Promotion Gap Threshold",
        min_value=0.0, max_value=1.0,
        value=0.5, step=0.05,
        help="Employees above this Promotion Gap Ratio are flagged as high-gap"
    )

    st.markdown("---")
    st.markdown("#### 📊 Navigation")
    page = st.radio("Go to", [
        "🏠 Executive Overview",
        "🔵 Career Path Clustering",
        "📊 Promotion Gap Monitor",
        "🎯 Retention Opportunity Panel",
        "👥 Managerial Insight Dashboard",
    ])

# ─── FILTER DATA ─────────────────────────────────────────────────────────────
dff = df.copy()
if sel_dept    != "All": dff = dff[dff["Department"]     == sel_dept]
if sel_role    != "All": dff = dff[dff["JobRole"]        == sel_role]
if sel_cluster != "All": dff = dff[dff["Career Cluster"] == sel_cluster]
if sel_stage   != "All": dff = dff[dff["Tenure Band"]    == sel_stage]

# Re-apply threshold dynamically
dff["Gap Score (Dynamic)"] = dff["Promotion Gap Ratio"].apply(
    lambda x: "High" if x >= gap_threshold else ("Medium" if x >= gap_threshold/2 else "Low")
)

# ─── HELPER KPI CARD ─────────────────────────────────────────────────────────
def kpi(value, label):
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Overview":
    st.markdown('<div class="page-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Organisation-wide workforce snapshot — Palo Alto Networks HR Dataset</div>', unsafe_allow_html=True)

    total       = len(dff)
    active      = (dff["Attrition"] == 0).sum()
    left        = (dff["Attrition"] == 1).sum()
    attr_rate   = f"{left/total*100:.1f}%" if total else "—"
    avg_tenure  = f"{dff['YearsAtCompany'].mean():.2f} yrs" if total else "—"
    avg_gap     = f"{dff['Promotion Gap Ratio'].mean():.2f}" if total else "—"

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, val, lbl in zip(
        [c1,c2,c3,c4,c5,c6],
        [total, active, left, attr_rate, avg_tenure, avg_gap],
        ["Total Employees","Active","Attrition Count","Attrition Rate %","Avg Tenure","Avg Promo Gap Ratio"]
    ):
        col.markdown(kpi(val, lbl), unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Attrition Count by Department</div>', unsafe_allow_html=True)
        dept_attr = dff[dff["Attrition"]==1]["Department"].value_counts().reset_index()
        dept_attr.columns = ["Department","Count"]
        fig = px.bar(dept_attr, x="Department", y="Count",
                     color="Department", color_discrete_sequence=PALETTE)
        fig = dark_layout(fig)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Employee Distribution by Department</div>', unsafe_allow_html=True)
        dept_dist = dff["Department"].value_counts().reset_index()
        dept_dist.columns = ["Department","Count"]
        fig = px.pie(dept_dist, names="Department", values="Count",
                     hole=0.55, color_discrete_sequence=PALETTE)
        fig = dark_layout(fig)
        fig.update_traces(textfont_color=TEXT)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Career Cluster Distribution</div>', unsafe_allow_html=True)
        clust = dff["Career Cluster"].value_counts().reset_index()
        clust.columns = ["Cluster","Count"]
        fig = px.bar(clust, x="Count", y="Cluster",
                     color="Cluster",
                     color_discrete_map=CLUSTER_COLORS)
        fig = dark_layout(fig)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Attrition by Attrition Status</div>', unsafe_allow_html=True)
        status = dff["Attrition Status"].value_counts().reset_index()
        status.columns = ["Status","Count"]
        fig = px.pie(status, names="Status", values="Count",
                     hole=0.55, color_discrete_sequence=["#3fb950","#f85149"])
        fig = dark_layout(fig)
        fig.update_traces(textfont_color=TEXT)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CAREER PATH CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Career Path Clustering":
    st.markdown('<div class="page-title">Career Path Clustering Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Rule-based employee segmentation across 5 career trajectory types</div>', unsafe_allow_html=True)

    # KPIs
    total = len(dff)
    cluster_counts = dff["Career Cluster"].value_counts()
    fast  = cluster_counts.get("Fast-Track Performer", 0)
    risk  = cluster_counts.get("High-Risk Stagnation", 0)
    stall = cluster_counts.get("Promotion-Stalled", 0)

    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1,c2,c3,c4],
        [total, f"{fast} ({fast/total*100:.1f}%)" if total else "—",
         f"{risk} ({risk/total*100:.1f}%)" if total else "—",
         f"{stall} ({stall/total*100:.1f}%)" if total else "—"],
        ["Total Employees","Fast-Track Performers","High-Risk Stagnation","Promotion-Stalled"]
    ):
        col.markdown(kpi(val, lbl), unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown('<div class="section-header">Promotion Gap vs Role Stagnation by Cluster</div>', unsafe_allow_html=True)
        fig = px.scatter(
            dff, x="Promotion Gap Ratio", y="Role Stagnation Index",
            color="Career Cluster", color_discrete_map=CLUSTER_COLORS,
            hover_data=["EmployeeID","Department","JobRole","YearsAtCompany"],
            opacity=0.75
        )
        fig = dark_layout(fig, height=420)
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Cluster-wise Employee Share</div>', unsafe_allow_html=True)
        clust_pie = dff["Career Cluster"].value_counts().reset_index()
        clust_pie.columns = ["Cluster","Count"]
        fig = px.pie(clust_pie, names="Cluster", values="Count",
                     hole=0.5, color="Cluster",
                     color_discrete_map=CLUSTER_COLORS)
        fig = dark_layout(fig, height=420)
        fig.update_traces(textfont_color=TEXT, textinfo="percent+label")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Career Cluster Profile Summary</div>', unsafe_allow_html=True)
    profile = dff.groupby("Career Cluster").agg(
        Employees=("EmployeeID","count"),
        Avg_Promo_Gap=("Promotion Gap Ratio","mean"),
        Attrition_Rate=("Attrition","mean"),
        Avg_Performance=("PerformanceRating","mean"),
        Avg_Tenure=("YearsAtCompany","mean")
    ).round(2).reset_index()
    profile["Attrition_Rate"] = (profile["Attrition_Rate"]*100).round(1).astype(str) + "%"
    profile.columns = ["Career Cluster","Employees","Avg Promo Gap Ratio","Attrition Rate %","Avg Performance Rating","Avg Tenure (Yrs)"]
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Career Pattern Summaries</div>', unsafe_allow_html=True)
    summaries = {
        "⚡ Fast-Track Performer":        "Lowest promotion gap (0.06) + lowest attrition (10.7%). Rapid career movement is the dominant retention driver.",
        "🔴 High-Risk Stagnation":        "Highest promotion gap (0.79) combined with low satisfaction. Most urgent segment for immediate HR intervention.",
        "🟡 Promotion-Stalled":           "High promotion gap (0.81) but still satisfied. Proactive career conversation can prevent future disengagement.",
        "🔵 Early-Career Explorer":       "Highest attrition (34.9%) driven by early-tenure churn, not stagnation. Needs structured onboarding programmes.",
        "🟣 Stable Long-Term Contributor":"Moderate gap (0.33), low attrition (11.4%). Monitor to prevent drift into Promotion-Stalled territory.",
    }
    c1, c2 = st.columns(2)
    items = list(summaries.items())
    for i, (name, desc) in enumerate(items):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f'<div class="insight-box"><strong>{name}</strong><br>{desc}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PROMOTION GAP MONITOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Promotion Gap Monitor":
    st.markdown('<div class="page-title">Promotion Gap Monitor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">Threshold: Promotion Gap Ratio ≥ {gap_threshold:.2f} (adjust in sidebar)</div>', unsafe_allow_html=True)

    high_gap   = dff[dff["Promotion Gap Ratio"] >= gap_threshold]
    high_count = len(high_gap)
    high_pct   = f"{high_count/len(dff)*100:.1f}%" if len(dff) else "—"
    avg_promo  = f"{dff['YearsSinceLastPromotion'].mean():.2f} yrs" if len(dff) else "—"

    c1,c2,c3 = st.columns(3)
    for col, val, lbl in zip(
        [c1,c2,c3],
        [high_count, high_pct, avg_promo],
        ["High Promotion Gap Count","% High Promotion Gap","Avg Yrs Since Last Promotion"]
    ):
        col.markdown(kpi(val, lbl), unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Avg Years Since Last Promotion by Job Role</div>', unsafe_allow_html=True)
        role_gap = dff.groupby("JobRole")["YearsSinceLastPromotion"].mean().sort_values(ascending=True).reset_index()
        role_gap.columns = ["Job Role","Avg Years"]
        fig = px.bar(role_gap, x="Avg Years", y="Job Role", orientation="h",
                     color="Avg Years", color_continuous_scale=["#1f6feb","#58a6ff","#f85149"])
        fig = dark_layout(fig)
        fig.update_coloraxes(showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Promotion Gap Score Distribution</div>', unsafe_allow_html=True)
        gap_dist = dff["Promotion Gap Score"].value_counts().reset_index()
        gap_dist.columns = ["Score","Count"]
        color_map = {"High":"#f85149","Medium":"#d29922","Low":"#3fb950"}
        fig = px.bar(gap_dist, x="Score", y="Count",
                     color="Score", color_discrete_map=color_map)
        fig = dark_layout(fig)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Promotion Gap Heatmap: Department × Job Role</div>', unsafe_allow_html=True)
    heat = dff.pivot_table(
        index="JobRole", columns="Department",
        values="Promotion Gap Ratio", aggfunc="mean"
    ).round(3)
    fig = go.Figure(data=go.Heatmap(
        z=heat.values,
        x=heat.columns.tolist(),
        y=heat.index.tolist(),
        colorscale=[[0,"#0d2137"],[0.5,"#1f6feb"],[1,"#f85149"]],
        text=np.round(heat.values, 2),
        texttemplate="%{text}",
        textfont=dict(color=TEXT, size=11),
        hoverongaps=False
    ))
    fig = dark_layout(fig, height=320)
    fig.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">High Promotion Gap Employees</div>', unsafe_allow_html=True)
    high_tbl = high_gap[[
        "EmployeeID","Department","JobRole",
        "YearsAtCompany","YearsSinceLastPromotion",
        "Promotion Gap Ratio","PerformanceRating","Attrition Status"
    ]].sort_values("Promotion Gap Ratio", ascending=False).reset_index(drop=True)
    high_tbl.columns = ["Emp ID","Department","Job Role","Tenure","Yrs Since Promo","Gap Ratio","Perf Rating","Status"]
    st.dataframe(high_tbl, use_container_width=True, hide_index=True, height=300)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RETENTION OPPORTUNITY PANEL
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Retention Opportunity Panel":
    st.markdown('<div class="page-title">Retention Opportunity Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Active, satisfied employees with high promotion gaps — intervention required before disengagement</div>', unsafe_allow_html=True)

    flagged = dff[dff["Retention Opportunity"] == "Yes"]
    active  = dff[dff["Attrition"] == 0]
    ret_count = len(flagged)
    ret_rate  = f"{ret_count/len(active)*100:.1f}%" if len(active) else "—"

    c1,c2,c3 = st.columns(3)
    for col, val, lbl in zip(
        [c1,c2,c3],
        [ret_count, ret_rate, f"{flagged['YearsSinceLastPromotion'].mean():.1f} yrs" if ret_count else "—"],
        ["Retention Opportunity Count","Retention Opportunity Rate %","Avg Yrs Since Last Promo (Flagged)"]
    ):
        col.markdown(kpi(val, lbl), unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Retention Opportunity by Department</div>', unsafe_allow_html=True)
        dep_ret = flagged["Department"].value_counts().reset_index()
        dep_ret.columns = ["Department","Count"]
        fig = px.bar(dep_ret, x="Department", y="Count",
                     color="Department", color_discrete_sequence=PALETTE)
        fig = dark_layout(fig)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Suggested Action Breakdown</div>', unsafe_allow_html=True)
        action_dist = flagged["Suggested Action"].value_counts().reset_index()
        action_dist.columns = ["Action","Count"]
        action_colors = {
            "Skill Development / Training":"#58a6ff",
            "Lateral Rotation":"#3fb950",
            "Promotion Review":"#d29922"
        }
        fig = px.pie(action_dist, names="Action", values="Count",
                     hole=0.55, color="Action", color_discrete_map=action_colors)
        fig = dark_layout(fig)
        fig.update_traces(textfont_color=TEXT, textinfo="percent+label")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Retention Opportunity by Job Role</div>', unsafe_allow_html=True)
    role_ret = flagged["JobRole"].value_counts().reset_index()
    role_ret.columns = ["Job Role","Count"]
    fig = px.bar(role_ret, x="Count", y="Job Role", orientation="h",
                 color="Job Role", color_discrete_sequence=PALETTE)
    fig = dark_layout(fig, height=280)
    fig.update_traces(marker_line_width=0)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Employees Flagged for Career Intervention</div>', unsafe_allow_html=True)
    flagged_tbl = flagged[[
        "EmployeeID","Department","JobRole",
        "YearsSinceLastPromotion","JobSatisfaction",
        "PerformanceRating","Suggested Action"
    ]].sort_values("YearsSinceLastPromotion", ascending=False).reset_index(drop=True)
    flagged_tbl.columns = ["Emp ID","Department","Job Role","Yrs Since Promo","Job Satisfaction","Perf Rating","Suggested Action"]
    st.dataframe(flagged_tbl, use_container_width=True, hide_index=True, height=320)

    st.markdown('<div class="section-header">Key Insight</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-box">
    💡 <strong>{ret_count} employees</strong> are currently active and satisfied (Job Satisfaction ≥ 3)
    yet have been waiting an average of <strong>{flagged['YearsSinceLastPromotion'].mean():.1f} years</strong> for a promotion.
    Standard attrition models would not flag these employees — this panel surfaces them before disengagement occurs.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MANAGERIAL INSIGHT DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif page == "👥 Managerial Insight Dashboard":
    st.markdown('<div class="page-title">Managerial Insight Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Manager tenure vs career growth and team-level stagnation signals</div>', unsafe_allow_html=True)

    avg_mgr     = f"{dff['YearsWithCurrManager'].mean():.2f} yrs" if len(dff) else "—"
    stable_pct  = f"{(dff['Manager Stability']=='Stable').mean()*100:.1f}%" if len(dff) else "—"
    avg_stag    = f"{dff['Role Stagnation Index'].mean():.2f}" if len(dff) else "—"

    c1,c2,c3 = st.columns(3)
    for col, val, lbl in zip(
        [c1,c2,c3],
        [avg_mgr, stable_pct, avg_stag],
        ["Avg Yrs With Manager","Stable Manager %","Avg Role Stagnation Index"]
    ):
        col.markdown(kpi(val, lbl), unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Manager Stability Distribution</div>', unsafe_allow_html=True)
        mgr_dist = dff["Manager Stability"].value_counts().reset_index()
        mgr_dist.columns = ["Stability","Count"]
        stability_colors = {"Stable":"#3fb950","Frequent Change":"#f85149","New Joiner":"#58a6ff"}
        fig = px.bar(mgr_dist, x="Stability", y="Count",
                     color="Stability", color_discrete_map=stability_colors)
        fig = dark_layout(fig)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Role Stagnation by Department & Manager Stability</div>', unsafe_allow_html=True)
        stag_grp = dff.groupby(["Department","Manager Stability"])["Role Stagnation Index"].mean().round(3).reset_index()
        fig = px.bar(stag_grp, x="Department", y="Role Stagnation Index",
                     color="Manager Stability", barmode="group",
                     color_discrete_map=stability_colors,
                     text="Role Stagnation Index")
        fig = dark_layout(fig)
        fig.update_traces(marker_line_width=0, textfont=dict(color=TEXT), textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Team-Level Stability Overview</div>', unsafe_allow_html=True)
    team_matrix = pd.crosstab(dff["Department"], dff["Manager Stability"])
    team_matrix = team_matrix.reset_index()
    st.dataframe(team_matrix, use_container_width=True, hide_index=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Manager Tenure vs Avg Promotion Gap (by Department)</div>', unsafe_allow_html=True)
        mgr_gap = dff.groupby("Department").agg(
            Avg_Mgr_Tenure=("YearsWithCurrManager","mean"),
            Avg_Promo_Gap=("Promotion Gap Ratio","mean")
        ).round(3).reset_index()
        fig = px.scatter(mgr_gap, x="Avg_Mgr_Tenure", y="Avg_Promo_Gap",
                         color="Department", size=[30]*len(mgr_gap),
                         color_discrete_sequence=PALETTE,
                         hover_data=["Department"],
                         labels={"Avg_Mgr_Tenure":"Avg Yrs With Manager","Avg_Promo_Gap":"Avg Promotion Gap Ratio"})
        fig = dark_layout(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Attrition Rate by Manager Stability</div>', unsafe_allow_html=True)
        attr_stab = dff.groupby("Manager Stability")["Attrition"].mean().reset_index()
        attr_stab["Attrition %"] = (attr_stab["Attrition"]*100).round(1)
        fig = px.bar(attr_stab, x="Manager Stability", y="Attrition %",
                     color="Manager Stability", color_discrete_map=stability_colors,
                     text="Attrition %")
        fig = dark_layout(fig, height=300)
        fig.update_traces(marker_line_width=0, texttemplate="%{text}%", textposition="outside", textfont=dict(color=TEXT))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    insights = [
        "📌 Stable manager relationships show <strong>higher</strong> Role Stagnation Index than Frequent Change — counter-intuitive, but suggests long-term stable relationships correlate with longer time in the same role.",
        "📌 New Joiners have the highest attrition rate (34.9%), confirming early-tenure churn — not managerial instability — is the primary attrition driver.",
        "📌 Only 36.39% of the workforce has a Stable manager relationship, making consistent managerial continuity a minority experience at Palo Alto Networks.",
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#8b949e; font-size:0.75rem;">'
    'Career Progression & Promotion Gap Analysis — Palo Alto Networks &nbsp;|&nbsp; '
    'Unified Mentor Data Analytics Program &nbsp;|&nbsp; Pawan Kumar Lakhera'
    '</div>',
    unsafe_allow_html=True
)
