"""
FloodSense AI — Dark Frontend
================================
Streamlit app connected to RiskEngine, AllocationEngine,
and postprocess utilities.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_engine import RiskEngine
from core.allocation_engine import AllocationEngine
from utils.postprocess import refine_mask, compute_flood_statistics, extract_flood_zones

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FloodSense AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  DARK THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base dark background ── */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}

[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d !important;
}

[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* ── Header ── */
.fs-header {
    padding: 2rem 0 1rem 0;
    text-align: center;
}
.fs-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #58a6ff, #3fb950, #58a6ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 3s linear infinite;
    letter-spacing: -1px;
}
@keyframes shine {
    to { background-position: 200% center; }
}
.fs-subtitle {
    color: #8b949e;
    font-size: 1.05rem;
    margin-top: 0.3rem;
}

/* ── Cards ── */
.fs-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.fs-card-title {
    font-size: 1rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.6rem;
}

/* ── Metric tiles ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.4rem;
}
.metric-tile {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1;
}
.metric-blue  { color: #58a6ff; }
.metric-green { color: #3fb950; }
.metric-amber { color: #d29922; }
.metric-red   { color: #f85149; }

/* ── Allocation table ── */
.alloc-header {
    display: grid;
    grid-template-columns: 1.2fr 1.2fr 1.4fr 1.4fr 0.8fr;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: #21262d;
    border-radius: 8px 8px 0 0;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}
.alloc-row {
    display: grid;
    grid-template-columns: 1.2fr 1.2fr 1.4fr 1.4fr 0.8fr;
    gap: 0.5rem;
    padding: 0.65rem 1rem;
    border-bottom: 1px solid #21262d;
    font-size: 0.9rem;
    align-items: center;
}
.alloc-row:hover { background: #1c2128; }
.alloc-zone  { font-family: monospace; color: #58a6ff; }
.alloc-risk  { color: #f85149; font-weight: 600; }
.alloc-food  { color: #3fb950; }
.alloc-med   { color: #d29922; }
.alloc-boat  { color: #58a6ff; }

/* ── Risk bar ── */
.risk-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
}
.risk-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #3fb950, #d29922, #f85149);
}

/* ── Zone severity badge ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-critical { background: #3d1a1a; color: #f85149; }
.badge-high     { background: #2e2005; color: #d29922; }
.badge-moderate { background: #0d2d1a; color: #3fb950; }

/* ── Buttons ── */
.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
}

/* ── Inputs / selects ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider {
    background: #21262d !important;
    color: #e6edf3 !important;
    border-color: #30363d !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr { border-color: #30363d !important; }

/* ── Section headers ── */
h1, h2, h3, h4 { color: #e6edf3 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #58a6ff !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #161b22; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def severity_badge(score: float) -> str:
    if score >= 0.75:
        return '<span class="badge badge-critical">CRITICAL</span>'
    if score >= 0.5:
        return '<span class="badge badge-high">HIGH</span>'
    return '<span class="badge badge-moderate">MODERATE</span>'


def risk_bar(score: float) -> str:
    pct = int(score * 100)
    return f"""
    <div class="risk-bar-bg">
      <div class="risk-bar-fill" style="width:{pct}%"></div>
    </div>
    """


def dark_fig(figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    return fig, ax


# ─────────────────────────────────────────────
#  BACKEND INIT
# ─────────────────────────────────────────────
@st.cache_resource
def get_engine():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    return RiskEngine(data_dir=data_dir)

engine = get_engine()


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="fs-header">
  <div class="fs-title">🌊 FloodSense AI</div>
  <div class="fs-subtitle">Flood Risk Intelligence · Real-Time Detection · Smart Resource Allocation</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂 Data Source")

    regions = engine.get_maps_by_region()
    use_synthetic = False

    if not regions:
        st.info("No `.npy` flood maps found — using synthetic demo data.")
        use_synthetic = True
    else:
        selected_region = st.selectbox("Region", sorted(regions.keys()))
        region_maps = regions[selected_region]
        selected_map = st.selectbox(f"Map ({len(region_maps)} available)", region_maps)

    st.markdown("---")
    st.markdown("### 📦 Available Resources")
    food_packets = st.number_input("🍞 Food Packets",   min_value=0, value=5000, step=100)
    medical_kits = st.number_input("🏥 Medical Kits",   min_value=0, value=1200, step=50)
    boats        = st.number_input("🚤 Rescue Boats",   min_value=0, value=50,   step=5)

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")
    population_method = st.selectbox(
        "Population Model",
        ["gradient", "uniform"],
        help="Gradient = urban-centre weighted. Uniform = equal density."
    )
    flood_weight = st.slider("Flood vs Population weight", 0.0, 1.0, 0.6, 0.05,
                              help="Higher = flood probability drives risk more than population density.")
    refine = st.checkbox("Apply mask refinement (morphological)", value=True)
    top_k  = st.slider("Top-K zones to allocate", 5, 20, 10)

    st.markdown("---")
    analyse_btn = st.button("🔍 Run Analysis", use_container_width=True)


# ─────────────────────────────────────────────
#  MAIN ANALYSIS
# ─────────────────────────────────────────────
run = analyse_btn or ("analysed" not in st.session_state)

if run:
    st.session_state.analysed = True

    with st.spinner("Running flood risk analysis…"):
        # Load map
        if use_synthetic:
            engine.flood_mask      = np.random.rand(256, 256).astype(np.float32)
            engine.current_map_name = "synthetic_demo"
        else:
            engine.load_flood_mask(selected_map)

        # Optional refinement
        if refine:
            engine.flood_mask = refine_mask(engine.flood_mask)

        engine.generate_population_grid(method=population_method)
        engine.compute_risk(weights={"flood": flood_weight,
                                      "population": 1 - flood_weight})

        risk_stats   = engine.get_risk_statistics()
        flood_stats  = compute_flood_statistics(engine.flood_mask)
        flood_zones  = extract_flood_zones(engine.flood_mask)

        resources    = {"food_packets": food_packets,
                        "medical_kits": medical_kits,
                        "boats": boats}
        allocator    = AllocationEngine(engine.risk_map, resources)
        alloc_plan   = allocator.allocate(top_k=top_k)

    # ── Metric tiles ──────────────────────────
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-tile">
        <div class="metric-label">Flood Coverage</div>
        <div class="metric-value metric-blue">{flood_stats['coverage_pct']:.1f}%</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Mean Risk</div>
        <div class="metric-value metric-green">{risk_stats['mean_risk']:.4f}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">High Risk Cells</div>
        <div class="metric-value metric-amber">{risk_stats['high_risk_cells']:,}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Critical Cells</div>
        <div class="metric-value metric-red">{risk_stats['critical_cells']:,}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Maps row ──────────────────────────────
    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown('<div class="fs-card">', unsafe_allow_html=True)
        st.markdown('<div class="fs-card-title">🌊 Flood Probability Map</div>', unsafe_allow_html=True)
        fig, ax = dark_fig((6, 5))
        im = ax.imshow(engine.flood_mask, cmap="Blues", vmin=0, vmax=1)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
        cbar.set_label("Probability", color="#8b949e")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="fs-card">', unsafe_allow_html=True)
        st.markdown('<div class="fs-card-title">🔥 Composite Risk Heatmap</div>', unsafe_allow_html=True)
        fig, ax = dark_fig((6, 5))
        im = ax.imshow(engine.risk_map, cmap="inferno", vmin=0, vmax=1)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
        cbar.set_label("Risk Intensity", color="#8b949e")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Flood zones detail ────────────────────
    st.markdown('<div class="fs-card">', unsafe_allow_html=True)
    st.markdown('<div class="fs-card-title">📍 Top Risk Zone Locations</div>', unsafe_allow_html=True)
    fig, ax = dark_fig((12, 5))
    ax.imshow(engine.risk_map, cmap="inferno", alpha=0.85)
    for i, zone in enumerate(alloc_plan[:top_k]):
        r, c = zone["coordinates"]
        ax.scatter(c, r, s=160, color="#58a6ff", edgecolors="#e6edf3",
                   linewidths=1.2, zorder=5)
        ax.text(c + 3, r - 3, f"#{i+1}", color="#e6edf3", fontsize=8,
                fontweight="bold", zorder=6)
    ax.axis("off")
    ax.set_title(f"Top {top_k} Allocation Zones", color="#8b949e", fontsize=11, pad=10)
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Resource Allocation table ─────────────
    st.markdown('<div class="fs-card">', unsafe_allow_html=True)
    st.markdown('<div class="fs-card-title">📦 Smart Resource Allocation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="alloc-header">
      <span>Zone</span>
      <span>Risk Score</span>
      <span>Food Packets</span>
      <span>Medical Kits</span>
      <span>Boats</span>
    </div>
    """, unsafe_allow_html=True)

    for i, zone in enumerate(alloc_plan):
        r, c    = zone["coordinates"]
        score   = zone["risk_score"]
        pct     = int(score * 100)
        st.markdown(f"""
        <div class="alloc-row">
          <span class="alloc-zone">({r}, {c}) &nbsp;{severity_badge(score)}</span>
          <span class="alloc-risk">
            {score:.4f}
            {risk_bar(score)}
          </span>
          <span class="alloc-food">🍞 {zone['food_packets']:,}</span>
          <span class="alloc-med">🏥 {zone['medical_kits']:,}</span>
          <span class="alloc-boat">🚤 {zone['boats']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Flood zone detail cards ───────────────
    if flood_zones:
        st.markdown('<div class="fs-card">', unsafe_allow_html=True)
        st.markdown('<div class="fs-card-title">🗺 Detected Flood Zones (from postprocessor)</div>',
                    unsafe_allow_html=True)

        cols = st.columns(min(len(flood_zones[:6]), 3))
        for i, zone in enumerate(flood_zones[:6]):
            with cols[i % 3]:
                sev = zone["mean_severity"]
                st.markdown(f"""
                <div class="fs-card" style="margin-bottom:0.6rem">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-weight:700;color:#e6edf3">Zone #{zone['zone_id']}</span>
                    {severity_badge(sev)}
                  </div>
                  <div style="margin-top:0.6rem;font-size:0.85rem;color:#8b949e">
                    Area: <b style="color:#58a6ff">{zone['area_pixels']:,} px</b><br>
                    Severity: <b style="color:#f85149">{sev:.3f}</b><br>
                    Centroid: <b style="color:#3fb950">({zone['centroid'][1]:.0f}, {zone['centroid'][0]:.0f})</b>
                  </div>
                  {risk_bar(sev)}
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Summary stats ─────────────────────────
    st.markdown('<div class="fs-card">', unsafe_allow_html=True)
    st.markdown('<div class="fs-card-title">📊 Analysis Summary</div>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(f"""
        **Flood Statistics**
        - Coverage: `{flood_stats['coverage_pct']:.2f}%`
        - Affected pixels: `{flood_stats['affected_pixels']:,}`
        - Mean severity: `{flood_stats['mean_severity']:.4f}`
        - Max severity: `{flood_stats['max_severity']:.4f}`
        - Detected zones: `{flood_stats['num_zones']}`
        """)
    with sc2:
        st.markdown(f"""
        **Risk Statistics**
        - Mean risk: `{risk_stats['mean_risk']:.4f}`
        - Max risk: `{risk_stats['max_risk']:.4f}`
        - Std deviation: `{risk_stats['std_risk']:.4f}`
        - High risk (>0.5): `{risk_stats['high_risk_cells']:,}`
        - Critical (>0.8): `{risk_stats['critical_cells']:,}`
        """)
    with sc3:
        total_food = sum(z["food_packets"] for z in alloc_plan)
        total_med  = sum(z["medical_kits"] for z in alloc_plan)
        total_boat = sum(z["boats"] for z in alloc_plan)
        st.markdown(f"""
        **Resources Dispatched**
        - Food packets: `{total_food:,}` / `{food_packets:,}`
        - Medical kits: `{total_med:,}` / `{medical_kits:,}`
        - Rescue boats: `{total_boat}` / `{boats}`
        - Zones covered: `{len(alloc_plan)}`
        - Map: `{engine.current_map_name}`
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:0.8rem;padding:0.5rem 0 1rem 0">
  FloodSense AI · Built with PyTorch · U-Net ResNet50 · OpenStreetMap
</div>
""", unsafe_allow_html=True)
