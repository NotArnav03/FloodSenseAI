"""
FloodSense AI — Dark Frontend
================================
Tab 1: Risk Dashboard  (RiskEngine + AllocationEngine + postprocess)
Tab 2: Upload & Predict (InferenceEngine — live U-Net inference)
"""

import os, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_engine import RiskEngine
from core.allocation_engine import AllocationEngine
from utils.postprocess import refine_mask, compute_flood_statistics, extract_flood_zones

# ── page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodSense AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"],.stApp{background:#0d1117!important;color:#e6edf3!important}
[data-testid="stSidebar"]{background:#161b22!important;border-right:1px solid #30363d!important}
[data-testid="stSidebar"] *{color:#c9d1d9!important}
.fs-title{font-size:2.8rem;font-weight:800;background:linear-gradient(90deg,#58a6ff,#3fb950,#58a6ff);
  background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:shine 3s linear infinite;text-align:center;letter-spacing:-1px}
@keyframes shine{to{background-position:200% center}}
.fs-sub{text-align:center;color:#8b949e;font-size:1rem;margin-bottom:1rem}
.fs-card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.3rem 1.5rem;margin-bottom:1.1rem}
.fs-card-title{font-size:.82rem;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.7rem}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.2rem}
.metric-tile{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.1rem;text-align:center}
.metric-label{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.3rem}
.metric-value{font-size:1.85rem;font-weight:700;line-height:1}
.m-blue{color:#58a6ff}.m-green{color:#3fb950}.m-amber{color:#d29922}.m-red{color:#f85149}
.alloc-header{display:grid;grid-template-columns:1.3fr 1.3fr 1.4fr 1.4fr .7fr;gap:.5rem;
  padding:.5rem 1rem;background:#21262d;border-radius:8px 8px 0 0;
  font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;font-weight:600}
.alloc-row{display:grid;grid-template-columns:1.3fr 1.3fr 1.4fr 1.4fr .7fr;gap:.5rem;
  padding:.6rem 1rem;border-bottom:1px solid #21262d;font-size:.88rem;align-items:center}
.alloc-row:hover{background:#1c2128}
.alloc-zone{font-family:monospace;color:#58a6ff}
.alloc-risk{color:#f85149;font-weight:600}
.alloc-food{color:#3fb950}.alloc-med{color:#d29922}.alloc-boat{color:#58a6ff}
.rbar-bg{background:#21262d;border-radius:4px;height:5px;margin-top:3px}
.rbar-fill{height:5px;border-radius:4px;background:linear-gradient(90deg,#3fb950,#d29922,#f85149)}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.7rem;font-weight:600}
.bc{background:#3d1a1a;color:#f85149}.bh{background:#2e2005;color:#d29922}.bm{background:#0d2d1a;color:#3fb950}
.pred-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1rem}
.pred-tile{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1rem;text-align:center}
.pred-val{font-size:2rem;font-weight:700}
.stButton>button{background:#238636!important;color:#fff!important;border:1px solid #2ea043!important;
  border-radius:8px!important;font-weight:600!important;padding:.45rem 1.1rem!important}
.stButton>button:hover{background:#2ea043!important}
.stTabs [data-baseweb="tab-list"]{background:#161b22!important;border-bottom:1px solid #30363d!important;gap:.3rem}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#8b949e!important;
  border-radius:6px 6px 0 0!important;padding:.5rem 1.2rem!important;font-weight:600}
.stTabs [aria-selected="true"]{background:#21262d!important;color:#e6edf3!important;
  border-bottom:2px solid #58a6ff!important}
h1,h2,h3,h4{color:#e6edf3!important}
hr{border-color:#30363d!important}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#161b22}
::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────
def badge(score):
    if score >= .75: return '<span class="badge bc">CRITICAL</span>'
    if score >= .5:  return '<span class="badge bh">HIGH</span>'
    return '<span class="badge bm">MODERATE</span>'

def rbar(score):
    return f'<div class="rbar-bg"><div class="rbar-fill" style="width:{int(score*100)}%"></div></div>'

def dark_fig(fs=(7,5)):
    fig,ax = plt.subplots(figsize=fs)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e")
    return fig,ax

CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "best_model.pth"
)

@st.cache_resource
def get_risk_engine():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    return RiskEngine(data_dir=data_dir)

@st.cache_resource
def get_inference_engine():
    """Returns (engine, error_message). engine is None if checkpoint missing."""
    try:
        from core.inference import InferenceEngine
        eng = InferenceEngine(checkpoint_path=CKPT, input_size=256)
        return eng, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Failed to load model: {e}"

risk_engine = get_risk_engine()

# ── header ────────────────────────────────────────────────────────────
st.markdown('<div class="fs-title">🌊 FloodSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="fs-sub">Flood Risk Intelligence · Real-Time Detection · Smart Resource Allocation</div>',
            unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────
tab_dash, tab_predict = st.tabs(["📊 Risk Dashboard", "🤖 Upload & Predict"])

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — RISK DASHBOARD
# ══════════════════════════════════════════════════════════════════════
with tab_dash:
    with st.sidebar:
        st.markdown("### 🗂 Data Source")
        regions   = risk_engine.get_maps_by_region()
        use_synth = not bool(regions)

        if use_synth:
            st.info("No `.npy` maps found — using synthetic demo data.")
        else:
            sel_region = st.selectbox("Region", sorted(regions.keys()))
            sel_map    = st.selectbox(f"Map ({len(regions[sel_region])} available)", regions[sel_region])

        st.markdown("---")
        st.markdown("### 📦 Resources")
        food_packets = st.number_input("🍞 Food Packets", 0, value=5000, step=100)
        medical_kits = st.number_input("🏥 Medical Kits", 0, value=1200, step=50)
        boats        = st.number_input("🚤 Rescue Boats", 0, value=50,   step=5)

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        pop_method   = st.selectbox("Population Model", ["gradient","uniform"])
        flood_weight = st.slider("Flood weight", 0.0, 1.0, 0.6, 0.05)
        do_refine    = st.checkbox("Mask refinement", value=True)
        top_k        = st.slider("Top-K zones", 5, 20, 10)
        st.markdown("---")
        run_btn = st.button("🔍 Run Analysis", use_container_width=True)

    run = run_btn or ("dash_run" not in st.session_state)
    if run:
        st.session_state.dash_run = True
        with st.spinner("Running risk analysis…"):
            if use_synth:
                risk_engine.flood_mask       = np.random.rand(256,256).astype(np.float32)
                risk_engine.current_map_name = "synthetic_demo"
            else:
                risk_engine.load_flood_mask(sel_map)
            if do_refine:
                risk_engine.flood_mask = refine_mask(risk_engine.flood_mask)
            risk_engine.generate_population_grid(method=pop_method)
            risk_engine.compute_risk(weights={"flood":flood_weight,"population":1-flood_weight})

            rstats  = risk_engine.get_risk_statistics()
            fstats  = compute_flood_statistics(risk_engine.flood_mask)
            fzones  = extract_flood_zones(risk_engine.flood_mask)
            res     = {"food_packets":food_packets,"medical_kits":medical_kits,"boats":boats}
            alloc   = AllocationEngine(risk_engine.risk_map, res).allocate(top_k=top_k)

        # metric tiles
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-tile"><div class="metric-label">Flood Coverage</div>
            <div class="metric-value m-blue">{fstats['coverage_pct']:.1f}%</div></div>
          <div class="metric-tile"><div class="metric-label">Mean Risk</div>
            <div class="metric-value m-green">{rstats['mean_risk']:.4f}</div></div>
          <div class="metric-tile"><div class="metric-label">High Risk Cells</div>
            <div class="metric-value m-amber">{rstats['high_risk_cells']:,}</div></div>
          <div class="metric-tile"><div class="metric-label">Critical Cells</div>
            <div class="metric-value m-red">{rstats['critical_cells']:,}</div></div>
        </div>""", unsafe_allow_html=True)

        # maps
        c1,c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown('<div class="fs-card"><div class="fs-card-title">🌊 Flood Probability</div>',
                        unsafe_allow_html=True)
            fig,ax = dark_fig((6,5))
            im = ax.imshow(risk_engine.flood_mask, cmap="Blues", vmin=0, vmax=1)
            ax.axis("off")
            cb = fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
            cb.ax.yaxis.set_tick_params(color="#8b949e")
            plt.setp(cb.ax.yaxis.get_ticklabels(),color="#8b949e")
            cb.set_label("Probability",color="#8b949e")
            st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="fs-card"><div class="fs-card-title">🔥 Risk Heatmap</div>',
                        unsafe_allow_html=True)
            fig,ax = dark_fig((6,5))
            im = ax.imshow(risk_engine.risk_map, cmap="inferno", vmin=0, vmax=1)
            ax.axis("off")
            cb = fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
            cb.ax.yaxis.set_tick_params(color="#8b949e")
            plt.setp(cb.ax.yaxis.get_ticklabels(),color="#8b949e")
            cb.set_label("Risk",color="#8b949e")
            st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        # zone map
        st.markdown('<div class="fs-card"><div class="fs-card-title">📍 Top Risk Zones</div>',
                    unsafe_allow_html=True)
        fig,ax = dark_fig((12,5))
        ax.imshow(risk_engine.risk_map, cmap="inferno", alpha=.85)
        for i,z in enumerate(alloc):
            r,c = z["coordinates"]
            ax.scatter(c,r,s=150,color="#58a6ff",edgecolors="#e6edf3",linewidths=1.2,zorder=5)
            ax.text(c+3,r-3,f"#{i+1}",color="#e6edf3",fontsize=8,fontweight="bold",zorder=6)
        ax.axis("off")
        ax.set_title(f"Top {top_k} Allocation Zones",color="#8b949e",fontsize=11,pad=10)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        # allocation table
        st.markdown('<div class="fs-card"><div class="fs-card-title">📦 Resource Allocation</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="alloc-header">
          <span>Zone</span><span>Risk</span><span>Food Packets</span>
          <span>Medical Kits</span><span>Boats</span></div>""", unsafe_allow_html=True)
        for z in alloc:
            r,c = z["coordinates"]
            st.markdown(f"""<div class="alloc-row">
              <span class="alloc-zone">({r},{c}) {badge(z['risk_score'])}</span>
              <span class="alloc-risk">{z['risk_score']:.4f}{rbar(z['risk_score'])}</span>
              <span class="alloc-food">🍞 {z['food_packets']:,}</span>
              <span class="alloc-med">🏥 {z['medical_kits']:,}</span>
              <span class="alloc-boat">🚤 {z['boats']}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # flood zone detail cards
        if fzones:
            st.markdown('<div class="fs-card"><div class="fs-card-title">🗺 Detected Flood Zones</div>',
                        unsafe_allow_html=True)
            cols = st.columns(min(len(fzones[:6]),3))
            for i,z in enumerate(fzones[:6]):
                with cols[i%3]:
                    sev = z["mean_severity"]
                    st.markdown(f"""
                    <div class="fs-card" style="margin-bottom:.5rem">
                      <div style="display:flex;justify-content:space-between">
                        <b style="color:#e6edf3">Zone #{z['zone_id']}</b>{badge(sev)}</div>
                      <div style="margin-top:.5rem;font-size:.84rem;color:#8b949e">
                        Area: <b style="color:#58a6ff">{z['area_pixels']:,} px</b><br>
                        Severity: <b style="color:#f85149">{sev:.3f}</b><br>
                        Centroid: <b style="color:#3fb950">({z['centroid'][1]:.0f},{z['centroid'][0]:.0f})</b>
                      </div>{rbar(sev)}</div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # summary
        st.markdown('<div class="fs-card"><div class="fs-card-title">📊 Summary</div>',
                    unsafe_allow_html=True)
        s1,s2,s3 = st.columns(3)
        with s1:
            st.markdown(f"""**Flood Stats**
- Coverage: `{fstats['coverage_pct']:.2f}%`
- Affected px: `{fstats['affected_pixels']:,}`
- Mean severity: `{fstats['mean_severity']:.4f}`
- Zones detected: `{fstats['num_zones']}`""")
        with s2:
            st.markdown(f"""**Risk Stats**
- Mean: `{rstats['mean_risk']:.4f}`
- Max: `{rstats['max_risk']:.4f}`
- Std: `{rstats['std_risk']:.4f}`
- Critical cells: `{rstats['critical_cells']:,}`""")
        with s3:
            tf = sum(z["food_packets"] for z in alloc)
            tm = sum(z["medical_kits"] for z in alloc)
            tb = sum(z["boats"] for z in alloc)
            st.markdown(f"""**Dispatched**
- Food: `{tf:,}` / `{food_packets:,}`
- Medical: `{tm:,}` / `{medical_kits:,}`
- Boats: `{tb}` / `{boats}`
- Map: `{risk_engine.current_map_name}`""")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — UPLOAD & PREDICT
# ══════════════════════════════════════════════════════════════════════
with tab_predict:
    inf_engine, inf_error = get_inference_engine()

    # ── model status banner ───────────────────────────────────────────
    if inf_engine is None:
        st.markdown(f"""
        <div class="fs-card" style="border-color:#f85149">
          <div style="display:flex;align-items:center;gap:.8rem">
            <span style="font-size:1.8rem">⚠️</span>
            <div>
              <div style="color:#f85149;font-weight:700;font-size:1rem">Model Not Trained Yet</div>
              <div style="color:#8b949e;font-size:.88rem;margin-top:.2rem">
                Train the model first, then come back to this tab for live inference.
              </div>
            </div>
          </div>
          <div style="margin-top:1rem;background:#0d1117;border-radius:8px;padding:.8rem 1rem;
                      font-family:monospace;font-size:.82rem;color:#3fb950">
            py models/train.py --data_dir data/flood_dataset --epochs 30 --batch_size 4
            --input_size 256 --workers 0
          </div>
        </div>""", unsafe_allow_html=True)

        # still show a demo with synthetic data so UI is visible
        st.markdown("#### 👇 Preview (demo mode — no model loaded)")
        demo_mode = True
    else:
        meta = inf_engine.info
        st.markdown(f"""
        <div class="fs-card" style="border-color:#3fb950">
          <div style="display:flex;align-items:center;gap:.8rem">
            <span style="font-size:1.8rem">✅</span>
            <div>
              <div style="color:#3fb950;font-weight:700;font-size:1rem">Model Ready</div>
              <div style="color:#8b949e;font-size:.88rem;margin-top:.2rem">
                U-Net · ResNet50 · Epoch <b style="color:#58a6ff">{meta['epoch']}</b>
                · Best IoU <b style="color:#58a6ff">{float(meta['best_iou']):.4f}</b>
                · Device <b style="color:#58a6ff">{meta['device'].upper()}</b>
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        demo_mode = False

    st.markdown("---")

    # ── uploader ──────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a satellite image (JPG / PNG)",
        type=["jpg","jpeg","png"],
        help="Drop any RGB satellite or aerial photo. The model will segment flood regions."
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("Running inference…"):
            if demo_mode:
                # synthetic result for UI preview
                h, w  = 256, 256
                prob  = np.random.rand(h, w).astype(np.float32) * 0.6
                prob  = refine_mask(prob)
                bmask = ((prob > 0.5) * 255).astype(np.uint8)
                flood_px   = (bmask > 0).sum()
                flood_pct  = float(flood_px / bmask.size * 100)
                confidence = float(prob[bmask > 0].mean()) if flood_px > 0 else 0.0
                orig_r     = cv2.resize(img_rgb, (256,256))
                overlay    = orig_r.copy()
                overlay[bmask > 0] = (
                    0.4 * overlay[bmask > 0] +
                    0.6 * np.array([30,100,220],dtype=np.float32)
                ).astype(np.uint8)
                hmap_u8    = (prob * 255).astype(np.uint8)
                heatmap    = cv2.cvtColor(cv2.applyColorMap(hmap_u8, cv2.COLORMAP_INFERNO),
                                          cv2.COLOR_BGR2RGB)
                result = {"prob_map":prob,"binary_mask":bmask,"flood_pct":flood_pct,
                          "confidence":confidence,"overlay":overlay,
                          "heatmap":heatmap,"orig_resized":orig_r}
            else:
                result = inf_engine.predict_with_overlay(img_rgb)

        # ── prediction metrics ────────────────────────────────────────
        flood_pct  = result["flood_pct"]
        confidence = result["confidence"]
        sev_label  = "CRITICAL" if flood_pct>30 else ("HIGH" if flood_pct>10 else "MODERATE")
        sev_color  = "#f85149" if flood_pct>30 else ("#d29922" if flood_pct>10 else "#3fb950")

        st.markdown(f"""
        <div class="pred-grid">
          <div class="pred-tile">
            <div class="metric-label">Flood Coverage</div>
            <div class="pred-val m-blue">{flood_pct:.1f}%</div>
          </div>
          <div class="pred-tile">
            <div class="metric-label">Model Confidence</div>
            <div class="pred-val m-green">{confidence:.3f}</div>
          </div>
          <div class="pred-tile">
            <div class="metric-label">Severity</div>
            <div class="pred-val" style="color:{sev_color}">{sev_label}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── four-panel visualisation ──────────────────────────────────
        st.markdown('<div class="fs-card"><div class="fs-card-title">🖼 Visual Analysis</div>',
                    unsafe_allow_html=True)

        p1,p2,p3,p4 = st.columns(4, gap="small")

        with p1:
            st.markdown("**Original Image**")
            st.image(result["orig_resized"], use_column_width=True)

        with p2:
            st.markdown("**Flood Overlay**")
            st.image(result["overlay"], use_column_width=True)

        with p3:
            st.markdown("**Probability Heatmap**")
            st.image(result["heatmap"], use_column_width=True)

        with p4:
            st.markdown("**Binary Mask**")
            st.image(result["binary_mask"], use_column_width=True, clamp=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── probability histogram ─────────────────────────────────────
        st.markdown('<div class="fs-card"><div class="fs-card-title">📈 Probability Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig((9, 3))
        ax.hist(result["prob_map"].flatten(), bins=80, color="#58a6ff",
                edgecolor="#0d1117", alpha=0.85)
        ax.axvline(0.5, color="#f85149", linewidth=1.5, linestyle="--", label="threshold=0.5")
        ax.set_xlabel("Flood Probability", color="#8b949e")
        ax.set_ylabel("Pixel Count", color="#8b949e")
        ax.set_title("Distribution of per-pixel flood probabilities", color="#8b949e", fontsize=10)
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── risk engine on prediction ─────────────────────────────────
        st.markdown('<div class="fs-card"><div class="fs-card-title">⚡ Auto Risk & Allocation from Prediction</div>',
                    unsafe_allow_html=True)

        risk_engine.flood_mask       = result["prob_map"]
        risk_engine.current_map_name = uploaded.name
        risk_engine.generate_population_grid(method="gradient")
        risk_engine.compute_risk()
        auto_alloc = AllocationEngine(
            risk_engine.risk_map,
            {"food_packets": 5000, "medical_kits": 1200, "boats": 50}
        ).allocate(top_k=5)

        st.markdown("**Top 5 zones auto-identified from model output:**")
        st.markdown("""<div class="alloc-header">
          <span>Zone</span><span>Risk</span><span>Food</span><span>Medical</span><span>Boats</span>
        </div>""", unsafe_allow_html=True)
        for z in auto_alloc:
            r,c = z["coordinates"]
            st.markdown(f"""<div class="alloc-row">
              <span class="alloc-zone">({r},{c}) {badge(z['risk_score'])}</span>
              <span class="alloc-risk">{z['risk_score']:.4f}{rbar(z['risk_score'])}</span>
              <span class="alloc-food">🍞 {z['food_packets']:,}</span>
              <span class="alloc-med">🏥 {z['medical_kits']:,}</span>
              <span class="alloc-boat">🚤 {z['boats']}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="fs-card" style="text-align:center;padding:3rem">
          <div style="font-size:3rem">🛰️</div>
          <div style="color:#8b949e;margin-top:1rem;font-size:1rem">
            Upload a satellite image above to run flood detection
          </div>
          <div style="color:#484f58;font-size:.85rem;margin-top:.5rem">
            Supports JPG and PNG · Any resolution
          </div>
        </div>""", unsafe_allow_html=True)

# ── footer ────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:.78rem;padding:.4rem 0 1rem 0">
  FloodSense AI · U-Net ResNet50 · PyTorch · OpenStreetMap
</div>""", unsafe_allow_html=True)
