import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_engine import RiskEngine
from core.allocation_engine import AllocationEngine

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Flood Risk Intelligence",
    layout="wide"
)

# ----------------------------
# BRIGHT CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
/* Force light blue background everywhere */
.stApp {
    background: linear-gradient(135deg, #e0f4ff 0%, #b8e4ff 50%, #d0f0ff 100%) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #cce7ff, #b3d9ff) !important;
}

.big-title {
    font-size: 52px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #ff3366, #ff6b35, #ffcc00, #00d4aa, #00aaff, #aa66ff);
    background-size: 300% 300%;
    animation: rainbow 3s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 4px 20px rgba(255,51,102,0.3);
}

@keyframes rainbow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #ff6b9d;
    font-weight: 600;
    margin-bottom: 30px;
}

.card {
    background: linear-gradient(145deg, #e8f6ff, #d4edff) !important;
    padding: 25px;
    border-radius: 24px;
    box-shadow: 0px 15px 40px rgba(0,150,255,0.15), 0px 5px 15px rgba(0,200,255,0.1);
    margin-bottom: 25px;
    border: 2px solid rgba(100,180,255,0.3);
}

.metric-box {
    background: linear-gradient(135deg, #00d4ff, #00ffcc, #00ff88) !important;
    color: white;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,212,255,0.4);
    border: none;
}

.metric-box2 {
    background: linear-gradient(135deg, #ff6b6b, #ff8e53, #ffcc00) !important;
    color: white;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(255,107,107,0.4);
    border: none;
}

.metric-box3 {
    background: linear-gradient(135deg, #a855f7, #ec4899, #f43f5e) !important;
    color: white;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(168,85,247,0.4);
    border: none;
}

.metric-box4 {
    background: linear-gradient(135deg, #10b981, #059669, #047857) !important;
    color: white;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(16,185,129,0.4);
    border: none;
}

/* Bright buttons and inputs */
.stButton>button {
    background: linear-gradient(90deg, #ff6b9d, #c44cff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* Headers */
h1, h2, h3, h4 {
    color: #ff3399 !important;
}

/* Success messages */
.stSuccess {
    background: linear-gradient(90deg, #00ff88, #00d4aa) !important;
    border-radius: 12px !important;
}

.allocation-row {
    background: linear-gradient(90deg, #f0f9ff, #e0f2fe);
    padding: 12px 20px;
    border-radius: 12px;
    margin: 8px 0;
    border-left: 4px solid #0ea5e9;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# INITIALIZE ENGINE
# ----------------------------
@st.cache_resource
def get_engine():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    return RiskEngine(data_dir=data_dir)

engine = get_engine()

# ----------------------------
# HEADER
# ----------------------------
st.markdown('<div class="big-title">🌊 Flood Risk Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Based Disaster Detection & Smart Resource Allocation</div>', unsafe_allow_html=True)

st.write("")

# ----------------------------
# SIDEBAR - Map Selection
# ----------------------------
with st.sidebar:
    st.header("📍 Map Selection")
    
    regions = engine.get_maps_by_region()
    
    if not regions:
        st.error("No flood maps found in data/ directory")
        st.stop()
    
    # Region selector
    selected_region = st.selectbox(
        "Select Region",
        options=sorted(regions.keys()),
        index=0
    )
    
    # Map selector for region
    region_maps = regions[selected_region]
    selected_map = st.selectbox(
        f"Select Map ({len(region_maps)} available)",
        options=region_maps,
        index=0
    )
    
    st.divider()
    
    st.header("📦 Resources")
    food_packets = st.number_input("Food Packets", min_value=0, value=5000)
    medical_kits = st.number_input("Medical Kits", min_value=0, value=1200)
    boats = st.number_input("Rescue Boats", min_value=0, value=50)
    
    st.divider()
    
    st.header("⚙️ Settings")
    population_method = st.selectbox(
        "Population Model",
        options=["gradient", "uniform"],
        index=0,
        help="Gradient simulates urban-to-rural distribution"
    )
    
    flood_weight = st.slider("Flood Weight", 0.0, 1.0, 0.6)
    
    analyze_btn = st.button("🔍 Analyze Risk", use_container_width=True)

# ----------------------------
# MAIN ANALYSIS
# ----------------------------
if analyze_btn or 'analyzed' not in st.session_state:
    st.session_state.analyzed = True
    
    # Load and process
    with st.spinner("Loading flood probability map..."):
        engine.load_flood_mask(selected_map)
        engine.generate_population_grid(method=population_method)
        engine.compute_risk(weights={'flood': flood_weight, 'population': 1 - flood_weight})
    
    # Get statistics
    stats = engine.get_risk_statistics()
    
    # Display maps
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🌊 Flood Probability Map")
        fig1, ax1 = plt.subplots(figsize=(8, 8), facecolor='#e0f4ff')
        im1 = ax1.imshow(engine.flood_mask, cmap="Blues")
        ax1.axis("off")
        ax1.set_title(selected_map, fontsize=10, color='#666')
        plt.colorbar(im1, ax=ax1, label="Flood Probability", shrink=0.8)
        fig1.patch.set_facecolor('#e0f4ff')
        st.pyplot(fig1)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔥 Composite Risk Heatmap")
        fig2, ax2 = plt.subplots(figsize=(8, 8), facecolor='#e0f4ff')
        im2 = ax2.imshow(engine.risk_map, cmap="inferno")
        ax2.axis("off")
        ax2.set_title("Risk = Flood × Population Density", fontsize=10, color='#666')
        plt.colorbar(im2, ax=ax2, label="Risk Intensity", shrink=0.8)
        fig2.patch.set_facecolor('#e0f4ff')
        st.pyplot(fig2)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("")
    
    # METRICS
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Mean Risk</h4>
            <h2>{stats['mean_risk']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        st.markdown(f"""
        <div class="metric-box2">
            <h4>Max Risk</h4>
            <h2>{stats['max_risk']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        st.markdown(f"""
        <div class="metric-box3">
            <h4>High Risk Cells</h4>
            <h2>{stats['high_risk_cells']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with m4:
        st.markdown(f"""
        <div class="metric-box4">
            <h4>Critical Cells</h4>
            <h2>{stats['critical_cells']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    
    # ----------------------------
    # RESOURCE ALLOCATION
    # ----------------------------
    resources = {
        "food_packets": food_packets,
        "medical_kits": medical_kits,
        "boats": boats
    }
    
    allocator = AllocationEngine(engine.risk_map, resources)
    allocation_plan = allocator.allocate(top_k=10)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📦 Smart Resource Allocation (Top 10 High-Risk Zones)")
    
    # Create allocation table
    col_zone, col_risk, col_food, col_med, col_boat = st.columns([2, 2, 2, 2, 1])
    
    with col_zone:
        st.markdown("**Zone**")
    with col_risk:
        st.markdown("**Risk Score**")
    with col_food:
        st.markdown("**Food Packets**")
    with col_med:
        st.markdown("**Medical Kits**")
    with col_boat:
        st.markdown("**Boats**")
    
    for i, zone in enumerate(allocation_plan[:10]):
        coord = zone['coordinates']
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
        
        with c1:
            st.code(f"({coord[0]}, {coord[1]})")
        with c2:
            st.progress(zone['risk_score'], text=f"{zone['risk_score']:.4f}")
        with c3:
            st.info(f"🍞 {zone['food_packets']}")
        with c4:
            st.warning(f"🏥 {zone['medical_kits']}")
        with c5:
            st.success(f"🚤 {zone['boats']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ----------------------------
    # TOP RISK ZONES VISUALIZATION
    # ----------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📍 Top 10 Risk Zone Locations")
    
    fig3, ax3 = plt.subplots(figsize=(10, 10), facecolor='#e0f4ff')
    ax3.imshow(engine.risk_map, cmap="inferno", alpha=0.7)
    
    # Plot top zones
    for i, zone in enumerate(allocation_plan[:10]):
        coord = zone['coordinates']
        ax3.scatter(coord[1], coord[0], s=200, c='cyan', edgecolors='white', linewidth=2, marker='o')
        ax3.annotate(f"#{i+1}", (coord[1], coord[0]), color='white', fontsize=10, 
                    ha='center', va='center', fontweight='bold')
    
    ax3.axis("off")
    ax3.set_title("High-Risk Zone Locations", fontsize=14, color='#333')
    fig3.patch.set_facecolor('#e0f4ff')
    st.pyplot(fig3)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ----------------------------
    # REGION SUMMARY
    # ----------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"📊 {selected_region} Region Summary")
    
    st.write(f"**Total maps in region:** {len(region_maps)}")
    st.write(f"**Current map:** {selected_map}")
    st.write(f"**Map dimensions:** {engine.flood_mask.shape[0]} × {engine.flood_mask.shape[1]} pixels")
    st.write(f"**Total cells analyzed:** {stats['total_cells']:,}")
    
    st.markdown('</div>', unsafe_allow_html=True)