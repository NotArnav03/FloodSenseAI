import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Flood Risk Intelligence System",
    layout="wide"
)

# -------------------------
# HEADER
# -------------------------
st.title("🌊 Flood Risk & Resource Allocation System")
st.markdown("AI-powered disaster prioritization and response planning")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("Controls")

uploaded_image = st.sidebar.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "png"]
)

total_resources = st.sidebar.number_input(
    "Total Relief Kits Available",
    min_value=0,
    value=100
)

# -------------------------
# MAIN CONTENT
# -------------------------
if uploaded_image is not None:

    col1, col2 = st.columns(2)

    # Show Uploaded Image
    with col1:
        st.subheader("Original Image")
        st.image(uploaded_image, use_column_width=True)

    # Dummy flood mask (replace later with model output)
    flood_mask = np.random.rand(128, 128)

    with col2:
        st.subheader("Flood Severity Map")
        fig1, ax1 = plt.subplots()
        ax1.imshow(flood_mask, cmap="Blues")
        ax1.axis("off")
        st.pyplot(fig1)

    # -------------------------
    # Risk Calculation
    # -------------------------
    population_density = np.random.rand(128, 128)
    risk = flood_mask * population_density
    total_risk = np.sum(risk)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Risk Heatmap")
        fig2, ax2 = plt.subplots()
        ax2.imshow(risk, cmap="Reds")
        ax2.axis("off")
        st.pyplot(fig2)

    with col4:
        st.subheader("Risk Metrics")
        st.metric("Total Risk Score", f"{total_risk:.2f}")
        st.metric("Highest Risk Cell", f"{np.max(risk):.4f}")

    # -------------------------
    # Allocation Engine
    # -------------------------
    if total_risk > 0 and total_resources > 0:
        allocation = (risk / total_risk) * total_resources

        st.divider()
        st.subheader("Resource Allocation (Top Risk Zones)")

        flat_indices = np.argsort(risk.flatten())[::-1][:5]
        rows, cols = np.unravel_index(flat_indices, risk.shape)

        for i in range(5):
            st.write(
                f"Zone ({rows[i]}, {cols[i]}) → "
                f"Allocated: {allocation[rows[i], cols[i]]:.2f} kits"
            )

else:
    st.info("Upload a satellite image from the sidebar to begin.")