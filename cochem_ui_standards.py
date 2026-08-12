from typing import Any, Dict, List, Optional
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from tqdm import tqdm
import math

# ==========================================
# 1. ACS Standards & Accessibility
# ==========================================
def apply_acs_standards() -> Any:
    """Applies ACS standard formatting and color-blind friendly palettes."""
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 8.0
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.width'] = 1.0
    mpl.rcParams['ytick.minor.width'] = 1.0
    mpl.rcParams['lines.linewidth'] = 1.5
    # Color-blind-friendly default cycle (Viridis derived)
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#440154', '#31688e', '#35b779', '#fde725'])
    mpl.rcParams['image.cmap'] = 'viridis'

# ==========================================
# 2. LTTB Downsampling Algorithm
# ==========================================
def lttb_downsample(data, n_out) -> Any:
    """
    Largest Triangle Three Buckets (LTTB) algorithm for downsampling.
    data: 2D numpy array [x, y]
    n_out: number of output points
    """
    if n_out >= len(data) or n_out == 0:
        return data

    sampled = np.zeros((n_out, 2))
    sampled[0] = data[0]
    sampled[n_out - 1] = data[len(data) - 1]

    every = (len(data) - 2) / (n_out - 2)
    a = 0
    next_a = 0

    # Ensure tqdm progress bar is included for all python scripts as directed
    for i in tqdm(range(n_out - 2), desc="LTTB Downsampling"):
        avg_x = 0
        avg_y = 0
        avg_range_start = int(math.floor((i + 1) * every) + 1)
        avg_range_end = int(math.floor((i + 2) * every) + 1)
        avg_range_end = min(avg_range_end, len(data))
        avg_range_length = avg_range_end - avg_range_start

        while avg_range_start < avg_range_end:
            avg_x += data[avg_range_start][0]
            avg_y += data[avg_range_start][1]
            avg_range_start += 1

        avg_x /= avg_range_length
        avg_y /= avg_range_length

        range_offs = int(math.floor((i + 0) * every) + 1)
        range_to = int(math.floor((i + 1) * every) + 1)

        point_a_x = data[a][0]
        point_a_y = data[a][1]

        max_area = -1
        max_area_point = np.zeros(2)

        while range_offs < range_to:
            area = math.fabs(
                (point_a_x - avg_x) * (data[range_offs][1] - point_a_y) -
                (point_a_x - data[range_offs][0]) * (avg_y - point_a_y)
            ) * 0.5
            if area > max_area:
                max_area = area
                max_area_point = data[range_offs]
                next_a = range_offs
            range_offs += 1

        sampled[i + 1] = max_area_point
        a = next_a

    return sampled

# ==========================================
# 3. Automatic Unit Conversions
# ==========================================
UNIT_CONVERSIONS = {
    "Hartrees": 1.0,
    "kcal/mol": 627.509,
    "eV": 27.2114,
    "cm^-1": 219474.63
}

def convert_units(value, from_unit, to_unit) -> Any:
    """Converts dynamically between quantum chemistry units."""
    value_in_hartrees = value / UNIT_CONVERSIONS[from_unit]
    return value_in_hartrees * UNIT_CONVERSIONS[to_unit]

# ==========================================
# 4. Streamlit UI Build
# ==========================================
def run_ui() -> Any:
    st.set_page_config(page_title="CoChem Spectra UI", layout="wide")
    
    st.title("CoChem Accessible Spectra Viewer")
    
    # Ensure WCAG 2.1 AA compliant aria-live region and 4.5:1 contrast
    st.markdown(
        '<div aria-live="polite" style="color: #000000; background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-weight: bold;">'
        'Dashboard initialized. WCAG 2.1 AA constraints active (Contrast > 4.5:1).'
        '</div>', 
        unsafe_allow_html=True
    )
    
    apply_acs_standards()

    # Sidebar: Unit Conversions
    st.sidebar.header("Unit Converter")
    from_unit = st.sidebar.selectbox("From", list(UNIT_CONVERSIONS.keys()), index=0)
    to_unit = st.sidebar.selectbox("To", list(UNIT_CONVERSIONS.keys()), index=1)
    val = st.sidebar.number_input("Value", value=1.0)
    converted_val = convert_units(val, from_unit, to_unit)
    st.sidebar.success(f"{val} {from_unit} = {converted_val:.4f} {to_unit}")

    st.header("Spectra Rendering: 1M Points to 1K")
    if st.button("Generate & Downsample Spectra"):
        # 1. Generate 1,000,000 points of raw spectra
        st.write("Generating 1,000,000-point raw spectra data...")
        x = np.linspace(0, 4000, 1000000)
        y = np.sin(x / 100) * np.exp(-x / 2000) + np.random.normal(0, 0.05, 1000000)
        raw_data = np.column_stack((x, y))
        
        # 2. LTTB Downsampling
        st.write("Executing LTTB Downsampling to 1,000 points...")
        downsampled = lttb_downsample(raw_data, 1000)
        
        col1, col2 = st.columns(2)
        
        # Matplotlib ACS Standard View
        with col1:
            st.subheader("Matplotlib (ACS Format)")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(downsampled[:, 0], downsampled[:, 1], color='#440154') # Viridis palette used
            ax.set_xlabel("Wavenumber (cm$^{-1}$)")
            ax.set_ylabel("Intensity (a.u.)")
            st.pyplot(fig)
            
            # Save strictly as SVG
            svg_path = "cochem_spectra_plot.svg"
            fig.savefig(svg_path, format="svg", bbox_inches='tight')
            st.markdown(f"**Saved:** `{svg_path}`")

        # Plotly WebGL View
        with col2:
            st.subheader("Plotly (WebGL)")
            fig_ply = go.Figure(data=go.Scattergl(
                x=downsampled[:, 0], y=downsampled[:, 1],
                mode='lines',
                line=dict(color='#21918c', width=1.5) # Colorblind friendly cividis/viridis palette
            ))
            fig_ply.update_layout(
                xaxis_title="Wavenumber (cm^-1)",
                yaxis_title="Intensity (a.u.)",
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            # Ensure high contrast in axes
            fig_ply.update_xaxes(showline=True, linewidth=1.5, linecolor='black', gridcolor='lightgrey')
            fig_ply.update_yaxes(showline=True, linewidth=1.5, linecolor='black', gridcolor='lightgrey')
            st.plotly_chart(fig_ply, use_container_width=True)

if __name__ == "__main__":
    run_ui()