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
    Vectorized Largest Triangle Three Buckets (LTTB) algorithm for downsampling.
    data: 2D numpy array [x, y]
    n_out: number of output points
    """
    if n_out >= len(data) or n_out <= 2:
        return data

    sampled = np.zeros((n_out, 2))
    sampled[0] = data[0]
    sampled[-1] = data[-1]

    bucket_size = (len(data) - 2) / (n_out - 2)
    bucket_indices = np.floor(np.arange(0, n_out - 1) * bucket_size).astype(int) + 1
    bucket_indices[-1] = len(data)

    next_bucket_indices = bucket_indices[1:]
    sums = np.add.reduceat(data, next_bucket_indices[:-1])
    counts = np.diff(next_bucket_indices)[:, np.newaxis]
    avgs = sums / counts

    a = 0
    for i in tqdm(range(n_out - 2), desc="LTTB Downsampling"):
        bucket_data = data[bucket_indices[i]:bucket_indices[i+1]]
        point_a = data[a]
        avg_point = avgs[i]

        areas = np.abs(
            (point_a[0] - avg_point[0]) * (bucket_data[:, 1] - point_a[1]) -
            (point_a[0] - bucket_data[:, 0]) * (avg_point[1] - point_a[1])
        )

        max_idx = np.argmax(areas)
        sampled[i + 1] = bucket_data[max_idx]
        a = bucket_indices[i] + max_idx

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
