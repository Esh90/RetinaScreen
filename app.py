"""
app.py — RetinaScreen Clinical Interface
Production-grade Streamlit application for Diabetic Retinopathy grading.

Architecture:
  - EfficientNet-B4 backbone
  - Asymmetric Clinical Severity-Weighted CORN loss (ACS-CORN)
  - Monte Carlo Dropout for uncertainty quantification
  - Directional Uncertainty Decomposition (DUD)
  - Uncertainty-Modulated Grad-CAM (UM-GradCAM)
  - OpenStreetMap Overpass specialist referral
"""

from __future__ import annotations

import io
import os
from datetime import datetime

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from theme_overrides import RS_THEME_DARK, RS_THEME_LIGHT

st.set_page_config(
    page_title="RetinaScreen",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "RetinaScreen — Research prototype. Not for clinical use.",
    },
)


def _load_css() -> None:
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


_load_css()


def _st_image(image, *, caption: str | None = None) -> None:
    """Streamlit <1.29 compatibility: st.image may not support use_container_width."""
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)


def _apply_theme_overrides(dark: bool) -> None:
    """Inject comprehensive theme skin (see theme_overrides.py). Toggle uses key `rs_dark_mode`."""
    st.markdown(RS_THEME_DARK if dark else RS_THEME_LIGHT, unsafe_allow_html=True)


try:
    from src.clinical_report import build_clinical_pdf_bytes, heatmap_overlay_or_none
    from src.gradcam import compute_um_gradcam, overlay_heatmap as overlay_heatmap_on_image
    from src.model import load_model, preprocess_image
    from src.referral import find_nearest_ophthalmologists, render_specialist_map
    from src.uncertainty import mc_predict_with_dud

    _MODULES_OK = True
except ImportError as _e:
    _MODULES_OK = False
    _IMPORT_ERROR = str(_e)

try:
    from streamlit_image_comparison import image_comparison

    _IMAGE_COMPARISON_OK = True
except ImportError:
    _IMAGE_COMPARISON_OK = False

GRADE_INFO = {
    0: ("No Diabetic Retinopathy", "#0F766E", "Annual screening sufficient."),
    1: ("Mild Non-Proliferative DR", "#047857", "Microaneurysms present. 12-month follow-up advised."),
    2: ("Moderate Non-Proliferative DR", "#CA8A04", "Haemorrhages or hard exudates. Ophthalmologist within 1 month."),
    3: ("Severe Non-Proliferative DR", "#EA580C", "Extensive haemorrhages, IRMA. Urgent ophthalmology."),
    4: ("Proliferative DR", "#B91C1C", "Neovascularisation or vitreous haemorrhage. Emergency referral."),
}

# Color-blind–friendly grade scale (approx. viridis / plasma for discrete classes)
GRADE_CHART_COLORS = ["#3B4994", "#3C5AB4", "#4A90A4", "#73B879", "#CFCF1A"]

URGENCY_INFO = {
    "CLEAR": ("#0F766E", "No referral required. Maintain annual diabetic eye screening."),
    "MONITOR": ("#047857", "Early findings detected. Repeat examination in 6 months."),
    "ROUTINE": ("#CA8A04", "Ophthalmology consultation recommended within 4 weeks."),
    "PRIORITY": ("#EA580C", "Significant findings trending toward severity. Consult within 1 week."),
    "URGENT": ("#B91C1C", "Sight-threatening findings. Same-day or next-day specialist review required."),
}


def _grade_quick_ref_html() -> str:
    """ETDRS quick reference with severity-colored rows (see .rs-grade-ref-table in style.css)."""
    short = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
    actions = {
        0: "Annual screen",
        1: "6-month review",
        2: "Refer < 4 wk",
        3: "Refer < 1 wk",
        4: "Emergency",
    }
    rows = []
    for g in range(5):
        color = GRADE_INFO[g][1]
        rows.append(
            f'<tr data-grade="{g}">'
            f'<td><span class="rs-grade-pill" style="background:linear-gradient(145deg,{color},#0f172a);">{g}</span></td>'
            f"<td>{short[g]}</td>"
            f"<td>{actions[g]}</td>"
            f"</tr>"
        )
    return (
        '<p class="rs-grade-ref-caption">ETDRS severity ladder (quick reference)</p>'
        '<table class="rs-data-table rs-grade-ref-table" style="margin-top:0">'
        "<thead><tr><th>Grade</th><th>Class</th><th>Action</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _ben_graham_visual(img_rgb: np.ndarray) -> np.ndarray:
    """Ben Graham preprocessing for side-by-side comparison (same op as model path, visualization only)."""
    return cv2.addWeighted(img_rgb, 4, cv2.GaussianBlur(img_rgb, (0, 0), 30), -4, 128)


def _plotly_clinical_theme() -> dict:
    """Light plot canvas for readability even when the app shell is dark."""
    paper = "#FFFFFF"
    fg = "#0F172A"
    muted = "#475569"
    grid = "rgba(15,23,42,0.09)"
    return {
        "paper_bgcolor": paper,
        "plot_bgcolor": paper,
        "font": dict(family="Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif", size=12, color=muted),
        "title_font": dict(family="Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif", size=14, color=fg),
        "margin": dict(l=48, r=28, t=52, b=48),
        "showlegend": False,
        "xaxis": dict(
            showgrid=False,
            zeroline=False,
            linewidth=0,
            linecolor=grid,
            tickfont=dict(size=11, color=muted),
            title_font=dict(color=muted, size=12),
        ),
        "yaxis": dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=grid,
            zeroline=False,
            linewidth=0,
            tickfont=dict(size=11, color=muted),
            title_font=dict(color=muted, size=12),
        ),
        "_fg": fg,
        "_muted": muted,
        "_grid": grid,
        "_paper": paper,
    }


def _clinical_radar_values(result: dict, grade: int) -> tuple[list[float], list[str]]:
    """
    Three 0-100 indices derived from ordinal grade, CAS, MC variance, and U↑.
    Interpretable as relative screening emphasis, not stand-alone diagnosis.
    """
    g = grade / 4.0
    cas = float(result["CAS"])
    tv = min(float(result["total_variance"]), 1.0)
    uu = float(result["U_up"])
    # Slightly lifted baselines so clear / low-grade cases still show a readable triangle
    micro = 18.0 + 82.0 * float(
        np.clip(0.35 * g + 0.38 * cas + 0.22 * min(uu * 3.0, 1.0) + 0.12 * tv, 0.0, 1.0)
    )
    exu = 18.0 + 82.0 * float(np.clip(0.42 * g + 0.48 * tv + 0.18 * cas, 0.0, 1.0))
    hem = 18.0 + 82.0 * float(
        np.clip(0.52 * g + 0.28 * (1.0 - tv) + 0.22 * min(uu * 2.5, 1.0), 0.0, 1.0)
    )
    cats = ["Microaneurysm load", "Exudate load", "Hemorrhage emphasis"]
    return [micro, exu, hem], cats


def _fig_grade_distribution(dark: bool, grade_probs: list, pred_grade: int, mc_passes: int) -> go.Figure:
    t = _plotly_clinical_theme()
    x_labels = [f"G{g}" for g in range(5)]
    accent = "#14B8A6" if dark else "#0D9488"
    fig = go.Figure(
        go.Bar(
            x=x_labels,
            y=grade_probs,
            marker=dict(color=GRADE_CHART_COLORS, line=dict(width=0)),
            text=[f"{v:.0%}" for v in grade_probs],
            textposition="outside",
            textfont=dict(size=11, color=t["_fg"], family=t["font"]["family"]),
            hovertemplate="Grade %{x}<br>p = %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Monte Carlo grade mass (mean histogram)",
        paper_bgcolor=t["paper_bgcolor"],
        plot_bgcolor=t["plot_bgcolor"],
        font=t["font"],
        title_font=t["title_font"],
        margin=t["margin"],
        showlegend=t["showlegend"],
        yaxis=dict(
            **t["yaxis"],
            tickformat=".0%",
            range=[0, max(0.08, max(grade_probs) * 1.22)],
            title=dict(text="Probability", font=dict(color=t["_muted"])),
        ),
        xaxis=dict(
            **t["xaxis"],
            title=dict(text="ETDRS class", font=dict(color=t["_muted"])),
        ),
        height=340,
    )
    x_center = x_labels[pred_grade]
    fig.add_shape(
        type="line",
        x0=x_center,
        x1=x_center,
        y0=0,
        y1=max(grade_probs) * 1.05,
        line=dict(color=accent, width=1, dash="dot"),
    )
    fig.add_annotation(
        x=x_center,
        y=max(grade_probs) * 1.12,
        text=f"Posterior mode · {x_center}",
        showarrow=False,
        font=dict(size=11, color=accent, family=t["font"]["family"]),
    )
    fig.update_annotations(dict(font=dict(size=10, color=t["_muted"])))
    return fig


def _fig_mc_histogram(dark: bool, all_grades: list, mc_passes: int, pred_grade: int) -> go.Figure:
    t = _plotly_clinical_theme()
    accent = GRADE_CHART_COLORS[pred_grade]
    fig = go.Figure(
        go.Histogram(
            x=all_grades,
            xbins=dict(start=-0.5, end=4.5, size=1),
            marker=dict(color=accent, line=dict(width=0)),
            opacity=0.92,
        )
    )
    fig.update_layout(
        title=f"Pass-level grade draws (T = {mc_passes})",
        paper_bgcolor=t["paper_bgcolor"],
        plot_bgcolor=t["plot_bgcolor"],
        font=t["font"],
        title_font=t["title_font"],
        margin=t["margin"],
        showlegend=t["showlegend"],
        xaxis=dict(
            **t["xaxis"],
            tickmode="linear",
            tick0=0,
            dtick=1,
            tickvals=[0, 1, 2, 3, 4],
            title=dict(text="Discrete grade draw", font=dict(color=t["_muted"])),
        ),
        yaxis=dict(
            **t["yaxis"],
            title=dict(text="Count", font=dict(color=t["_muted"])),
            rangemode="tozero",
        ),
        height=280,
    )
    return fig


def _fig_cas_gauge(dark: bool, cas_val: float, cas_color: str) -> go.Figure:
    t = _plotly_clinical_theme()
    fg = t["_fg"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=cas_val,
            number=dict(
                font=dict(
                    family="Inter, 'SF Pro Display', Helvetica, Arial, sans-serif",
                    size=26,
                    color=fg,
                ),
                valueformat=".3f",
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 1],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    tickfont=dict(size=10, color=t["_muted"]),
                    tickcolor=t["_grid"],
                ),
                bar=dict(color=cas_color, thickness=0.2),
                bgcolor=t["_paper"],
                borderwidth=0,
                steps=[
                    {"range": [0.0, 0.40], "color": "rgba(15,118,110,0.14)"},
                    {"range": [0.40, 0.60], "color": "rgba(202,138,4,0.12)"},
                    {"range": [0.60, 1.00], "color": "rgba(185,28,28,0.12)"},
                ],
                threshold=dict(line=dict(color=fg, width=1), thickness=0.65, value=0.5),
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor=t["paper_bgcolor"],
        plot_bgcolor=t["plot_bgcolor"],
        font=t["font"],
        title=dict(
            text="Clinical asymmetry score (CAS)",
            font=dict(family=t["title_font"]["family"], size=14, color=fg),
        ),
        margin=dict(l=24, r=24, t=56, b=24),
        height=300,
    )
    return fig


def _fig_clinical_radar(dark: bool, values: list[float], categories: list[str]) -> go.Figure:
    """Radar with explicit 120° axes so Plotly always draws a triangle (not a stray chord)."""
    t = _plotly_clinical_theme()
    fg = t["_fg"]
    muted = t["_muted"]
    grid_major = "rgba(15,23,42,0.14)"
    grid_minor = "rgba(15,23,42,0.07)"
    r = list(values) + [values[0]]
    degs = [90, 210, 330, 90]
    vtx_colors = ["#34D399", "#FBBF24", "#FB7185", "#34D399"]
    hover_lbl = list(categories) + [categories[0]]
    customdata = [[values[i % 3]] for i in range(4)]

    fig = go.Figure(
        go.Scatterpolar(
            r=r,
            theta=degs,
            thetaunit="degrees",
            mode="lines+markers",
            fill="toself",
            fillcolor="rgba(45,212,191,0.35)" if dark else "rgba(13,148,136,0.28)",
            line=dict(color="#2DD4BF" if dark else "#0D9488", width=3),
            marker=dict(size=12, color=vtx_colors, line=dict(width=1.5, color=fg)),
            hovertext=hover_lbl,
            customdata=customdata,
            hovertemplate="<b>%{hovertext}</b><br>Index: %{customdata[0]:.1f} / 100<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text="<b>Derived screening emphasis</b><br><sup>0–100 composite from grade · variance · CAS · U↑</sup>",
            font=dict(size=15, color=fg, family=t["title_font"]["family"]),
        ),
        paper_bgcolor=t["paper_bgcolor"],
        plot_bgcolor=t["plot_bgcolor"],
        font=t["font"],
        margin=dict(l=54, r=54, t=72, b=48),
        height=440,
        polar=dict(
            bgcolor=t["paper_bgcolor"],
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                tickfont=dict(size=11, color=muted),
                gridcolor=grid_major,
                gridwidth=1,
                showline=True,
                linecolor=grid_major,
                angle=90,
            ),
            angularaxis=dict(
                tickvals=[90, 210, 330],
                ticktext=categories,
                tickfont=dict(size=12, color=fg),
                linecolor=grid_major,
                gridcolor=grid_minor,
                direction="counterclockwise",
            ),
        ),
        showlegend=False,
    )
    return fig


if "rs_dark_mode" not in st.session_state:
    st.session_state.rs_dark_mode = True
if "result" not in st.session_state:
    st.session_state.result = None
if "cam_maps" not in st.session_state:
    st.session_state.cam_maps = None
if "img_np" not in st.session_state:
    st.session_state.img_np = None
if "upload_fingerprint" not in st.session_state:
    st.session_state.upload_fingerprint = None


@st.cache_resource(show_spinner=False)
def _get_model():
    return load_model("weights/best_model.pth")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p class="rs-sidebar-kicker">Display</p>',
        unsafe_allow_html=True,
    )
    st.toggle("Dark mode", key="rs_dark_mode")
    _apply_theme_overrides(bool(st.session_state.rs_dark_mode))

    st.markdown('<div class="rs-divider" style="margin:1rem 0"></div>', unsafe_allow_html=True)

    st.markdown("#### Inference", unsafe_allow_html=False)
    mc_passes = st.slider(
        "MC Dropout passes",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        help="More passes improve uncertainty calibration at the cost of latency.",
    )
    show_cam = st.toggle(
        "UM-GradCAM analysis",
        value=True,
        help="Compute uncertainty-modulated attention maps (adds inference time).",
    )

    st.markdown('<div class="rs-divider" style="margin:1rem 0"></div>', unsafe_allow_html=True)

    st.markdown("#### Referral search", unsafe_allow_html=False)
    radius_km = st.slider("Search radius (km)", min_value=5, max_value=150, value=30, step=5)

    st.markdown('<div class="rs-divider" style="margin:1rem 0"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rs-disclaimer">Research prototype only. Not approved for clinical use.</p>',
        unsafe_allow_html=True,
    )


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="rs-header">
      <div class="rs-wordmark">
        <div class="rs-wordmark-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="3.5"/>
            <path d="M2 12C2 12 6 5 12 5s10 7 10 7-4 7-10 7S2 12 2 12z" stroke-linejoin="round"/>
            <circle cx="12" cy="12" r="7" stroke-dasharray="2 3"/>
          </svg>
        </div>
        <div class="rs-wordmark-text">
          <h1 class="rs-title-hero">RetinaScreen</h1>
          <span class="rs-title-sub">Diabetic retinopathy grading system</span>
        </div>
      </div>
      <div class="rs-header-meta">
        <span class="rs-tag"><span class="rs-tag-dot"></span>EfficientNet-B4</span>
        <span class="rs-tag"><span class="rs-tag-dot"></span>ACS-CORN · MC Dropout</span>
        <span class="rs-tag"><span class="rs-tag-dot"></span>Hospital-grade UI</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not _MODULES_OK:
    st.error(
        f"**Module import error.** One or more source modules could not be loaded.\n\n"
        f"`{_IMPORT_ERROR}`\n\n"
        f"Install dependencies and ensure `src/` is on PYTHONPATH."
    )
    st.stop()

# ── Upload ───────────────────────────────────────────────────────────────────
st.markdown('<p class="rs-section-label">Input image</p>', unsafe_allow_html=True)

col_upload, col_ref = st.columns([1.35, 1], gap="large")

with col_upload:
    st.markdown('<div class="rs-upload-wrapper">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload retinal fundus photograph",
        type=["png", "jpg", "jpeg", "tiff"],
        label_visibility="collapsed",
        help="Colour fundus photographs. Recommended ≥ 1024 px.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_ref:
    st.markdown(_grade_quick_ref_html(), unsafe_allow_html=True)

if uploaded_file is not None:
    fp = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', '')}"
    if st.session_state.upload_fingerprint != fp:
        st.session_state.upload_fingerprint = fp
        st.session_state.cam_maps = None

    img_bytes = uploaded_file.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img_pil)
    st.session_state.img_np = img_np

    img_tensor = preprocess_image(img_np)

    with st.spinner("Running Monte Carlo inference…"):
        model = _get_model()
        result = mc_predict_with_dud(model, img_tensor, n_passes=mc_passes)

    st.session_state.result = result

    grade = result["final_grade"]
    g_label, g_color, g_desc = GRADE_INFO[grade]
    urgency = result["referral_urgency"]
    u_color, u_desc = URGENCY_INFO[urgency]

    st.markdown('<div class="rs-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="rs-section-label">Grading result</p>', unsafe_allow_html=True)

    confidence_pct = max(0, min(100, int((1 - min(result["total_variance"] * 5, 1)) * 100)))

    st.markdown(
        f"""
        <div class="rs-result-banner" style="--grade-color:{g_color}">
          <div class="rs-grade-numeral">{grade}</div>
          <div class="rs-grade-meta">
            <h3>{g_label}</h3>
            <p>{g_desc}</p>
          </div>
          <span class="rs-urgency-badge" style="--urgency-color:{u_color}">{urgency}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cas_val = result["CAS"]
    crps_val = result["CRPS"]
    if cas_val > 0.62:
        cas_color = "#EA580C"
    elif cas_val > 0.52:
        cas_color = "#CA8A04"
    else:
        cas_color = "#0F766E"

    crps_color = (
        "#B91C1C"
        if crps_val > 0.75
        else "#EA580C"
        if crps_val > 0.50
        else "#CA8A04"
        if crps_val > 0.30
        else "#0F766E"
    )

    st.markdown(
        f"""
        <div class="rs-metric-grid">
          <div class="rs-metric-card" style="--accent-color:{g_color}">
            <div class="rs-metric-label">DR grade</div>
            <div class="rs-metric-value" style="color:{g_color}">{grade}/4</div>
            <div class="rs-metric-sub">{g_label}</div>
          </div>
          <div class="rs-metric-card" style="--accent-color:var(--teal)">
            <div class="rs-metric-label">Confidence</div>
            <div class="rs-metric-value" style="color:var(--teal)">{confidence_pct}%</div>
            <div class="rs-metric-sub">MC variance {result['total_variance']:.4f}</div>
          </div>
          <div class="rs-metric-card" style="--accent-color:{cas_color}">
            <div class="rs-metric-label">CAS</div>
            <div class="rs-metric-value" style="color:{cas_color}">{cas_val:.3f}</div>
            <div class="rs-metric-sub">{"↑ severity skew" if cas_val > 0.55 else "symmetric" if cas_val > 0.45 else "↓ skew"}</div>
          </div>
          <div class="rs-metric-card" style="--accent-color:{crps_color}">
            <div class="rs-metric-label">CRPS</div>
            <div class="rs-metric-value" style="color:{crps_color}">{crps_val:.3f}</div>
            <div class="rs-metric-sub">Composite referral index</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alert_class = (
        "rs-alert-urgent"
        if urgency in ("URGENT", "PRIORITY")
        else "rs-alert-warning"
        if urgency == "ROUTINE"
        else "rs-alert-info"
        if urgency == "MONITOR"
        else "rs-alert-success"
    )

    st.markdown(
        f"""
        <div class="rs-alert {alert_class}">
          <span class="rs-alert-marker">{urgency}</span>
          <span>{u_desc}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rs-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="rs-section-label">Clinical review</p>', unsafe_allow_html=True)

    dark = bool(st.session_state.rs_dark_mode)
    review_l, review_r = st.columns([1.2, 1], gap="large")

    img_cmp = cv2.resize(img_np, (640, 640), interpolation=cv2.INTER_AREA)
    ben_cmp = _ben_graham_visual(img_cmp)

    with review_l:
        if _IMAGE_COMPARISON_OK:
            image_comparison(
                img1=img_cmp,
                img2=ben_cmp,
                label1="Raw fundus",
                label2="Ben Graham preprocessing",
                width=640,
                starting_position=50,
                make_responsive=True,
                in_memory=True,
            )
        else:
            alpha = st.slider("Blend toward clinical preprocessing", 0.0, 1.0, 0.5, 0.01)
            blend = cv2.addWeighted(img_cmp, 1.0 - alpha, ben_cmp, alpha, 0)
            _st_image(blend)
            st.caption("Install `streamlit-image-comparison` for draggable comparison.")

    rad_vals, rad_cats = _clinical_radar_values(result, grade)
    with review_r:
        st.plotly_chart(
            _fig_clinical_radar(dark, rad_vals, rad_cats),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        mcols = st.columns(3)
        for i, mc in enumerate(mcols):
            with mc:
                st.metric(rad_cats[i], f"{rad_vals[i]:.0f}")
        st.caption(
            "Larger area = more model emphasis on that clinical pattern given grade, variance, CAS, and upward uncertainty. "
            "Open **Attention maps** for spatial UM-GradCAM."
        )

    pdf_overlay = None
    if show_cam:
        if st.session_state.cam_maps is None:
            with st.spinner("Computing UM-GradCAM…"):
                cam_maps = compute_um_gradcam(model, img_tensor, grade, n_passes=10)
            st.session_state.cam_maps = cam_maps
        else:
            cam_maps = st.session_state.cam_maps
        disp_pdf = cv2.resize(img_np, (380, 380), interpolation=cv2.INTER_AREA)
        pdf_overlay = heatmap_overlay_or_none(
            disp_pdf, cam_maps["mean_attention"], overlay_heatmap_on_image
        )
    else:
        st.session_state.cam_maps = None
        cam_maps = None

    try:
        pdf_bytes = build_clinical_pdf_bytes(
            fundus_rgb=img_np,
            heatmap_overlay_rgb=pdf_overlay,
            grade=grade,
            grade_label=g_label,
            urgency=urgency,
            cas=float(cas_val),
            crps=float(crps_val),
            u_up=float(result["U_up"]),
            u_down=float(result["U_down"]),
            total_variance=float(result["total_variance"]),
            mean_grade=float(result["mean_grade"]),
            mc_passes=mc_passes,
        )
        if isinstance(pdf_bytes, bytearray):
            pdf_bytes = bytes(pdf_bytes)
        st.download_button(
            label="Download clinical PDF report",
            key="rs_clinical_pdf",
            data=pdf_bytes,
            file_name=f"RetinaScreen_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=False,
        )
    except Exception as ex:
        st.warning(f"PDF export unavailable: {ex}")

    st.markdown('<div class="rs-divider"></div>', unsafe_allow_html=True)

    tab_cam, tab_dud, tab_dist, tab_ref = st.tabs(
        ["Attention maps", "Uncertainty decomposition", "MC distribution", "Specialist referral"]
    )

    with tab_cam:
        if show_cam:
            m = st.session_state.cam_maps
            if m is None:
                with st.spinner("Computing UM-GradCAM…"):
                    m = compute_um_gradcam(model, img_tensor, grade, n_passes=10)
                st.session_state.cam_maps = m

            img_display = cv2.resize(img_np, (380, 380), interpolation=cv2.INTER_AREA)

            def _overlay(heatmap, cmap=cv2.COLORMAP_TURBO):
                return overlay_heatmap_on_image(img_display, heatmap, colormap=cmap, alpha=0.42)

            c1, c2, c3, c4 = st.columns(4, gap="small")

            with c1:
                st.markdown(
                    '<div class="rs-cam-header"><span>Source</span></div>',
                    unsafe_allow_html=True,
                )
                _st_image(img_display)
                st.markdown('<p class="rs-cam-caption">Original fundus</p>', unsafe_allow_html=True)

            with c2:
                st.markdown(
                    '<div class="rs-cam-header"><span>Mean attention</span></div>',
                    unsafe_allow_html=True,
                )
                _st_image(_overlay(m["mean_attention"]))
                st.markdown(
                    '<p class="rs-cam-caption">Spatial attention (T=10)</p>',
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    '<div class="rs-cam-header"><span>Attention variance</span></div>',
                    unsafe_allow_html=True,
                )
                _st_image(_overlay(m["uncertainty_map"], cmap=cv2.COLORMAP_VIRIDIS))
                st.markdown(
                    '<p class="rs-cam-caption">Pass-to-pass fluctuation</p>',
                    unsafe_allow_html=True,
                )

            with c4:
                st.markdown(
                    '<div class="rs-cam-header"><span>Certain attention</span></div>',
                    unsafe_allow_html=True,
                )
                _st_image(_overlay(m["certain_attention"]))
                st.markdown(
                    '<p class="rs-cam-caption">Variance-downweighted signal</p>',
                    unsafe_allow_html=True,
                )

            st.caption(
                "UM-GradCAM — uncertain attention is down-weighted when forming the certain-attention map."
            )
        else:
            st.info("Enable UM-GradCAM in the sidebar to view attention maps.")

    with tab_dud:
        col_gauge, col_table = st.columns([1, 1], gap="large")

        with col_gauge:
            st.plotly_chart(
                _fig_cas_gauge(dark, cas_val, cas_color),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with col_table:
            st.markdown(
                f"""
                <table class="rs-data-table" style="margin-top:0.5rem">
                  <thead>
                    <tr><th>Parameter</th><th>Symbol</th><th style="text-align:right">Value</th></tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Mean grade (continuous)</td>
                      <td>g&#x0305;</td>
                      <td class="rs-val">{result['mean_grade']:.3f}</td>
                    </tr>
                    <tr>
                      <td>Total MC variance</td>
                      <td>Var</td>
                      <td class="{"rs-val-danger" if result["total_variance"] > 0.5 else "rs-val"}">{result['total_variance']:.4f}</td>
                    </tr>
                    <tr>
                      <td>Upward uncertainty</td>
                      <td>U&#x2191;</td>
                      <td class="{"rs-val-warn" if result["U_up"] > result["U_down"] else "rs-val"}">{result['U_up']:.4f}</td>
                    </tr>
                    <tr>
                      <td>Downward uncertainty</td>
                      <td>U&#x2193;</td>
                      <td class="rs-val">{result['U_down']:.4f}</td>
                    </tr>
                    <tr>
                      <td>Clinical asymmetry score</td>
                      <td>CAS</td>
                      <td class="{"rs-val-danger" if cas_val > 0.62 else "rs-val-warn" if cas_val > 0.52 else "rs-val"}">{cas_val:.4f}</td>
                    </tr>
                    <tr>
                      <td>Composite referral priority</td>
                      <td>CRPS</td>
                      <td class="{"rs-val-danger" if crps_val > 0.5 else "rs-val-warn" if crps_val > 0.3 else "rs-val"}">{crps_val:.4f}</td>
                    </tr>
                    <tr>
                      <td>MC passes</td>
                      <td>T</td>
                      <td class="rs-val">{mc_passes}</td>
                    </tr>
                  </tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

            if float(cas_val) > 0.60:
                interp = "Uncertainty skewed toward higher DR grades. Higher risk of under-call relative to model variance."
                interp_cls = "rs-alert-warning"
            elif float(cas_val) < 0.40:
                interp = "Uncertainty skewed toward lower DR grades. Trends conservative."
                interp_cls = "rs-alert-info"
            else:
                interp = "Directional uncertainty is near-symmetric about the mean grade draw."
                interp_cls = "rs-alert-success"

            st.markdown(
                f'<div class="rs-alert {interp_cls}" style="margin-top:1rem">'
                f'<span class="rs-alert-marker">Note</span>'
                f'<span style="font-size:0.81rem">{interp}</span></div>',
                unsafe_allow_html=True,
            )

    with tab_dist:
        all_grades = result["all_grades"]
        grade_counts = [all_grades.count(g) for g in range(5)]
        grade_probs = [c / len(all_grades) for c in grade_counts]

        st.plotly_chart(
            _fig_grade_distribution(dark, grade_probs, grade, mc_passes),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            _fig_mc_histogram(dark, all_grades, mc_passes, grade),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with tab_ref:
        if urgency not in ("ROUTINE", "PRIORITY", "URGENT"):
            st.markdown(
                f'<div class="rs-alert rs-alert-success">'
                f'<span class="rs-alert-marker">Clear</span>'
                f"<span>{u_desc}</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="rs-referral-header">
                  <h4>Specialist referral · {urgency.title()}</h4>
                  <span class="rs-urgency-badge" style="--urgency-color:{u_color}">{urgency}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="rs-alert rs-alert-{"urgent" if urgency=="URGENT" else "warning"}">'
                f'<span class="rs-alert-marker">Action</span>'
                f"<span>{u_desc}</span></div>",
                unsafe_allow_html=True,
            )

            user_city = st.text_input(
                "City / metro region",
                placeholder="e.g. Boston, London, Karachi",
                help="Geocoded with OpenStreetMap Nominatim; facilities from Overpass.",
            )

            if user_city and user_city.strip():
                with st.spinner(f"Querying OpenStreetMap near “{user_city.strip()}”…"):
                    try:
                        specialists = find_nearest_ophthalmologists(user_city.strip(), radius_km)
                    except Exception:
                        specialists = []

                if not specialists:
                    st.markdown(
                        '<div class="rs-alert rs-alert-warning">'
                        "<span class='rs-alert-marker'>Network</span>"
                        "<span>No ophthalmology-tagged facilities returned. Increase radius, try a nearby "
                        "larger city, or retry — public Overpass endpoints can throttle during peak hours.</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="rs-ref-toolbar">'
                        f'<span class="rs-section-label" style="border:none;padding:0;margin:0">Facilities</span>'
                        f'<span class="rs-count-badge">{len(specialists)} matches</span></div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown('<div class="rs-map-container">', unsafe_allow_html=True)
                    render_specialist_map(specialists, user_city, dark_map=dark)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown('<div class="rs-divider" style="margin:0.75rem 0"></div>', unsafe_allow_html=True)

                    for sp in specialists:
                        dist = float(sp.get("distance_km", 0.0))
                        phone_str = sp["phone"] or "Not listed"
                        addr = sp["address"] or "Address not available"
                        kind = sp.get("kind", "Facility")
                        st.markdown(
                            f"""
                            <div class="rs-hospital-card">
                              <div class="rs-hospital-card-main">
                                <div class="rs-hospital-title-row">
                                  <span class="rs-hospital-name">{sp['name']}</span>
                                  <span class="rs-hospital-chip">{kind}</span>
                                </div>
                                <div class="rs-hospital-meta">{dist:.1f} km · {addr}</div>
                                <div class="rs-hospital-phone">{phone_str}</div>
                              </div>
                              <a class="rs-spec-link" href="{sp['maps_url']}" target="_blank" rel="noopener noreferrer">Get directions</a>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

else:
    st.markdown(
        """
        <div class="rs-empty-state">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" aria-hidden="true"
               stroke="currentColor" stroke-width="1.2" style="opacity:0.35">
            <circle cx="12" cy="12" r="3.5"/>
            <path d="M2 12C2 12 6 5 12 5s10 7 10 7-4 7-10 7S2 12 2 12z" stroke-linejoin="round"/>
          </svg>
          <p class="rs-empty-kicker">Upload a fundus image to begin</p>
          <p class="rs-empty-copy">PNG, JPEG, or TIFF. The stack runs EfficientNet-B4 with Monte Carlo dropout (DropPath) for calibrated ordinal grades.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
