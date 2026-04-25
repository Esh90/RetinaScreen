"""
app.py — RetinaScreen Clinical Interface
Production-grade Streamlit application for Diabetic Retinopathy grading.

Architecture:
  - EfficientNet-B4 backbone
  - Asymmetric Clinical Severity-Weighted CORN loss (ACS-CORN)
  - Monte Carlo Dropout for uncertainty quantification
  - Directional Uncertainty Decomposition (DUD)
  - Uncertainty-Modulated Grad-CAM (UM-GradCAM)
  - OpenStreetMap / Overpass specialist referral
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import torch
import cv2
import io
import os
from PIL import Image

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
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

# ── Inject CSS ────────────────────────────────────────────────────────────────
def _load_css() -> None:
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()


def _apply_theme_overrides(dark: bool) -> None:
        if not dark:
                return

        st.markdown(
                """
                <style>
                :root {
                    --bg-canvas:      #090E1A;
                    --bg-surface:     #0F1623;
                    --bg-sunken:      #0B1120;
                    --bg-overlay:     #141E30;
                    --border-subtle:  #1A2540;
                    --border-default: #233153;
                    --border-strong:  #2D4070;
                    --text-primary:   #F1F5F9;
                    --text-secondary: #94A3B8;
                    --text-muted:     #475569;
                    --text-inverse:   #0F172A;
                    --teal:           #14A085;
                    --teal-light:     #1BC5A8;
                    --teal-dim:       #0D7377;
                    --teal-ghost:     rgba(20, 160, 133, 0.09);
                    --teal-ghost-md:  rgba(20, 160, 133, 0.16);
                    --shadow-xs:  0 1px 2px rgba(0,0,0,0.4);
                    --shadow-sm:  0 1px 4px rgba(0,0,0,0.5), 0 1px 2px rgba(0,0,0,0.3);
                    --shadow-md:  0 4px 16px rgba(0,0,0,0.5), 0 2px 6px rgba(0,0,0,0.3);
                    --shadow-lg:  0 12px 40px rgba(0,0,0,0.6), 0 4px 12px rgba(0,0,0,0.4);
                }

                html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"], .main {
                    background-color: var(--bg-canvas) !important;
                    color: var(--text-primary) !important;
                }

                [data-testid="stSidebar"] {
                    background-color: var(--bg-surface) !important;
                    border-right: 1px solid var(--border-subtle) !important;
                }

                [data-testid="stHeader"],
                [data-testid="stToolbar"] {
                    background: transparent !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
        )

# ── Local module imports ──────────────────────────────────────────────────────
try:
    from src.model        import load_model, preprocess_image
    from src.uncertainty  import mc_predict_with_dud
    from src.gradcam      import compute_um_gradcam, overlay_heatmap as overlay_heatmap_on_image
    from src.referral     import find_nearest_ophthalmologists, render_specialist_map
    _MODULES_OK = True
except ImportError as _e:
    _MODULES_OK = False
    _IMPORT_ERROR = str(_e)

# ── Clinical constants ────────────────────────────────────────────────────────
GRADE_INFO = {
    0: ("No Diabetic Retinopathy",          "#16A34A", "Annual screening sufficient."),
    1: ("Mild Non-Proliferative DR",        "#65A30D", "Microaneurysms present. 12-month follow-up advised."),
    2: ("Moderate Non-Proliferative DR",    "#CA8A04", "Haemorrhages or hard exudates. Ophthalmologist within 1 month."),
    3: ("Severe Non-Proliferative DR",      "#EA580C", "Extensive haemorrhages, IRMA. Urgent ophthalmology."),
    4: ("Proliferative DR",                 "#DC2626", "Neovascularisation or vitreous haemorrhage. Emergency referral."),
}

URGENCY_INFO = {
    "CLEAR":    ("#16A34A", "No referral required. Maintain annual diabetic eye screening."),
    "MONITOR":  ("#65A30D", "Early findings detected. Repeat examination in 6 months."),
    "ROUTINE":  ("#CA8A04", "Ophthalmology consultation recommended within 4 weeks."),
    "PRIORITY": ("#EA580C", "Significant findings trending toward severity. Consult within 1 week."),
    "URGENT":   ("#DC2626", "Sight-threatening findings. Same-day or next-day specialist review required."),
}

# Plotly base layout — clinical journal style
def _base_plotly_layout(dark: bool) -> dict:
    bg     = "#0F1623" if dark else "#FFFFFF"
    paper  = "#0F1623" if dark else "#FFFFFF"
    grid   = "#1A2540" if dark else "#F1F5F9"
    text   = "#F1F5F9" if dark else "#0F172A"
    tick   = "#94A3B8" if dark else "#64748B"
    return dict(
        paper_bgcolor = paper,
        plot_bgcolor  = bg,
        font          = dict(family="IBM Plex Mono, monospace", size=11, color=tick),
        title_font    = dict(family="IBM Plex Sans, sans-serif", size=13, color=text),
        xaxis=dict(gridcolor=grid, linecolor=grid, zerolinecolor=grid, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=grid, linecolor=grid, zerolinecolor=grid, tickfont=dict(size=10)),
        margin=dict(l=36, r=20, t=40, b=36),
        showlegend=False,
    )


# ── Session state initialisation ──────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True          # default dark
if "result"    not in st.session_state:
    st.session_state.result = None
if "cam_maps"  not in st.session_state:
    st.session_state.cam_maps = None
if "img_np"    not in st.session_state:
    st.session_state.img_np = None

# ── Apply theme override EARLY (before any page content renders) ──────────────
_apply_theme_overrides(st.session_state.dark_mode)


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_model():
    return load_model("weights/best_model.pth")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Theme toggle ──────────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;'
        'letter-spacing:0.12em;text-transform:uppercase;color:var(--text-muted);'
        'margin-bottom:0.5rem">Display</p>',
        unsafe_allow_html=True,
    )
    dark_toggle = st.toggle(
        "Dark mode",
        value=st.session_state.dark_mode,
        key="dark_toggle",
    )
    st.session_state.dark_mode = dark_toggle

    _apply_theme_overrides(dark_toggle)

    st.markdown('<div class="rs-divider" style="margin:1rem 0"></div>', unsafe_allow_html=True)

    # ── Inference settings ────────────────────────────────────────────────────
    st.markdown("#### Inference", unsafe_allow_html=False)

    mc_passes = st.slider(
        "MC Dropout passes",
        min_value=10, max_value=50, value=20, step=5,
        help="More passes improve uncertainty calibration at the cost of latency.",
    )
    show_cam = st.toggle(
        "UM-GradCAM analysis",
        value=True,
        help="Compute uncertainty-modulated attention maps (adds ~15 s).",
    )

    st.markdown('<div class="rs-divider" style="margin:1rem 0"></div>', unsafe_allow_html=True)

    # ── Referral settings ─────────────────────────────────────────────────────
    st.markdown("#### Referral search", unsafe_allow_html=False)

    radius_km = st.slider(
        "Search radius (km)",
        min_value=5, max_value=150, value=30, step=5,
    )

    st.markdown('<div class="rs-divider" style="margin:1rem 0"></div>', unsafe_allow_html=True)

    # ── System information ─────────────────────────────────────────────────────
    st.markdown("#### System", unsafe_allow_html=False)

    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;'
        f'color:var(--text-secondary)">'
        f'Compute : <span style="color:var(--teal)">{device_str}</span><br>'
        f'Model   : <span style="color:var(--teal)">EfficientNet-B4</span><br>'
        f'Loss    : <span style="color:var(--teal)">ACS-CORN</span></p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rs-divider" style="margin:1rem 0"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;'
        'color:var(--text-muted);line-height:1.6">'
        'Research prototype only.<br>Not approved for clinical use.<br>'
        'Do not use for diagnosis.</p>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="rs-header">
      <div class="rs-wordmark">
        <div class="rs-wordmark-icon">
          <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="3.5"/>
            <path d="M2 12C2 12 6 5 12 5s10 7 10 7-4 7-10 7S2 12 2 12z" stroke-linejoin="round"/>
            <circle cx="12" cy="12" r="7" stroke-dasharray="2 3"/>
          </svg>
        </div>
        <div class="rs-wordmark-text">
          <h1>RetinaScreen</h1>
          <span>Diabetic Retinopathy Grading System</span>
        </div>
      </div>
      <div class="rs-header-meta">
        <span class="rs-tag"><span class="rs-tag-dot"></span>EfficientNet-B4</span>
        <span class="rs-tag"><span class="rs-tag-dot"></span>ACS-CORN &amp; MC Dropout</span>
        <span class="rs-tag"><span class="rs-tag-dot"></span>Research Prototype</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Module import guard ───────────────────────────────────────────────────────
if not _MODULES_OK:
    st.error(
        f"**Module import error.** One or more source modules could not be loaded.\n\n"
        f"`{_IMPORT_ERROR}`\n\n"
        f"Ensure all dependencies are installed and `src/` modules are present."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="rs-section-label">Input image</p>', unsafe_allow_html=True)

col_upload, col_ref = st.columns([3, 1], gap="large")

with col_upload:
    st.markdown('<div class="rs-upload-wrapper">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload retinal fundus photograph",
        type=["png", "jpg", "jpeg", "tiff"],
        label_visibility="collapsed",
        help="Accepts colour fundus photographs. Recommended resolution: 1024 x 1024 px or higher.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_ref:
    st.markdown(
        """
        <table class="rs-data-table" style="margin-top:0.15rem">
          <thead>
            <tr><th>Grade</th><th>Classification</th><th>Action</th></tr>
          </thead>
          <tbody>
            <tr><td>0</td><td>No DR</td><td>Screen annually</td></tr>
            <tr><td>1</td><td>Mild</td><td>6-month review</td></tr>
            <tr><td>2</td><td>Moderate</td><td>Refer &lt; 4 wk</td></tr>
            <tr><td>3</td><td>Severe</td><td>Refer &lt; 1 wk</td></tr>
            <tr><td>4</td><td>Proliferative</td><td>Emergency</td></tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if uploaded_file is not None:

    # -- Load image --
    img_bytes = uploaded_file.read()
    img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np    = np.array(img_pil)
    st.session_state.img_np = img_np

    img_tensor = preprocess_image(img_np)

    # -- Run inference --
    with st.spinner("Running inference..."):
        model  = _get_model()
        result = mc_predict_with_dud(model, img_tensor, n_passes=mc_passes)

    st.session_state.result = result

    grade    = result["final_grade"]
    g_label, g_color, g_desc = GRADE_INFO[grade]
    urgency  = result["referral_urgency"]
    u_color, u_desc = URGENCY_INFO[urgency]

    # ──────────────────────────────────────────────────────────────────────────
    # RESULT BANNER
    # ──────────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────────
    # METRIC GRID
    # ──────────────────────────────────────────────────────────────────────────
    cas_val  = result["CAS"]
    crps_val = result["CRPS"]

    # Determine CAS colour
    if   cas_val > 0.62: cas_color = "#EA580C"
    elif cas_val > 0.52: cas_color = "#CA8A04"
    else:                cas_color = "#16A34A"

    crps_color = (
        "#DC2626" if crps_val > 0.75 else
        "#EA580C" if crps_val > 0.50 else
        "#CA8A04" if crps_val > 0.30 else
        "#16A34A"
    )

    st.markdown(
        f"""
        <div class="rs-metric-grid">
          <div class="rs-metric-card" style="--accent-color:{g_color}">
            <div class="rs-metric-label">DR Grade</div>
            <div class="rs-metric-value" style="color:{g_color}">{grade}/4</div>
            <div class="rs-metric-sub">{g_label}</div>
          </div>
          <div class="rs-metric-card" style="--accent-color:var(--teal)">
            <div class="rs-metric-label">Confidence</div>
            <div class="rs-metric-value" style="color:var(--teal)">{confidence_pct}%</div>
            <div class="rs-metric-sub">MC variance {result['total_variance']:.4f}</div>
          </div>
          <div class="rs-metric-card" style="--accent-color:{cas_color}">
            <div class="rs-metric-label">CAS Score</div>
            <div class="rs-metric-value" style="color:{cas_color}">{cas_val:.3f}</div>
            <div class="rs-metric-sub">
              {"Trending to higher severity" if cas_val > 0.55 else
               "Symmetric uncertainty"        if cas_val > 0.45 else
               "Trending to lower severity"}
            </div>
          </div>
          <div class="rs-metric-card" style="--accent-color:{crps_color}">
            <div class="rs-metric-label">CRPS</div>
            <div class="rs-metric-value" style="color:{crps_color}">{crps_val:.3f}</div>
            <div class="rs-metric-sub">Composite referral priority</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # CLINICAL ALERT
    # ──────────────────────────────────────────────────────────────────────────
    alert_class = (
        "rs-alert-urgent"  if urgency in ("URGENT", "PRIORITY") else
        "rs-alert-warning" if urgency == "ROUTINE" else
        "rs-alert-info"    if urgency == "MONITOR" else
        "rs-alert-success"
    )
    alert_marker = urgency

    st.markdown(
        f"""
        <div class="rs-alert {alert_class}">
          <span class="rs-alert-marker">{alert_marker}</span>
          <span>{u_desc}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rs-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════════
    tab_cam, tab_dud, tab_dist, tab_ref = st.tabs([
        "Attention Maps",
        "Uncertainty Decomposition",
        "Grade Distribution",
        "Specialist Referral",
    ])

    # ── TAB 1: UM-GradCAM ─────────────────────────────────────────────────────
    with tab_cam:
        if show_cam:
            if st.session_state.cam_maps is None:
                with st.spinner("Computing uncertainty-modulated attention maps..."):
                    cam_maps = compute_um_gradcam(model, img_tensor, grade, n_passes=10)
                st.session_state.cam_maps = cam_maps
            else:
                cam_maps = st.session_state.cam_maps

            img_display = cv2.resize(img_np, (380, 380))

            def _overlay(heatmap, cmap=cv2.COLORMAP_JET):
                return overlay_heatmap_on_image(img_display, heatmap, colormap=cmap, alpha=0.42)

            c1, c2, c3, c4 = st.columns(4, gap="small")

            with c1:
                st.markdown(
                    '<div class="rs-cam-panel"><div class="rs-cam-header">'
                    '<span>Source</span></div><div class="rs-cam-body">',
                    unsafe_allow_html=True,
                )
                st.image(img_display, use_column_width=True)
                st.markdown(
                    '<p class="rs-cam-caption">Original fundus image</p>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    '<div class="rs-cam-panel"><div class="rs-cam-header">'
                    '<span>Mean attention</span></div><div class="rs-cam-body">',
                    unsafe_allow_html=True,
                )
                st.image(_overlay(cam_maps["mean_attention"]), use_column_width=True)
                st.markdown(
                    '<p class="rs-cam-caption">Reliable spatial attention (T=10 passes)</p>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    '<div class="rs-cam-panel"><div class="rs-cam-header">'
                    '<span>Attention variance</span></div><div class="rs-cam-body">',
                    unsafe_allow_html=True,
                )
                st.image(
                    _overlay(cam_maps["uncertainty_map"], cmap=cv2.COLORMAP_COOL),
                    use_column_width=True,
                )
                st.markdown(
                    '<p class="rs-cam-caption">Where attention fluctuates across passes</p>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

            with c4:
                st.markdown(
                    '<div class="rs-cam-panel"><div class="rs-cam-header">'
                    '<span>Certain attention</span></div><div class="rs-cam-body">',
                    unsafe_allow_html=True,
                )
                st.image(_overlay(cam_maps["certain_attention"]), use_column_width=True)
                st.markdown(
                    '<p class="rs-cam-caption">Confident lesion-driving regions</p>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<p style="font-size:0.75rem;color:var(--text-muted);margin-top:0.75rem;'
                'font-family:\'IBM Plex Mono\',monospace">'
                'UM-GradCAM — uncertainty-modulated gradient-weighted class activation mapping. '
                'Certain-attention = mean activation weighted by (1 - normalised variance).</p>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Enable UM-GradCAM analysis in the sidebar to view attention maps.")

    # ── TAB 2: DUD ───────────────────────────────────────────────────────────
    with tab_dud:
        import plotly.graph_objects as go

        dark = st.session_state.dark_mode
        layout = _base_plotly_layout(dark)

        col_gauge, col_table = st.columns([1, 1], gap="large")

        with col_gauge:
            # CAS Gauge — styled as clinical instrument dial
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cas_val,
                domain={"x": [0, 1], "y": [0, 1]},
                number={
                    "font": {"family": "IBM Plex Mono", "size": 28,
                             "color": "#F1F5F9" if dark else "#0F172A"},
                    "valueformat": ".3f",
                },
                gauge={
                    "axis": {
                        "range": [0, 1],
                        "tickvals": [0, 0.25, 0.5, 0.75, 1.0],
                        "ticktext": ["0", "0.25", "0.50", "0.75", "1.0"],
                        "tickfont": {"size": 9, "family": "IBM Plex Mono",
                                     "color": "#94A3B8" if dark else "#64748B"},
                        "tickcolor": "#1A2540" if dark else "#E2E8F0",
                    },
                    "bar": {"color": cas_color, "thickness": 0.22},
                    "bgcolor": "#1A2540" if dark else "#F1F5F9",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0.00, 0.40], "color": "rgba(22,163,74,0.12)"},
                        {"range": [0.40, 0.60], "color": "rgba(202,138,4,0.12)"},
                        {"range": [0.60, 1.00], "color": "rgba(220,38,38,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "#94A3B8", "width": 1.5},
                        "thickness": 0.6,
                        "value": 0.5,
                    },
                },
            ))
            fig_gauge.update_layout(
                **layout,
                title=dict(
                    text="Clinical Asymmetry Score (CAS)",
                    font=dict(family="IBM Plex Sans", size=12,
                              color="#F1F5F9" if dark else "#0F172A"),
                ),
                height=280,
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

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
                      <td>&sigma;&sup2;</td>
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
                      <td>Clinical Asymmetry Score</td>
                      <td>CAS</td>
                      <td class="{"rs-val-danger" if cas_val > 0.62 else "rs-val-warn" if cas_val > 0.52 else "rs-val"}">{cas_val:.4f}</td>
                    </tr>
                    <tr>
                      <td>Composite Referral Priority</td>
                      <td>CRPS</td>
                      <td class="{"rs-val-danger" if crps_val > 0.5 else "rs-val-warn" if crps_val > 0.3 else "rs-val"}">{crps_val:.4f}</td>
                    </tr>
                    <tr>
                      <td>MC passes used</td>
                      <td>T</td>
                      <td class="rs-val">{mc_passes}</td>
                    </tr>
                  </tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

            # DUD interpretation note
            if cas_val > 0.60:
                interp = "Uncertainty skewed toward higher DR grades. Model is more likely under-grading than over-grading."
                interp_cls = "rs-alert-warning"
                interp_marker = "Note"
            elif cas_val < 0.40:
                interp = "Uncertainty skewed toward lower DR grades. Prediction may lean conservative."
                interp_cls = "rs-alert-info"
                interp_marker = "Note"
            else:
                interp = "Directional uncertainty is approximately symmetric around the predicted grade."
                interp_cls = "rs-alert-success"
                interp_marker = "Note"

            st.markdown(
                f'<div class="rs-alert {interp_cls}" style="margin-top:1rem">'
                f'<span class="rs-alert-marker">{interp_marker}</span>'
                f'<span style="font-size:0.81rem">{interp}</span></div>',
                unsafe_allow_html=True,
            )

    # ── TAB 3: Grade Distribution ────────────────────────────────────────────
    with tab_dist:
        import plotly.graph_objects as go

        dark   = st.session_state.dark_mode
        layout = _base_plotly_layout(dark)

        all_grades   = result["all_grades"]
        grade_counts = [all_grades.count(g) for g in range(5)]
        grade_probs  = [c / len(all_grades) for c in grade_counts]
        bar_colors   = [GRADE_INFO[g][1] for g in range(5)]

        fig_bar = go.Figure(go.Bar(
            x=[f"Grade {g}" for g in range(5)],
            y=grade_probs,
            marker=dict(
                color=bar_colors,
                line=dict(width=0),
            ),
            text=[f"{v:.0%}" for v in grade_probs],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10,
                          color="#F1F5F9" if dark else "#0F172A"),
        ))

        fig_bar.update_layout(
            title="Grade probability distribution across MC passes",
            paper_bgcolor=layout["paper_bgcolor"],
            plot_bgcolor=layout["plot_bgcolor"],
            font=layout["font"],
            title_font=layout["title_font"],
            margin=layout["margin"],
            showlegend=layout["showlegend"],
            yaxis=dict(
                tickformat=".0%",
                range=[0, max(grade_probs) * 1.25],
                gridcolor=layout["yaxis"]["gridcolor"],
                linecolor=layout["yaxis"]["linecolor"],
                zerolinecolor=layout["yaxis"]["zerolinecolor"],
                tickfont=layout["yaxis"]["tickfont"],
            ),
            xaxis=layout["xaxis"],
            height=320,
        )
        # Highlight predicted grade
        fig_bar.add_vline(
            x=grade,
            line_dash="dot",
            line_color="#94A3B8",
            line_width=1.2,
            annotation_text=f"Prediction: Grade {grade}",
            annotation_font=dict(size=10, family="IBM Plex Mono",
                                 color="#94A3B8"),
            annotation_position="top right",
        )

        st.plotly_chart(fig_bar, use_container_width=True,
                        config={"displayModeBar": False})

        # Distribution histogram (raw pass counts)
        fig_hist = go.Figure(go.Histogram(
            x=all_grades,
            nbinsx=5,
            marker=dict(
                color=GRADE_INFO[result["final_grade"]][1],
                opacity=0.75,
                line=dict(width=0),
            ),
            xbins=dict(start=-0.5, end=4.5, size=1),
        ))
        fig_hist.update_layout(
            title=f"Raw grade frequency — {mc_passes} MC Dropout passes",
            paper_bgcolor=layout["paper_bgcolor"],
            plot_bgcolor=layout["plot_bgcolor"],
            font=layout["font"],
            title_font=layout["title_font"],
            margin=layout["margin"],
            showlegend=layout["showlegend"],
            xaxis=dict(
                tickvals=[0, 1, 2, 3, 4],
                ticktext=["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"],
                gridcolor=layout["xaxis"]["gridcolor"],
                linecolor=layout["xaxis"]["linecolor"],
                zerolinecolor=layout["xaxis"]["zerolinecolor"],
                tickfont=layout["xaxis"]["tickfont"],
            ),
            yaxis=dict(**layout["yaxis"], title="Count"),
            height=260,
        )
        st.plotly_chart(fig_hist, use_container_width=True,
                        config={"displayModeBar": False})

    # ── TAB 4: Specialist Referral ───────────────────────────────────────────
    with tab_ref:

        if urgency in ("ROUTINE", "PRIORITY", "URGENT"):

            st.markdown(
                f"""
                <div class="rs-referral-header">
                  <h4>Specialist referral — {urgency.title()} priority</h4>
                  <span class="rs-urgency-badge" style="--urgency-color:{u_color}">{urgency}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="rs-alert rs-alert-{"urgent" if urgency=="URGENT" else "warning"}">'
                f'<span class="rs-alert-marker">Action</span>'
                f'<span>{u_desc}</span></div>',
                unsafe_allow_html=True,
            )

            user_city = st.text_input(
                "City or region for facility search",
                placeholder="e.g. Karachi, Lahore, Mumbai, Nairobi",
                help="Enter the patient's city to locate nearby ophthalmology services.",
            )

            if user_city and user_city.strip():
                with st.spinner(f"Searching for facilities near {user_city}..."):
                    try:
                        specialists = find_nearest_ophthalmologists(
                            user_city.strip(), radius_km
                        )
                    except Exception:
                        specialists = []

                if not specialists:
                    st.markdown(
                        '<div class="rs-alert rs-alert-warning">'
                        '<span class="rs-alert-marker">Info</span>'
                        '<span>Unable to locate facilities via the mapping service, or no results '
                        'found within the specified radius. Please try a larger nearby city, or '
                        'consult your local ophthalmology directory.</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Count badge
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem">'
                        f'<p class="rs-section-label" style="margin:0">Facilities found</p>'
                        f'<span class="rs-count-badge">{len(specialists)} results</span></div>',
                        unsafe_allow_html=True,
                    )

                    # Map
                    st.markdown('<div class="rs-map-container">', unsafe_allow_html=True)
                    render_specialist_map(specialists, user_city)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="rs-divider" style="margin:1rem 0"></div>',
                                unsafe_allow_html=True)

                    # Facility cards
                    for sp in specialists:
                        phone_str   = sp["phone"]   if sp["phone"]   != "Not listed" else "Not listed"
                        hours_str   = sp["hours"]   if sp["hours"]   != "Not listed" else "Not listed"
                        address_str = sp["address"] if sp["address"] != "Address not available" else "Address not available"

                        st.markdown(
                            f"""
                            <div class="rs-spec-card">
                              <div>
                                <div class="rs-spec-name">{sp['name']}</div>
                                <div class="rs-spec-detail">
                                  {address_str}<br>
                                  Tel: {phone_str}<br>
                                  Hours: {hours_str}
                                </div>
                              </div>
                              <a class="rs-spec-link"
                                 href="{sp['maps_url']}"
                                 target="_blank"
                                 rel="noopener noreferrer">
                                Get directions
                              </a>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        else:
            st.markdown(
                f'<div class="rs-alert rs-alert-success">'
                f'<span class="rs-alert-marker">Clear</span>'
                f'<span>{u_desc}</span></div>',
                unsafe_allow_html=True,
            )

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
          display:flex;
          flex-direction:column;
          align-items:center;
          justify-content:center;
          min-height:260px;
          color:var(--text-muted);
          text-align:center;
          gap:0.85rem;
          border:1px dashed var(--border-subtle);
          border-radius:var(--radius-lg);
          padding:3rem 2rem;
          background:var(--bg-surface);
        ">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="1.2"
               xmlns="http://www.w3.org/2000/svg" style="opacity:0.35">
            <circle cx="12" cy="12" r="3.5"/>
            <path d="M2 12C2 12 6 5 12 5s10 7 10 7-4 7-10 7S2 12 2 12z" stroke-linejoin="round"/>
          </svg>
          <p style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                    letter-spacing:0.08em;color:var(--text-muted);margin:0">
            Upload a retinal fundus photograph to begin analysis
          </p>
          <p style="font-size:0.76rem;color:var(--text-muted);margin:0;max-width:380px;line-height:1.6">
            Accepted formats: PNG, JPEG, TIFF.
            The system will grade diabetic retinopathy severity (0-4),
            quantify prediction uncertainty, and generate attention maps.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )