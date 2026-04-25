"""
RetinaScreen — Uncertainty-Aware Diabetic Retinopathy Grading
Clinical AI · Directional Uncertainty Decomposition · Triage Priority
"""
import streamlit as st

# ── MUST be first Streamlit call ──────────────────────────────────────
st.set_page_config(
    page_title="RetinaScreen · Clinical AI",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

import torch
import numpy as np
from PIL import Image
import cv2
import io
import plotly.graph_objects as go
import plotly.express as px

# ── Load CSS ──────────────────────────────────────────────────────────
def _load_css():
    with open("assets/style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()

# ── Lazy imports (avoid crashing if weights not found) ────────────────
from src.model      import load_model, preprocess_image
from src.uncertainty import mc_predict_with_dud
from src.gradcam    import compute_um_gradcam, overlay_heatmap
from src.referral   import find_nearest_ophthalmologists, render_specialist_map

# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════
GRADES = {
    0: ("No DR",           "#00E676", "0,230,118"),
    1: ("Mild DR",         "#69F0AE", "105,240,174"),
    2: ("Moderate DR",     "#FFD600", "255,214,0"),
    3: ("Severe DR",       "#FF6D00", "255,109,0"),
    4: ("Proliferative DR","#FF1744", "255,23,68"),
}

URGENCY = {
    'CLEAR':    ("#00E676", "0,230,118",  "Annual screening. No action required."),
    'MONITOR':  ("#69F0AE", "105,240,174","Early signs detected. Review in 6 months."),
    'ROUTINE':  ("#FFD600", "255,214,0",  "Ophthalmologist consultation within 1 month."),
    'PRIORITY': ("#FF6D00", "255,109,0",  "Significant findings. Consult within 1 week."),
    'URGENT':   ("#FF1744", "255,23,68",  "SAME-DAY specialist consultation required."),
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Mono, monospace', color='#7B9BB8'),
    margin=dict(t=40, b=20, l=20, r=20),
)

# ══════════════════════════════════════════════════════════════════════
# Model loading (cached)
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_model():
    """Load from local weights or HuggingFace Hub."""
    import os
    local = "weights/best_model.pth"
    if os.path.exists(local):
        return load_model(local, device='cpu')
    # Fallback: HF Hub (set HF_REPO in Streamlit secrets)
    try:
        repo = st.secrets.get("HF_REPO", "your-username/retinascreen")
        from src.model import load_model_from_hub
        return load_model_from_hub(repo, device='cpu')
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.3rem;
                  font-weight:800;color:#E8EAF6;letter-spacing:-0.02em">
        👁 RetinaScreen
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                  color:#3D5A73;letter-spacing:0.2em;text-transform:uppercase;
                  margin-top:0.3rem">
        v1.0 · Clinical AI
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="rs-sidebar-title">⚙ Inference Settings</div>',
                unsafe_allow_html=True)
    mc_passes = st.slider("MC Dropout Passes", 10, 50, 20, 5,
        help="More passes = better uncertainty estimate, ~0.5s per pass on CPU")
    show_gradcam = st.toggle("UM-GradCAM Analysis", value=True,
        help="Compute uncertainty-modulated attention maps (slower)")
    gradcam_passes = st.slider("GradCAM MC Passes", 5, 20, 10, 5,
        help="Passes for attention variance computation",
        disabled=not show_gradcam)

    st.markdown("---")
    st.markdown('<div class="rs-sidebar-title">🏥 Specialist Search</div>',
                unsafe_allow_html=True)
    radius_km = st.slider("Search Radius (km)", 5, 150, 30)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                color:#3D5A73;line-height:1.8">
      <div style="color:#7B9BB8;margin-bottom:0.5rem;
                  text-transform:uppercase;letter-spacing:0.1em;font-size:0.6rem">
        Novel Contributions
      </div>
      🧮 ACS-CORN Asymmetric Loss<br>
      🧭 DUD Directional Uncertainty<br>
      🔬 UM-GradCAM Attention Maps<br>
      📊 GURS Reliability Surface<br>
      🏥 CRPS Triage Priority Score
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#3D5A73;
                text-align:center;line-height:1.6">
      ⚠ For research purposes only.<br>
      Not a substitute for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# Hero Header
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="rs-hero">
  <div class="rs-logo-ring">
    <span style="font-size:1.8rem">👁</span>
  </div>
  <div class="rs-hero-text">
    <h1>RetinaScreen</h1>
    <p>Uncertainty-Aware Diabetic Retinopathy Grading · Directional Risk Triage</p>
  </div>
  <div class="rs-hero-badge">
    <span class="rs-badge rs-badge-cyan">EfficientNet-B4</span>
    <span class="rs-badge rs-badge-amber">ACS-CORN Loss</span>
    <span class="rs-badge rs-badge-cyan">DUD Uncertainty</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# Upload Section
# ══════════════════════════════════════════════════════════════════════
col_up, col_ref = st.columns([3, 2], gap="large")

with col_up:
    st.markdown("""
    <div class="rs-section-header">
      <div class="rs-section-dot"></div>
      <h3>Upload Fundus Photograph</h3>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop a retinal fundus image",
        type=['png', 'jpg', 'jpeg'],
        label_visibility="collapsed"
    )

with col_ref:
    st.markdown("""
    <div class="rs-section-header">
      <div class="rs-section-dot" style="background:#FFB347;box-shadow:0 0 8px #FFB347"></div>
      <h3>DR Grading Scale</h3>
    </div>
    <table class="rs-ref-table">
      <thead><tr><th>Grade</th><th>Severity</th><th>Action</th></tr></thead>
      <tbody>
        <tr><td><span style="color:#00E676">0</span></td>
            <td>No DR</td><td>Annual screen</td></tr>
        <tr><td><span style="color:#69F0AE">1</span></td>
            <td>Mild</td><td>6-month follow-up</td></tr>
        <tr><td><span style="color:#FFD600">2</span></td>
            <td>Moderate</td><td>Ophthalmologist</td></tr>
        <tr><td><span style="color:#FF6D00">3</span></td>
            <td>Severe</td><td>Urgent referral</td></tr>
        <tr><td><span style="color:#FF1744">4</span></td>
            <td>Proliferative</td><td>Same-day</td></tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# Analysis Pipeline
# ══════════════════════════════════════════════════════════════════════
if uploaded is not None:
    raw_bytes = uploaded.read()
    img_pil   = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
    img_np    = np.array(img_pil)
    img_tensor = preprocess_image(img_np)

    # ── Run inference ─────────────────────────────────────────────────
    with st.spinner(""):
        st.markdown("""
        <div style="margin:1rem 0">
          <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                      color:#00D4FF;letter-spacing:0.15em;margin-bottom:0.4rem">
            SCANNING · MC DROPOUT ACTIVE
          </div>
          <div class="rs-scan-bar"></div>
        </div>
        """, unsafe_allow_html=True)

        model  = get_model()
        result = mc_predict_with_dud(model, img_tensor,
                                      n_passes=mc_passes, device='cpu')

    grade   = result['final_grade']
    g_label, g_color, _ = GRADES[grade]
    urgency  = result['referral_urgency']
    u_color, u_rgb, u_desc = URGENCY[urgency]

    # ══════════════════════════════════════════════════════════════════
    # Grade Banner
    # ══════════════════════════════════════════════════════════════════
    confidence_pct = max(0, round((1 - min(result['total_variance'] * 3, 1)) * 100))
    cas_direction  = (
        "↑ Trending toward higher severity"
        if result['CAS'] > 0.55 else
        "↓ Trending toward lower severity"
        if result['CAS'] < 0.45 else
        "↔ Symmetric uncertainty"
    )

    st.markdown(f"""
    <div class="rs-grade-banner" style="--grade-color:{g_color}">
      <div class="rs-grade-number">{grade}</div>
      <div class="rs-grade-info">
        <h3>{g_label}</h3>
        <p>{cas_direction}</p>
      </div>
      <div class="rs-urgency" style="--urgency-color:{u_color};--urgency-rgb:{u_rgb}">
        <div class="rs-urgency-tag">{urgency}</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                    color:#7B9BB8;margin-top:0.4rem;max-width:180px;
                    text-align:center;line-height:1.4">
          {u_desc}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # Metrics Row
    # ══════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div class="rs-metrics-grid">
      <div class="rs-metric" style="--accent-color:{g_color}">
        <div class="rs-metric-label">DR Grade</div>
        <div class="rs-metric-value" style="color:{g_color}">{grade}</div>
        <div class="rs-metric-sub">{g_label}</div>
      </div>
      <div class="rs-metric" style="--accent-color:#00D4FF">
        <div class="rs-metric-label">Confidence</div>
        <div class="rs-metric-value" style="color:#00D4FF">{confidence_pct}%</div>
        <div class="rs-metric-sub">MC Dropout estimate</div>
      </div>
      <div class="rs-metric" style="--accent-color:#FFB347">
        <div class="rs-metric-label">CAS Score</div>
        <div class="rs-metric-value" style="color:#FFB347">{result['CAS']:.3f}</div>
        <div class="rs-metric-sub">Clinical Asymmetry</div>
      </div>
      <div class="rs-metric" style="--accent-color:{u_color}">
        <div class="rs-metric-label">CRPS Triage</div>
        <div class="rs-metric-value" style="color:{u_color}">{result['CRPS']:.3f}</div>
        <div class="rs-metric-sub">Priority score [0–1]</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # Main Analysis Tabs
    # ══════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 UM-GradCAM",
        "📊 DUD Analysis",
        "📈 MC Distribution",
        "🏥 Specialist Referral",
    ])

    # ── Tab 1: UM-GradCAM ─────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="rs-section-header">
          <div class="rs-section-dot"></div>
          <h3>Uncertainty-Modulated Grad-CAM</h3>
        </div>
        <p style="font-size:0.82rem;color:#7B9BB8;margin-bottom:1.2rem">
          Grad-CAM computed across multiple MC Dropout passes, decomposed into
          mean attention, attention variance, and certain-attention regions.
        </p>
        """, unsafe_allow_html=True)

        if show_gradcam:
            cam_maps = None
            with st.spinner("Computing attention maps across MC passes..."):
                try:
                    cam_maps = compute_um_gradcam(
                        model, img_tensor, grade,
                        n_passes=gradcam_passes, device='cpu'
                    )
                except Exception as e:
                    st.warning(f"UM-GradCAM could not be computed: {e}")

            if cam_maps is not None:
                img_disp = cv2.resize(img_np, (380, 380))

                c1, c2, c3, c4 = st.columns(4, gap="small")
                overlay_cfg = [
                    ("Original Fundus",       img_disp,              None),
                    ("Mean Attention",         cam_maps['mean_attention'],    cv2.COLORMAP_JET),
                    ("Attention Uncertainty",  cam_maps['uncertainty_map'],   cv2.COLORMAP_COOL),
                    ("Certain-Attention",      cam_maps['certain_attention'], cv2.COLORMAP_HOT),
                ]
                labels = [
                    "RAW INPUT",
                    "WHERE MODEL LOOKS",
                    "ATTENTION VARIANCE",
                    "CONFIDENT REGIONS",
                ]
                descs = [
                    "Original fundus photograph",
                    "Reliable spatial attention averaged across passes",
                    "Regions where attention fluctuates (possible artifacts)",
                    "High-confidence lesion regions driving the grade",
                ]

                for col, (title, data, cmap), lbl, desc in zip(
                    [c1, c2, c3, c4], overlay_cfg, labels, descs
                ):
                    with col:
                        if cmap is None:
                            st.image(data, use_column_width=True)
                        else:
                            ov = overlay_heatmap(img_disp, data, colormap=cmap)
                            st.image(ov, use_column_width=True)
                        st.markdown(f"""
                        <div class="rs-img-label">{lbl}</div>
                        <div style="font-size:0.68rem;color:#3D5A73;
                                    text-align:center;margin-top:0.2rem;
                                    line-height:1.4">{desc}</div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Enable **UM-GradCAM Analysis** in the sidebar to compute attention maps.")

    # ── Tab 2: DUD Analysis ───────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="rs-section-header">
          <div class="rs-section-dot" style="background:#FFB347;box-shadow:0 0 8px #FFB347"></div>
          <h3>Directional Uncertainty Decomposition</h3>
        </div>
        """, unsafe_allow_html=True)

        col_gauge, col_dud = st.columns([1, 1], gap="large")

        with col_gauge:
            # CAS Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['CAS'],
                number={'font': {'family': 'DM Mono', 'size': 36, 'color': '#FFB347'}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Clinical Asymmetry Score",
                       'font': {'family': 'Syne', 'size': 13, 'color': '#7B9BB8'}},
                gauge={
                    'axis': {'range': [0, 1], 'tickcolor': '#3D5A73',
                             'tickfont': {'family': 'DM Mono', 'size': 10}},
                    'bar': {'color': '#FFB347', 'thickness': 0.25},
                    'bgcolor': '#0a1628',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0.0, 0.4], 'color': '#0d2a1a'},
                        {'range': [0.4, 0.6], 'color': '#2a2000'},
                        {'range': [0.6, 1.0], 'color': '#2a0a00'},
                    ],
                    'threshold': {
                        'line': {'color': '#00D4FF', 'width': 3},
                        'thickness': 0.85,
                        'value': 0.5
                    }
                }
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=260)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Interpretation
            if result['CAS'] > 0.6:
                cas_msg = "⬆ Model uncertainty skews toward **higher severity**. Consider urgent referral."
                cas_col = "#FF6D00"
            elif result['CAS'] < 0.4:
                cas_msg = "⬇ Model uncertainty skews toward **lower severity**. Likely safe."
                cas_col = "#00E676"
            else:
                cas_msg = "↔ Uncertainty is **symmetric** around predicted grade."
                cas_col = "#FFD600"

            st.markdown(f"""
            <div style="background:#0d1e35;border:1px solid #1a3a5c;
                        border-left:3px solid {cas_col};border-radius:8px;
                        padding:0.8rem 1rem;font-size:0.8rem;color:#E8EAF6;
                        font-family:'DM Sans',sans-serif;line-height:1.5">
              {cas_msg}
            </div>
            """, unsafe_allow_html=True)

        with col_dud:
            # DUD breakdown table
            st.markdown("""
            <div class="rs-section-header" style="margin-top:0">
              <h3 style="font-size:0.85rem!important">Decomposition Components</h3>
            </div>
            """, unsafe_allow_html=True)

            rows = [
                ("U↑  Upward Variance",   result['U_up'],          "#FF6D00",
                 "Uncertainty toward higher (worse) grades"),
                ("U↓  Downward Variance", result['U_down'],         "#00E676",
                 "Uncertainty toward lower (better) grades"),
                ("σ²  Total Variance",    result['total_variance'], "#00D4FF",
                 "Standard MC Dropout scalar uncertainty"),
                ("CAS  Asymmetry Score",  result['CAS'],            "#FFB347",
                 "U↑ / (U↑ + U↓) — directional bias"),
                ("CRPS  Priority Score",  result['CRPS'],           "#E040FB",
                 "Composite triage score (grade + σ² + CAS)"),
            ]

            for key, val, color, desc in rows:
                bar_pct = min(100, val * 200) if key != "CAS  Asymmetry Score" and key != "CRPS  Priority Score" else val * 100
                st.markdown(f"""
                <div style="background:#0d1e35;border:1px solid #1a3a5c;
                            border-radius:8px;padding:0.8rem 1rem;
                            margin-bottom:0.5rem">
                  <div style="display:flex;justify-content:space-between;
                              align-items:center;margin-bottom:0.35rem">
                    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;
                                 color:#7B9BB8">{key}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.82rem;
                                 font-weight:500;color:{color}">{val:.4f}</span>
                  </div>
                  <div style="height:3px;background:#0a1628;border-radius:2px">
                    <div style="width:{bar_pct:.1f}%;height:100%;
                                background:{color};border-radius:2px;
                                box-shadow:0 0 6px {color}"></div>
                  </div>
                  <div style="font-size:0.65rem;color:#3D5A73;
                              margin-top:0.25rem;font-family:'DM Mono',monospace">
                    {desc}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # MC grades raw distribution bar chart
            g_vals = result['all_grades']
            grade_dist = [g_vals.count(i) / len(g_vals) for i in range(5)]
            g_colors = [GRADES[i][1] for i in range(5)]

            fig_bar = go.Figure(go.Bar(
                x=[f'G{i}' for i in range(5)],
                y=grade_dist,
                marker_color=g_colors,
                marker_line_width=0,
                text=[f'{v:.0%}' for v in grade_dist],
                textposition='auto',
                textfont={'family': 'DM Mono', 'size': 10},
            ))
            fig_bar.add_vline(
                x=result['mean_grade'],
                line_dash="dot", line_color="#00D4FF", line_width=2,
                annotation_text=f"μ={result['mean_grade']:.2f}",
                annotation_font={'color': '#00D4FF', 'family': 'DM Mono', 'size': 10}
            )
            fig_bar.update_layout(
                **PLOTLY_LAYOUT,
                title={'text': "MC Grade Distribution", 'font': {'size': 11}},
                yaxis_title="Probability",
                xaxis_title="DR Grade",
                height=220,
                showlegend=False,
                yaxis={'tickformat': '.0%'},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tab 3: MC Dropout Distribution ───────────────────────────────
    with tab3:
        st.markdown("""
        <div class="rs-section-header">
          <div class="rs-section-dot" style="background:#E040FB;box-shadow:0 0 8px #E040FB"></div>
          <h3>Monte Carlo Dropout Distribution</h3>
        </div>
        """, unsafe_allow_html=True)

        col_hist, col_stats = st.columns([2, 1], gap="large")

        with col_hist:
            grades_arr = np.array(result['all_grades'], dtype=float)

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=grades_arr, nbinsx=20,
                marker_color='#00D4FF',
                marker_line_color='#0a1628',
                marker_line_width=1,
                opacity=0.85,
                name='Grade Samples',
            ))
            fig_hist.add_vline(
                x=result['mean_grade'],
                line_dash="dash", line_color="#FFB347", line_width=2.5,
                annotation_text=f"Mean: {result['mean_grade']:.2f}",
                annotation_font={'color': '#FFB347', 'family': 'DM Mono', 'size': 11},
                annotation_position="top right"
            )
            fig_hist.add_vline(
                x=result['final_grade'],
                line_dash="solid", line_color="#FF1744", line_width=2,
                annotation_text=f"Final: {result['final_grade']}",
                annotation_font={'color': '#FF1744', 'family': 'DM Mono', 'size': 11},
                annotation_position="top left"
            )
            # Shade upward uncertainty
            if result['U_up'] > 0:
                fig_hist.add_vrect(
                    x0=result['mean_grade'], x1=4,
                    fillcolor="rgba(255,109,0,0.06)",
                    line_width=0,
                    annotation_text="U↑", annotation_position="top right",
                    annotation_font={'color': '#FF6D00', 'size': 9}
                )
            if result['U_down'] > 0:
                fig_hist.add_vrect(
                    x0=0, x1=result['mean_grade'],
                    fillcolor="rgba(0,230,118,0.05)",
                    line_width=0,
                    annotation_text="U↓", annotation_position="top left",
                    annotation_font={'color': '#00E676', 'size': 9}
                )

            fig_hist.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Grade Sampling Distribution ({mc_passes} MC passes)",
                xaxis={'tickvals': [0,1,2,3,4],
                       'ticktext': ['G0','G1','G2','G3','G4'],
                       'range': [-0.5, 4.5]},
                yaxis_title="Count",
                height=320,
                bargap=0.05,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_stats:
            st.markdown("""
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                        color:#3D5A73;text-transform:uppercase;letter-spacing:0.15em;
                        margin-bottom:0.8rem">
              Inference Statistics
            </div>
            """, unsafe_allow_html=True)

            stats = [
                ("Final Grade",    str(result['final_grade']),  GRADES[result['final_grade']][1]),
                ("Mean Grade",     f"{result['mean_grade']:.3f}", "#00D4FF"),
                ("Std Dev",        f"{np.std(result['all_grades']):.4f}", "#7B9BB8"),
                ("Variance σ²",    f"{result['total_variance']:.4f}", "#7B9BB8"),
                ("Min Prediction", str(int(min(result['all_grades']))), "#00E676"),
                ("Max Prediction", str(int(max(result['all_grades']))), "#FF1744"),
                ("MC Passes",      str(mc_passes), "#7B9BB8"),
                ("Certainty",      f"{confidence_pct}%", "#00D4FF"),
            ]
            for label, val, color in stats:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            padding:0.5rem 0;border-bottom:1px solid #0d1e35">
                  <span style="font-family:'DM Mono',monospace;font-size:0.72rem;
                               color:#7B9BB8">{label}</span>
                  <span style="font-family:'DM Mono',monospace;font-size:0.72rem;
                               font-weight:500;color:{color}">{val}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 4: Specialist Referral ────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="rs-section-header">
          <div class="rs-section-dot" style="background:#FF1744;box-shadow:0 0 8px #FF1744"></div>
          <h3>Specialist Referral</h3>
        </div>
        """, unsafe_allow_html=True)

        u_color_tab, _, u_desc_tab = URGENCY[urgency]

        if urgency in ('ROUTINE', 'PRIORITY', 'URGENT'):
            st.markdown(f"""
            <div style="background:#0d1e35;border:1px solid #1a3a5c;
                        border-left:4px solid {u_color_tab};border-radius:12px;
                        padding:1.2rem 1.5rem;margin-bottom:1.5rem">
              <div style="font-family:'Syne',sans-serif;font-size:1rem;
                          font-weight:700;color:{u_color_tab};margin-bottom:0.3rem">
                {urgency} — Referral Required
              </div>
              <div style="font-family:'DM Sans',sans-serif;font-size:0.85rem;
                          color:#7B9BB8">{u_desc_tab}</div>
            </div>
            """, unsafe_allow_html=True)

            city = st.text_input(
                "📍 Enter your city to find ophthalmologists",
                placeholder="e.g. Karachi, Pakistan",
                label_visibility="collapsed"
            )

            if city:
                with st.spinner(f"Finding ophthalmologists near {city}..."):
                    specialists = find_nearest_ophthalmologists(city, radius_km)

                if specialists:
                    render_specialist_map(specialists)

                    st.markdown(f"""
                    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                                color:#3D5A73;margin:1rem 0 0.5rem;
                                text-transform:uppercase;letter-spacing:0.15em">
                      Found {len(specialists)} facilities within {radius_km}km
                    </div>
                    """, unsafe_allow_html=True)

                    for sp in specialists[:6]:
                        st.markdown(f"""
                        <div class="rs-spec-card">
                          <div class="rs-spec-icon">🏥</div>
                          <div>
                            <div class="rs-spec-name">{sp['name']}</div>
                            <div class="rs-spec-detail">
                              📍 {sp['address']}<br>
                              📞 {sp['phone']} &nbsp;·&nbsp; 🕐 {sp['hours']}
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No facilities found in that area. Try increasing the search radius or use a nearby major city.")
        else:
            st.markdown(f"""
            <div style="background:#0a2a14;border:1px solid #1a5c2a;
                        border-radius:12px;padding:1.5rem;text-align:center">
              <div style="font-size:2rem;margin-bottom:0.5rem">✅</div>
              <div style="font-family:'Syne',sans-serif;font-size:1rem;
                          font-weight:700;color:#00E676;margin-bottom:0.4rem">
                No Referral Needed
              </div>
              <div style="font-family:'DM Sans',sans-serif;font-size:0.85rem;color:#69F0AE">
                {u_desc_tab}
              </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── Empty state ───────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;
                background:linear-gradient(135deg,#0a1628 0%,#050c1a 100%);
                border:1px solid #1a3a5c;border-radius:20px;margin-top:1rem">
      <div style="font-size:4rem;margin-bottom:1rem;
                  filter:drop-shadow(0 0 20px rgba(0,212,255,0.4))">👁</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.4rem;
                  font-weight:700;color:#E8EAF6;margin-bottom:0.5rem">
        Upload a fundus photograph to begin
      </div>
      <div style="font-family:'DM Sans',sans-serif;font-size:0.85rem;
                  color:#7B9BB8;max-width:480px;margin:0 auto;line-height:1.6">
        RetinaScreen will grade the image for diabetic retinopathy severity,
        decompose uncertainty into directional components, and generate a
        clinical triage priority score.
      </div>
      <div style="margin-top:2rem;display:flex;justify-content:center;gap:1rem;flex-wrap:wrap">
        <div style="background:#0d1e35;border:1px solid #1a3a5c;border-radius:8px;
                    padding:0.8rem 1.2rem;font-family:'DM Mono',monospace;
                    font-size:0.7rem;color:#00D4FF">
          PNG / JPG / JPEG
        </div>
        <div style="background:#0d1e35;border:1px solid #1a3a5c;border-radius:8px;
                    padding:0.8rem 1.2rem;font-family:'DM Mono',monospace;
                    font-size:0.7rem;color:#FFB347">
          Retinal Fundus Photos
        </div>
        <div style="background:#0d1e35;border:1px solid #1a3a5c;border-radius:8px;
                    padding:0.8rem 1.2rem;font-family:'DM Mono',monospace;
                    font-size:0.7rem;color:#00E676">
          Max 50 MB
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)