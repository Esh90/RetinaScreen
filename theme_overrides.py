"""
Large Streamlit theme injection strings (kept out of app.py for readability).
"""

RS_THEME_DARK = """
<style class="rs-theme-skin">
html { color-scheme: dark; }
:root {
  --bg-canvas:      #0B0F19;
  --bg-surface:     #121824;
  --bg-sunken:      #0E141F;
  --bg-overlay:     #161D2E;
  --border-subtle:  #1E293B;
  --border-default: #334155;
  --border-strong:  #475569;
  --text-primary:   #E2E8F0;
  --text-secondary: #94A3B8;
  --text-muted:     #64748B;
  --text-inverse:   #0F172A;
  --teal:           #14B8A6;
  --teal-light:     #2DD4BF;
  --teal-dim:       #0F766E;
  --teal-ghost:     rgba(20, 184, 166, 0.12);
  --teal-ghost-md:  rgba(20, 184, 166, 0.2);
  --accent-blue:    #38BDF8;
  --accent-amber:   #FBBF24;
  --accent-rose:    #FB7185;
  --shadow-xs:  0 1px 2px rgba(0,0,0,0.45);
  --shadow-sm:  0 1px 4px rgba(0,0,0,0.45), 0 1px 2px rgba(0,0,0,0.35);
  --shadow-md:  0 4px 16px rgba(0,0,0,0.5), 0 2px 6px rgba(0,0,0,0.35);
  --shadow-lg:  0 12px 40px rgba(0,0,0,0.55), 0 4px 12px rgba(0,0,0,0.4);
  --font-sans: Inter, -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, "SF Mono", "Cascadia Code", monospace;
}
.stApp {
  background: radial-gradient(ellipse 85% 50% at 95% 5%, rgba(56,189,248,0.10) 0%, transparent 48%),
              radial-gradient(ellipse 70% 50% at 5% 95%, rgba(20,184,166,0.11) 0%, transparent 48%),
              linear-gradient(168deg, #0d1526 0%, #0B0F19 45%, #080d18 100%) !important;
}
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stAppViewContainer"] .main,
[data-testid="stAppViewContainer"] .main > div,
[data-testid="stAppViewContainer"] .block-container,
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
section.main,
section.main > div,
.main .block-container,
div.block-container {
  background-color: #0B0F19 !important;
}
.stApp > div {
  background: linear-gradient(168deg, #0d1526 0%, #0B0F19 45%, #080d18 100%) !important;
}
[data-testid="element-container"] { background-color: transparent !important; }
[data-testid="stVerticalBlock"],
[data-testid="column"] { background-color: transparent !important; }
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] > div > div { background-color: transparent !important; }
[data-testid="stHeader"], header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(180deg, #141c2c 0%, #121824 40%, #0E141F 100%) !important;
  border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span:not([data-baseweb]) { color: #E2E8F0 !important; }
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h2 { color: #E2E8F0 !important; }
[data-testid="stAppViewContainer"] [data-testid="element-container"]:has([data-baseweb="tabs"]) { background-color: #0B0F19 !important; }
[data-testid="stTabs"],
section.main .stTabs,
section.main .stTabs > div,
section.main [data-testid="stTabs"] > div,
section.main [data-testid="stTabs"] > div > div,
section.main [data-testid="stVerticalBlock"]:has([data-baseweb="tabs"]) {
  background-color: #0B0F19 !important;
  background-image: none !important;
}
section.main div[data-baseweb="tabs"],
section.main div[data-baseweb="tabs"] > div {
  background-color: #0B0F19 !important;
  background-image: none !important;
}
[data-baseweb="tab-panel"],
section.main [data-baseweb="tab-panel"],
[data-baseweb="tab-panel"] > div {
  background-color: #0B0F19 !important;
  color: #E2E8F0 !important;
  padding-top: 0.5rem !important;
}
[data-baseweb="tab-panel"] [data-testid="element-container"],
[data-baseweb="tab-panel"] [data-testid="stVerticalBlock"],
[data-baseweb="tab-panel"] [data-testid="stVerticalBlock"] > div { background-color: #0B0F19 !important; }
[data-baseweb="tab-panel"] [data-testid="stMarkdownContainer"] { background: transparent !important; }
[data-baseweb="tab-panel"] [data-testid="stCaptionContainer"],
[data-baseweb="tab-panel"] .stCaption,
[data-baseweb="tab-panel"] [data-testid="stCaption"] {
  color: #E2E8F0 !important;
  -webkit-text-fill-color: #E2E8F0 !important;
  opacity: 1 !important;
}
.stTabs [data-baseweb="tab-panel"] { background-color: #0B0F19 !important; color: #E2E8F0 !important; }
.stTabs [data-baseweb="tab-list"] {
  background-color: #0E141F !important;
  border: 1px solid #334155 !important;
  border-radius: 10px !important;
}
.stTabs [data-baseweb="tab"] {
  color: #E2E8F0 !important;
  background: transparent !important;
  opacity: 1 !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: #F8FAFC !important;
  background: rgba(51, 65, 85, 0.45) !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(180deg, #134e4a 0%, #0f766e 100%) !important;
  color: #ECFEFF !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.35) !important;
}

/* Tab nav shell — Streamlit 1.4x+ light wrappers */
section.main [data-testid="stTabsNavContainer"],
section.main [data-testid="stTabsNavContainer"] > *,
section.main [data-testid="stTabs"] > div:first-child,
section.main [data-testid="stTabs"] [role="tablist"] {
  background-color: #0e141f !important;
  background-image: none !important;
  color: #e2e8f0 !important;
  border-color: #334155 !important;
}
section.main [data-testid="stTabs"] button[role="tab"] {
  color: #94a3b8 !important;
  -webkit-text-fill-color: #94a3b8 !important;
}
section.main [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: #ecfeff !important;
  -webkit-text-fill-color: #ecfeff !important;
}

/* PDF export — red/maroon (works for <a> and BaseWeb button) */
[data-testid="stDownloadButton"] a,
[data-testid="stDownloadButton"] button,
[data-testid="stDownloadButton"] [data-testid="baseButton-secondary"] {
  background: linear-gradient(145deg, #7f1d1d 0%, #991b1b 45%, #b91c1c 100%) !important;
  background-color: #991b1b !important;
  color: #fff7f7 !important;
  -webkit-text-fill-color: #fff7f7 !important;
  border: 1px solid rgba(254, 202, 202, 0.45) !important;
}
[data-testid="stDownloadButton"] a:hover,
[data-testid="stDownloadButton"] button:hover {
  background: linear-gradient(145deg, #991b1b 0%, #b91c1c 50%, #dc2626 100%) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}
.stTextInput input, .stNumberInput input, textarea {
  background-color: #0E141F !important;
  color: #E2E8F0 !important;
  border-color: #334155 !important;
}
[data-testid="stMetricContainer"] {
  background: linear-gradient(135deg, #121824 0%, #0E141F 100%) !important;
  border: 1px solid #1E293B !important;
  border-radius: 10px !important;
  padding: 0.75rem !important;
}
[data-testid="stMetricValue"] { color: #2DD4BF !important; }
[data-testid="stMetricLabel"] { color: #CBD5E1 !important; }
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stCaption"],
.stCaption,
[data-testid="stImage"] figcaption {
  color: #E2E8F0 !important;
  opacity: 1 !important;
  -webkit-text-fill-color: #E2E8F0 !important;
  font-size: 0.95rem !important;
}
[data-testid="stFileUploadDropzone"] {
  background-color: #1e293b !important;
  border: 2px dashed #64748B !important;
}
[data-testid="stFileUploadDropzone"]:hover {
  background-color: #243B53 !important;
  border-color: #2DD4BF !important;
}
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] label {
  color: #FFFFFF !important;
  opacity: 1 !important;
  font-size: 0.92rem !important;
}
[data-testid="stFileUploadDropzone"] button {
  background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
  color: #F8FAFC !important;
  border: 1px solid #475569 !important;
  font-weight: 600 !important;
}
[data-testid="stPlotlyChart"] {
  background: #FFFFFF !important;
  border: 1px solid #E2E8F0 !important;
  border-radius: 12px !important;
  padding: 6px 4px !important;
}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: #E2E8F0 !important; }
div[data-testid="stAlert"] { background-color: #162032 !important; border-color: #334155 !important; }
.rs-urgency-badge {
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
  color: inherit !important;
}
.rs-upload-wrapper {
  background: linear-gradient(160deg, #162032 0%, #0E141F 100%) !important;
  border: 1px solid rgba(100, 116, 139, 0.55) !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.35) !important;
  margin-top: 0.35rem !important;
  margin-right: 1.25rem !important;
}
.rs-tag {
  background: linear-gradient(135deg, rgba(30,58,78,0.9) 0%, rgba(15,118,110,0.15) 100%) !important;
  border-color: rgba(45,212,191,0.35) !important;
  color: #5EEAD4 !important;
}
.rs-result-banner {
  background: linear-gradient(135deg, #121824 0%, #1a2332 50%, #162032 100%) !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.35) !important;
}
.rs-metric-card {
  background: linear-gradient(180deg, #1a2332 0%, #121824 100%) !important;
  border-color: #334155 !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
}
.rs-header .rs-wordmark-text h1.rs-title-hero {
  color: #F8FAFC !important;
  font-size: clamp(2.4rem, 5vw, 4rem) !important;
  font-weight: 800 !important;
  letter-spacing: -0.04em !important;
  line-height: 1.05 !important;
  margin: 0 !important;
  text-shadow: 0 1px 3px rgba(0,0,0,0.45) !important;
}
.rs-title-sub, .rs-wordmark-text .rs-title-sub {
  color: #BAE6E0 !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.14em !important;
  margin-top: 0.35rem !important;
}
.rs-header { border-bottom-color: #334155 !important; }
.rs-wordmark-icon { width: 4.25rem !important; height: 4.25rem !important; }
.rs-wordmark-icon svg { width: 2.1rem !important; height: 2.1rem !important; }
.rs-section-label {
  color: #CBD5E1 !important;
  font-size: 0.78rem !important;
  border-bottom-color: #334155 !important;
}
.rs-section-label::before { background: #2DD4BF !important; }
.rs-sidebar-kicker { color: #CBD5E1 !important; font-size: 0.72rem !important; letter-spacing: 0.14em !important; }
.rs-disclaimer { color: #A8B8CC !important; font-size: 0.78rem !important; }
[data-testid="stSidebar"] label > div:last-child,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stSlider label p,
[data-testid="stSidebar"] .stToggle label p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-size: 1.02rem !important;
  color: #F1F5F9 !important;
  font-weight: 500 !important;
}
.rs-result-banner .rs-grade-meta h3 { color: #F1F5F9 !important; font-size: 1.15rem !important; }
[data-testid="stSlider"] { color: #F8FAFC !important; }
[data-testid="stSlider"] label,
[data-testid="stSlider"] label p,
[data-testid="stSlider"] small {
  color: #FFFFFF !important;
  opacity: 1 !important;
  -webkit-text-fill-color: #FFFFFF !important;
}
.rs-result-banner .rs-grade-meta p { color: #CBD5E1 !important; font-size: 0.95rem !important; }
.rs-metric-label { color: #94A3B8 !important; font-size: 0.72rem !important; }
.rs-metric-value { font-size: 1.85rem !important; }
.rs-metric-sub { color: #A8B8CC !important; font-size: 0.82rem !important; }
.rs-alert { color: #E2E8F0 !important; }
[data-testid="column"]:has(.rs-cam-header) {
  background: #121824 !important;
  border: 1px solid #334155 !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}
[data-testid="column"]:has(.rs-cam-header) > div { gap: 0 !important; }
[data-testid="column"]:has(.rs-cam-header) [data-testid="element-container"] { margin-bottom: 0 !important; }
[data-testid="column"]:has(.rs-cam-header) [data-testid="stImage"] { padding: 0.45rem 0.65rem 0 !important; }
.rs-cam-header {
  background: #161D2E !important;
  border-bottom: 1px solid #334155 !important;
  margin: 0 !important;
}
.rs-cam-header span {
  color: #F8FAFC !important;
  font-size: 0.74rem !important;
}
.rs-cam-caption {
  color: #E2E8F0 !important;
  border-top: 1px solid #334155 !important;
  margin: 0 !important;
  padding: 0.5rem 0.65rem 0.65rem !important;
}
.rs-grade-ref-caption { color: #94A3B8 !important; font-size: 0.72rem !important; }
.rs-grade-ref-table thead tr {
  background: linear-gradient(90deg, #115e59 0%, #0f766e 35%, #0d9488 70%, #155e75 100%) !important;
}
.rs-grade-ref-table thead th {
  color: #F8FAFC !important;
  font-size: 0.78rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.12em !important;
  padding: 0.9rem 1rem !important;
}
.rs-grade-ref-table tbody td {
  color: #F1F5F9 !important;
  font-size: 1.02rem !important;
  font-weight: 600 !important;
  text-shadow: 0 1px 2px rgba(0,0,0,0.35);
}
.rs-grade-ref-table tbody tr[data-grade="4"] td { color: #FFF7F7 !important; font-weight: 700 !important; }
.rs-empty-state {
  background: linear-gradient(165deg, #1e293b 0%, #121a2e 50%, #0f1629 100%) !important;
  border: 2px dashed #475569 !important;
  color: #F1F5F9 !important;
}
.rs-empty-kicker { color: #5EEAD4 !important; font-size: 0.95rem !important; }
.rs-empty-copy { color: #E2E8F0 !important; font-size: 1.05rem !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #F1F5F9 !important; }
section.main .rs-data-table {
  background: #121824 !important;
  border-color: #334155 !important;
}
section.main .rs-data-table th {
  background: #161D2E !important;
  color: #CBD5E1 !important;
  border-color: #334155 !important;
}
section.main .rs-data-table td {
  color: #F1F5F9 !important;
  border-color: #334155 !important;
}
[data-testid="stExpander"] {
  background: #121824 !important;
  border: 1px solid #334155 !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span { color: #F1F5F9 !important; }
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p { color: #F1F5F9 !important; }
div[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]) {
  margin-top: 1.75rem !important;
  gap: 2.5rem !important;
}
.rs-hospital-card {
  background: #0b0f19 !important;
  border: 1px solid #334155 !important;
}
.rs-hospital-name, .rs-hospital-meta, .rs-hospital-phone {
  color: #f8fafc !important;
  -webkit-text-fill-color: #f8fafc !important;
}
.rs-hospital-chip {
  color: #e2e8f0 !important;
  -webkit-text-fill-color: #e2e8f0 !important;
  background: #1e293b !important;
  border-color: #475569 !important;
}
section.main [data-baseweb="tab-panel"] .stTextInput > div > div > input {
  background-color: #0e141f !important;
  color: #f1f5f9 !important;
  -webkit-text-fill-color: #f1f5f9 !important;
  border-color: #334155 !important;
}
section.main > div { max-width: none !important; }
</style>
"""

RS_THEME_LIGHT = """
<style class="rs-theme-skin">
html { color-scheme: light; }
:root {
  --bg-canvas:      #F4FAFB;
  --bg-surface:     #FFFFFF;
  --bg-sunken:      #ECFDF5;
  --bg-overlay:     #F0FDFA;
  --border-subtle:  #E2E8F0;
  --border-default: #CBD5E1;
  --border-strong:  #94A3B8;
  --text-primary:   #0F172A;
  --text-secondary: #475569;
  --text-muted:     #64748B;
  --text-inverse:   #F8FAFC;
  --teal:           #0D9488;
  --teal-light:     #14B8A6;
  --teal-dim:       #0F766E;
  --teal-ghost:     rgba(13, 148, 136, 0.12);
  --teal-ghost-md:  rgba(13, 148, 136, 0.18);
  --accent-blue:    #0284C7;
  --accent-amber:   #CA8A04;
  --accent-rose:    #E11D48;
  --shadow-xs:  0 1px 2px rgba(15,23,42,0.05);
  --shadow-sm:  0 1px 4px rgba(15,23,42,0.07);
  --shadow-md:  0 4px 12px rgba(15,23,42,0.08);
  --shadow-lg:  0 12px 32px rgba(15,23,42,0.10);
  --font-sans: Inter, -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, "SF Mono", "Cascadia Code", monospace;
}
.stApp {
  background: linear-gradient(165deg, #ECFEFF 0%, #F8FAFC 35%, #FFFFFF 70%, #F0FDFA 100%) !important;
}
.stApp > div, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], section.main,
section.main > div, div.block-container {
  background-color: transparent !important;
  color: #0F172A !important;
}
[data-testid="stHeader"], header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(180deg, #FFFFFF 0%, #F0FDFA 100%) !important;
  border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stMetricContainer"] {
  background: #FFFFFF !important;
  border: 1px solid #E2E8F0 !important;
  border-left: 3px solid #14B8A6 !important;
  border-radius: 10px !important;
  box-shadow: 0 2px 8px rgba(13, 148, 136, 0.08) !important;
}
[data-testid="stMetricValue"] { color: #0F766E !important; }
.rs-header .rs-wordmark-text h1.rs-title-hero {
  font-size: clamp(2.4rem, 5vw, 4rem) !important;
  font-weight: 800 !important;
  color: #0C1E1B !important;
  letter-spacing: -0.04em !important;
}
.rs-title-sub { font-size: 0.95rem !important; color: #475569 !important; }
.rs-wordmark-icon { width: 4.25rem !important; height: 4.25rem !important; }
.rs-wordmark-icon svg { width: 2.1rem !important; height: 2.1rem !important; }
.rs-section-label { font-size: 0.78rem !important; }
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stSlider label p,
[data-testid="stSidebar"] .stToggle label p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 1.02rem !important; }
.rs-grade-ref-table thead tr {
  background: linear-gradient(90deg, #0f766e 0%, #0d9488 50%, #115e59 100%) !important;
}
.rs-grade-ref-table thead th {
  color: #f8fafc !important;
  font-size: 0.78rem !important;
  font-weight: 800 !important;
  background: transparent !important;
}
.rs-grade-ref-table tbody td { font-size: 1.02rem !important; font-weight: 600 !important; }
.rs-empty-kicker { font-size: 0.95rem !important; }
.rs-empty-copy { font-size: 1.05rem !important; }
[data-testid="column"]:has(.rs-cam-header) {
  background: #FFFFFF !important;
  border: 1px solid #E2E8F0 !important;
  border-radius: 10px !important;
  overflow: hidden !important;
  box-shadow: 0 1px 4px rgba(15,23,42,0.06) !important;
}
[data-testid="column"]:has(.rs-cam-header) > div { gap: 0 !important; }
[data-testid="column"]:has(.rs-cam-header) [data-testid="element-container"] { margin-bottom: 0 !important; }
[data-testid="column"]:has(.rs-cam-header) [data-testid="stImage"] { padding: 0.45rem 0.65rem 0 !important; }
.rs-cam-caption {
  margin: 0 !important;
  padding: 0.5rem 0.65rem 0.65rem !important;
  border-top: 1px solid #E2E8F0 !important;
}
div[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]) {
  margin-top: 1.75rem !important;
  gap: 2.5rem !important;
}
section.main > div { max-width: none !important; }
</style>
"""
