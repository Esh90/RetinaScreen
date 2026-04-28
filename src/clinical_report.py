"""
clinical_report.py — One-click clinical PDF summary (fpdf2).
ASCII-first output for built-in Helvetica; no external font assets required.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Optional

import cv2
import numpy as np
from fpdf import FPDF


def _encode_png_rgb(rgb: np.ndarray) -> bytes:
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode image for PDF")
    return buf.tobytes()


class _ClinicalPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(15, 23, 42)
        self.cell(0, 8, "RetinaScreen - Clinical Summary", ln=True)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(71, 85, 105)
        self.cell(0, 5, "Research prototype. Not for diagnostic or treatment use.", ln=True)
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(148, 163, 184)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def build_clinical_pdf_bytes(
    fundus_rgb: np.ndarray,
    heatmap_overlay_rgb: Optional[np.ndarray],
    grade: int,
    grade_label: str,
    urgency: str,
    cas: float,
    crps: float,
    u_up: float,
    u_down: float,
    total_variance: float,
    mean_grade: float,
    mc_passes: int,
) -> bytes:
    """
    Build a single-page PDF with fundus thumbnail, optional UM-GradCAM overlay,
    and structured clinical metrics.
    """
    pdf = _ClinicalPDF(format="Letter", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)

    # Thumbnails (side by side)
    max_w = 82
    h_img = 62
    x0, y0 = pdf.get_x(), pdf.get_y()

    fundus_small = cv2.resize(fundus_rgb, (380, 380), interpolation=cv2.INTER_AREA)
    png_f = _encode_png_rgb(fundus_small)
    pdf.image(BytesIO(png_f), x=x0, y=y0, w=max_w, h=h_img)

    if heatmap_overlay_rgb is not None:
        hm = cv2.resize(heatmap_overlay_rgb, (380, 380), interpolation=cv2.INTER_AREA)
        png_h = _encode_png_rgb(hm)
        pdf.image(BytesIO(png_h), x=x0 + max_w + 10, y=y0, w=max_w, h=h_img)
    else:
        pdf.set_xy(x0 + max_w + 10, y0 + 8)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.multi_cell(max_w, 4, "UM-GradCAM overlay not available for this export.")

    pdf.set_xy(x0, y0 + h_img + 6)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(max_w, 4, "Fundus (upload)", align="C")
    if heatmap_overlay_rgb is not None:
        pdf.set_xy(x0 + max_w + 10, y0 + h_img + 6)
        pdf.cell(max_w, 4, "UM-GradCAM (mean attention)", align="C")

    pdf.ln(12)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, f"Predicted DR grade: {grade} - {grade_label}", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 5, f"Referral urgency: {urgency}", ln=True)
    pdf.cell(0, 5, f"Clinical Asymmetry Score (CAS): {cas:.4f}", ln=True)
    pdf.cell(0, 5, f"Composite Referral Priority (CRPS): {crps:.4f}", ln=True)
    pdf.cell(0, 5, f"Mean grade (continuous): {mean_grade:.3f}", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 5, "Uncertainty decomposition (MC Dropout)", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 5, f"Total variance: {total_variance:.4f}", ln=True)
    pdf.cell(0, 5, f"U up (severity-increasing): {u_up:.4f}", ln=True)
    pdf.cell(0, 5, f"U down (severity-decreasing): {u_down:.4f}", ln=True)
    pdf.cell(0, 5, f"MC passes (T): {mc_passes}", ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(
        0,
        4,
        f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC. "
        "Findings are algorithmic outputs only; clinical correlation required.",
    )

    out = pdf.output(dest="S")
    if isinstance(out, bytearray):
        return bytes(out)
    if isinstance(out, bytes):
        return out
    return str(out).encode("latin-1")


def heatmap_overlay_or_none(
    img_display: np.ndarray,
    heatmap: Optional[np.ndarray],
    overlay_fn: Any,
) -> Optional[np.ndarray]:
    """Return RGB overlay image or None if heatmap missing."""
    if heatmap is None:
        return None
    try:
        return overlay_fn(img_display, heatmap)
    except Exception:
        return None
