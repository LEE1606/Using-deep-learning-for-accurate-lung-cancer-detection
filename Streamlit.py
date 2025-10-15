import os
import uuid
import cv2
import sqlite3
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from xml.dom import minidom
from pathlib import Path
from io import BytesIO

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes, disk, opening
import skfuzzy as fuzz
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ===============================
# Constants & Paths
# ===============================
IMG_SIZE = (240, 240)
# In Colab, save outputs inside /content
OUTPUT_FOLDER = Path("/content/drive/MyDrive/lung_project/segmentation_results")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

DB_PATH = OUTPUT_FOLDER / "cases.db"

# ===============================
# Load classifier model
# ===============================
# Load classifier




from tensorflow.keras.models import load_model
import streamlit as st

MODEL_PATH = "/content/drive/MyDrive/lung_project/models_b1/best_B1_model_training.keras"

@st.cache_resource  # ✅ 只載入一次，之後重複使用
def load_classifier():
    return load_model(MODEL_PATH, compile=False)

classifier_model = load_classifier()



# ===============================
# Database Setup
# ===============================
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            scan_date TEXT,
            filename TEXT,
            predicted_class TEXT,
            confidence REAL,
            xml_path TEXT,
            pdf_path TEXT,
            saved_at TEXT,
            review_status TEXT,
            review_comment TEXT,
            review_like INTEGER,
            UNIQUE (patient_id, filename, scan_date)
        );
        """)

def ensure_review_columns():
    with sqlite3.connect(DB_PATH) as con:
        cols = pd.read_sql("PRAGMA table_info(cases)", con)["name"].tolist()
        if "review_status" not in cols:
            con.execute("ALTER TABLE cases ADD COLUMN review_status TEXT;")
        if "review_comment" not in cols:
            con.execute("ALTER TABLE cases ADD COLUMN review_comment TEXT;")
        if "review_like" not in cols:
            con.execute("ALTER TABLE cases ADD COLUMN review_like INTEGER;")


init_db()
ensure_review_columns()

def delete_patient_cases(patient_id: str):
    """Delete all cases (rows + files) for a patient."""
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("SELECT * FROM cases WHERE patient_id=?", con, params=(patient_id,))
        for _, row in df.iterrows():
            try:
                if row['pdf_path'] and Path(row['pdf_path']).exists():
                    Path(row['pdf_path']).unlink()
                if row['xml_path'] and Path(row['xml_path']).exists():
                    Path(row['xml_path']).unlink()
            except Exception as e:
                print(f"⚠️ Could not delete files for {row['id']}: {e}")
        con.execute("DELETE FROM cases WHERE patient_id=?", (patient_id,))

def delete_all_cases():
    """Delete all cases and their files."""
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("SELECT * FROM cases", con)
        for _, row in df.iterrows():
            try:
                if row['pdf_path'] and Path(row['pdf_path']).exists():
                    Path(row['pdf_path']).unlink()
                if row['xml_path'] and Path(row['xml_path']).exists():
                    Path(row['xml_path']).unlink()
            except Exception as e:
                print(f"⚠️ Could not delete files for {row['id']}: {e}")
        con.execute("DELETE FROM cases")

import sqlite3
import streamlit as st
import re

DB_PATH = "/content/drive/MyDrive/lung_project/segmentation_results/cases.db"

@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def get_next_patient_id():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT patient_id 
        FROM cases 
        ORDER BY CAST(SUBSTR(patient_id, 2) AS INTEGER) DESC 
        LIMIT 1
    """)
    result = cursor.fetchone()

    if result and result[0]:
        last_id = result[0]
        match = re.match(r"P(\d+)", last_id)
        if match:
            next_num = int(match.group(1)) + 1
            return f"P{next_num:03d}"
    return "P001"

# =====================================================
# Radiologist Review Function (placed outside the page)
# =====================================================
def radiologist_review(case_key, gray_img, patient_id, patient_name, scan_date,
                       uploaded_file, predicted_class, confidence,
                       xml_path, pdf_path, edited_mask_path): 
    review_state = st.session_state.get(case_key, {"status": "", "comment": "", "like": False})

    st.subheader("👨‍⚕️ Radiologist Review")
    col1, col2, col3 = st.columns(3)

    if col1.button("✅ Accept"):
        review_state["status"] = "Accepted"
    if col2.button("❌ Reject"):
        review_state["status"] = "Rejected"
    if col3.button("✏️ Edit"):
        review_state["status"] = "Edited"

    st.caption(f"Current Review Status: **{review_state['status'] or '—'}**")
    review_state["comment"] = st.text_area("📝 Radiologist Comment", value=review_state["comment"])
    review_state["like"] = st.checkbox("👍 Like this AI prediction", value=review_state["like"])
    st.session_state[case_key] = review_state

    # --- Editing Mode ---
    if review_state["status"] == "Edited":
        st.subheader("✏️ Radiologist Editing Mode")

        ct_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        colL, colR = st.columns([1, 1])
        with colL:
            st.image(ct_rgb, caption="CT (reference)", use_container_width=True)
        with colR:
            canvas_result = st_canvas(
                fill_color="rgba(255,0,0,0.3)",
                stroke_width=3,
                stroke_color="red",
                background_color="rgba(0,0,0,0)",
                update_streamlit=True,
                height=IMG_SIZE[0],
                width=IMG_SIZE[1],
                drawing_mode="freedraw",
                key="canvas_edit",
            )

        if canvas_result.image_data is not None:
            edited_mask = (canvas_result.image_data[:, :, 0] > 0).astype(np.uint8)
            if edited_mask.shape[:2] != ct_rgb.shape[:2]:
                edited_mask = cv2.resize(
                    edited_mask, (ct_rgb.shape[1], ct_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            st.image(edited_mask * 255, caption="Edited Tumor Mask (binary)", use_container_width=True)
            overlay_preview = ct_rgb.copy()
            overlay_preview[edited_mask == 1] = [255, 0, 0]
            st.image(overlay_preview, caption="Overlay Preview (CT + Edited Mask)", use_container_width=True) 

            if st.button("💾 Save Edited Mask"):
                cv2.imwrite(str(edited_mask_path), edited_mask * 255)

                edited_lesions = compute_lesion_metrics(edited_mask.astype(bool), gray_img.shape)
                st.subheader("📏 Lesion Measurements (Edited Mask)")
                accepted = []
                for r in edited_lesions:
                    keep = st.checkbox(f"Lesion {r['index']} ({r['side']} {r['lobe']})",
                                       value=True, key=f"edit_{r['index']}")
                    if keep:
                        accepted.append(r)
                    with st.expander(f"Details for lesion {r['index']}"):
                        st.json(r)

                # Regenerate AIM + PDF
                generate_aim_xml(uploaded_file.name, edited_mask, edited_lesions, xml_path)
                overlay_bgr_for_pdf = cv2.cvtColor(overlay_preview, cv2.COLOR_RGB2BGR)
                make_pdf_report(
                    patient_id, patient_name, scan_date, uploaded_file.name,
                    predicted_class, confidence, edited_lesions,
                    overlay_bgr_for_pdf, edited_mask, pdf_path
                )

                # Update DB
                with sqlite3.connect(DB_PATH) as con:
                    con.execute("""
                        INSERT OR REPLACE INTO cases (
                            patient_id, patient_name, scan_date, filename,
                            predicted_class, confidence,
                            xml_path, pdf_path, saved_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        patient_id, patient_name, scan_date, uploaded_file.name,
                        predicted_class, confidence,
                        str(xml_path), str(pdf_path), datetime.now().isoformat()
                    ))
                st.success("✅ Edited mask saved. AIM XML, PDF, and DB row updated!")

    # --- Save Review ---
    if st.button("💾 Save Radiologist Review"):
        with sqlite3.connect(DB_PATH) as con:
            con.execute("""
                UPDATE cases
                SET review_status=?, review_comment=?, review_like=?
                WHERE patient_id=? AND filename=? AND scan_date=?;
            """, (review_state["status"], review_state["comment"], int(review_state["like"]),
                  patient_id, uploaded_file.name, scan_date))
        st.success("✅ Radiologist review saved!")



# --- Custom CSS for Professional Medical Look ---

# --- Navbar ---
st.markdown("""
    <style>
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 999;
        background-color: #0077b6;
        color: white;
        padding: 14px 25px;
        font-size: 22px;
        font-weight: 600;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.3);
    }
    .navbar span {
        font-size: 16px;
        font-weight: 400;
        margin-left: 15px;
        color: #dff6ff;
    }
    .block-container {
        padding-top: 80px !important; /* push content down below navbar */
    }
    .card {
        background: #ffffff;
        padding: 25px;
        margin: 20px 0;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    </style>
    <div class="navbar">
        🫁 Using ML for Accurate Lung Cancer Detection 
        <span>| Professional Medical Dashboard</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Global background */
    .stApp {
        background: linear-gradient(to right, #f8fbff, #e6f2ff);
        font-family: "Segoe UI", "Helvetica", sans-serif;
        color: #2c3e50;
    }


    /* Add padding so content doesn’t overlap navbar */
    .block-container {
        padding-top: 70px !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #e6f2ff;
    }
    section[data-testid="stSidebar"] h1, h2, h3 {
        color: #0077b6 !important;
    }

    /* Headings */
    h1, h2, h3 {
        color: #004d80 !important;
        font-weight: 600;
    }

    /* White card panels */
    .card {
        background: #ffffff;
        padding: 20px;
        margin: 15px 0;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }

    /* Buttons */
    div.stButton > button {
        background-color: #0077b6;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 500;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #0096c7;
        color: white;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] section {
        border: 2px dashed #0077b6;
        background-color: #f0f8ff;
    }

    /* Success and info boxes */
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)



# ===============================
# FCM Segmenter
# ===============================
class FCM_Combined_Segmenter:
    def __init__(
        self,
        image: np.ndarray,
        num_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 150,
        threshold: float = 1e-5,
        tumor_cluster: int | None = None,
        confine_to_lungs: bool = True,
        target_lobe: str = "none",
        lobe_margin: float = 0.07,
        brightness_min: float = 0.35,
        min_area: int = 120,
        max_area_frac_of_lung: float = 0.35,
    ) -> None:
        self.image = image
        self.num_clusters = num_clusters
        self.m = m
        self.max_iter = max_iter
        self.threshold = threshold
        self.tumor_cluster = tumor_cluster
        self.confine_to_lungs = confine_to_lungs
        self.target_lobe = target_lobe
        self.lobe_margin = lobe_margin
        self.brightness_min = brightness_min
        self.min_area = min_area
        self.max_area_frac_of_lung = max_area_frac_of_lung

    def _normalize(self):
        img = self.image.astype(np.float32)
        mn, mx = float(img.min()), float(img.max())
        norm = (img - mn) / (mx - mn + 1e-8)
        flat = norm.reshape(-1, 1)
        return norm, flat

    def _lung_mask(self):
        _, thr = cv2.threshold(self.image, 100, 255, cv2.THRESH_BINARY_INV)
        mask = thr > 0
        mask = remove_small_objects(mask, min_size=1500)
        mask = remove_small_holes(mask, area_threshold=2000)
        return mask.astype(np.bool_)

    def _overlay_with_centroids(self, mask, centroids, color_mask=(255, 0, 0), color_dot=(0, 0, 255)):
        ct_rgb = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        tumor_layer = np.zeros_like(ct_rgb)
        tumor_layer[mask == 1] = color_mask
        overlay = cv2.addWeighted(ct_rgb, 0.7, tumor_layer, 0.5, 0)
        for (cy, cx) in centroids:
            cv2.circle(overlay, (int(cx), int(cy)), 5, color_dot, -1)
        return overlay

    def _passes_morph_heuristics(self, prop):
        return prop.eccentricity <= 0.95 and prop.solidity < 0.99

    def _passes_lobe_filter(self, centroid):
        if self.target_lobe == "none": return True
        h, w = self.image.shape[:2]
        cy, cx = centroid
        x_mid, y_mid = w * 0.5, h * 0.5
        y_upper, y_lower = y_mid * (1.0 - self.lobe_margin), y_mid * (1.0 + self.lobe_margin)
        if self.target_lobe == "left_lower": return (cx < x_mid) and (cy > y_lower)
        if self.target_lobe == "left_upper": return (cx < x_mid) and (cy < y_upper)
        if self.target_lobe == "right_lower": return (cx >= x_mid) and (cy > y_lower)
        if self.target_lobe == "right_upper": return (cx >= x_mid) and (cy < y_upper)
        return True

    def segment(self):
        np.random.seed(42)  # Fix randomness
        norm, flat = self._normalize()
        cntr, u, *_ = fuzz.cluster.cmeans(flat.T, self.num_clusters, self.m, error=self.threshold, maxiter=self.max_iter)
        lung = self._lung_mask()

        # Fix tumor cluster selection
        if self.tumor_cluster is None:
            self.tumor_cluster = int(np.argmax(cntr.ravel()))  # Always choose brightest cluster

        membership_map = u[self.tumor_cluster].reshape(self.image.shape)
        t = 0.1  # Fixed threshold instead of Otsu
        binary = (membership_map > t).astype(np.uint8)

        cleaned = remove_small_objects(binary.astype(bool), min_size=150)
        cleaned = remove_small_holes(cleaned, area_threshold=600)
        cleaned = opening(cleaned, disk(3))
        mask = cleaned.astype(np.uint8)

        if self.confine_to_lungs:
            mask = np.logical_and(mask, lung).astype(np.uint8)

        lbl = label(mask)
        out = np.zeros_like(mask)
        lung_area = int(lung.sum()) if lung.any() else mask.size
        max_area = int(self.max_area_frac_of_lung * lung_area)
        centroids = []

        for prop in regionprops(lbl):
            if not (self.min_area <= prop.area <= max_area):
                continue
            rr, cc = np.where(lbl == prop.label)
            mean_intensity = float(np.mean(norm[rr, cc]))
            if mean_intensity < self.brightness_min:
                continue
            if not self._passes_morph_heuristics(prop):
                continue
            if not self._passes_lobe_filter(prop.centroid):
                continue
            out[lbl == prop.label] = 1
            centroids.append(prop.centroid)
            print(f"Tumor centroid detected at (x={prop.centroid[1]:.1f}, y={prop.centroid[0]:.1f})")

        overlay = self._overlay_with_centroids(out, centroids)
        return self.image, membership_map, out, overlay, centroids


# ===============================
# Measurement & Utils
# ===============================
def compute_lesion_metrics(mask_bool, img_shape):
    """
    Compute lesion metrics without asking radiologist for pixel spacing/slice thickness.
    Assume 1 pixel ≈ 1 mm for this dataset.
    """
    lbl = label(mask_bool)
    lesions = []
    for i, prop in enumerate(regionprops(lbl)):
        area_mm2 = prop.area  # pixel count ≈ mm²
        minr, minc, maxr, maxc = prop.bbox
        dy = (maxr - minr)
        dx = (maxc - minc)
        bbox_diag_mm = (dx**2 + dy**2) ** 0.5
        eq_diam_mm = prop.equivalent_diameter
        diameter_mm = max(bbox_diag_mm, eq_diam_mm)
        volume_mm3 = area_mm2  # 2D slice, treat area as volume proxy

        cy, cx = prop.centroid
        side = "Right lung" if cx >= img_shape[1] / 2 else "Left lung"
        y_mid = img_shape[0] / 2
        margin = 0.07 * img_shape[0]
        if cy < y_mid - margin:
            lobe = "Upper lobe"
        elif cy > y_mid + margin:
            lobe = "Lower lobe"
        else:
            lobe = "Mid zone"

        lesions.append({
            "index": i,
            "area_mm2": area_mm2,
            "diameter_mm": diameter_mm,
            "volume_mm3": volume_mm3,
            "side": side,
            "lobe": lobe
        })
    return lesions


def save_case(patient_id, patient_name, scan_date, filename, pred_class,
              confidence, xml_path, pdf_path):
    now = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as con:
       
            con.execute("""
                INSERT OR REPLACE INTO cases (patient_id, patient_name, scan_date, filename,
                                   predicted_class, confidence, xml_path, pdf_path, saved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, patient_name, scan_date, filename, pred_class, float(confidence),
                  str(xml_path), str(pdf_path), now))

def query_cases(patient_id=None, pred_class=None, start_date=None, end_date=None):
    query = "SELECT * FROM cases WHERE 1=1"
    params = []
    if patient_id:
        query += " AND patient_id=?"
        params.append(patient_id)
    if pred_class:
        query += " AND predicted_class=?"
        params.append(pred_class)
    if start_date and end_date:
        query += " AND scan_date BETWEEN ? AND ?"
        params.extend([start_date, end_date])
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(query, con, params=params)
    return df

def make_pdf_report(patient_id, patient_name, scan_date, filename,
                    pred_class, confidence, lesions, overlay_bgr, mask, save_path):
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    ok1, png_overlay = cv2.imencode(".png", overlay_rgb)

    mask_rgb = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    ok2, png_mask = cv2.imencode(".png", mask_rgb)

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 40

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Lung Cancer AI Report")
    y -= 25
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Patient ID: {patient_id}")
    y -= 16
    c.drawString(40, y, f"Name: {patient_name}")
    y -= 16
    c.drawString(40, y, f"Scan Date: {scan_date}")
    y -= 16
    c.drawString(40, y, f"File: {filename}")
    y -= 16
    c.drawString(40, y, f"Prediction: {pred_class} ({confidence*100:.1f}%)")

    # Tumor mask and overlay
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Tumor Segmentation Results:")

    from reportlab.lib.utils import ImageReader
    y -= 20
    img_overlay = ImageReader(BytesIO(png_overlay.tobytes()))
    c.drawImage(img_overlay, 40, y - 200, width=250, height=200)

    img_mask = ImageReader(BytesIO(png_mask.tobytes()))
    c.drawImage(img_mask, 310, y - 200, width=250, height=200)

    y -= 220

    # Lesion info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Lesions:")
    y -= 14
    c.setFont("Helvetica", 10)
    for r in lesions:
        line = f"#{r['index']} | Area {r['area_mm2']:.1f} mm² | Diam {r['diameter_mm']:.1f} mm | Vol {r['volume_mm3']:.1f} mm³ | {r['side']} {r['lobe']}"
        c.drawString(40, y, line)
        y -= 12

    c.showPage()
    c.save()

    with open(save_path, "wb") as f:
        f.write(buf.getvalue())


def generate_aim_xml(image_name, mask, lesions, save_path):
    root = ET.Element("ImageAnnotation")
    root.set("name", "FCM Tumor Annotation")
    root.set("dateTime", datetime.utcnow().isoformat())

    geo_col = ET.SubElement(root, "geometricShapeCollection")
    lbl = label(mask)
    for region in regionprops(lbl):
        cy, cx = region.centroid
        geo = ET.SubElement(geo_col, "GeometricShape", {"type": "Point"})
        ET.SubElement(geo, "x").text = str(cx)
        ET.SubElement(geo, "y").text = str(cy)

    lesion_col = ET.SubElement(root, "lesionCollection")
    for r in lesions:
        lesion = ET.SubElement(lesion_col, "Lesion", {"id": str(r["index"])})
        ET.SubElement(lesion, "Area").text = f"{r['area_mm2']:.1f}"
        ET.SubElement(lesion, "Diameter").text = f"{r['diameter_mm']:.1f}"
        ET.SubElement(lesion, "Volume").text = f"{r['volume_mm3']:.1f}"
        ET.SubElement(lesion, "Side").text = r["side"]
        ET.SubElement(lesion, "Lobe").text = r["lobe"]

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(save_path, "w") as f:
        f.write(xml_str)

def cleanup_duplicates():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            DELETE FROM cases
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM cases
                GROUP BY patient_id, filename, scan_date
            )
        """)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="wide")
st.sidebar.title("🩺 Lung Cancer Detection System")
page = st.sidebar.radio("Navigation", ["🏠 Introduction", "📂 About the Dataset", "🧪 Prediction", "🗂️ Case Browser"])

# -------------------------------
# -------------------------------
if page == "🏠 Introduction":
    st.title("Introduction to Lung Cancer")
    st.image(
        "/content/drive/MyDrive/lung_project/lung cancer intro.png",
        caption="Introduction to Lung Cancer",
         use_container_width=True
    )

    # Card 1: Threat
    st.markdown(
        """
        <div class="card">
          <h3>&#128204; Lung Cancer: A Persistent Threat</h3>
          <p>
            Lung cancer is one of the <b>leading causes of cancer death worldwide</b>, 
            with an estimated <b>1.8 million deaths every year (WHO)</b>. 
            Prompt and accurate diagnosis is essential to enhance survival rates. 
            CT scans are the most effective non-invasive imaging tool 
            for detecting pulmonary nodules and abnormal growths.
          </p>
                 </div>
        """,
        unsafe_allow_html=True,
    )

         # Display image BELOW the card
    st.image(
    "cancer statistic.png",  # Make sure the image path is correct!
    use_container_width=True,
    caption="Worldwide Cancer Mortality (2020)"
)

    # Card 2: Traditional Methods
    st.markdown(
        """
        <div class="card">
          <h3>&#9888; The Challenge of Traditional Methods</h3>
          <ul>
            <li>Chest X-rays often miss early or subtle lung cancer signs.</li>
            <li>CT scans are effective but rely on <b>manual interpretation</b>.</li>
            <li>Manual review of large imaging datasets is <b>time-consuming</b> and prone to errors.</li>
            <li>Conventional CAD systems are <b>inefficient—either</b> segmenting all images or running segmentation and classification in parallel, both causing redundant computation.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Card 3: Deep Learning
    st.markdown(
        """
        <div class="card">
          <h3>&#129302; The Promise of Deep Learning</h3>
          <p>
            Advances in <b>Artificial Intelligence (AI)</b>, especially 
            <b>Deep Learning (DL)</b>, enable automated feature extraction 
            directly from CT scans.
          </p>
          <ul>
            <li><b>EfficientNetB1</b>: lightweight yet accurate classifier.</li>
            <li><b>FCM Segmentation</b>: Lightweight, unsupervised segmentation triggered only when cancer is detected, reducing computation load.</li>
            <li><b>Transfer Learning</b>: boosts performance with limited datasets.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Card 4: Types of Lung Cancer
    st.markdown(
        """
        <div class="card">
          <h3>&#128138; Types of Lung Cancer Covered</h3>
          <h4>1. Adenocarcinoma</h4>
          <ul>
            <li>Most common form (~30% of all lung cancers, ~40% of NSCLC).</li>
            <li>Found in the outer lung region, originates in mucus-secreting glands.</li>
            <li><b>Symptoms:</b> coughing, hoarseness, weight loss, weakness.</li>
          </ul>
          <h4>2. Large Cell Carcinoma</h4>
          <ul>
            <li>A fast-growing, undifferentiated carcinoma (~10-15% of NSCLC).</li>
            <li>Can develop anywhere in the lung.</li>
            <li>Often detected late due to rapid spread.</li>
          </ul>
          <h4>3. Squamous Cell Carcinoma</h4>
          <ul>
            <li>Found centrally near the bronchi and trachea (~30% of NSCLC).</li>
            <li>Strongly associated with smoking.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "📂 About the Dataset":
    st.title("About the Dataset")
    st.markdown("""
    - ✅ The dataset contains **three types of lung cancer**: Adenocarcinoma, Large Cell Carcinoma, and Squamous Cell Carcinoma.
    - ✅ A fourth class represents **normal CT scans**.
    - ✅ Images are **not in DICOM format** but in `.jpg` or `.png` to suit model training.
    - ✅ Dataset is organized into:
        - `train` (70%)
        - `valid` (10%)
        - `test` (20%)
    """)

    # Add path to your actual dataset
    base_path = "/content/drive/MyDrive/Data"
    train_dir = os.path.join(base_path, "train")
    val_dir = os.path.join(base_path, "valid")
    test_dir = os.path.join(base_path, "test")

    def GetDatasetSize(path):
        return {folder: len(os.listdir(os.path.join(path, folder))) for folder in os.listdir(path)}

    train_set = GetDatasetSize(train_dir)
    val_set = GetDatasetSize(val_dir)
    test_set = GetDatasetSize(test_dir)

    import pandas as pd
    import matplotlib.pyplot as plt

    st.subheader("📊 Dataset Distribution per Class")

    df = pd.DataFrame({
        "Train": pd.Series(train_set),
        "Validation": pd.Series(val_set),
        "Test": pd.Series(test_set)
    })

    st.dataframe(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Image Count")
    ax.set_title("Number of Images per Class (Train, Validation, Test)")
    ax.legend(loc="upper right")
    st.pyplot(fig)


elif page == "🧪 Prediction":
    from streamlit_drawable_canvas import st_canvas
    import base64

    st.title("Lung Cancer Detection & Segmentation")

    # --- Case meta ---
    if "current_patient_id" not in st.session_state:
        st.session_state.current_patient_id = get_next_patient_id()

    patient_id = st.text_input("Patient ID", value=st.session_state.current_patient_id)
    patient_name = st.text_input("Patient Name", "John Doe")
    scan_date = st.date_input("Scan Date", datetime.today()).isoformat()

    uploaded_file = st.file_uploader("Upload CT Image", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file:
        # ---------- Load & preprocess ----------
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        resized_rgb = cv2.resize(original, IMG_SIZE)
        gray_img = cv2.cvtColor(resized_rgb, cv2.COLOR_BGR2GRAY)

        img_arr = image.img_to_array(resized_rgb)
        input_arr = preprocess_input(img_arr)
        input_batch = np.expand_dims(input_arr, axis=0)

        # ---------- Predict ----------
        class_names = ['Adenocarcinoma', 'Large cell carcinoma', 'normal', 'Squamous cell carcinoma']
        preds = classifier_model.predict(input_batch)
        predicted_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        # ---------- Non-CT / low-confidence guard ----------
        CONF_THRESH = 0.70
        if confidence < CONF_THRESH:
            st.error("❌ This upload doesn’t look like a valid CT scan for this model.")
            st.info("Tip: Use an axial chest CT slice with good contrast and minimal annotations.")
            st.stop()

        # ---------- Proceed only for confident predictions ----------
        st.markdown(f"### 🔍 Predicted: `{predicted_class}` ({confidence*100:.1f}%)")

        # Case key
        filename = os.path.splitext(uploaded_file.name)[0]
        case_key = f"{patient_id}_{filename}_{scan_date}"

        # --- Load review defaults from DB if exists ---
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("""
                SELECT review_status, review_comment, review_like
                FROM cases
                WHERE patient_id=? AND filename=? AND scan_date=? 
                LIMIT 1;
            """, (patient_id, uploaded_file.name, scan_date))
            row = cur.fetchone()

        if case_key not in st.session_state:
            if row:
                st.session_state[case_key] = {
                    "status": row[0],
                    "comment": row[1] or "",
                    "like": bool(row[2]) if row[2] is not None else False
                }
            else:
                st.session_state[case_key] = {"status": None, "comment": "", "like": False}

        # ---------- Output paths ----------
        out_dir = OUTPUT_FOLDER / filename
        out_dir.mkdir(exist_ok=True)
        xml_path = out_dir / "annotation.xml"
        pdf_path = out_dir / "report.pdf"
        edited_mask_path = out_dir / "edited_mask.png"

        # ---------- NORMAL CASE ----------
        if predicted_class.lower() == "normal":
            st.info("No segmentation required. Saving case for record keeping.")

            empty_mask = np.zeros_like(gray_img, dtype=np.uint8)
            overlay_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            lesions = []

            # Generate files
            generate_aim_xml(uploaded_file.name, empty_mask, lesions, xml_path)
            make_pdf_report(
                patient_id, patient_name, scan_date, uploaded_file.name,
                predicted_class, confidence, lesions, overlay_bgr, empty_mask, pdf_path
            )

            # Save one row per patient/scan
            with sqlite3.connect(DB_PATH) as con:
                con.execute("""
                    INSERT OR REPLACE INTO cases (
                        patient_id, patient_name, scan_date, filename,
                        predicted_class, confidence,
                        xml_path, pdf_path, saved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    patient_id, patient_name, scan_date, uploaded_file.name,
                    predicted_class, confidence,
                    str(xml_path), str(pdf_path), datetime.now().isoformat()
                ))

            # Preview
            st.subheader("📄 Report & AIM XML Preview (Normal)")
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button("⬇️ Download PDF Report", pdf_bytes, file_name="report.pdf")
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                        'width="900" height="800" type="application/pdf"></iframe>',
                        unsafe_allow_html=True)
            with open(xml_path, "r", encoding="utf-8") as f:
                xml_text = f.read()
            st.download_button("⬇️ Download AIM XML", xml_text,
                               file_name=os.path.basename(xml_path), mime="application/xml")
            st.code(xml_text, language="xml")

        # ---------- CANCER CLASSES ----------
        else:
            if edited_mask_path.exists():
                st.info("Using radiologist-edited mask for this case.")
                mask = cv2.imread(str(edited_mask_path), cv2.IMREAD_GRAYSCALE)
                mask = (mask > 0).astype(np.uint8)

                lesions = compute_lesion_metrics(mask.astype(bool), gray_img.shape)
                overlay_preview = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
                overlay_preview[mask == 1] = [255, 0, 0]

                generate_aim_xml(uploaded_file.name, mask, lesions, xml_path)
                make_pdf_report(
                    patient_id, patient_name, scan_date, uploaded_file.name,
                    predicted_class, confidence, lesions,
                    cv2.cvtColor(overlay_preview, cv2.COLOR_RGB2BGR),
                    mask, pdf_path
                )
            else:
                seg = FCM_Combined_Segmenter(gray_img)
                _, membership_map, final_mask, overlay, centroids = seg.segment()
                lesions = compute_lesion_metrics(final_mask.astype(bool), gray_img.shape)

                # Lesion Selection
                st.subheader("📏 Lesion Measurements")
                accepted = []
                for r in lesions:
                    keep = st.checkbox(f"Lesion {r['index']} ({r['side']} {r['lobe']})", value=True)
                    if keep:
                        accepted.append(r)
                    with st.expander(f"Details for lesion {r['index']}"):
                        st.json(r)

                # Preview segmentation
                fig, axs = plt.subplots(1, 4, figsize=(24, 5))
                axs[0].imshow(gray_img, cmap="gray"); axs[0].set_title("Original CT")
                axs[1].imshow(membership_map, cmap="hot"); axs[1].contour(final_mask, colors="lime"); axs[1].set_title("Membership + Mask")
                axs[2].imshow(final_mask, cmap="gray"); axs[2].set_title("Tumor Mask")
                axs[3].imshow(overlay); axs[3].set_title("Overlay with Centroids")
                for ax in axs: ax.axis("off")
                st.pyplot(fig)

                generate_aim_xml(uploaded_file.name, final_mask, accepted, xml_path)
                make_pdf_report(
                    patient_id, patient_name, scan_date, uploaded_file.name,
                    predicted_class, confidence, accepted, overlay, final_mask, pdf_path
                )

                with sqlite3.connect(DB_PATH) as con:
                    con.execute("""
                        INSERT OR REPLACE INTO cases (
                            patient_id, patient_name, scan_date, filename,
                            predicted_class, confidence,
                            xml_path, pdf_path, saved_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        patient_id, patient_name, scan_date, uploaded_file.name,
                        predicted_class, confidence,
                        str(xml_path), str(pdf_path), datetime.now().isoformat()
                    ))

            # Preview
            st.subheader("📄 Report & AIM XML Preview")
            view = st.radio("Preview", ["PDF Report", "AIM XML"], horizontal=True)
            if view == "PDF Report":
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button("⬇️ Download PDF Report", pdf_bytes, file_name="report.pdf")
                base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                            'width="900" height="800" type="application/pdf"></iframe>',
                            unsafe_allow_html=True)
            else:
                with open(xml_path, "r", encoding="utf-8") as f:
                    xml_text = f.read()
                st.download_button("⬇️ Download AIM XML", xml_text,
                                   file_name=os.path.basename(xml_path), mime="application/xml")
                st.code(xml_text, language="xml")

           

            # 🔔 Call the review function
            radiologist_review(case_key, gray_img, patient_id, patient_name,
                               scan_date, uploaded_file, predicted_class,
                               confidence, xml_path, pdf_path, edited_mask_path)

# -------------------------------
# -------------------------------
elif page == "🗂️ Case Browser":
    st.title("Case Browser")
    

    # --- Filters ---
    cleanup_duplicates()
    st.success("✅ Duplicate rows cleaned.")
    pid = st.text_input("Filter by Patient ID")
    cancer = st.selectbox("Filter by Cancer Type", ["","Adenocarcinoma","Large cell carcinoma","normal","Squamous cell carcinoma"])
    start = st.date_input("Start Date", None)
    end = st.date_input("End Date", None)

    start_str = start.isoformat() if start else None
    end_str = end.isoformat() if end else None

    df = query_cases(pid if pid else None,
                     cancer if cancer else None,
                     start_str, end_str)

    # --- Bulk delete warning ---
    st.markdown("### ⚠️ Danger Zone")
    confirm_all = st.checkbox("Confirm bulk delete of ALL patients")
    if st.button("🗑️ Delete ALL Cases"):
        if confirm_all:
            delete_all_cases()
            st.success("✅ All cases and files deleted")
            st.rerun()
        else:
            st.warning("Please tick confirm before bulk delete")

    # --- Display Case Records with Selection ---
    if not df.empty:
        st.subheader("📋 Patient Records")

        # Add checkbox column
        df_display = df.copy()
        df_display["Select"] = False

        selection = st.data_editor(
            df_display[["patient_id","patient_name","scan_date","predicted_class","confidence","review_status", "review_comment", "review_like","pdf_path","xml_path","Select"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", default=False),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", format="%.2f", min_value=0, max_value=1
                )
            }
        )

        # Collect selected rows
        selected_patients = selection[selection["Select"] == True]

        if not selected_patients.empty:
            st.markdown(f"### ✅ {len(selected_patients)} Record(s) Selected")

            col1, col2, col3, col4 = st.columns(4)

            # --- Batch View ---
            with col1:
                if st.button("👁️ Batch View"):
                    for _, row in selected_patients.iterrows():
                        st.markdown(f"#### 🧑 Patient {row['patient_id']} - {row['patient_name']}")
                        st.write(f"Prediction: {row['predicted_class']} ({row['confidence']*100:.1f}%)")
                        tab1, tab2 = st.tabs(["📰 PDF Report", "🧾 AIM XML"])
   


                        # PDF View
                        with tab1:
                            if Path(row["pdf_path"]).exists():
                                with open(row["pdf_path"], "rb") as f:
                                    pdf_bytes = f.read()
                                import base64
                                base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                                st.markdown(
                                    f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                                                                                             'width="900" height="500" type="application/pdf"></iframe>',
                                    unsafe_allow_html=True,
                                )
                        # XML View
                        with tab2:
                            if Path(row["xml_path"]).exists():
                                with open(row["xml_path"], "r") as f:
                                    xml_text = f.read()
                                st.markdown(
    f"""
    <textarea style="width:900px; height:500px; font-family:monospace; font-size:14px;">
{xml_text}
    </textarea>
    """,
    unsafe_allow_html=True,
)

            # --- Batch PDF Download ---
            with col2:
                import zipfile, io
                if not selected_patients.empty:
                    if len(selected_patients) == 1:
                         # Single patient → direct PDF
                         row = selected_patients.iloc[0]
                         pdf_path = Path(row["pdf_path"])
                         if pdf_path.exists():
                             st.download_button("⬇️ Download PDF Report",
                                                 pdf_path.read_bytes(),
                                                file_name=f"{row['patient_id']}_report.pdf",
                                                key="single_pdf")
                    else:
                        # Multiple patients → ZIP
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w") as zf:
                               for _, row in selected_patients.iterrows():
                                     pdf_path = Path(row["pdf_path"])
                                     if pdf_path.exists():
                                          zf.write(pdf_path, arcname=f"{row['patient_id']}_report.pdf")
                        st.download_button("📦 Download Selected PDFs (ZIP)",
                                           zip_buffer.getvalue(),
                                           file_name="selected_reports.zip",
                                           key="batch_pdf")
                    
              

            # --- Batch XML Download ---
            with col3:
                import zipfile, io
                if not selected_patients.empty:
                        if len(selected_patients) == 1:
                            # Single patient → direct XML
                            row = selected_patients.iloc[0]
                            xml_path = Path(row["xml_path"])
                            if xml_path.exists():
                                with open(xml_path, "r") as f:
                                      xml_text = f.read()
                                st.download_button("⬇️ Download AIM XML",
                                                       xml_text,
                                                       file_name=f"{row['patient_id']}_annotation.xml",
                                                       mime="application/xml",
                                                       key="single_xml")
                        else:
                    # Multiple patients → ZIP
                           zip_buffer = io.BytesIO()
                           with zipfile.ZipFile(zip_buffer, "w") as zf:
                              for _, row in selected_patients.iterrows():
                                     xml_path = Path(row["xml_path"])
                                     if xml_path.exists():
                                             zf.write(xml_path, arcname=f"{row['patient_id']}_annotation.xml")
                                         
                           st.download_button("📦 Download Selected AIM XMLs (ZIP)",
                                              zip_buffer.getvalue(),
                                              file_name="selected_annotations.zip",
                                              key="batch_xml")
                                           
                    
                    
                                                   
               

            # --- Batch Delete ---
            with col4:
                if st.button("🗑️ Delete Selected"):
                    for _, row in selected_patients.iterrows():
                        delete_patient_cases(row["patient_id"])
                    st.success("✅ Deleted selected records")
                    # Refresh dataframe
                    st.success("✅ Deleted selected records")
                    st.rerun()
    else:
        st.info("No cases found.")
