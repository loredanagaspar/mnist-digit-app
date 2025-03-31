# app.py
import os
import sys
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import torch
import torch.nn.functional as F
from model.train import DigitCNN
import psycopg2
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse
import torchvision.transforms.functional as TF

# === Streamlit Setup ===
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🧠 Digit Recognizer by Loredana Gaspar")

# === DB Connection Helper ===
def get_db_connection():
    try:
        url = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(url, sslmode='require')
        return conn
    except Exception as e:
        st.error(f"DB connection failed: {e}")
        return None

# === Load Model Once ===
@st.cache_resource
def load_model():
    model = DigitCNN()
    model.load_state_dict(torch.load("model/model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# === Canvas Input ===
st.markdown("### ✍️ Draw a number")
canvas = st_canvas(
    fill_color="#000000",
    stroke_color="#FFFFFF",
    stroke_width=20,
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# === Utility: Center & Pad cropped digit ===
def center_pad(img_tensor, size=28):
    h, w = img_tensor.shape
    pad_top = (size - h) // 2
    pad_left = (size - w) // 2
    pad_bottom = size - h - pad_top
    pad_right = size - w - pad_left
    return TF.pad(img_tensor, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

# === Prediction Logic ===
if canvas.image_data is not None:
    img = canvas.image_data[:, :, 0]
    img = 255 - img
    img = img / 255.0
    img = torch.tensor(img).float()

    mask = img > 0.1
    if mask.any():
       coords = torch.nonzero(mask)
       top_left = coords.min(dim=0)[0]
       bottom_right = coords.max(dim=0)[0]
        # Add margin
       margin = 10
       top = max(top_left[0] - margin, 0)
       left = max(top_left[1] - margin, 0)
       bottom = min(bottom_right[0] + margin, img.shape[0] - 1)
       right = min(bottom_right[1] + margin, img.shape[1] - 1)

       cropped = img[top:bottom+1, left:right+1]
    else:
        cropped = img

    padded = center_pad(cropped)
    resized = F.interpolate(
          padded.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False)
    img = (resized - 0.1307) / 0.3081
    display_img = img * 0.3081 + 0.1307
    st.image(display_img.squeeze().numpy(), caption="Input to model", width=150, clamp=True)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
        conf = torch.max(F.softmax(output, dim=1)).item()

    st.markdown("### 🔍 Prediction")
    st.write(f"**Prediction:** {pred}")
    st.write(f"**Confidence:** {conf:.2%}")

    # === Log Prediction ===
    st.markdown("### ✏️ Enter True Label")
    true_label = st.number_input("True label (0–9)", 0, 9, step=1)
    if st.button("✅ Submit"):
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO predictions (timestamp, prediction, confidence, true_label)
                        VALUES (%s, %s, %s, %s)
                    """, (datetime.now(), pred, conf, true_label))
                conn.commit()
                st.success("Logged to database ✅")
            except Exception as e:
                st.error(f"Database error: {e}")
            finally:
                conn.close()
        else:
            st.warning("DB not available")

# === Show History ===
st.markdown("### 📜 History")
conn = get_db_connection()
if conn:
    try:
        df = pd.read_sql("""
            SELECT timestamp, prediction, true_label
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 5
        """, conn)
        st.dataframe(df)
    except Exception as e:
        st.warning(f"Error reading history: {e}")
    finally:
        conn.close()
else:
    st.warning("DB not connected")
