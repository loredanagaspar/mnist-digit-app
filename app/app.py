import sys, os
sys.path.append(os.path.abspath("."))

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import torch
import torch.nn.functional as F
from model.train import DigitCNN
import psycopg2
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🧠 Digit Recognizer")

# === 1. DRAWING CANVAS ===
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

# === Load Model ===
model = DigitCNN()
model.load_state_dict(torch.load("model/model.pt", map_location=torch.device("cpu")))
model.eval()

if canvas.image_data is not None:
    img = canvas.image_data[:, :, 0]
    img = 255 - img
    img = img / 255.0
    img = torch.tensor(img).float()

    # Center crop
    threshold = 0.1
    mask = img > threshold
    if mask.any():
        coords = torch.nonzero(mask)
        top_left = coords.min(dim=0)[0]
        bottom_right = coords.max(dim=0)[0]
        cropped = img[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    else:
        cropped = img

    # Resize + Normalize
    resized = torch.nn.functional.interpolate(
        cropped.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False
    )
    normalized = (resized - 0.1307) / 0.3081
    img = normalized

    # === Predict ===
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
        conf = torch.max(F.softmax(output, dim=1)).item()

    st.markdown("### 🔍 Prediction")
    st.write(f"**Prediction:** {pred}")
    st.write(f"**Confidence:** {conf:.2%}")

    # === 2. TRUE LABEL INPUT + SUBMIT ===
    st.markdown("### ✏️ Enter True Label")
    true_label = st.number_input("True label (0–9)", 0, 9, step=1)
    if st.button("✅ Submit"):
        try:
            conn = psycopg2.connect(
                host="db", dbname="postgres", user="postgres", password="postgres"
            )
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO predictions (timestamp, prediction, confidence, true_label) VALUES (%s, %s, %s, %s)",
                (datetime.now(), pred, conf, true_label)
            )
            conn.commit()
            cur.close()
            conn.close()
            st.success("Logged to database ✅")
        except Exception as e:
            st.error(f"Database error: {e}")

# === 3. DISPLAY HISTORY ===
st.markdown("### 📜 History")
try:
    conn = psycopg2.connect(
        host="db", dbname="postgres", user="postgres", password="postgres"
    )
    df = pd.read_sql("SELECT timestamp, prediction, true_label FROM predictions ORDER BY timestamp DESC LIMIT 5", conn)
    st.dataframe(df)
    conn.close()
except:
    st.warning("Database not ready or no data yet.")
