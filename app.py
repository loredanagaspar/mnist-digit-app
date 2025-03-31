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
from pathlib import Path

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

# === Prediction Logic ===
if canvas.image_data is not None:
    img = canvas.image_data[:, :, 0]
    img = 255 - img
    img = img / 255.0
    img = torch.tensor(img).float()

    resized = F.interpolate(
        img.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False
    )
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save input tensor and label locally
        Path("data/logged").mkdir(parents=True, exist_ok=True)
        torch.save(img, f"data/logged/{timestamp}.pt")
        with open("data/logged/labels.csv", "a") as f:
            f.write(f"{timestamp}.pt,{true_label}\n")

        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO predictions (timestamp, prediction, confidence, true_label)
                        VALUES (%s, %s, %s, %s)
                    """, (datetime.now(), pred, conf, true_label))
                conn.commit()
                st.success("Logged to database and saved input image ✅")
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
