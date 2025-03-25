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
from urllib.parse import urlparse

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("\U0001F9E0 Digit Recognizer")

# === Setup database connection ===
db_url = os.environ.get("DATABASE_URL")
if db_url:
    parsed_url = urlparse(db_url)
    db_config = {
        "dbname": parsed_url.path[1:],
        "user": parsed_url.username,
        "password": parsed_url.password,
        "host": parsed_url.hostname,
        "port": parsed_url.port,
    }
else:
    db_config = None
    st.warning("Failed to init DB table: 'DATABASE_URL'")

# === Drawing Canvas ===
st.markdown("### \u270D\ufe0f Draw a number")
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

    threshold = 0.1
    mask = img > threshold
    if mask.any():
        coords = torch.nonzero(mask)
        top_left = coords.min(dim=0)[0]
        bottom_right = coords.max(dim=0)[0]
        cropped = img[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    else:
        cropped = img

    resized = F.interpolate(
        cropped.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False
    )
    normalized = (resized - 0.1307) / 0.3081
    img = normalized

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
        conf = torch.max(F.softmax(output, dim=1)).item()

    st.markdown("### \U0001F50D Prediction")
    st.write(f"**Prediction:** {pred}")
    st.write(f"**Confidence:** {conf:.2%}")

    st.markdown("### \u270F\ufe0f Enter True Label")
    true_label = st.number_input("True label (0–9)", 0, 9, step=1)
    if st.button("✅ Submit") and db_config:
        try:
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    prediction INT,
                    confidence FLOAT,
                    true_label INT
                )
                """
            )
            cur.execute(
                "INSERT INTO predictions (prediction, confidence, true_label) VALUES (%s, %s, %s)",
                (pred, conf, true_label)
            )
            conn.commit()
            cur.close()
            conn.close()
            st.success("Logged to database ✅")
        except Exception as e:
            st.error(f"Database error: {e}")

# === 3. Display History ===
st.markdown("### \ud83d\udcdc History")
try:
    if db_config:
        conn = psycopg2.connect(**db_config)
        df = pd.read_sql("SELECT timestamp, prediction, true_label FROM predictions ORDER BY timestamp DESC LIMIT 5", conn)
        st.dataframe(df)
        conn.close()
    else:
        st.warning("Missing DB config.")
except Exception as e:
    st.warning(f"Database not ready or no data yet.\n{e}")
