import os
import sys
from urllib.parse import urlparse
import psycopg2
import streamlit as st
import logging

# Optional log file to capture DB-related errors
LOG_FILE = "db_connection.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Health check ===
def validate_db_connection():
    url = os.environ.get("DATABASE_URL")
    if not url:
        logging.error("Missing DATABASE_URL in environment variables.")
        return False, "DATABASE_URL not set"

    try:
        parsed = urlparse(url)
        with psycopg2.connect(
            dbname=parsed.path[1:],
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True, "DB connection successful"
    except Exception as e:
        logging.exception("Failed to connect to DB")
        return False, str(e)

# === Streamlit Health Page ===
if __name__ == "__main__":
    st.set_page_config(page_title="DB Health Check", layout="centered")
    st.title("🔎 Database Health Check")

    success, message = validate_db_connection()
    if success:
        st.success(message)
    else:
        st.error(f"Database connection failed: {message}")

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            logs = f.read()
        with st.expander("View logs"):
            st.code(logs, language="text")
