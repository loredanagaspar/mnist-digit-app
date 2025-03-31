import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def create_predictions_table():
    try:
        url = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(url, sslmode='require')
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                prediction SMALLINT CHECK (prediction BETWEEN 0 AND 9),
                confidence REAL CHECK (confidence BETWEEN 0 AND 1),
                true_label SMALLINT CHECK (true_label BETWEEN 0 AND 9)
            );
        """)

        conn.commit()
        cur.close()
        conn.close()
        print("✅ predictions table created successfully.")

    except Exception as e:
        print("❌ Error creating table:", e)

if __name__ == "__main__":
    create_predictions_table()
