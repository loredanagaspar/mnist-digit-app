CREATE TABLE IF NOT EXISTS predictions (
  id SERIAL PRIMARY KEY,
  timestamp TIMESTAMP DEFAULT NOW(),
  prediction INT,
  confidence FLOAT,
  true_label INT
);
