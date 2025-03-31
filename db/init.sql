CREATE TABLE IF NOT EXISTS predictions (
  id SERIAL PRIMARY KEY,
  timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
  prediction SMALLINT CHECK (prediction BETWEEN 0 AND 9),
  confidence REAL CHECK (confidence BETWEEN 0 AND 1),
  true_label SMALLINT CHECK (true_label BETWEEN 0 AND 9)
);
