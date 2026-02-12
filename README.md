#  DE2 – End-to-End Finance ML Data Engineering Pipeline (GCP)

##  Project Overview

This project implements a complete **end-to-end data engineering + ML pipeline** on Google Cloud Platform.

It demonstrates:

* Synthetic financial tick data generation
* Batch data cleaning and aggregation
* Feature engineering
* Machine learning training (Random Forest)
* Batch predictions
* Cloud Storage (GCS) integration
* BigQuery analytics tables
* REST API for live prediction

The objective is to show a **fully working system**, not just isolated scripts.

---

#  Architecture Overview

```
Data Generator (1000+ records/hour, 6+ hours)
        ↓
Raw CSV files
        ↓
Batch Cleaning
        ↓
Hourly Aggregation
        ↓
Feature Engineering
        ↓
Model Training (RandomForest)
        ↓
Batch Prediction
        ↓
GCS (Raw + Model)
        ↓
BigQuery (Analytics + Predictions)
        ↓
Flask API (Live Prediction Endpoint)
```

---

#  Project Structure

```
de2-final-project/
│
├── generator/          # Synthetic data generation
├── batch/              # Cleaning, aggregation, ML, prediction
├── api/                # Flask ML API
├── gcp/                # GCS upload utilities
├── data/
│   ├── raw/
│   ├── analytics/
│   ├── features/
│   ├── models/
│   └── predictions/
├── logs/
└── requirements.txt
```

---

#  Data Generation Requirement

✔ Generates **1000+ records per hour**
✔ Runs for **6+ hours**

Example:

```bash
python -m generator.generate_orders --hours 6 --records-per-hour 1000
```

This produces 6 hourly CSV files with ~1000 rows each.

---

#  Batch Processing

## 1️ Clean Data

```bash
python -m batch.clean_validate
```

## 2️ Aggregate to Hourly Level

```bash
python -m batch.aggregate
```

Creates:

```
data/analytics/hourly_analytics.csv
```

---

#  Feature Engineering

```bash
python -m batch.features
```

Features created:

* avg_price
* total_volume
* trades
* return_1h
* ma_3
* vol_3
* label_up_next (target)

Latest hour rows are marked as **NaN for label** (prediction-only).

---

#  Model Training

```bash
python -m batch.train_model
```

Model:

* RandomForestClassifier
* 300 estimators
* class_weight balanced
* random_state=42

Outputs:

```
data/models/rf_model.joblib
data/models/rf_model_meta.json
```

---

#  Batch Prediction

```bash
python -m batch.batch_predict
```

Creates:

```
data/predictions/predictions_TIMESTAMP.csv
```

Predicts:

* pred_up_next (0 or 1)
* pred_up_next_proba (probability)

Meaning:

> Will price increase in the next hour?

---

#  Google Cloud Setup

## Set Credentials

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/c/Users/Asus/Desktop/.../de2-gcp-key.json"
```

Test:

```bash
python -c "from google.cloud import storage; print(storage.Client().project)"
```

---

#  Upload Raw Data to GCS

```bash
python gcp/gcs_upload.py de2-finance-raw-487115 data/raw/batch_input
```

Bucket:

```
de2-finance-raw-487115
```

---

#  Upload Analytics to BigQuery

## Hourly Analytics

```bash
python -c "
from google.cloud import bigquery
import pandas as pd
client=bigquery.Client()
df=pd.read_csv('data/analytics/hourly_analytics.csv')
table_id='de2-final-project-487115.de2_dataset.hourly_analytics'
job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')
job=client.load_table_from_dataframe(df, table_id, job_config=job_config)
job.result()
print('Uploaded', len(df), 'rows')
"
```

---

## Batch Predictions

```bash
python -c "
from google.cloud import bigquery
import pandas as pd, os
client=bigquery.Client()
fn=os.popen('ls -t data/predictions | head -n 1').read().strip()
df=pd.read_csv(f'data/predictions/{fn}')
table_id='de2-final-project-487115.de2_dataset.batch_predictions'
job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')
job=client.load_table_from_dataframe(df, table_id, job_config=job_config)
job.result()
print('Uploaded', len(df), 'rows')
"
```

---

#  ML REST API

Start server:

```bash
python api/app.py
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

Prediction:

```bash
curl "http://127.0.0.1:8080/predict" \
-H "Content-Type: application/json" \
-d "{\"avg_price\":35000,\"total_volume\":500,\"trades\":290,\"return_1h\":0.02,\"ma_3\":34000,\"vol_3\":0.8}"
```

Response:

```json
{
  "prediction": 1,
  "prob_up_next": 0.97
}
```

---

# 📊 BigQuery Tables

Dataset:

```
de2_dataset
```

Tables:

* hourly_analytics
* batch_predictions

---

#  What This Demonstrates

✔ Live end-to-end working pipeline
✔ Data flowing from generation → cleaning → ML → cloud
✔ Integration with GCS + BigQuery
✔ REST prediction endpoint
✔ Reproducible setup
✔ Scalable architecture (increase hours to increase training volume)

---

#  Notes

* Latest hour rows are prediction-only (no observable next hour).
* Accuracy depends on historical window size.
* Increasing `--hours` increases training robustness.

---

#  Scaling

To increase training volume:

```bash
python -m generator.generate_orders --hours 168 --records-per-hour 1000
```

Architecture requires no changes.

---




