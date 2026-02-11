from __future__ import annotations

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

# Runs your existing batch modules in correct order
with DAG(
    dag_id="de2_batch_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,          # manual trigger (best for demo)
    catchup=False,
    tags=["de2", "batch"],
) as dag:

    generate = BashOperator(
        task_id="generate_data",
        bash_command="cd /opt/airflow/de2-final-project && python -m generator.generate_orders --hours 6 --records-per-hour 1000",
    )

    clean_validate = BashOperator(
        task_id="clean_validate",
        bash_command="cd /opt/airflow/de2-final-project && python -m batch.clean_validate",
    )

    aggregate = BashOperator(
        task_id="aggregate",
        bash_command="cd /opt/airflow/de2-final-project && python -m batch.aggregate",
    )

    features = BashOperator(
        task_id="features",
        bash_command="cd /opt/airflow/de2-final-project && python -m batch.features",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow/de2-final-project && python -m batch.train_model",
    )

    batch_predict = BashOperator(
        task_id="batch_predict",
        bash_command="cd /opt/airflow/de2-final-project && python -m batch.batch_predict",
    )

    generate >> clean_validate >> aggregate >> features >> train_model >> batch_predict
