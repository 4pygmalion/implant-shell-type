import os
from typing import Tuple

import mlflow
from matplotlib import pyplot as plt


TRACKING_URI = "http://localhost:5000"
EXP_S_T = "texture_classification"


def get_expid_client(
    tracking_uri: str, experiment_name: str
) -> Tuple[str, mlflow.MlflowClient]:
    mlflow.set_tracking_uri(tracking_uri)

    mlflow_client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    exp = mlflow_client.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow_client.create_experiment(experiment_name)
        exp = mlflow_client.get_experiment_by_name(experiment_name)

    exp_id = exp.experiment_id

    return exp_id, mlflow_client


def save_plot_and_clear(filename):
    """현재 플롯팅된 Figure을 저장하고, Mlflow로깅후 Clear"""
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    os.remove(filename)
    plt.clf()
