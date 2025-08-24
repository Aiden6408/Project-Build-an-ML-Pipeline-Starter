import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: do not include this by default; run explicitly after promoting a model to prod
    # "test_regression_model",
]

# This automatically reads in the configuration
@hydra.main(config_name="config")
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        # --- download ---
        if "download" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version="main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
            )

        # --- basic_cleaning ---
        if "basic_cleaning" in active_steps:
            with mlflow.start_run(run_name="basic_cleaning"):
                mlflow.run(
                    uri=os.path.join(get_original_cwd(), "src", "basic_cleaning"),
                    entry_point="main",
                    env_manager="local",
                    parameters={
                        "input_artifact": "sample.csv:latest",
                        "output_artifact": "clean_sample.csv",
                        "output_type": "clean_data",
                        "output_description": "Data after basic cleaning",
                        "min_price": config["etl"]["min_price"],
                        "max_price": config["etl"]["max_price"],
                    },
                )

        # --- data_check ---
        if "data_check" in active_steps:
            with mlflow.start_run(run_name="data_check"):
                mlflow.run(
                    uri=os.path.join(get_original_cwd(), "src", "data_check"),
                    entry_point="main",
                    env_manager="local",
                    parameters={
                        "csv": "clean_sample.csv:latest",
                        "ref": "clean_sample.csv:reference",
                        "kl_threshold": config["data_check"]["kl_threshold"],
                        "min_price": config["etl"]["min_price"],
                        "max_price": config["etl"]["max_price"],
                    },
                )

        # --- data_split ---
        if "data_split" in active_steps:
            with mlflow.start_run(run_name="data_split"):
                _ = mlflow.run(
                    f"{config['main']['components_repository']}/train_val_test_split",
                    "main",
                    version="main",
                    env_manager="conda",
                    parameters={
                        # component expects these names exactly
                        "input": "clean_sample.csv:latest",
                        "test_size": str(config["modeling"]["test_size"]),
                        "random_seed": str(config["modeling"]["random_seed"]),
                        "stratify_by": config["modeling"]["stratify_by"],
                    },
                )

        # --- train_random_forest ---
        if "train_random_forest" in active_steps:
            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config_path = os.path.abspath("rf_config.json")
            with open(rf_config_path, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # Train step
            with mlflow.start_run(run_name="train_random_forest"):
                mlflow.run(
                    uri=os.path.join(get_original_cwd(), "src", "train_random_forest"),
                    entry_point="main",
                    env_manager="local",
                    parameters={
                        "trainval_artifact": "trainval_data.csv:latest",
                        "val_size": str(config["modeling"]["val_size"]),
                        "random_seed": str(config["modeling"]["random_seed"]),
                        "stratify_by": config["modeling"]["stratify_by"],
                        "rf_config": rf_config_path,  # use ONLY inside this block
                        "max_tfidf_features": str(config["modeling"]["max_tfidf_features"]),
                        "output_artifact": "random_forest_export",
                    },
                )

        # --- test_regression_model ---
        if "test_regression_model" in active_steps:
            with mlflow.start_run(run_name="test_regression_model"):
                _ = mlflow.run(
                    uri=f"{config['main']['components_repository']}/test_regression_model",
                    entry_point="main",
                    version="main",
                    env_manager="conda",
                    parameters={
                        "mlflow_model": "random_forest_export:prod",
                        "test_dataset": "test_data.csv:latest",
                    },
                )


if __name__ == "__main__":
    go()