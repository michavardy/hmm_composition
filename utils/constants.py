from pathlib import Path

trained_model_path = ".trained_models"
trained_model_metadata_path = ".trained_metadata"

Path(trained_model_path).mkdir(parents=True, exist_ok=True)
Path(trained_model_metadata_path).mkdir(parents=True, exist_ok=True)

