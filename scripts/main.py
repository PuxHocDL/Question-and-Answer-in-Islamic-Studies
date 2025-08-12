import os
import pandas as pd
import yaml
from functools import partial
from typing import List, Dict, Callable
from scripts.inference import process_csv_file
from scripts.utils import get_filename_suffix, save_mcq_file
from scripts.models import (
    load_documents_and_index,
    get_prediction_mistral,
    get_prediction_fanar,
    get_prediction_gemini,
)

# Initialize document index
DOCUMENTS, FAISS_INDEX = load_documents_and_index()

# Model function mapping
MODEL_FUNCTIONS: Dict[str, Callable] = {
    "mistral": partial(
        get_prediction_mistral,
        model_version="mistral-saba-24b",
        top_k=10,
        documents=DOCUMENTS,
        faiss_index=FAISS_INDEX,
        task_type='knowledge',
    ),
    "fanar_rag": partial(
        get_prediction_fanar,
        model_version="Islamic-RAG",
        top_k=10,
        documents=DOCUMENTS,
        faiss_index=FAISS_INDEX,
        task_type='knowledge',
    ),
    "gemini": partial(
        get_prediction_gemini,
        model_version="gemini-2.0-flash",
        top_k=10,
        documents=DOCUMENTS,
        faiss_index=FAISS_INDEX,
        task_type='knowledge'
    ),
}

# Initialize Fanar API configuration
FANAR_API_KEY = os.getenv("FANAR_API_KEY")
FANAR_API_KEY_2 = os.getenv("FANAR_API_KEY_2")
if 'get_prediction_fanar' in globals():
    get_prediction_fanar.fanar_failure_count = 0
    get_prediction_fanar.current_fanar_key = FANAR_API_KEY
    get_prediction_fanar.use_secondary_key = False
    print(f"Initialized Fanar global state.\nFANAR_API_KEY: {FANAR_API_KEY}\nFANAR_API_KEY_2: {FANAR_API_KEY_2}")


def load_config(config_path: str = "../config.yaml") -> tuple[str, str, List[str]]:
    """
    Load configuration from a YAML file and select the first valid model.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        tuple: Input directory, output directory, and list of selected models (single model).

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If no models are selected in the configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    input_dir = config.get("paths", {}).get("input_dir", "")
    output_dir = config.get("paths", {}).get("output_dir", "")

    # Select models marked as "Y"
    selected_models = [
        model for model, status in config.get("models", {}).items()
        if status.strip().upper() == "Y"
    ]

    if not selected_models:
        raise ValueError("No models selected in configuration file.")

    # Use only the first selected model
    if len(selected_models) > 1:
        print(f"Multiple models selected {selected_models}, using only: {selected_models[0]}")

    return input_dir, output_dir, [selected_models[0]]


def predict_from_directory(config_path: str = "../config.yaml") -> pd.DataFrame:
    """
    Process all CSV files in the input directory, generate predictions, and save results.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        pd.DataFrame: Combined predictions from all processed files.
    """
    all_predictions_df = pd.DataFrame()
    input_dir, output_dir, selected_models = load_config(config_path)
    model_suffix = "_".join(selected_models)
    print(f"Model selected: {selected_models[0]}")

    if not selected_models:
        return all_predictions_df

    # Map selected model to its prediction function
    models_to_evaluate = {
        model: MODEL_FUNCTIONS[model]
        for model in selected_models
        if model in MODEL_FUNCTIONS
    }

    for file in os.listdir(input_dir):
        if file.endswith(".csv") and "_prediction" not in file:
            try:
                print(f"Processing file: {file}")
                input_path = os.path.join(input_dir, file)
                df = process_csv_file(input_path, models_to_evaluate)

                if df.empty:
                    print(f"Warning: No predictions generated for {file}")
                    continue

                all_predictions_df = pd.concat([all_predictions_df, df], ignore_index=True)

                # Rename the last column to 'prediction'
                df = df.rename(columns={df.columns[-1]: "prediction"})

                # Reorder columns to place 'prediction' after 'level' if present
                if "level" in df.columns:
                    cols = list(df.columns)
                    cols.remove("prediction")
                    level_idx = cols.index("level")
                    cols = cols[:level_idx + 1] + ["prediction"] + cols[level_idx + 1:]
                    df = df[cols]

                # Generate output filename with subtask suffix
                suffix = get_filename_suffix(df)
                output_file = os.path.join(
                    output_dir,
                    f"{os.path.splitext(file)[0]}_{model_suffix}_subtask{suffix}_prediction.csv",
                )
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # Save output
                df.to_csv(output_file, index=False)
                save_mcq_file(output_file)
                print(f"Saved predictions to: {output_file}")

            except Exception as e:
                print(f"Error processing file {file}: {e}")

    return all_predictions_df


if __name__ == "__main__":
    predict_from_directory()