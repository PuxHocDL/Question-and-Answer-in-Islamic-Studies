import pandas as pd
import os
import re


def save_mcq_file(file_path, model_column="prediction"):
    """
    Save two CSV files:
    1. Full version with all columns (overwrite original).
    2. Submission file with only id_question and prediction columns.
    """
    try:
        # Load full dataframe
        df = pd.read_csv(file_path)

        # Always re-save the full version (optional but safer for encoding)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')

        # Check required columns
        if "id_question" not in df.columns or model_column not in df.columns:
            print(f"⚠ Required columns not found: id_question or {model_column}")
            return

        # Prepare submission file
        submission_df = df[["id_question", model_column]].rename(columns={model_column: "prediction"})

        # Detect subtask based on number of options
        suffix = get_filename_suffix(df)

        submission_path = os.path.join(
            os.path.dirname(file_path),
            f"subtask{suffix}_predictions.csv"
        )

        submission_df.to_csv(submission_path, index=False, encoding='utf-8-sig')
        print(f"✅ Submission file saved: {submission_path}")

    except Exception as e:
        print(f"❌ Error while saving CSV files: {e}")


def get_filename_suffix(df: pd.DataFrame) -> str:
    """
    Detect the subtask based on number of options.

    Returns:
        str: suffix (1 or 2), default unknown.
    """
    option_cols = [col for col in df.columns if col.lower().startswith("option")]
    options_count = len(option_cols)

    if options_count == 6:
        return "1"
    elif options_count == 4:
        return "2"
    else:
        return "unknown"


def clean_and_validate_response(response, valid_responses):
    """
    Clean model output and extract valid answer.
    More robust parsing using regex.
    """
    if not response:
        return None

    response = response.strip().upper()

    # Strong regex to capture most formats like:
    # "Answer: B", "Correct is: C", "The answer is B", etc.
    match = re.search(r"(?:answer|correct)?\s*(?:is)?\s*[:\-]?\s*([A-F])", response, re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        if candidate in valid_responses:
            return candidate

    # Fallback basic extraction for isolated single letter A-F
    matches = re.findall(r"\b([A-F])\b", response)
    if len(matches) == 1 and matches[0] in valid_responses:
        return matches[0]

    return None
