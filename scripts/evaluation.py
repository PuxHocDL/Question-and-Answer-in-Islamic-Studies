import os
import csv
import json
import shutil

def read_csv_column(filepath, column_name):
    """
    Read a specific column from a CSV file, skipping empty or invalid rows.

    Args:
        filepath (str): Path to the CSV file.
        column_name (str): Name of the column to extract.

    Returns:
        list: List of non-empty, stripped values from the specified column.
    """
    values = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Verify that the column exists
        if column_name not in reader.fieldnames:
            raise ValueError(f"Column '{column_name}' not found in {filepath}")
        for row in reader:
            if column_name in row:
                value = row[column_name]
                # Skip empty, None, or whitespace-only values
                if value is not None and value.strip():
                    values.append(value.strip())
    return values
ARABIC_TO_ENGLISH = {
    'أ': 'A',
    'ب': 'B',
    'ج': 'C',
    'د': 'D', 
    'A': 'A',
    'B': 'B',
    'C': 'C',
    'D': 'D',
    'E': 'E',
    'F': 'F',
}

def map_arabic_to_english(labels, mapping=ARABIC_TO_ENGLISH):
    """
    Map a list of Arabic labels to their corresponding English labels.
    
    Args:
        labels (list): List of Arabic labels (e.g., ['أ', 'ب', 'ج']).
        mapping (dict): Dictionary mapping Arabic to English labels.
        
    Returns:
        list: List of mapped English labels (e.g., ['A', 'B', 'C']).
    """
    return [mapping.get(label, label) for label in labels]

def evaluate(reference_dir, prediction_dir, output_dir=None):
    """
    Evaluate predictions against reference and compute full evaluation metrics.
    Optionally write score file and backup prediction files into output_dir.
    """
    # Read reference file (first 5 examples only)
    truth = read_csv_column(reference_dir, 'label')
    # Read predictions
    preds = read_csv_column(prediction_dir, 'prediction')

    # Length check
    if len(truth) != len(preds):
        raise ValueError(f"Mismatch in number of predictions: {len(preds)} vs {len(truth)}")

    # Compute metrics
    total_examples = len(truth)
    correct = sum(1 for t, p in zip(truth, preds) if t == p or  ARABIC_TO_ENGLISH[t] == p)
    accuracy = correct / total_examples

    submission_copied = False

    # Write scores and copy prediction files if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        scores_path = os.path.join(output_dir, 'scores.json')

        scores = {
            'total_examples': total_examples,
            'correct_predictions': correct,
            'accuracy': accuracy
        }

        with open(scores_path, 'w') as f:
            json.dump(scores, f, indent=2)

        basename = os.path.basename(prediction_dir)
        # Copy predictions files to output
        shutil.copy(prediction_dir, os.path.join(output_dir, basename))

    # Rename submission file (in same directory as prediction_dir)
        # Rename submission file (in same directory as prediction_dir)
    submission_file = os.path.join(os.path.dirname(prediction_dir), 'subtask1_predictions.csv')
    if os.path.exists(submission_file):
        original_basename = os.path.basename(prediction_dir)
        prefix = "Task1_QCM_Dev_"
        suffix = "_subtask1_prediction.csv"
        if original_basename.startswith(prefix) and original_basename.endswith(suffix):
            middle_part = original_basename[len(prefix):-len(suffix)]
            new_submission_name = f"subtask1_{middle_part}_predictions.csv"
        else:
            new_submission_name = "subtask1_predictions.csv"  # fallback

        new_submission_path = os.path.join(os.path.dirname(prediction_dir), new_submission_name)
        if os.path.exists(new_submission_path):
            os.remove(new_submission_path)  # avoid FileExistsError
        os.rename(submission_file, new_submission_path)
        print(f"✅ Submission file renamed to {new_submission_name}")
        submission_copied = True


    print("\n----------------------------------")
    print("       Evaluation Report:")
    print("----------------------------------")
    print(f"✅ Total examples = {total_examples}")
    print(f"✅ Correct predictions = {correct}")
    print(f"✅ Accuracy = {100*round(accuracy, 4)}")

    if output_dir:
        print(f"Score file written to {scores_path}")
        print("Full prediction file copied to output directory.")


    return accuracy