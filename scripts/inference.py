import pandas as pd

def detect_question_type(df):
    """
    Detect the type of MCQ based on available options (4 or 6 choices).
    """
    required_columns = {"question", "option1", "option2", "option3", "option4"}
    df_columns = set(df.columns)

    if not required_columns.issubset(df_columns):
        print("❌ Error: Missing required columns for a valid MCQ format.")
        print(f"Detected columns: {df_columns}")
        return None

    if {"option5", "option6"}.issubset(df_columns):
        return "MCQ_6"
    else:
        return "MCQ_4"


def process_csv_file(input_file, models_to_evaluate):
    """
    Process a CSV file and apply selected model to generate predictions.

    Args:
        input_file (str): Path to CSV file.
        models_to_evaluate (dict): {model_name: model_function}

    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    df = pd.read_csv(input_file)

    question_type = detect_question_type(df)
    if not question_type:
        return None

    # ✅ There should be only one model at a time (already enforced upstream)
    model_name, model_function = list(models_to_evaluate.items())[0]
    predictions = []

    for idx, row in df.iterrows():
        question = row['question']

        # Prepare options depending on type
        if question_type == "MCQ_6":
            options = [row[f'option{i}'] for i in range(1, 7)]
        else:
            options = [row[f'option{i}'] for i in range(1, 5)]

        try:
            prediction = model_function(question, *options)
        except Exception as e:
            print(f"❌ Error with model {model_name} on row {idx}: {e}")
            prediction = None

        predictions.append(prediction)

    # Add prediction column named after model
    df[model_name] = predictions

    return df
