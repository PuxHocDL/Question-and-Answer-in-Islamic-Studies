# 🕌 QIAS 2025: Islamic Q&A Shared Task
### Benchmarking LLMs on Islamic Inheritance and General Knowledge

## 📖 Overview

The **QIAS 2025 Shared Task** evaluates Large Language Models (LLMs) on their ability to understand, reason, and answer questions grounded in Islamic knowledge.

It features two subtasks:
- **Subtask 1:** Islamic Inheritance (ʿIlm al-Mawārīth)
- **Subtask 2:** General Islamic Knowledge

Participants may use prompting, fine-tuning, retrieval-augmented generation (RAG), or any other technique. All datasets are bilingual (Arabic-English) and verified by Islamic scholars.

---

## 🧪 Subtasks

### 📜 Subtask 1: Islamic Inheritance (Mīrāth)

- Focuses on inheritance-related problems (ʿIlm al-Mawārīth), requiring precise rule-based reasoning aligned with Islamic jurisprudence.
- Dataset:
  - 9450 MCQs  (training))
  - 1500 MCQs (validation)
  - 1500 MCQs (test)
- Extra data:
  - 32,000 IslamWeb fatwas
- Levels: Beginner, Intermediate, Advanced

### 📚 Subtask 2: General Islamic Knowledge

- Covers a broad spectrum of Islamic knowledge, including theology, jurisprudence, biography, and ethics. The difficulty levels reflect increasing depth and complexity.
- Dataset:
  - 1,200 MCQs (200 for validation and 800 for final test)
- Extra data:
  - Source: major Islamic classical books. The answers to the multiple-choice questions in the validation and test sets are derived from these books.  
- Levels: Beginner, Intermediate, Advanced

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://gitlab.com/islamgpt1/qias_shared_task_2025.git
cd qias_shared_task_2025
```
### 2. Create a new Conda environment and activate it: (optional)
1. Download and install MiniConda from [here](https://www.anaconda.com/docs/getting-started/miniconda/main#quick-command-line-install).
2. Create a new Conda environment and activate it:
```bash
conda create -n qias python=3.12
conda activate qias

```
### Install dependencies
```bash
pip install -r requirements.txt
``` 
###  🔐 API Key Configuration
Before running the project, you need to provide an API key in a .env file.

1. An example file named .env_exemple is provided.

2. You must rename it to .env:
```bash
cp .env_exemple .env
```
3.Open the .env file and add your API key to the appropriate variable: 
```bash
API_KEY=your_api_key_here
```
###  ⚙️ Configuration File
We provide an example configuration file:
```bash
cp example.yaml config.yaml
```
You must edit config.yaml to specify:
```bash
input_dir: "/path/to/data"

```
The output directory where prediction results will be stored:
```bash
output_dir: "/path/to/results"

```

## 🧭 Baselines
The current baseline supports several models using few-shot prompting-based inference:


Fanar LLM (API) — designed for Arabic tasks 
🔑 You can request free API access here: https://api.fanar.qa/docs

Mistral (Groq API) — a  open-weight model, accessed via the free Groq API: https://groq.com/

**ALLAM LLM (Local Inference with GPU)** — Arabic large language model available on Hugging Face.  
  🔗 Model: [ALLaM-AI/ALLaM-7B-Instruct-preview](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview)

  > ⚠ You need a GPU-equipped machine to run this model locally.


We are currently working on adding inference support for an open-source, small-sized Arabic LLM that requires fewer resources. It will be available soon as part of the shared task baseline.
### ▶️ How to Run the Code
#### Configure the Models to Use

- Open the config.yaml file.
- Enable or disable the LLM models by setting "Y" or "N" under the models section:
```bash
models:
  mistral: "Y"
  fanar_rag: "N"
  fanar: "Y"
```
 Customize the Configuration

- Modify config.yaml to change input/output paths, enable/disable models, or set parameters.
- To add new models, edit scripts/models.py and update the MODEL_FUNCTIONS dictionary in scripts/main.py.
#### Run Predictions from the Notebook

Open the prediction.ipynb notebook in Jupyter or VS Code.

Run the first code cell::
```bash
from scripts.inference import process_csv_file
from scripts.models import get_prediction_mistral, get_prediction_fanar
from scripts.utils import get_filename_suffix, save_mcq_file
from scripts.main import load_config,  predict_from_directory

predict_from_directory(config_path="../config.yaml")

```
This script will:

- Load the Excel files from the input directory.
- Apply the selected models to generate predictions.
- Automatically detect the number of options (4 or 6).
- Save the output files in the output directory (output_dir), appending the appropriate suffix (_subtask1.xlsx or _subtask2.xlsx).
- Format the Excel files for better readability.

#### 📊 Evaluation
1. Ensure Ground Truth Answers Are Available
Make sure each Excel file in the input directory contains a column named answer with the correct answers.

2. Run Evaluation from the Notebook
After generating predictions, add and run the following cell in the prediction.ipynb notebook:
```bash
from  scripts.evaluation import evaluate

# ✅ Specify your file paths
prediction_dir  =  "../results/prediction/Task1_QCM_Dev_fanar_rag_subtask1_prediction.csv"
reference_dir = "../data/Task1_QCM_Dev.csv"
output_dir = '../results/prediction/output' 

# ✅ Call the evaluation function
accuracy = evaluate(reference_dir, prediction_dir, output_dir)
```


## 📈 Evaluation Metrics

### 📄 Submission Format

Participants must submit a UTF-8 encoded CSV file, with one row per question.

**File Naming Convention:**

- `subtask1_<team_name>_predictions.csv` for SubTask 1 (Islamic Inheritance Reasoning)
- `subtask2_<team_name>_predictions.csv` for SubTask 2 (Islamic Knowledge Assessment)

### 📊 Required Columns (exact order)

**SubTask 1:**

| Column Name | Description |
|-------------|-------------|
| `id_question` | Unique identifier for each question |
| `prediction` | Model’s predicted answer (A, B, C, D, E, or F) |

**SubTask 2:**

| Column Name | Description |
|-------------|-------------|
| `id_question` | Unique identifier for each question |
| `prediction` | Model’s predicted answer (A, B, C, or D) |

### 🧮 Evaluation Metric

Model performance will be evaluated based on **accuracy**:

> The percentage of questions for which the model’s prediction exactly matches the correct answer.

Once submitted, predictions will be automatically evaluated by the QIAS 2025 organizers, and results will be shared with all participating teams.

---

✅ **READY TO PARTICIPATE?**

We are excited to see your contributions to Islamic AI benchmarking! 🚀
