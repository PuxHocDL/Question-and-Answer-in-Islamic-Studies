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

### 📜 Subtask 1: Islamic Inheritance (Mirāth)

- Focuses on inheritance-related problems (ʿIlm al-Mawārīth), requiring precise rule-based reasoning aligned with Islamic jurisprudence.
- Dataset:
  - 8,000 MCQs (train)
  - 2,000 MCQs (validation)
  - 500 MCQs (test)
- Extra data:
  - 32,000 IslamWeb fatwas
  - 2,500 open-ended Q&A pairs
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
