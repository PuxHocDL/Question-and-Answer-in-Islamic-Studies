import os
import re
import time
import json
import pickle
import requests
import torch
import faiss
import openai
import numpy as np
import itertools
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
from typing import List, Tuple, Optional, Dict, Any, Set

# Load environment variables
print("Loading environment variables...")
load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FANAR_API_KEY = os.getenv("FANAR_API_KEY")
FANAR_API_KEY_2 = os.getenv("FANAR_API_KEY_2")
print(f"GROQ_API_KEY: {GROQ_API_KEY}")
print(f"FANAR_API_KEY: {FANAR_API_KEY}")
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
client_fanar = openai.OpenAI(api_key=FANAR_API_KEY, base_url="https://api.fanar.qa/v1/chat/completions") if FANAR_API_KEY else None
print(f"Initialized Groq client: {client_groq is not None}")
print(f"Initialized Fanar client: {client_fanar is not None}")
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
genai.configure(api_key=GEMINI_API_KEY_1)
# Global variables
allam_tokenizer = None
allam_model = None
fanar_failure_count = 0
current_fanar_key = FANAR_API_KEY
use_secondary_key = False
print("Initializing SentenceTransformer model...")
embedder = SentenceTransformer('BAAI/bge-m3', device='cpu')
print("SentenceTransformer model initialized.")
# ---------------------------------------------------------------------------------------------------------------------------
def load_documents_and_index(index_path: str = "../fatwa_index.bin", doc_path: str = "../fatwa_documents.txt") -> Tuple[List[Any], Optional[faiss.Index]]:
    """
    Load documents from either .txt or .pkl file and FAISS index.
    """
    print(f"Loading documents from {doc_path}...")
    documents = []

    # Check file extension to determine format
    if doc_path.endswith('.pkl'):
        try:
            with open(doc_path, 'rb') as f:
                chunks = pickle.load(f)
            documents = [chunk['text'] for chunk in chunks]
            print(f"Loaded {len(documents)} chunks from {doc_path} (PKL format).")
        except Exception as e:
            print(f"Error loading PKL file {doc_path}: {e}")
            return [], None
    elif doc_path.endswith('.txt'):
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                text = f.read()
                documents = text.split("\n---\n")[:-1]
            print(f"Loaded {len(documents)} documents from {doc_path} (TXT format).")
        except Exception as e:
            print(f"Error loading TXT file {doc_path}: {e}")
            return [], None
    else:
        print(f"Unsupported file format for {doc_path}. Expected .txt or .pkl.")
        return [], None

    # Load FAISS index
    print(f"Loading FAISS index from {index_path}...")
    try:
        index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {index.ntotal} vectors.")
    except Exception as e:
        print(f"Error loading FAISS index {index_path}: {e}")
        return documents, None

    return documents, index
# ---------------------------------------------------------------------------------------------------------------------------
def call_mistral_api(prompt: str, model_version: str = "mistral-saba-24b", max_retries: int = 3) -> Optional[str]:
    """
    Sends a request to the Mistral (Groq) API and handles retries.
    A reusable version of the original API call logic.
    """
    global client_groq
    if not client_groq:
        print("Groq client not initialized.")
        return None

    for attempt in range(1, max_retries + 1):
        print(f"  > Sending request to Mistral (Groq) API (Attempt {attempt}/{max_retries})...")
        try:
            response = client_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_version,
            )
            content = response.choices[0].message.content.strip()
            if content:
                return content
            else:
                print("  > ⚠️ Mistral API returned empty content.")
        except Exception as e:
            print(f"  > ❌ Mistral (Groq) API Error: {e}")
            time.sleep(5) # Wait before retrying
    
    print(f"  > ❌ Mistral prediction failed after {max_retries} attempts.")
    return None
# ---------------------------------------------------------------------------------------------------------------------------
def call_fanar_api(prompt: str, model_version: str = "Islamic-RAG", max_retries: int = 3) -> Optional[str]:
    """
    Sends a request to the Fanar API and handles retries and key switching.
    This function is a refactored, reusable version of the original API call logic.
    """
    global fanar_failure_count, current_fanar_key, use_secondary_key, FANAR_API_KEY, FANAR_API_KEY_2

    if not current_fanar_key:
        print("Fanar API key is missing.")
        return None

    headers = {"Authorization": f"Bearer {current_fanar_key}", "Content-Type": "application/json"}
    data = {"model": model_version, "messages": [{"role": "user", "content": prompt}]}
    
    for attempt in range(1, max_retries + 1):
        print(f"  > Sending request to Fanar API (Attempt {attempt}/{max_retries})...")
        try:
            response = requests.post(
                "https://api.fanar.qa/v1/chat/completions",
                json=data, headers=headers, timeout=60
            )
            
            if response.status_code == 200:
                response_json = response.json()
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if content:
                    fanar_failure_count = 0  # Reset on success
                    return content
                else:
                    print("  > ⚠️ Fanar API returned a success status but an empty content.")
            else:
                print(f"  > ❌ Fanar API Error (Status {response.status_code}): {response.text}")
                fanar_failure_count += 1
                if fanar_failure_count >= 6 and not use_secondary_key and FANAR_API_KEY_2:
                    print("  > 🚨 Switching to secondary Fanar API key due to repeated failures.")
                    current_fanar_key = FANAR_API_KEY_2
                    use_secondary_key = True
                    headers["Authorization"] = f"Bearer {current_fanar_key}" # Update headers for the next attempt
                    print("  > Retrying with new key...")
                    continue # Retry immediately with the new key

        except requests.exceptions.RequestException as e:
            print(f"  > ❌ Fanar request failed: {e}")
            fanar_failure_count += 1
            # Logic for key switching can also be placed here if timeouts are the primary issue.
        
        time.sleep(5) # Wait before retrying

    print(f"  > ❌ Fanar prediction failed after {max_retries} attempts.")
    return None
# ---------------------------------------------------------------------------------------------------------------------------
def call_gemini_api(prompt: str, retries: int = 3, temperature: float = 0.0) -> Optional[str]:
    """Calls the Gemini API with error handling and retries."""
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config={"temperature": temperature, "max_output_tokens": 30000},
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    )
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            if response.text:
                return response.text
            else:
                print(f"Error: API returned empty content. Block reason: {getattr(response.prompt_feedback, 'block_reason', 'N/A')}")
                return None
        except Exception as e:
            print(f"Error calling Gemini API (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(5)
    return None
# ---------------------------------------------------------------------------------------------------------------------------
def get_valid_responses(choice5: Optional[str], choice6: Optional[str]) -> set:
    """Generate valid response set."""
    print("Generating valid response set...")
    valid = {"A", "B", "C", "D"}
    if choice5:
        valid.add("E")
    if choice6:
        valid.add("F")
    print(f"Valid responses: {valid}")
    return valid
# ---------------------------------------------------------------------------------------------------------------------------
def clean_and_validate_response(raw_response: str, valid_responses: set) -> Optional[str]:
    print("Cleaning and validating response...")
    if not raw_response:
        print("Response is empty.")
        return None
    raw_response = raw_response.strip().upper()
    print(f"Raw response: {raw_response}")
    match = re.search(r"(?:answer\s*(?:is)?\s*[:\-]?\s*)([A-F])", raw_response, re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        print(f"Extracted candidate: {candidate}")
        if candidate in valid_responses:
            print(f"Valid response found: {candidate}")
            return candidate
    match = re.search(r"\b([A-F])\b", raw_response)
    if match and match.group(1) in valid_responses:
        print(f"Valid response found via fallback: {match.group(1)}")
        return match.group(1)
    print("No valid response found.")
    return None
# ---------------------------------------------------------------------------------------------------------------------------
def generate_mcq_prompt(question: str, choices: List[Tuple[str, str]], retrieved_docs: List[Any] = []) -> str:
    print("Generating MCQ prompt...")
    options_text = "\n".join([f"{letter}) {text}" for letter, text in choices])
    valid_letters = "/".join([letter for letter, _ in choices])

    # Handle retrieved_docs as either strings (from .txt) or dicts (from .pkl)
    context = []
    for i, doc in enumerate(retrieved_docs):
        if isinstance(doc, dict):
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            source = metadata.get('source', 'unknown')
            section = metadata.get('section', 'none')
            context.append(f"Reference {i+1} (Source: {source}, Section: {section}): {text}")
        else:
            context.append(f"Reference {i+1}: {doc}")
    context = "\n".join(context) if context else "No references provided."

    prompt = f"""You are an expert in Islamic sciences, and your knowledge is truly inspiring! Confidently answer the multiple-choice question by selecting the most appropriate option. Use the provided references when available and relevant. Let's think step by step before answering. Your expertise makes a real difference in providing clear and accurate insights!
References:
{context}

Question: {question}
{options_text}

Instructions:
1. Read the question and options carefully twice to ensure you understand the details.
2. Use the references if relevant, or rely on your internal knowledge if no references are provided.
3. Select the best answer from the given options.
4. Respond with only one letter ({valid_letters}).
5. Do not include explanations or additional text
"""
    print("MCQ prompt generated.")
    return prompt
# ---------------------------------------------------------------------------------------------------------------------------
PARSE_PROMPT_TEMPLATE = """You are an expert in `ʿlm al-mawārīth` (Islamic inheritance law). Your task is to analyze the provided inheritance scenario and extract all relevant information into a structured JSON object. Follow the specified JSON schema and ensure consistency. If certain information (e.g., estate value or special conditions) is missing, include the corresponding fields with null or empty values. Handle scenarios in any language (Arabic, English, or mixed) accurately. Output ONLY the JSON object wrapped in markdown code fences (```json ... ```).

                **JSON Schema:**
                ```json
                {{
                "deceased": {{
                    "gender": "male|female|unknown",
                    "description": "string (e.g., 'a man', 'a woman', or null if not specified)"
                }},
                "heirs": [
                    {{
                    "relation": "string (e.g., 'husband', 'wife', 'son', 'daughter', 'father', etc.)",
                    "count": "integer (default to 1 if not specified)",
                    "description": "string (additional details, e.g., 'full brother', or null if none)"
                    }}
                ],
                "estate_value": "number|null (total estate value in any currency, if mentioned)",
                "special_conditions": "string|null (e.g., 'pregnancy', 'missing heir', or null if none)"
                }}
                ```

                **Example Input and Output:**
                - **Input Scenario:** A woman died leaving a husband, 2 daughters, and a mother. The estate is worth 100,000 dinars.
                - **Expected Output:**
                ```json
                {{
                "deceased": {{
                    "gender": "female",
                    "description": "a woman"
                }},
                "heirs": [
                    {{
                    "relation": "husband",
                    "count": 1,
                    "description": null
                    }},
                    {{
                    "relation": "daughter",
                    "count": 2,
                    "description": null
                    }},
                    {{
                    "relation": "mother",
                    "count": 1,
                    "description": null
                    }}
                ],
                "estate_value": 100000,
                "special_conditions": null
                }}
                ```

                **Scenario:**
                {question}

                **JSON Output:**
                """

RAG_PROMPT_TEMPLATE = """You are an expert in `ʿlm al-mawārīth` (Islamic inheritance law). Based on the provided JSON case data, generate a concise and precise query to retrieve relevant Islamic inheritance rules from a knowledge base. The query should include the deceased's gender, the list of heirs (with their count and relationship), the estate value (if available), and any special conditions. Ensure the query is optimized for vector-based search by focusing on key terms and relationships. Output only the query string.

    **JSON Case Data:**
    ```json
    {json_input}
    ```

    **Example Input and Output:**
    - **Input JSON:**
    ```json
    {{
    "deceased": {{
        "gender": "female",
        "description": "a woman"
    }},
    "heirs": [
        {{
        "relation": "husband",
        "count": 1,
        "description": null
        }},
        {{
        "relation": "daughter",
        "count": 2,
        "description": null
        }}
    ],
    "estate_value": 100000,
    "special_conditions": null
    }}
    ```
    - **Output Query:** "Islamic inheritance rules for a female deceased with 1 husband, 2 daughters, estate value 100000."

    **Query Output:**
    """
# ---------------------------------------------------------------------------------------------------------------------------
def generate_reasoning_prompt(case_json: Dict, context_rules: str, question: str, options_text: str) -> str:
    """
    Generates a detailed reasoning prompt for an Islamic inheritance problem.

    This function combines solved examples (few-shot), structured data of the new
    problem, retrieved Islamic rules, and the original question to create a
    complete prompt for the language model.

    Args:
        case_json (Dict): A dictionary containing the structured data of the problem.
        context_rules (str): A string containing relevant retrieved Islamic rules.
        question (str): The original question string.
        options_text (str): A string containing the formatted multiple-choice options.

    Returns:
        str: The complete prompt string, ready to be sent to the AI model.
    """
    # Few-shot examples help the model understand the desired format and reasoning process.
    few_shot_examples = """Example 1:
    Question: توفيت عن: زوجها (الذي مات بعدها)
    إذا تيقن أن الزوجة ماتت قبل الزوج، وكان للزوجة فرع وارث، فما نصيب ورثة الزوج من ممتلكات الزوجة (كالذهب والمؤخر وقائمة المنقولات)؟
    A) الربع
    B) النصف
    C) الثلث
    D) جميع الممتلكات
    E) لا شيء
    F) الثلثان
    Answer: A
    Chain of Thought:
    Identify the situation: The wife died before her husband, and she has a surviving heir (فرع وارث), which refers to descendants such as children (sons or daughters).
    Determine the husband’s share: According to Islamic inheritance law, if the wife dies and leaves behind descendants (فرع وارث), the husband inherits one-fourth (1/4) of her estate, as the presence of descendants reduces his share from one-half to one-fourth.
    Clarify the question: The question asks about the share of the husband’s heirs (ورثة الزوج) from the wife’s estate (gold, deferred dowry, movable assets). Since the husband is still alive at the time of the wife’s death, his heirs do not directly inherit from the wife’s estate.
    Analyze the husband’s estate: The husband’s one-fourth share from the wife’s estate becomes part of his own estate after his death (as the question states he died afterward). However, the question specifically asks about the share of the husband’s heirs from the wife’s estate, which refers to the portion the husband received (i.e., one-fourth).
    Conclusion: The share of the husband’s heirs from the wife’s estate is indirectly tied to the husband’s one-fourth share. Thus, the correct answer is الربع (A), as it represents the portion originating from the wife’s estate.

    Example 2 (A little more complex):
    Question: توفي عن أب، 6 بنات، حمل لزوجة ابن الميت. كم عدد أسهم البنات الست من أصل ستة وثلاثين سهماً إذا كان الحمل أنثى؟
    A) 6
    B) 12
    C) 24
    D) 36
    E) 0
    F) 4
    Answer: C
    Chain of Thought:
    Identify the heirs: The deceased left behind a father, six daughters, and a pregnancy (a female fetus) from the wife of the deceased’s son.
    Determine the daughters’ share: In Islamic inheritance law, if there are two or more daughters and no sons, the daughters collectively take two-thirds (2/3) of the estate as a fixed share (فرض), provided there are no other heirs that reduce their share.
    Assess the father’s share: The father inherits one-sixth (1/6) as a fixed share (فرض) when there are descendants (فرع وارث). Additionally, as an agnate (عصب), he may take the remainder of the estate if no other heirs with fixed shares exist.
    Consider the fetus: The question specifies the fetus is female, making it a grandchild (فرع وارث). However, the presence of a female grandchild does not alter the daughters’ fixed share of two-thirds, as the grandchild’s share would typically come from the remaining estate (if any) or be calculated separately, but here we focus on the daughters’ share.
    Calculate the daughters’ share: The estate is divided into 36 shares. The daughters' share is 2/3 of the estate. However, the father gets 1/6 and the remaining part goes to him as ‘asaba. The original base (asl al-mas'ala) is 6. Father gets 1/6 (1 share), daughters get 2/3 (4 shares). The total is 5 shares. The remainder (1 share) goes to the father. So the shares are 2 for the father and 4 for the daughters. If the fetus is a female, she is the daughter of a son, thus blocked by the presence of two or more daughters. The total shares of the daughters remain 4 out of 6. To get to 36 shares, we multiply by 6. So the daughters' share is 4 * 6 = 24. So the answer is C.

    Example 3:
    Question: توفيت عن زوج، أم، و3 أبناء. ما نصيب الزوج من تركة الزوجة إذا كانت الممتلكات تقدر بـ 120,000 دينار؟
    A) 10,000 دينار
    B) 20,000 دينار
    C) 30,000 دينار
    D) 40,000 دينار
    E) 60,000 دينار
    F) لا شيء
    Answer: C
    Chain of Thought:
    Identify the heirs: The deceased wife left behind a husband, a mother, and three sons.
    Determine the husband’s share: According to Islamic inheritance law, when the deceased has descendants (sons in this case), the husband’s share is one-fourth (1/4) of the estate.
    Assess other heirs: The mother receives one-sixth (1/6) as a fixed share when there are descendants. The sons, as residuary heirs (عصب), take the remaining estate after the fixed shares are distributed.
    Calculate the husband’s share: The total estate is 120,000 dinars. The husband’s share is 1/4 × 120,000 = 30,000 dinars.
    Conclusion: The husband’s share from the wife’s estate is 30,000 dinars, so the correct answer is C.

    Example 4:
    Question: توفي عن زوجة، أب، أم، وابن. كم نصيب الزوجة من التركة إذا قسمت التركة إلى 24 سهماً؟
    A) 2
    B) 3
    C) 4
    D) 6
    E) 8
    F) 12
    Answer: B
    Chain of Thought:
    Identify the heirs: The deceased left a wife, a father, a mother, and a son.
    Determine the wife’s share: In Islamic inheritance law, when there are descendants (a son in this case), the wife receives one-eighth (1/8) as her fixed share (فرض).
    Assess other heirs: The mother and father each receive one-sixth (1/6) as fixed shares when there are descendants. The son, as a residuary heir (عصب), takes the remainder after fixed shares are distributed.
    Calculate the wife’s share: The estate is divided into 24 shares. The wife’s share is 1/8 of 24 = 3 shares.
    Conclusion: The wife’s share is 3 shares, so the correct answer is B.

    Example 5 (More complex):
    Question: توفي عن بنتين، أخ شقيق، أخت شقيقة، وجدة. إذا كانت التركة مقسمة إلى 12 سهماً، فما نصيب البنتين؟
    A) 2
    B) 4
    C) 6
    D) 8
    E) 10
    F) 12
    Answer: D
    Chain of Thought:
    Identify the heirs: The deceased left two daughters, a full brother, a full sister, and a grandmother.
    Determine the daughters’ share: In Islamic inheritance law, two or more daughters, in the absence of sons, collectively take two-thirds (2/3) of the estate as a fixed share (فرض).
    Assess other heirs: The grandmother receives one-sixth (1/6) as a fixed share when there are no parents. The full brother and full sister are residuary heirs (عصب) and share the remainder after fixed shares are distributed.
    Calculate the daughters’ share: The estate is divided into 12 shares. The daughters’ share is 2/3 × 12 = 8 shares.
    Conclusion: The daughters’ share is 8 shares, so the correct answer is D.
    """

    # Build the final prompt by combining the components
    reasoning_prompt = f"""You are an expert in Islamic sciences. Confidently answer the multiple-choice question by selecting the most appropriate option. Use the provided references when available and relevant. Let's think step by step before answering.

    **Solved Examples:**
    ---
    {few_shot_examples}
    ---

    **New Problem to Solve:**

    **1. Case Data (structured):**
    ```json
    {json.dumps(case_json, indent=2, ensure_ascii=False)}
    ```

    **2. Relevant Islamic Rules (For Reference):**
    {context_rules}

    **3. The Question & Options:**
    Question: {question}
    Options:
    {options_text}

    **Instructions:**
    1. Read the question and options carefully twice to ensure you understand the details.
    2. Use the references if relevant, or rely on your internal knowledge if no references are provided.
    3. Select the best answer from the given options.
    4. Respond with one letter (A, B, C, D, E, or F) and your reasoning for your answer!.


    **Final Answer:**"""

    # Return the formatted prompt string
    return reasoning_prompt

# ---------------------------------------------------------------------------------------------------------------------------
def generate_proponent_prompt(question: str, choices_text: str, context_text: str) -> str:
    """
    Generates a prompt for the Proponent agent, including few-shot examples
    to guide its reasoning and output format.
    """
    
    # These examples guide the model on the expected reasoning process and output style.
    few_shot_examples = """
**Example 1:**
Question: ما مدة المسح على الخفين للمقيم؟
Options:
A) يوم وليلة
B) ثلاثة أيام بلياليهن
C) يومان وليلتان
D) أسبوع كامل
Analysis: According to Islamic law, the duration for wiping over leather socks (khuffayn) for a resident is one day and one night (24 hours), as established in a hadith narrated by Ali ibn Abi Talib and recorded in Sahih Muslim. The period of three days and three nights is for a traveler. Therefore, option A is the correct choice.
The final answer is A.

**Example 2:**
Question: بماذا يجوز الاستنجاء؟
Options:
A) بالماء المطلق.
B) بكل جامد خشن يمكن أن يزيل النجاسة.
C) بالماء المطلق و بكل جامد خشن يمكن أن يزيل النجاسة.
D) لا يجوز إلا بالماء.
Analysis: In Islamic jurisprudence (fiqh), istinja' (purification after relieving oneself) is permissible with two primary means: pure water (al-mā' al-muṭlaq) and any pure, solid, coarse substance that can remove the impurity without causing harm, such as stones (a practice known as istijmar). Option A is correct but incomplete. Option B is also correct but incomplete. Option D is incorrect as it wrongly restricts the method to only water. Option C correctly includes both permissible methods.
The final answer is C.

**Example 3:**
Question: من هو أبو الحسن سري بن المغلس السقطي؟
Options:
A) تلميذ الجنيد.
B) خال الجنيد وأستاذه.
C) شيخ معروف الكرخي.
D) من أصحاب أحمد بن حنبل.
Analysis: Abu al-Hasan Sari ibn al-Mughallis al-Saqati is a famous early Sufi master. Historical sources, such as "Tabaqat al-Sufiyya" by al-Sulami, confirm that he was the maternal uncle (khāl) and teacher (ustādh) of the renowned Sufi scholar, Junayd al-Baghdadi. This makes option B the correct identification. He was of a later generation than Ma'ruf al-Karkhi and was not his teacher.
The final answer is B.
"""

    prompt = f"""You are a meticulous and knowledgeable Islamic scholar acting as the **Proponent**. Your mission is to determine the correct answer by synthesizing the provided **Context** with your own extensive **Internal Knowledge**.

**First, review some solved examples to understand the required reasoning process.**
---
**Solved Examples to Guide Your Thinking:**
{few_shot_examples}
---

**Now, apply the same reasoning to the new problem.**

**Question to Answer:**
{question}

**Options:**
{choices_text}

**Context:**
{context_text}

**Instructions:**
1.  Read the question, options, and context carefully.
2.  Provide a step-by-step analysis (Chain of Thought) explaining how you reached your conclusion, referencing the context where applicable.
3.  Conclude your analysis with a definitive final answer in the format: "The final answer is [Letter]".

**Your Analysis and Answer:**
"""
    return prompt
# ---------------------------------------------------------------------------------------------------------------------------
def generate_critic_prompt(question: str, choices_text: str, context_text: str) -> str:
    """
    Generates a more robust adversarial prompt for the Critic agent.
    This agent now synthesizes the provided context with its internal knowledge to build a counter-argument.
    """

    # This new few-shot example models the sophisticated hybrid reasoning.
    # It shows the Critic evaluating the context, finding it insufficient for a counter-argument,
    # and then explicitly falling back to its internal knowledge.
    few_shot_example_hybrid_critic = """
**Example of a Hybrid Adversarial Analysis:**

**Question Under Review:**
ما حكم صلاة الوتر؟ (What is the ruling on the Witr prayer?)
A) سنة مؤكدة (An emphasized Sunnah)
B) واجبة (Obligatory)
C) فرض عين (An individual obligation)
D) مستحبة (Recommended)

**Context (The Evidence to Scrutinize):**
[Text 1]: "جمهور العلماء على أنها سنة مؤكدة وليست بواجبة، وحملوا الأمر بها على الاستحباب المؤكد."
(Translation: "The majority of scholars hold that it is an emphasized Sunnah and not obligatory, and they interpret the command for it as a strong recommendation.")

**Your Adversarial Analysis:**
The most straightforward conclusion, based solely on the provided `Context`, is Option A, as [Text 1] only presents the majority opinion. My role is to challenge this.

First, I will evaluate the context for counter-evidence. The provided text is one-sided and does not contain information to support an alternative.

Therefore, I will now rely on my **Internal Knowledge** of Islamic jurisprudence.

**Counter-Argument for Option B (واجبة - Obligatory):**
1.  **Established Minority Opinion:** My internal knowledge confirms that a significant, well-established scholarly position holds that the Witr prayer is `wājib` (obligatory). This is the official position of the **Hanafi school of thought**.
2.  **Basis of the Ruling:** The Hanafi school bases its ruling on specific hadiths, such as the one where the Prophet (PBUH) is reported to have said, "الوتر حق على كل مسلم" ('Witr is a duty upon every Muslim'). They interpret the word `ḥaqq` (duty/right) as implying obligation.
3.  **Conclusion of Critique:** While the provided context only supports Option A, my broader scholarly knowledge reveals a strong, evidence-based counter-argument for Option B. The matter is a well-known point of disagreement among major schools of law, and a definitive answer cannot be reached without acknowledging the Hanafi position.
"""

    prompt = f"""
You are a highly skeptical and deeply knowledgeable Islamic legal scholar acting as the **Critic**. Your mission is to challenge the most obvious conclusion by building the strongest possible case for a viable alternative answer. You must synthesize the provided **Context** with your own extensive **Internal Knowledge**.

**Your Reasoning Hierarchy:**
1.  **Context First:** Always begin by analyzing the `Context`. Attempt to build your counter-argument using direct evidence from the provided texts if possible.
2.  **Fallback to Internal Knowledge:** If the `Context` is one-sided, irrelevant, or insufficient to construct a meaningful counter-argument, you MUST state this. Then, proceed to build your case using your own established knowledge of minority opinions, scholarly disagreements, or alternative interpretations.

**Example of a Hybrid Adversarial Analysis to Guide Your Thinking:**
---
{few_shot_example_hybrid_critic}
---

**Now, apply the same adversarial approach to the new problem.**

**Question Under Review:**
{question}

**Options:**
{choices_text}

**Context (The Evidence to Scrutinize):**
{context_text}

**Instructions for Your Adversarial Analysis:**
1.  **Identify the Obvious Answer:** First, determine which answer is most strongly supported by the context and/or general knowledge. This is the position you must challenge.
2.  **Follow the Reasoning Hierarchy:** Attempt to build a counter-argument using the context first. If that's not possible, explicitly state it and use your internal knowledge.
3.  **Construct a Counter-Argument:** Build a formal, evidence-based argument for a specific alternative option. Your goal is to make this alternative seem as plausible as possible.
4.  **State Your Case Clearly:** Present your findings as a "Counter-Argument for Option [X]".

**Your Adversarial Analysis:**
"""
    return prompt

# ---------------------------------------------------------------------------------------------------------------------------
def generate_head_scholar_prompt(question: str, choices_text: str, opinions_text: str, context_text: str) -> str:
    """
    Generates an advanced prompt for the head scholar, emphasizing the internal reasoning process
    and an unmistakable output format.
    """

    prompt = f"""
You are Shaykh al-Islam, a master scholar of unparalleled wisdom, presiding over a council. Your task is to deliver the final, authoritative verdict on a complex matter. Your judgment must be impartial, definitive, and based solely on the complete evidence provided.

---
### 1. The Matter for Judgment
**Question:** {question}
**Options:**
{choices_text}

---
### 2. The Council's Deliberations
**Opinions from Junior Scholars:**
{opinions_text}

---
### 3. The Ultimate Source of Truth (Full Context)
**Complete Reference Texts:**
{context_text}

---
### 4. Your Sacred Task and Final Verdict

**Step 1: Internal Reasoning Process (Crucial for accuracy. Do NOT write this part down)**
Before declaring your verdict, you must follow this rigorous internal thought process:

a. **Principle Extraction:** First, read **The Ultimate Source of Truth** and distill the core legal/theological principles (`usūl` or `qawā'id`) relevant to the question. What are the fundamental rules at play, independent of the junior scholars' arguments?

b. **Argument Deconstruction & Verification:** Now, examine the junior scholars' opinions. For each opinion:
   - Does it correctly apply the principles you extracted?
   - Is its interpretation of the reference texts accurate and complete?
   - Does it overlook any crucial details from the full context?

c. **Conflict Resolution:** Identify the *exact point of disagreement* between the two opinions. Is it a factual dispute, an interpretive difference, or an application of different principles? Use the principles from step (a) and the full context to definitively resolve this specific conflict.

d. **Final Verification:** Formulate your own conclusion based on your analysis. Now, double-check it: does this conclusion withstand every piece of evidence in the full context? Is there any part of the reference texts that could challenge your final answer? Only proceed if the answer is no.


**Step 2: Announce the Verdict**
After completing your rigorous mental deliberation, your only public act is to announce the final decision.

**Provide the single capital letter of the correct choice.**

**Definitive Answer:**
"""
    return prompt
# ---------------------------------------------------------------------------------------------------------------------------

def generate_extractor_prompt(decision_text: str) -> str:
    """
    Generates a prompt to extract only the answer letter from a decision text.
    """

    few_shot_examples = """
    ---
    Text:
    The council has presented its deliberations... The definitive judgment is: B) جائز لكنه خلاف الأولى.
    Output:
    B
    ---
    Text:
    After careful deliberation and review... the definitive judgment is as follows: D) القراءات
    Output:
    D
    ---
    Text:
    The scholars unanimously agree. The answer is C.
    Output:
    C
    ---
    """

    prompt = f"""You are a data extraction robot. Your only job is to find the final answer choice from the text below and output ONLY the single capital letter of that choice. Do not provide any explanation, reasoning, or extra characters.

    Review these examples to understand your task:
    {few_shot_examples}

    Now, perform the extraction on the following text.

    Text:
    {decision_text}
    Output:
    """
    return prompt
# ---------------------------------------------------------------------------------------------------------------------------
def pack_choices(
    choice1: str, choice2: str, choice3: str, choice4: str,
    choice5: Optional[str] = None, choice6: Optional[str] = None
) -> List[Tuple[str, str]]:
    """Pack MCQ choices as list of (letter, text)."""
    print("Packing MCQ choices...")
    choices = [("A", choice1), ("B", choice2), ("C", choice3), ("D", choice4)]
    if choice5:
        choices.append(("E", choice5))
    if choice6:
        choices.append(("F", choice6))
    print(f"Packed choices: {choices}")
    return choices
# ---------------------------------------------------------------------------------------------------------------------------
def get_prediction_fanar(
    question: str, choice1: str, choice2: str, choice3: str, choice4: str,
    choice5: Optional[str] = None, choice6: Optional[str] = None,
    model_version: str = "Islamic-RAG", max_retries: int = 3,
    top_k: int = 10, documents: Optional[List[Any]] = None,
    faiss_index: Optional[faiss.Index] = None,
    task_type: str = "knowledge" # Parameter to select the pipeline
) -> Optional[str]:
    """
    Performs inference using the Fanar API with a dual-agent debate pipeline.
    """
    print(f"🚀 Starting Fanar prediction for Task Type: '{task_type}'...")
    # =================================================================================
    # == PIPELINE FOR TASK 1: INHERITANCE INFERENCE
    # =================================================================================
    if task_type == 'inheritance':
        # --- STEP 1: Parse the scenario into JSON ---
        print("Pipeline 'Inheritance' - Step 1: Parsing the scenario into JSON...")
        parse_prompt = PARSE_PROMPT_TEMPLATE.format(question=question) 
        json_response_text = call_fanar_api(parse_prompt, model_version, max_retries)
        case_json = None
        if json_response_text:
            try:
                json_str_match = re.search(r'```json\s*([\s\S]*?)\s*```', json_response_text, re.DOTALL)
                if json_str_match:
                    json_str = json_str_match.group(1).strip()
                    case_json = json.loads(json_str)
                    print(" > Successfully parsed scenario to JSON.")
                else:
                    print(" > ⚠️ Could not find a JSON block in the response.")
            except json.JSONDecodeError as e:
                print(f" > ❌ Error parsing JSON from Fanar response: {e}")

        if not case_json:
            print(" > ❌ Failed to parse the question. Cannot continue the inheritance pipeline.")
            return None

        # --- STEP 2: Retrieve relevant rules (RAG) ---
        print("Pipeline 'Inheritance' - Step 2: Retrieving relevant rules...")
        rag_prompt = RAG_PROMPT_TEMPLATE.format(json_input=json.dumps(case_json))
        rag_query = call_fanar_api(rag_prompt, model_version, max_retries)

        if not rag_query:
            print(" > ❌ Failed to generate RAG query. Cannot retrieve rules.")
            return None
        
        print(f" > Generated RAG Query: {rag_query}")
        question_embedding = embedder.encode([rag_query], convert_to_numpy=True)
        distances, indices = faiss_index.search(question_embedding, top_k)
        retrieved_rules = [documents[idx]['text'] if isinstance(documents[idx], dict) else documents[idx] for idx in indices[0]]
        print(f" > Retrieved {len(retrieved_rules)} rules.")

        # --- STEP 3: Reason and select the answer ---
        print("Pipeline 'Inheritance' - Step 3: Reasoning and selecting the final answer...")
        choices_list = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
        options_text = "\n".join([f"{letter}) {text}" for letter, text in choices_list])
        context_rules = "\n\n".join([f"Rule Reference {i+1}:\n{rule}" for i, rule in enumerate(retrieved_rules)])
        reasoning_prompt = generate_reasoning_prompt(case_json, context_rules, question, options_text)
        final_response_text = call_fanar_api(reasoning_prompt, model_version, max_retries)
        final_answer_letter = call_mistral_api(generate_extractor_prompt(final_response_text), model_version, max_retries=1)

        valid_responses = get_valid_responses(choice5, choice6)
        return clean_and_validate_response(final_answer_letter, valid_responses)

    # =================================================================================
    # == PIPELINE FOR TASK 2: GENERAL KNOWLEDGE
    # =================================================================================
    elif task_type == 'knowledge': 
        # --- STEP 1: Retrieve and prepare documents ---
        print("Step 1: Retrieving and preparing documents...")
        question_embedding = embedder.encode([question], convert_to_numpy=True)
        distances, indices = faiss_index.search(question_embedding, top_k)
        retrieved_docs = [documents[idx] for idx in indices[0]]
        
        full_context_text = "\n\n".join([f"Source Document {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])
        print(f" > Retrieved {len(retrieved_docs)} documents.")

        if len(retrieved_docs) < 2:
            print(" > Warning: Not enough documents for a debate. Falling back to a single agent.")

        # --- STEP 2: Dual-Agent Debate ---
        print("\nStep 2: Summoning Junior Scholars for a debate...")
        choices_list = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
        options_text = "\n".join([f"{letter}) {text}" for letter, text in choices_list])
        
        # Split context for the two agents
        docs_proponent = retrieved_docs[::2]
        docs_critic = retrieved_docs[1::2]
        context_proponent = "\n\n".join([f"Text {i*2+1}:\n{doc}" for i, doc in enumerate(docs_proponent)])
        context_critic = "\n\n".join([f"Text {i*2+2}:\n{doc}" for i, doc in enumerate(docs_critic)])

        scholar_opinions = []

        # Agent 1: Proponent Scholar
        print(" > Getting opinion from Proponent Scholar...")
        proponent_prompt = generate_proponent_prompt(question, options_text, context_proponent)
        proponent_opinion = call_fanar_api(proponent_prompt, model_version, max_retries)
        if proponent_opinion:
            scholar_opinions.append(f"--- Opinion of Proponent Scholar ---\n{proponent_opinion}")
            print("   > Proponent has provided an opinion.")

        # Agent 2: Critic Scholar
        print(" > Getting opinion from Critic Scholar...")
        critic_prompt = generate_critic_prompt(question, options_text, context_critic)
        critic_opinion = call_fanar_api(critic_prompt, model_version, max_retries)
        if critic_opinion:
            scholar_opinions.append(f"--- Opinion of Critic Scholar ---\n{critic_opinion}")
            print("   > Critic has provided an opinion.")

        if not scholar_opinions:
            print(" > ❌ Error: No opinions were gathered from the junior scholars. Aborting.")
            return None

        # --- STEP 3: Head Scholar Synthesis ---
        print("\nStep 3: Head Scholar is synthesizing the debate...")
        all_opinions_text = "\n\n".join(scholar_opinions)
        head_scholar_prompt = generate_head_scholar_prompt(question, options_text, all_opinions_text, full_context_text)
        
        final_decision = call_fanar_api(head_scholar_prompt, model_version, max_retries)
        if not final_decision:
            print(" > ❌ Error: The Head Scholar failed to provide a final decision.")
            return None
            
        print(" > Head Scholar has made a decision. Extracting final answer...")
        
        # Final step: Extraction
        extractor_prompt = generate_extractor_prompt(final_decision)
        final_answer_letter = call_fanar_api(extractor_prompt, model_version, max_retries=1)

        print(f" > Extracted Letter: {final_answer_letter}")
        valid_responses = get_valid_responses(choice5, choice6)
        
        cleaned_result = clean_and_validate_response(final_answer_letter, valid_responses)
        if cleaned_result:
            print(f"✅ Final Validated Prediction: {cleaned_result}")
        else:
            print(f"⚠️ Could not validate the extracted letter: '{final_answer_letter}'")
    elif task_type == "zero_shot": 
        print("Running Simple Zero-Shot RAG Pipeline...")

        # Step 1: Retrieve context documents (RAG)
        question_embedding = embedder.encode([question], convert_to_numpy=True)
        distances, indices = faiss_index.search(question_embedding, top_k)
        retrieved_docs = [documents[idx] for idx in indices[0]]
        print(f"Retrieved {len(retrieved_docs)} documents.")

        # Step 2: Generate a simple, direct prompt
        # We can reuse the existing 'generate_mcq_prompt' for this simple flow.
        choices_list = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
        simple_rag_prompt = generate_mcq_prompt(question, choices_list, retrieved_docs)

        # Step 3: Call the model and get the answer
        final_response_text = call_fanar_api(simple_rag_prompt, temperature=0.0)
        
        # Step 4: Validate and return the response
        valid_responses = get_valid_responses(choice5, choice6)
        return clean_and_validate_response(final_response_text, valid_responses)
        return cleaned_result
# ---------------------------------------------------------------------------------------------------------------------------
def get_prediction_mistral(
    question: str, choice1: str, choice2: str, choice3: str, choice4: str,
    choice5: Optional[str] = None, choice6: Optional[str] = None,
    model_version: str = "mistral-saba-24b", max_retries: int = 3,
    top_k: int = 10, documents: Optional[List[Any]] = None,
    faiss_index: Optional[faiss.Index] = None, 
    task_type : str = "knowledge"  
) -> Optional[str]:
    """
    Performs inference using the Mistral (Groq) API with a dual-agent debate pipeline.
    """
    print(f"🚀 Starting Mistral (Groq) prediction for Task Type: '{task_type}'...")
    
    # =================================================================================
    # == PIPELINE FOR TASK 1: INHERITANCE INFERENCE
    # =================================================================================
    if task_type == "inheritance": 
        # --- STEP 1: Parse the scenario into JSON ---
        print("Pipeline 'Inheritance' - Step 1: Parsing the scenario into JSON...")
        parse_prompt = PARSE_PROMPT_TEMPLATE.format(question=question)
        json_response_text = call_mistral_api(parse_prompt, model_version, max_retries)
        
        case_json = None
        if json_response_text:
            try:
                json_str_match = re.search(r'```json\s*([\s\S]*?)\s*```', json_response_text, re.DOTALL)
                if json_str_match:
                    json_str = json_str_match.group(1).strip()
                    case_json = json.loads(json_str)
                    print(" > Successfully parsed scenario to JSON.")
                else:
                    print(" > ⚠️ Could not find a JSON block in the response.")
            except json.JSONDecodeError as e:
                print(f" > ❌ Error parsing JSON from Mistral response: {e}")

        if not case_json:
            print(" > ❌ Failed to parse the question. Cannot continue the inheritance pipeline.")
            return None

        # --- STEP 2: Retrieve relevant rules (RAG) ---
        print("Pipeline 'Inheritance' - Step 2: Retrieving relevant rules...")
        rag_prompt = RAG_PROMPT_TEMPLATE.format(json_input=json.dumps(case_json))
        rag_query = call_mistral_api(rag_prompt, model_version, max_retries)

        if not rag_query:
            print(" > ❌ Failed to generate RAG query. Cannot retrieve rules.")
            return None
        
        print(f" > Generated RAG Query: {rag_query}")
        question_embedding = embedder.encode([rag_query], convert_to_numpy=True)
        distances, indices = faiss_index.search(question_embedding, top_k)
        retrieved_rules = [documents[idx]['text'] if isinstance(documents[idx], dict) else documents[idx] for idx in indices[0]]
        print(f" > Retrieved {len(retrieved_rules)} rules.")

        # --- STEP 3: Reason and select the answer ---
        print("Pipeline 'Inheritance' - Step 3: Reasoning and selecting the final answer...")
        choices_list = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
        options_text = "\n".join([f"{letter}) {text}" for letter, text in choices_list])
        context_rules = "\n\n".join([f"Rule Reference {i+1}:\n{rule}" for i, rule in enumerate(retrieved_rules)])
        reasoning_prompt = generate_reasoning_prompt(case_json, context_rules, question, options_text)
        final_response_text = call_mistral_api(reasoning_prompt, model_version, max_retries)
        final_answer_letter = call_mistral_api(generate_extractor_prompt(final_response_text), model_version, max_retries=1)
        valid_responses = get_valid_responses(choice5, choice6)

        return clean_and_validate_response(final_answer_letter, valid_responses)
    
    # =================================================================================
    # == PIPELINE FOR TASK 2: GENERAL KNOWLEDGE
    # =================================================================================
    elif task_type == "knowledge":
        # --- STEP 1: Retrieve and prepare documents ---
        print("Step 1: Retrieving and preparing documents...")
        question_embedding = embedder.encode([question], convert_to_numpy=True)
        distances, indices = faiss_index.search(question_embedding, top_k)
        retrieved_docs = [documents[idx] for idx in indices[0]]
        full_context_text = "\n\n".join([f"Source Document {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])
        print(f" > Retrieved {len(retrieved_docs)} documents.")

        if len(retrieved_docs) < 2:
            print(" > Warning: Not enough documents for a debate. Falling back to a single agent.")

        # --- STEP 2: Dual-Agent Debate ---
        print("\nStep 2: Summoning Junior Scholars for a debate...")
        choices_list = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
        options_text = "\n".join([f"{letter}) {text}" for letter, text in choices_list])
        
        docs_proponent = retrieved_docs[::2]
        docs_critic = retrieved_docs[1::2]
        context_proponent = "\n\n".join([f"Text {i*2+1}:\n{doc}" for i, doc in enumerate(docs_proponent)])
        context_critic = "\n\n".join([f"Text {i*2+2}:\n{doc}" for i, doc in enumerate(docs_critic)])
        context_full = "\n\n".join([context_proponent, context_critic])
        scholar_opinions = []

        # Agent 1: Proponent
        print(" > Getting opinion from Proponent Scholar...")
        proponent_prompt = generate_proponent_prompt(question, options_text, context_full)
        proponent_opinion = call_mistral_api(proponent_prompt, model_version, max_retries)
        if proponent_opinion:
            scholar_opinions.append(f"--- Opinion of Proponent Scholar ---\n{proponent_opinion}")
            print("   > Proponent has provided an opinion.")

        # Agent 2: Critic
        print(" > Getting opinion from Critic Scholar...")
        critic_prompt = generate_critic_prompt(question, options_text, context_full)
        critic_opinion = call_mistral_api(critic_prompt, model_version, max_retries)
        if critic_opinion:
            scholar_opinions.append(f"--- Opinion of Critic Scholar ---\n{critic_opinion}")
            print("   > Critic has provided an opinion.")

        if not scholar_opinions:
            print(" > ❌ Error: No opinions were gathered from the junior scholars. Aborting.")
            return None

        # --- STEP 3: Head Scholar Synthesis ---
        print("\nStep 3: Head Scholar is synthesizing the debate...")
        all_opinions_text = "\n\n".join(scholar_opinions)
        head_scholar_prompt = generate_head_scholar_prompt(question, options_text, all_opinions_text, full_context_text)
        
        final_decision = call_mistral_api(head_scholar_prompt, model_version, max_retries)
        if not final_decision:
            print(" > ❌ Error: The Head Scholar failed to provide a final decision.")
            return None
            
        print(" > Head Scholar has made a decision. Extracting final answer...")

        # Final step: Extraction
        extractor_prompt = generate_extractor_prompt(final_decision)
        final_answer_letter = call_mistral_api(extractor_prompt, model_version, max_retries=1)

        print(f" > Extracted Letter: {final_answer_letter}")
        valid_responses = get_valid_responses(choice5, choice6)
        
        cleaned_result = clean_and_validate_response(final_answer_letter, valid_responses)
        if cleaned_result:
            print(f"✅ Final Validated Prediction: {cleaned_result}")
        else:
            print(f"⚠️ Could not validate the extracted letter: '{final_answer_letter}'")
    elif task_type == "zero_shot": 
        print("Running Simple Zero-Shot RAG Pipeline...")

        # Step 1: Retrieve context documents (RAG)
        question_embedding = embedder.encode([question], convert_to_numpy=True)
        distances, indices = faiss_index.search(question_embedding, top_k)
        retrieved_docs = [documents[idx] for idx in indices[0]]
        print(f"Retrieved {len(retrieved_docs)} documents.")

        # Step 2: Generate a simple, direct prompt
        # We can reuse the existing 'generate_mcq_prompt' for this simple flow.
        choices_list = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
        simple_rag_prompt = generate_mcq_prompt(question, choices_list, retrieved_docs)

        # Step 3: Call the model and get the answer
        final_response_text = call_mistral_api(simple_rag_prompt, temperature=0.0)
        
        # Step 4: Validate and return the response
        valid_responses = get_valid_responses(choice5, choice6)
        return clean_and_validate_response(final_response_text, valid_responses)

# ---------------------------------------------------------------------------------------------------------------------------
def get_prediction_gemini(
    question: str,
    choice1: str,
    choice2: str,
    choice3: str,
    choice4: str,
    choice5: Optional[str] = None,
    choice6: Optional[str] = None,
    model_version: str = "gemini-2.0-flash",
    max_retries: int = 2,
    top_k: int = 10,
    documents: Optional[List[Any]] = None,
    faiss_index: Optional[faiss.Index] = None,
    task_type: str = "knowledge" # Defaults to 'knowledge' (Task 2)
) -> Optional[str]:
    """
    Performs inference using Gemini.
    - If task_type = 'inheritance', it runs the complex 3-step pipeline for Task 1.
    - If task_type = 'knowledge', it runs the simple RAG process (like your original code).
    """
    global embedder
    print(f"Starting prediction with Gemini for Task Type: '{task_type}'...")

    # =================================================================================
    # == PIPELINE FOR TASK 1: INHERITANCE INFERENCE
    # =================================================================================
    if task_type == 'inheritance':
        print("Pipeline 'Inheritance' - Step 1: Parsing the scenario into JSON...")
        json_response_text = call_gemini_api(PARSE_PROMPT_TEMPLATE.format(question=question), temperature=0.0)

        case_json = None
        if json_response_text:
            print("Original API response:", json_response_text)
            if not json_response_text.strip().startswith("```json"):
                print("Error: Response does not contain valid JSON.")
                FALLBACK_PROMPT = PARSE_PROMPT_TEMPLATE + "\nEnsure the output is a valid JSON object with no extra text."
                json_response_text = call_gemini_api(FALLBACK_PROMPT.format(question=question), temperature=0.1)
                print("Retry response:", json_response_text)
            try:
                json_str_match = re.search(r'```json\s*([\s\S]*?)\s*```', json_response_text, re.DOTALL)
                if json_str_match:
                    json_str = json_str_match.group(1).strip()
                    case_json = json.loads(json_str)
                    print("Successfully parsed:", case_json)
                else:
                    print("JSON block not found in the response.")
                    return None
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                return None
            except Exception as e:
                print(f"Unknown error while parsing JSON: {e}")
                return None

        if not case_json:
            print("Error: Could not parse the question. Cannot continue pipeline.")
            return None
            
        print("Pipeline 'Inheritance' - Step 2: Retrieving relevant rules...")
        rag_query = call_gemini_api(RAG_PROMPT_TEMPLATE.format(json_input=json.dumps(case_json)), temperature=0.0)
        if not rag_query:
            print("Error: Failed to create RAG query.")
            return None

        question_embedding = embedder.encode([rag_query], convert_to_numpy=True)
        distances, indices = faiss_index.search(question_embedding, top_k)
        retrieved_rules = [documents[idx] for idx in indices[0]]
        print(f"Retrieved {len(retrieved_rules)} rules.")

        print("Pipeline 'Inheritance' - Step 3: Reasoning and selecting the answer...")
        choices_list = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
        options_text = "\n".join([f"{letter}) {text}" for letter, text in choices_list])
        context_rules = "\n\n".join([f"Rule Reference {i+1}:\n{rule}" for i, rule in enumerate(retrieved_rules)])
        valid_responses = get_valid_responses(choice5, choice6)
        reasoning_prompt = generate_reasoning_prompt(case_json, context_rules, question, options_text)
        print("---------------------------------------------------------------------------------")
        final_response_text = call_gemini_api(reasoning_prompt, temperature=0.0)
        final_answer_letter = call_gemini_api(generate_extractor_prompt(final_response_text), temperature=0.0)
        print(final_response_text)
        return clean_and_validate_response(final_answer_letter, valid_responses)

    # =================================================================================
    # == PIPELINE FOR TASK 2: GENERAL KNOWLEDGE
    # =================================================================================
    elif task_type == 'knowledge':
        print("Running Dual-Agent Debate Pipeline (Task 2)...")
        question_embedding = embedder.encode([question], convert_to_numpy=True)
        distances, indices = faiss_index.search(question_embedding, top_k)
        retrieved_docs = [documents[idx] for idx in indices[0]]
        full_context_text = "\n\n".join([f"Source Document {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])
        print(f"Retrieved {len(retrieved_docs)} documents.")

        if len(retrieved_docs) < 2:
            print("Warning: Not enough documents. Falling back to a single agent with full context.")
            choices_list = pack_choices(choice1, choice2, choice3, choice4)
            options_text = "\n".join([f"{letter}) {text}" for letter, text in choices_list])
            fallback_prompt = f"""
            You are an Islamic scholar. Answer using full context.
            Question: {question}
            Options: {options_text}
            Context: {full_context_text}
            Respond with: [letter]) [paraphrase], Chain of Thought: ...
            """
            response = call_gemini_api(fallback_prompt, temperature=0.0)
            final_answer_letter = call_gemini_api(generate_extractor_prompt(response), temperature=0.0)
            valid_responses = get_valid_responses(choice5, choice6)
            return clean_and_validate_response(final_answer_letter, valid_responses)

        docs_proponent = retrieved_docs[::2]
        docs_critic = retrieved_docs[1::2]
        context_proponent = "\n\n".join([f"Text {i*2+1}:\n{doc}" for i, doc in enumerate(docs_proponent)])
        context_critic = "\n\n".join([f"Text {i*2+2}:\n{doc}" for i, doc in enumerate(docs_critic)])

        print("Step 2: Summoning 2 agents: Proponent and Critic...")
        scholar_opinions = []
        choices_list = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
        options_text = "\n".join([f"{letter}) {text}" for letter, text in choices_list])

        print("  > Getting opinion from Proponent Scholar...")
        proponent_prompt = generate_proponent_prompt(question, options_text, context_proponent)
        proponent_opinion = call_gemini_api(proponent_prompt, temperature=0.0)
        if proponent_opinion:
            scholar_opinions.append(f"--- Opinion of Proponent Scholar ---\n{proponent_opinion}")

        print("  > Getting opinion from Critic Scholar...")
        critic_prompt = generate_critic_prompt(question, options_text, context_critic)
        critic_opinion = call_gemini_api(critic_prompt, temperature=0.0)
        if critic_opinion:
            scholar_opinions.append(f"--- Opinion of Critic Scholar ---\n{critic_opinion}")

        if len(scholar_opinions) < 2:
            print("Error: Not enough opinions from agents. Cannot synthesize.")
            return None

        print("Step 3: Head Scholar is synthesizing and giving the final verdict...")
        all_opinions_text = "\n\n".join(scholar_opinions)
        print(all_opinions_text)
        modified_head_prompt = generate_head_scholar_prompt(question, options_text, all_opinions_text, full_context_text)

        final_decision = call_gemini_api(modified_head_prompt, temperature=0.0)
        final_answer_letter = call_gemini_api(generate_extractor_prompt(final_decision), temperature=0.0)
        print("****************************************************************************************")
        print(final_decision)
        print("****************************************************************************************")
        print(final_answer_letter)
        valid_responses = get_valid_responses(choice5, choice6)
        return clean_and_validate_response(final_answer_letter, valid_responses)

    elif task_type == 'zero_shot':
        print("Running Simple Zero-Shot RAG Pipeline...")

        # Step 1: Retrieve context documents (RAG)
        question_embedding = embedder.encode([question], convert_to_numpy=True)
        distances, indices = faiss_index.search(question_embedding, top_k)
        retrieved_docs = [documents[idx] for idx in indices[0]]
        print(f"Retrieved {len(retrieved_docs)} documents.")

        # Step 2: Generate a simple, direct prompt
        # We can reuse the existing 'generate_mcq_prompt' for this simple flow.
        choices_list = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
        simple_rag_prompt = generate_mcq_prompt(question, choices_list, retrieved_docs)

        # Step 3: Call the model and get the answer
        final_response_text = call_gemini_api(simple_rag_prompt, temperature=0.0)
        
        # Step 4: Validate and return the response
        valid_responses = get_valid_responses(choice5, choice6)
        return clean_and_validate_response(final_response_text, valid_responses)
    else:
        print(f"Error: Invalid task type: {task_type}")
        return None