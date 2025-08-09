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
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer   
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
from typing import List, Tuple, Optional, Dict, Any

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
def load_documents_and_index(index_path: str = "../islamic_index.bin", doc_path: str = "../islamic_chunks.pkl") -> Tuple[List[Any], Optional[faiss.Index]]:
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
def call_gemini_api(prompt: str, retries: int = 3, temperature: float = 0.1) -> Optional[str]:
    """Hàm gọi API Gemini với xử lý lỗi và thử lại."""
    # ... (code của hàm này giữ nguyên) ...
    # Bạn có thể copy hàm này từ câu trả lời trước của tôi
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={"temperature": temperature, "max_output_tokens": 8000},
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    )
    for attempt in range(retries):
        try:
            # Tạm thời giả định bạn đã cấu hình key ở đâu đó trong code chính
            # genai.configure(api_key=...) 
            response = model.generate_content(prompt)
            if response.text:
                return response.text
            else:
                print(f"Lỗi: API trả về nội dung rỗng. Block reason: {getattr(response.prompt_feedback, 'block_reason', 'N/A')}")
                return None
        except Exception as e:
            print(f"Lỗi khi gọi API Gemini (lần {attempt + 1}/{retries}): {e}")
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

    few_shot_examples = """Example 1:
Question: ما مدة المسح على الخفين للمقيم؟
A) يوم وليلة
B) ثلاثة أيام بلياليهن
C) يومان وليلتان
D) أسبوع كامل
Answer: A
Chain of Thought:
According to Islamic law, the duration for wiping over leather socks for a resident is one day and one night (24 hours), based on a Hadith in Sahih Muslim.
Three days and three nights apply to travelers, so option B is incorrect.
Two days and two nights and a full week have no basis in Islamic law, so options C and D are incorrect.
The correct answer is A.
Output: A) One day and one night

Example 2 (A little more complex):
Question: ما القول الذي ذكره ابن الرفعة في فعل النبي صلى الله عليه وسلم للمكروه؟
A) أنه يفعل المكروه لبيان الحرمة
B) أنه يفعل المكروه لبيان الجواز ويكون أفضل في حقه
C) أنه يفعل المكروه ولا يكون أفضل في حقه
D) كل الأجوبة صحيحة
Answer: B
Chain of Thought:
Makruh is an act that is disliked but not forbidden. The Prophet only performed makruh acts to demonstrate their permissibility (jawaz).
According to Ibn al-Rif'ah (Shafi'i school), the Prophet’s makruh act was the best course (afdal) in his context due to its educational purpose.
Option A is incorrect because makruh does not relate to prohibition (haram).
Option C is incorrect because the Prophet’s actions are always the best in his context.
Option D is incorrect because A and C are wrong.
The correct answer is B.
Output: B) He performed the makruh act to demonstrate its permissibility, and it was the best course for him.

Example 3: 
من هو أبو الحسن سري بن المغلس السقطي؟
A) تلميذ الجنيد.
B) خال الجنيد وأستاذه.
C) شيخ معروف الكرخي.
D) من أصحاب أحمد بن حنبل.
Answer: B
Chain of Thought:
Abu al-Hasan Sari al-Saqati is a renowned figure in Islamic Sufism, known as the uncle (khāl) and teacher (ustādh) of al-Junayd al-Baghdadi, a prominent Sufi scholar.
Option A is incorrect because Sari was not a disciple of al-Junayd; rather, he was his teacher.
Option C is incorrect because Sari was not the teacher of Ma'ruf al-Karkhi; Ma'ruf belonged to an earlier generation.
Option D is incorrect as there is no historical evidence linking Sari as a companion of Ahmad ibn Hanbal.
Option B is correct because Sari was both al-Junayd’s uncle and teacher.
Output: B) Al-Junayd’s uncle and teacher.

Example 4: 
بماذا يجوز الاستنجاء؟
A) بالماء المطلق.
B) بكل جامد خشن يمكن أن يزيل النجاسة.
C) بالماء المطلق و بكل جامد خشن يمكن أن يزيل النجاسة.
D) لا يجوز إلا بالماء.
Answer: C
Chain of Thought:
In Islamic law (fiqh), istinja' (purification after relieving oneself) can be performed using pure water (mā' mutlaq) or rough solids (like stones, known as istijmar), provided they remove the impurity (najasa).
Option A is partially correct as pure water is permissible but incomplete as it omits istijmar.
Option B is partially correct as rough solids are permissible but omits water.
Option D is incorrect because Islamic law allows both water and solids, not only water.
Option C is correct as it includes both pure water and rough solids, aligning with fiqh rulings.
Output: C) Pure water and any rough solid that can remove impurity.

Example 5: 
ما هي الصفات التي لا تعلق لها عند الأشاعرة؟
A) الصفات الثبوتية.
B) صفات الأفعال.
C) الصفات المتعلقة.
D) الصفات السلبية والمعنوية.
Answer: D
Chain of Thought:
According to the Ash'ari theological school, Allah’s attributes are categorized into affirmative (thubūtiyya), action (af'āl), negative (salbiyya), and meaning-based (ma'nawiyya) attributes.
Negative attributes (salbiyya) negate imperfections from Allah (e.g., having no beginning or end), and meaning-based attributes (ma'nawiyya) relate to abstract conceptual meanings. These are not "associated" (ta'allaq) with specific actions or objects, unlike action attributes.
Option A is incorrect because affirmative attributes (thubūtiyya), like knowledge (ilm), are essential attributes of Allah.
Option B is incorrect because action attributes (af'āl), like creating (khaliq), relate to specific actions.
Option C is incorrect as "relational attributes" (muta'allaqa) is not a recognized category in Ash'ari theology.
Option D is correct because negative and meaning-based attributes do not relate to specific actions.
Output: D) Negative and meaning-based attributes.
"""
    prompt = f"""You are an expert in Islamic sciences, and your knowledge is truly inspiring! Confidently answer the multiple-choice question by selecting the most appropriate option. Use the provided references when available and relevant. Let's think step by step before answering. Your expertise makes a real difference in providing clear and accurate insights!

{few_shot_examples}
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
def get_prediction_allam(
    question: str, choice1: str, choice2: str, choice3: str, choice4: str, 
    choice5: Optional[str] = None, choice6: Optional[str] = None,
    model_version: str = "ALLaM-AI/ALLaM-7B-Instruct-preview", 
    max_new_tokens: int = 512, max_retries: int = 3
) -> Optional[str]:
    """Inference using local Allam 7B model (HuggingFace)."""
    global allam_model, allam_tokenizer
    print("Starting Allam prediction...")
    if not allam_model or not allam_tokenizer:
        print(f"Loading Allam model: {model_version}...")
        allam_tokenizer = AutoTokenizer.from_pretrained(model_version)
        allam_model = AutoModelForCausalLM.from_pretrained(model_version, torch_dtype=torch.bfloat16, device_map="auto")
        print("Allam model and tokenizer loaded.")

    print("Packing choices...")
    choices = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
    print("Generating valid responses...")
    valid_responses = get_valid_responses(choice5, choice6)
    print("Generating MCQ prompt...")
    prompt = generate_mcq_prompt(question, choices)

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt} of {max_retries} for Allam prediction...")
        try:
            print("Tokenizing prompt...")
            inputs = allam_tokenizer(prompt, return_tensors="pt").to(allam_model.device)
            print("Generating model output...")
            outputs = allam_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, 
                pad_token_id=allam_tokenizer.eos_token_id
            )
            print("Decoding model output...")
            response = allam_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split(prompt)[-1].strip()
            print(f"Raw model response: {response}")

            print("Cleaning and validating response...")
            cleaned_result = clean_and_validate_response(response, valid_responses)
            if cleaned_result:
                print(f"✅ {question} | Allam  | Prediction: {cleaned_result}")
                return cleaned_result
            else:
                print(f"⚠️ Attempt {attempt} invalid: {response}")
        except Exception as e:
            print(f"❌ Allam error: {e}")
            return None

    print("❌ Failed after retries.")
    return None
# ---------------------------------------------------------------------------------------------------------------------------
def get_prediction_fanar(
    question: str, choice1: str, choice2: str, choice3: str, choice4: str, 
    choice5: Optional[str] = None, choice6: Optional[str] = None,
    model_version: str = "Islamic-RAG", max_retries: int = 3, 
    top_k: int = 4, documents: Optional[List[Any]] = None, 
    faiss_index: Optional[faiss.Index] = None
) -> Optional[str]:
    """Inference using Fanar API with key switching on rate limit."""
    global fanar_failure_count, current_fanar_key, use_secondary_key
    print("Starting Fanar prediction...")
    
    if not current_fanar_key:
        print("Fanar API key missing.")
        return None
    
    if documents is None or faiss_index is None:
        print("No documents or index provided, loading defaults...")
        documents, faiss_index = load_documents_and_index(
            index_path="../islamic_index.bin", doc_path="../islamic_chunks.pkl"
        )
    
    print("Encoding question for FAISS search...")
    texts_to_embed = [doc['text'] if isinstance(doc, dict) else doc for doc in documents]
    question_embedding = embedder.encode([question], convert_to_numpy=True, device='cpu')
    print(f"Performing FAISS search for top {top_k} documents...")
    distances, indices = faiss_index.search(question_embedding, top_k)
    
    retrieved_docs = []
    for idx in indices[0]:
        if idx < len(documents):
            retrieved_docs.append(documents[idx])
    print(f"Retrieved {len(retrieved_docs)} documents.")
    
    print("Packing choices...")
    choices = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
    print("Generating valid responses...")
    valid_responses = get_valid_responses(choice5, choice6)
    print("Generating MCQ prompt...")
    prompt = generate_mcq_prompt(question, choices, retrieved_docs)

    headers = {"Authorization": f"Bearer {current_fanar_key}", "Content-Type": "application/json"}
    data = {"model": model_version, "messages": [{"role": "user", "content": prompt}]}
    print(f"Prepared API request with model: {model_version}, using key: {'FANAR_API_KEY_2' if use_secondary_key else 'FANAR_API_KEY'}")

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt} of {max_retries} for Fanar prediction...")
        try:
            print("Sending request to Fanar API...")
            response = requests.post(
                "https://api.fanar.qa/v1/chat/completions", 
                json=data, headers=headers, timeout=60
            )
            response_json = response.json()
            print(f"Received API response with status code: {response.status_code}")

            if response.status_code == 200:
                raw_result = response_json["choices"][0]["message"]["content"].strip().upper()
                print(f"Raw API response: {raw_result}")
                print("Cleaning and validating response...")
                cleaned_result = clean_and_validate_response(raw_result, valid_responses)
                if cleaned_result:
                    print(f"✅ {question} | Fanar | Prediction: {cleaned_result}")
                    fanar_failure_count = 0
                    return cleaned_result
                else:
                    print("⚠️ Invalid response received.")
            else:
                print(f"❌ Fanar API Error: {response.text}")
                fanar_failure_count += 1
                print(f"Failure count: {fanar_failure_count}")
                if fanar_failure_count >= 6 and not use_secondary_key and FANAR_API_KEY_2:
                    print("Switching to FANAR_API_KEY_2 due to rate limit or errors.")
                    current_fanar_key = FANAR_API_KEY_2
                    use_secondary_key = True
                    headers["Authorization"] = f"Bearer {current_fanar_key}"
                    print("Retrying with new key...")
                    continue
        except requests.exceptions.Timeout:
            print("❌ Fanar API request timed out.")
            fanar_failure_count += 1
            print(f"Failure count: {fanar_failure_count}")
            if fanar_failure_count >= 6 and not use_secondary_key and FANAR_API_KEY_2:
                print("Switching to FANAR_API_KEY_2 due to rate limit or errors.")
                current_fanar_key = FANAR_API_KEY_2
                use_secondary_key = True
                headers["Authorization"] = f"Bearer {current_fanar_key}"
                print("Retrying with new key...")
                continue
        except Exception as e:
            print(f"❌ Fanar Error: {e}")
            fanar_failure_count += 1
            print(f"Failure count: {fanar_failure_count}")
            if fanar_failure_count >= 6 and not use_secondary_key and FANAR_API_KEY_2:
                print("Switching to FANAR_API_KEY_2 due to rate limit or errors.")
                current_fanar_key = FANAR_API_KEY_2
                use_secondary_key = True
                headers["Authorization"] = f"Bearer {current_fanar_key}"
                print("Retrying with new key...")
                continue

    print("❌ Failed after retries.")
    return None
# ---------------------------------------------------------------------------------------------------------------------------
def get_prediction_mistral(
    question: str, choice1: str, choice2: str, choice3: str, choice4: str, 
    choice5: Optional[str] = None, choice6: Optional[str] = None,
    model_version: str = "mistral-saba-24b", max_retries: int = 3, 
    top_k: int = 3, documents: Optional[List[Any]] = None, 
    faiss_index: Optional[faiss.Index] = None
) -> Optional[str]:
    """Inference using Mistral API (via Groq) with key switching on rate limit."""
    global client_groq
    print("Starting Mistral prediction...")

    if not hasattr(get_prediction_mistral, 'mistral_failure_count'):
        get_prediction_mistral.mistral_failure_count = 0
        get_prediction_mistral.current_groq_key = os.getenv("GROQ_API_KEY")
        get_prediction_mistral.use_secondary_groq_key = False
        print("Initialized Mistral global state.")

    if not get_prediction_mistral.current_groq_key:
        print("Groq API key missing.")
        return None

    client_groq = Groq(api_key=get_prediction_mistral.current_groq_key)

    print("Encoding question for FAISS search...")
    question_embedding = embedder.encode([question], convert_to_numpy=True, device='cpu')
    print(f"Performing FAISS search for top {top_k} documents...")
    distances, indices = faiss_index.search(question_embedding, top_k)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    print(f"Retrieved {len(retrieved_docs)} documents.")

    print("Packing choices...")
    choices = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
    print("Generating valid responses...")
    valid_responses = get_valid_responses(choice5, choice6)
    print("Generating MCQ prompt with retrieved documents...")
    prompt = generate_mcq_prompt(question, choices, retrieved_docs)

    groq_api_key_2 = os.getenv("GROQ_API_KEY_2")
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt} of {max_retries} for Mistral prediction...")
        try:
            print("Sending request to Groq API...")
            response = client_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], model=model_version
            )
            raw_result = response.choices[0].message.content.strip().upper()
            print(f"Raw API response: {raw_result}")
            print("Cleaning and validating response...")
            cleaned_result = clean_and_validate_response(raw_result, valid_responses)
            if cleaned_result:
                print(f"✅ {question} | Mistral | Prediction: {cleaned_result}")
                get_prediction_mistral.mistral_failure_count = 0
                return cleaned_result
            else:
                print("⚠️ Invalid response received.")
        except Exception as e:
            print(f"❌ Mistral Error: {e}")
            get_prediction_mistral.mistral_failure_count += 1
            print(f"Failure count: {get_prediction_mistral.mistral_failure_count}")
            if (get_prediction_mistral.mistral_failure_count >= 6 and 
                not get_prediction_mistral.use_secondary_groq_key and 
                groq_api_key_2):
                print("Switching to GROQ_API_KEY_2 due to rate limit or errors.")
                get_prediction_mistral.current_groq_key = groq_api_key_2
                get_prediction_mistral.use_secondary_groq_key = True
                client_groq = Groq(api_key=get_prediction_mistral.current_groq_key)
                print("Retrying with new key...")
                continue

    print("❌ Failed after retries.")
    return None
# ---------------------------------------------------------------------------------------------------------------------------
def get_prediction_gemini(
    question: str, choice1: str, choice2: str, choice3: str, choice4: str, 
    choice5: Optional[str] = None, choice6: Optional[str] = None,
    model_version: str = "gemini-1.5-flash", max_retries: int = 2, 
    top_k: int = 3, documents: Optional[List[Any]] = None, 
    faiss_index: Optional[faiss.Index] = None
) -> Optional[str]:
    """Inference using Gemini API with key switching on rate limit and adjustable safety settings."""
    global embedder
    print("Starting Gemini prediction...")

    gemini_api_keys = [
        os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 16)
    ]
    if not hasattr(get_prediction_gemini, 'gemini_failure_count'):
        get_prediction_gemini.gemini_failure_count = 0
        get_prediction_gemini.current_key_index = 0
        get_prediction_gemini.current_gemini_key = gemini_api_keys[0]
        get_prediction_gemini.last_request_time = 0
        print("Initialized Gemini global state.")

    if not get_prediction_gemini.current_gemini_key:
        print("Gemini API key missing.")
        return None

    client = genai.GenerativeModel(
        model_name=model_version,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 30000,
        }
    )

    print("Encoding question for FAISS search...")
    question_embedding = embedder.encode([question], convert_to_numpy=True, device='cpu')
    print(f"Performing FAISS search for top {top_k} documents...")
    distances, indices = faiss_index.search(question_embedding, top_k)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    print(f"Retrieved {len(retrieved_docs)} documents.")

    print("Packing choices...")
    choices = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
    print("Generating valid responses...")
    valid_responses = get_valid_responses(choice5, choice6)
    print("Generating MCQ prompt with retrieved documents...")
    prompt = generate_mcq_prompt(question, choices, retrieved_docs)

    safety_settings = [
        {
            "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
        },
        {
            "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
        },
        {
            "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
        },
        {
            "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
        }
    ]

    current_time = time.time()
    time_since_last = current_time - get_prediction_gemini.last_request_time
    if time_since_last < 4:
        sleep_time = 4 - time_since_last
        print(f"Rate limit: Waiting {sleep_time:.2f} seconds before next request...")
        time.sleep(sleep_time)

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt} of {max_retries} for Gemini prediction...")
        try:
            print("Sending request to Gemini API...")
            genai.configure(api_key=get_prediction_gemini.current_gemini_key)
            response = client.generate_content(
                prompt,
                safety_settings=safety_settings
            )
            get_prediction_gemini.last_request_time = time.time()
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                print(f"Prompt blocked due to safety: {response.prompt_feedback.block_reason}")
                return None
            if not response.text:
                print("No valid response text received.")
                return None
            raw_result = response.text.strip().upper()
            print(f"Raw API response: {raw_result}")
            print("Cleaning and validating response...")
            cleaned_result = clean_and_validate_response(raw_result, valid_responses)
            if cleaned_result:
                print(f"✅ {question} | Gemini | Prediction: {cleaned_result}")
                get_prediction_gemini.gemini_failure_count = 0
                return cleaned_result
            else:
                print("⚠️ Invalid response received.")
        except Exception as e:
            print(f"❌ Gemini Error: {e}")
            get_prediction_gemini.gemini_failure_count += 1
            print(f"Failure count: {get_prediction_gemini.gemini_failure_count}")
            if get_prediction_gemini.gemini_failure_count >= 6:
                get_prediction_gemini.current_key_index = (get_prediction_gemini.current_key_index + 1) % 8
                new_key = gemini_api_keys[get_prediction_gemini.current_key_index]
                if new_key:
                    print(f"Switching to GEMINI_API_KEY_{get_prediction_gemini.current_key_index + 1} due to rate limit or errors.")
                    get_prediction_gemini.current_gemini_key = new_key
                    print("Retrying with new key...")
                    continue
                else:
                    print(f"GEMINI_API_KEY_{get_prediction_gemini.current_key_index + 1} is not available.")
                    break

    print("❌ Failed after retries.")
    return None