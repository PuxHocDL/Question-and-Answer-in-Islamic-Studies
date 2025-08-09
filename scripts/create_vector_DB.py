import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Kiểm tra GPU
if not torch.cuda.is_available():
    raise RuntimeError("GPU not available. Please check CUDA installation.")
device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Khởi tạo BGE-M3 trên GPU
embedder = SentenceTransformer('BAAI/bge-m3', device='cuda')

# Hàm chia nhỏ văn bản
def chunk_text(text, max_length=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Hàm tải và tiền xử lý fatwa từ file JSON
def load_fatwas(json_files):
    documents = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if "فقه المواريث" in item['Category']:  # Lọc fatwa về thừa kế
                        text = f"Question: {item['Question']}\nAnswer: {item['Answer']}"
                        chunks = chunk_text(text, max_length=500)
                        documents.extend(chunks)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    print(f"Loaded {len(documents)} document chunks.")
    return documents

# Hàm nhúng tài liệu bằng BGE-M3 trên GPU
def embed_documents(documents):
    embeddings = embedder.encode(
        documents,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
        device='cuda'  # Rõ ràng sử dụng GPU
    )
    return embeddings

# Hàm tạo và lưu chỉ mục FAISS trên GPU
def create_faiss_index(embeddings, output_path="fatwa_index.bin"):
    dimension = embeddings.shape[1]
    # Sử dụng FAISS GPU
    res = faiss.StandardGpuResources()  # Tài nguyên GPU
    index = faiss.IndexFlatL2(dimension)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Chuyển chỉ mục sang GPU
    embeddings = embeddings.astype('float32')  # FAISS yêu cầu float32
    gpu_index.add(embeddings)
    # Chuyển chỉ mục về CPU để lưu
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index, output_path)
    print(f"FAISS index saved to {output_path}")
    return cpu_index

# Main: Tạo vector database
def main():
    # Đường dẫn tới 4 file JSON
    json_files = [
        "data/فقه_المواريث_batch_1.json",
        "data/فقه_المواريث_batch_2.json",
        "data/فقه_المواريث_batch_3.json",
        "data/فقه_المواريث_batch_4.json"
    ]
    
    # Tải và tiền xử lý fatwa
    documents = load_fatwas(json_files)
    
    # Nhúng tài liệu trên GPU
    doc_embeddings = embed_documents(documents)
    
    # Tạo và lưu chỉ mục FAISS trên GPU
    create_faiss_index(doc_embeddings, "fatwa_index.bin")
    
    # Lưu danh sách tài liệu để sử dụng sau
    with open("fatwa_documents.txt", "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(f"{doc}\n---\n")
    print("Documents saved to fatwa_documents.txt")

if __name__ == "__main__":
    main()