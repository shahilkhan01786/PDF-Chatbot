from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import numpy as np
import faiss
import os
import pickle

from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ==========================
# LOAD ENV FILE
# ==========================
load_dotenv()

# ==========================
# APP SETUP
# ==========================
app = Flask(__name__)
CORS(app)

# ==========================
# GROQ CLIENT (SECURE)
# ==========================
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ==========================
# MODEL
# ==========================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================
# FILE STORAGE
# ==========================
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.pkl"

# ==========================
# SAFE LOAD DATA
# ==========================
def load_data():
    try:
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            index = faiss.read_index(INDEX_FILE)
            with open(CHUNKS_FILE, "rb") as f:
                chunks = pickle.load(f)
            return index, chunks
    except:
        pass
    return None, None

# ==========================
# SAVE DATA
# ==========================
def save_data(index, chunks):
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

# ==========================
# PDF TEXT EXTRACTION
# ==========================
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ==========================
# CHUNK TEXT
# ==========================
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# ==========================
# VECTOR STORE
# ==========================
def create_vector_store(text_chunks):
    embeddings = embedding_model.encode(text_chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index

# ==========================
# LOAD ON START
# ==========================
index, chunks = load_data()

# ==========================
# ROUTES
# ==========================
@app.route("/")
def home():
    return "PDF Chatbot Running 🚀"

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global index, chunks

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    text = extract_text(file)

    if not text.strip():
        return jsonify({"error": "No readable text"}), 400

    chunks = chunk_text(text)
    index = create_vector_store(chunks)

    save_data(index, chunks)

    return jsonify({
        "message": "PDF uploaded successfully",
        "chunks": len(chunks)
    })

@app.route("/chat", methods=["POST"])
def chat():
    global index, chunks

    if index is None:
        return jsonify({"error": "Upload PDF first"}), 400

    question = request.json.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question required"}), 400

    query_embedding = embedding_model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    _, indices = index.search(query_embedding, k=3)

    context = ""
    for i in indices[0]:
        if i < len(chunks):
            context += chunks[i] + "\n"

    context = context[:3000]

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the context below:

{context}

Question: {question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return jsonify({"answer": response.choices[0].message.content})

# ==========================
# RUN SERVER
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)