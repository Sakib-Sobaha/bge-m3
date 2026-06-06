import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import faiss
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# --- Load Data ---
qa_df = pd.read_csv('train.csv')   # expects 'question','tag'
ans_df = pd.read_csv('tag_answer.csv')  # expects 'tag','answer'
questions = qa_df['question'].tolist()
tags = qa_df['tag'].tolist()
tag2answer = dict(zip(ans_df['tag'], ans_df['answer']))

# --- Load BGE-M3 Model ---
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # will use GPU if available

# --- Encode Questions and Build Index ---
embeddings = model.encode(questions, batch_size=32, max_length=8192)['dense_vecs']
emb_arr = np.vstack(embeddings).astype('float32')
faiss.normalize_L2(emb_arr)
d = emb_arr.shape[1]
cpu_index = faiss.IndexFlatIP(d)
cpu_index.add(emb_arr)

# --- FastAPI App ---
app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    k: int = 5  # top-k

def encode_long_query(text, model, max_chunk_words=128, stride=64):
    sentences = sent_tokenize(text)
    chunks = []
    current = []

    for s in sentences:
        current += s.split()
        if len(current) >= max_chunk_words:
            chunks.append(" ".join(current[:max_chunk_words]))
            current = current[stride:]
    if current:
        chunks.append(" ".join(current))

    # Embed each chunk, then mean-pool
    out = model.encode(chunks, return_dense=True)
    chunk_vecs = out['dense_vecs']
    avg_vec = np.mean(chunk_vecs, axis=0)
    norm_vec = avg_vec / np.linalg.norm(avg_vec)
    return norm_vec.astype('float32')

@app.post("/search")
def search_endpoint(req: SearchRequest):
    query_vec = encode_long_query(req.query, model)
    D, I = cpu_index.search(query_vec[np.newaxis, :], 1)

    results = []
    for score, idx in zip(D[0], I[0]):
        tag = tags[idx]
        answer = tag2answer.get(tag, "")
        results.append({
            "tag": tag,
            "answer": answer,
            "score": float(score)
        })
    return {"results": results}
