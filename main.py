import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import faiss

# --- Data loading ---
qa_df = pd.read_csv('train.csv')   # expects columns 'question','tag'
ans_df = pd.read_csv('tag_answer.csv')    # expects columns 'tag','answer'
questions = qa_df['question'].tolist()
tags = qa_df['tag'].tolist()
tag2answer = dict(zip(ans_df['tag'], ans_df['answer']))

# --- Embedding model ---
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # BGE-M3 on GPU:contentReference[oaicite:17]{index=17}

# Encode and index questions
embeddings = model.encode(questions, batch_size=32, max_length=8192)['dense_vecs']
emb_arr = np.vstack(embeddings).astype('float32')
faiss.normalize_L2(emb_arr)      # normalize vectors before indexing:contentReference[oaicite:18]{index=18}

d = emb_arr.shape[1]  # embedding dimension (1024)
# res = faiss.StandardGpuResources()
cpu_index = faiss.IndexFlatIP(d)  # inner-product index for cosine similarity:contentReference[oaicite:19]{index=19}
cpu_index.add(emb_arr)
# gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # move index to GPU:contentReference[oaicite:20]{index=20}
# gpu_index.add(emb_arr)

# --- FastAPI setup ---
app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    k: int = 5

@app.post("/search")
def search_endpoint(req: SearchRequest):
    vec = model.encode([req.query])['dense_vecs']
    vec = np.array(vec, dtype='float32')
    faiss.normalize_L2(vec)
    D, I = cpu_index.search(vec, 1)  # search on GPU:contentReference[oaicite:21]{index=21}

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
