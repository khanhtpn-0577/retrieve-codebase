import os
import math
import sqlite3
from typing import List, Dict, Any

import torch
from unixcoder import UniXcoder
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# ==========================
# Config
# ==========================

CHROMA_DIR = "./semantic_embedding_db"
SQLITE_PATH = "code_index.db"
NGRAM_SIZE = 3

SEM_WEIGHT = 0.7
INDEX_WEIGHT = 0.3

MAX_SEM_RESULTS = 10   # top-K từ embedding
MAX_LEX_RESULTS = 30   # cắt bớt candidate lexical cho nhanh
TOP_K_FINAL = 5        # top-K trả về cuối cùng

#bm25 parameters
K1 = 1.5
B = 0.75

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ==========================
# Init global objects
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading UniXcoder model for query embedding...")
ux_model = UniXcoder("microsoft/unixcoder-base")
ux_model.to(device)
ux_model.eval()

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="semantic_embedding_db")
collection = chroma_client.get_collection(name="code_chunks")

print("Connecting to SQLite index...")
sql_conn = sqlite3.connect(SQLITE_PATH)
sql_cursor = sql_conn.cursor()

print("Initializing OpenAI client...")
openai_client = OpenAI()


# ==========================
# Helper: embed query with UniXcoder
# ==========================

def embed_query(text: str) -> List[float]:
    """
    Tạo embedding cho query NL bằng UniXcoder.
    Trả về 1 vector (list float).
    """
    with torch.no_grad():
        token_ids = ux_model.tokenize(
            [text],
            max_length=512,
            mode="<encoder-only>"
        )
        input_ids = torch.tensor(token_ids).to(device)
        token_embeds, sentence_embedding = ux_model(input_ids)
        sentence_embedding = torch.nn.functional.normalize(
            sentence_embedding, p=2, dim=1
        )
    return sentence_embedding[0].cpu().tolist()


# ==========================
# Semantic retrieval via Chroma
# ==========================

def semantic_retrieve(query: str, top_k: int = MAX_SEM_RESULTS):
    """
    Dùng embedding query để truy vấn Chroma, trả về:
    - sem_scores: dict[chunk_id] = sem_score (0..1)
    """
    query_emb = embed_query(query)

    result = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k)

    sem_scores: Dict[str, float] = {}

    ids = result["ids"][0]
    distances = result["distances"][0]  # cosine distance nếu space=cosine

    for chunk_id, dist in zip(ids, distances):
        # cosine distance d = 1 - cos_sim => cos_sim = 1 - d
        sem_score = 1.0 - dist
        # Clamp về [0,1] nếu cần
        sem_score = max(0.0, min(1.0, sem_score))
        sem_scores[chunk_id] = sem_score

    return sem_scores


# ==========================
# Keyword extraction via GPT
# ==========================

def extract_keywords_with_gpt(query: str, max_keywords: int = 5) -> List[str]:
    """
    Gọi GPT để trích xuất keywords từ NL query.
    Trả về list string.
    """
    system_prompt = (
        "You are a coding assistant that extracts ONLY high-filter programming-related keywords "
        "from natural language queries. "
        "IMPORTANT: Ignore common words like 'the', 'a', 'to', 'and', 'or', 'upgrade', 'implement', 'fix', 'add', 'remove', 'update', 'modify', 'improve', 'function', 'code', 'application', 'system'. "
        "Extract ONLY specific technical terms: library/package names, framework names, API names, specific function/class names, protocols, technical concepts, file names. "
        "Return ONLY comma-separated keywords, no explanation, no JSON, no common words."
    )
    user_prompt = (
        f"Extract up to {max_keywords} HIGH-SPECIFICITY technical keywords "
        f"from the query (ignore generic action words):\n\n"
        f"\"{query}\"\n\n"
        f"ONLY extract: library names, framework names, API names, specific technical terms, protocols, concepts. "
        f"Return as comma-separated values (e.g., 'mcp, database, api'). "
        f"Example: query='upgrade the mcp function' -> mcp (NOT 'upgrade, function, the'). "
        f"Example: query='fix database connection issue' -> database (NOT 'fix, issue'). "
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
    )

    content = resp.choices[0].message.content.strip()

    # Tách theo dấu phẩy
    if "," in content:
        kws = [x.strip() for x in content.split(",")]
        # Lọc các keyword không rỗng
        return [k for k in kws if k]

    # Nếu không có dấu phẩy, trả về content duy nhất (hoặc rỗng nếu không có)
    return [content] if content else []


# ==========================
# N-gram & lexical retrieval
# ==========================

def generate_ngrams_from_keyword(keyword: str, n: int = NGRAM_SIZE) -> List[str]:
    k = keyword.lower()
    if len(k) <= n:
        return [k]
    return [k[i:i+n] for i in range(len(k) - n + 1)]

# ==========================
# BM25 HELPERS
# ==========================

def compute_idf(term: str, total_docs: int) -> float:
    sql_cursor.execute(
        "SELECT COUNT(DISTINCT chunk_id) FROM ngram_index WHERE gram = ?",
        (term,)
    )
    df = sql_cursor.fetchone()[0] or 0

    return math.log((total_docs - df + 0.5) / (df + 0.5) + 1e-6)


def bm25_score(text: str, keywords: List[str], idf_map: Dict[str, float],
               avg_len: float, file_path: str) -> float:

    tokens = text.lower()
    doc_len = len(tokens)

    score = 0.0
    for kw in keywords:
        tf = tokens.count(kw.lower())
        if tf == 0:
            continue

        idf = idf_map.get(kw, 0.0)

        # BM25 formula
        denom = tf + K1 * (1 - B + B * (doc_len / avg_len))
        score += idf * (tf * (K1 + 1)) / denom

    # path bonus
    path_bonus = 0.3 if any(kw.lower() in (file_path or "").lower() for kw in keywords) else 0.0

    return score + path_bonus

def lexical_retrieve_from_ngrams(keywords: List[str], max_results=MAX_LEX_RESULTS):

    if not keywords:
        return {}, {}

    # 1) Collect candidate chunk_ids
    hits = {}
    grams = {g for kw in keywords for g in generate_ngrams_from_keyword(kw)}
    print(f"List of ngrams from keywords: {grams}")

    for gram in grams: #dem so gram xuat hien trong moi chunk
        sql_cursor.execute("SELECT chunk_id FROM ngram_index WHERE gram = ?", (gram,))
        for (cid,) in sql_cursor.fetchall():
            hits[cid] = hits.get(cid, 0) + 1

    
    print("Found", len(hits), "candidate chunks from n-gram index.")

    if not hits:
        return {}, {}

    # top candidate chunks
    candidate_ids = [cid for cid, _ in sorted(hits.items(),
                        key=lambda x: x[1], reverse=True)[:max_results]]

    # 2) Prepare BM25 stats
    sql_cursor.execute("SELECT COUNT(*) FROM chunks")
    TOTAL_DOCS = sql_cursor.fetchone()[0]

    avg_len = 200  # approximate, or compute dynamically (fastest to use constant)

    # idf map
    idf_map = {kw.lower(): compute_idf(kw.lower(), TOTAL_DOCS) for kw in keywords}

    index_scores = {}
    chunk_meta = {}

    for cid in candidate_ids:
        sql_cursor.execute(
            "SELECT file_path, start_line, end_line, code FROM chunks WHERE chunk_id = ?",
            (cid,)
        )
        row = sql_cursor.fetchone()
        if not row:
            continue

        file_path, start_line, end_line, code = row

        raw_bm25 = bm25_score(code, keywords, idf_map, avg_len, file_path)

        # normalize
        idx_score = min(1.0, raw_bm25 / 10.0)

        index_scores[cid] = idx_score
        chunk_meta[cid] = {
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "code": code,
        }

    return index_scores, chunk_meta


# ==========================
# Hybrid retrieve
# ==========================

def hybrid_retrieve(query: str, top_k=TOP_K_FINAL):

    sem_scores = semantic_retrieve(query)
    print(f"Semmantic scores of chunks: {sem_scores}")
    print("[semantic]", len(sem_scores), "chunks")

    
    keywords = extract_keywords_with_gpt(query)
    print("[keywords]", keywords)

    index_scores, chunk_meta = lexical_retrieve_from_ngrams(keywords)
    print("[lexical]", len(index_scores), "chunks")

    all_ids = set(sem_scores) | set(index_scores)
    results = []

    for cid in all_ids:
        sem = sem_scores.get(cid, 0.0)
        idx = index_scores.get(cid, 0.0)
        hybrid = SEM_WEIGHT * sem + INDEX_WEIGHT * idx
        
        if cid not in chunk_meta:
            sql_cursor.execute(
                "SELECT file_path, start_line, end_line, code FROM chunks WHERE chunk_id=?",
                (cid,)
            )
            row = sql_cursor.fetchone()
            if not row:
                continue
            file_path, start, end, code = row
            chunk_meta[cid] = dict(
                file_path=file_path,
                start_line=start,
                end_line=end,
                code=code
            )

        meta = chunk_meta[cid]
        results.append({
            "chunk_id": cid,
            "hybrid_score": hybrid,
            "semantic_score": sem,
            "index_score": idx,
            **meta
        })

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:top_k]


# ==========================
# CLI demo
# ==========================

if __name__ == "__main__":
    import textwrap

    q = input("input: ").strip()
    top_chunks = hybrid_retrieve(q, top_k=TOP_K_FINAL)

    print(f"\n=== TOP {len(top_chunks)} RESULTS ===")
    for i, item in enumerate(top_chunks, 1):
        print(f"\n#{i}  score={item['hybrid_score']:.3f} "
              f"(sem={item['semantic_score']:.3f}, idx={item['index_score']:.3f})")
        print(f"File: {item['file_path']}:{item['start_line']}–{item['end_line']}")
        print("-" * 80)
        # in tóm tắt vài dòng đầu
        snippet = "\n".join(item["code"].splitlines()[:80])
        print(snippet)
        print("-" * 80)
