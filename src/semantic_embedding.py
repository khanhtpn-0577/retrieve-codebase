import torch
from unixcoder import UniXcoder
import chromadb
from chromadb.config import Settings

from src.chunker import create_all_chunks

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading UniXcoder model...")

model = UniXcoder("microsoft/unixcoder-base")
model.to(device)

# =======================
# Setup ChromaDB
# =======================
client = chromadb.PersistentClient(
        path="semantic_embedding_db",
        settings=Settings(allow_reset=True)
    )
client.reset()

collection = client.create_collection(
    name="code_chunks",
    metadata={"hnsw:space": "cosine"}
)

# =======================
# Embed function
# =======================
def embed_text_list(text_list):
    """
    Tạo embedding từ list đoạn text dùng UniXcoder (batch),
    padding để tất cả sequence có chiều bằng nhau.
    """
    model.eval()

    all_embeddings = []

    for text in text_list:
        tokens_ids = model.tokenize(
            [text],
            max_length=512,
            mode="<encoder-only>"
        )

        # Convert to tensor — đây là 2D: (1, seq_len)
        source_ids = torch.tensor(tokens_ids).to(device)

        with torch.no_grad():
            token_reps, sentence_embedding = model(source_ids)

            # chuẩn hoá vector để dùng cosine similarity
            sentence_embedding = torch.nn.functional.normalize(
                sentence_embedding, p=2, dim=1
            )

        all_embeddings.append(sentence_embedding[0].cpu().tolist())

    return all_embeddings



# =======================
# Main indexing pipeline
# =======================
def index_codebase(folder="sample-repo"):
    chunks = create_all_chunks(folder)
    print(f"Embedding {len(chunks)} chunks...")

    texts = [c["code"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadata = [{"file_path": c["file_path"],
                 "start_line": c["start_line"],
                 "end_line": c["end_line"]} for c in chunks]

    # Embed theo batch
    BATCH = 16
    for i in range(0, len(texts), BATCH):
        batch_text = texts[i:i+BATCH]
        batch_ids = ids[i:i+BATCH]
        batch_meta = metadata[i:i+BATCH]

        embeds = embed_text_list(batch_text)

        collection.add(
            ids=batch_ids,
            embeddings=embeds,
            metadatas=batch_meta,
            documents=batch_text
        )

        print(f"Embedded chunks {i} → {i+len(batch_text)}")

    print("DONE embedding all chunks!")


if __name__ == "__main__":
    index_codebase("sample-repo")
