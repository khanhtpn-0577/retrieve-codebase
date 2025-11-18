import sqlite3
from src.chunker import create_all_chunks

N = 3   # n-gram

# Create SQLite schema
def create_tables(conn):
    c = conn.cursor()

    # Chunk table
    c.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            file_path TEXT,
            start_line INTEGER,
            end_line INTEGER,
            code TEXT
        );
    """)

    # N-gram index
    c.execute("""
        CREATE TABLE IF NOT EXISTS ngram_index (
            gram TEXT,
            chunk_id TEXT,
            FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
        );
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_gram ON ngram_index(gram);")
    conn.commit()


# ==========================
# Generate n-grams
# ==========================
def generate_ngrams(text, n=3):
    text = text.lower()
    grams = []
    for i in range(len(text) - n + 1):
        grams.append(text[i:i+n])
    return grams


# ==========================
# Build index
# ==========================
def build_ngram_index(folder="sample-repo"):
    chunks = create_all_chunks(folder)

    conn = sqlite3.connect("code_index.db")
    create_tables(conn)
    cur = conn.cursor()

    print("Inserting chunks and indexing...")

    for c in chunks:
        # Insert chunk
        cur.execute("""
            INSERT OR REPLACE INTO chunks (chunk_id, file_path, start_line, end_line, code)
            VALUES (?, ?, ?, ?, ?)
        """, (c["chunk_id"], c["file_path"], c["start_line"], c["end_line"], c["code"]))

        # Create n-grams
        grams = generate_ngrams(c["code"], N)
        grams = set(grams)  # Remove duplicates in same chunk

        for g in grams:
            cur.execute("""
                INSERT INTO ngram_index (gram, chunk_id)
                VALUES (?, ?)
            """, (g, c["chunk_id"]))

    conn.commit()
    conn.close()
    print("DONE building n-gram index!")


if __name__ == "__main__":
    build_ngram_index("./sample-repo")
