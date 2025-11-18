# chunker.py
import os
import glob
import uuid
import hashlib

CHUNK_SIZE = 80        # số dòng trong một chunk
CHUNK_OVERLAP = 20     # overlap giữa các chunk

def read_all_files(folder, extensions=None):
    """
    Đọc toàn bộ file code trong folder theo extensions.
    extensions: ['.py', '.js'] ...
    Skip: node_modules, .git, __pycache__, .venv, dist, build
    """
    skip_dirs = {'node_modules', '.git', '__pycache__', '.venv', 'dist', 'build', '.next', '.pytest_cache', 'venv'}
    
    all_files = []
    for root, dirs, files in os.walk(folder):
        # Loại bỏ skip_dirs khỏi danh sách để os.walk không đi vào
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for f in files:
            if extensions is None or any(f.endswith(ext) for ext in extensions):
                all_files.append(os.path.join(root, f))
    return all_files


def generate_chunk_id(file_path, start_line, end_line, code):
    """
    Tạo chunk_id xác định dựa trên file_path, start_line, end_line
    để đảm bảo ID giống nhau khi chạy lại.
    """
    # Normalize file_path: loại bỏ ./, convert \ thành /
    normalized_path = file_path.lstrip("./").replace("\\", "/")
    content = f"{normalized_path}:{start_line}:{end_line}:{code}"
    return hashlib.md5(content.encode()).hexdigest()


def chunk_code(code, file_path):
    lines = code.split("\n")
    chunks = []

    start = 0
    n = len(lines)

    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk_lines = lines[start:end]

        chunk_content = "\n".join(chunk_lines)
        chunk_id = generate_chunk_id(file_path, start, end, chunk_content)

        chunks.append({
            "chunk_id": chunk_id,
            "file_path": file_path,
            "start_line": start,
            "end_line": end,
            "code": chunk_content,
        })

        # Tính bước nhảy tiếp theo
        next_start = start + CHUNK_SIZE - CHUNK_OVERLAP

        # Nếu next_start không tăng → tránh lặp vô hạn
        if next_start <= start:
            break

        start = next_start

    return chunks



def create_all_chunks(folder, extensions=None):
    """
    Đọc toàn bộ file + chunk toàn bộ codebase.
    Return: list các chunk dict.
    """
    all_chunks = []
    files = read_all_files(folder, extensions)

    for file in files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            file_chunks = chunk_code(code, file)
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return all_chunks


if __name__ == "__main__":
    chunks = create_all_chunks("sample-repo", extensions=[".py", ".js", ".ts"])
    print(f"Total chunks created: {len(chunks)}")
    print("Sample chunk:", chunks[0] if chunks else "No chunks created")
