# retrieve-codebase

## Giới thiệu

`retrieve-codebase` là một công cụ để tìm kiếm và truy xuất mã nguồn từ các repository. Dự án kết hợp ba kỹ thuật tìm kiếm chính:

1. **N-gram Indexing** - Xây dựng chỉ số n-gram cho tìm kiếm dựa trên chuỗi ký tự nhanh chóng
2. **Semantic Embedding** - Sử dụng mô hình Microsoft UniXcoder để tạo ra các vector embedding ngữ nghĩa cho mã nguồn
3. **Hybrid Retrieval** - Kết hợp cả tìm kiếm n-gram và embedding ngữ nghĩa để đạt hiệu quả tối ưu

## Cấu trúc dự án

```
retrieve-codebase/
├── README.md
├── unixcoder.py
├── sample-repo/              # Thư mục chứa source code cần retrieve
├── src/
│   ├── chunker.py           # Chia nhỏ mã nguồn thành các đoạn
│   ├── hybrid_retrieve.py   # Logic tìm kiếm hybrid
│   ├── ngram_indexer.py     # Xây dựng chỉ số n-gram
│   ├── semantic_embedding.py # Tạo semantic embedding
│   └── __pycache__/
```

## Yêu cầu & Cài đặt

### Bước 0: Cài đặt Dependencies

Trước tiên, cài đặt các Python package cần thiết:

```bash
pip install -r requirements.txt
```

### Cài đặt Dependencies chi tiết

File `requirements.txt` chứa các package sau:
- **torch** (>=2.0.0) - PyTorch framework, cần thiết cho embedding và model inference
- **transformers** (>=4.30.0) - Hugging Face Transformers, cung cấp mô hình pre-trained
- **chromadb** (>=0.3.21) - Vector database cho lưu trữ và truy vấn semantic embeddings
- **openai** (>=1.0.0) - OpenAI API client cho keyword extraction
- **python-dotenv** (>=1.0.0) - Load environment variables từ `.env` file

### Cấu hình Environment

Tạo file `.env` trong thư mục gốc để lưu OpenAI API key:

```bash
OPENAI_API_KEY=your-api-key-here
```

## Hướng dẫn chạy

### Chuẩn bị

Trước tiên, bạn cần load source code cần truy xuất vào folder `sample-repo/`.

### Bước 1: Xây dựng chỉ số N-gram

Chạy lệnh sau từ thư mục gốc của dự án:

```bash
python -m src.ngram_indexer
```

Bước này sẽ xây dựng chỉ số n-gram cho tất cả các tệp mã nguồn trong `sample-repo/`.

### Bước 2: Tạo Semantic Embedding

Chạy lệnh sau từ thư mục gốc của dự án:

```bash
python -m src.semantic_embedding
```

Bước này sử dụng mô hình Microsoft UniXcoder để tạo ra các vector embedding cho từng đoạn mã, cho phép tìm kiếm dựa trên ý nghĩa.

### Bước 3: Chạy Hybrid Retrieval

Chạy lệnh sau từ thư mục gốc của dự án:

```bash
python -m src.hybrid_retrieve
```

Bước này sẽ khởi chạy hệ thống tìm kiếm hybrid, cho phép bạn truy vấn và lấy ra các đoạn mã liên quan nhất.





