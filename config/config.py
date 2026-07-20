class Config:
    collection_name = "rag"
    persist_directory = "./chroma_db"

    chunk_size = 1000
    chunk_overlap = 100

    embedding_model_name = "text-embedding-v4"
    chat_model_name = "qwen3-max"

    session_id = "user_001"
    
    top_k = 3
config = Config()