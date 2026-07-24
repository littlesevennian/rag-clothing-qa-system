from pathlib import Path

class Config:
    base_dir = Path(__file__).resolve().parent.parent

    collection_name = "fashion_knowledge"
    persist_directory = str(base_dir / "storage" / "chroma")

    chunk_size = 300
    chunk_overlap = 50

    embedding_model_name = "text-embedding-v4"
    chat_model_name = "qwen3-max"

    session_id = "user_001"

    supported_file_types = [".txt", ".md"]
    
    top_k = 3

config = Config()



    



    