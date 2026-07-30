import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import config
from utils.file_loader import KnowledgeBaseService


def remove_old_vector_db():
    persist_path = Path(config.persist_directory)

    if persist_path.exists():
        shutil.rmtree(persist_path)
        print(f"已删除旧向量库：{persist_path}")
    else:
        print(f"未发现旧向量库：{persist_path}")


def rebuild_from_data_dir():
    data_dir = PROJECT_ROOT / "data"

    if not data_dir.exists():
        raise FileNotFoundError(f"data 目录不存在：{data_dir}")

    kb_service = KnowledgeBaseService()

    total_files = 0
    total_chunks = 0

    for file_path in data_dir.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in config.supported_file_types:
            print(f"跳过不支持的文件：{file_path.name}")
            continue

        file_content = file_path.read_bytes()

        result = kb_service.upload_file(
            file_name=file_path.name,
            file_content=file_content,
        )

        total_files += 1
        total_chunks += result["chunk_count"]

        print(
            f"已入库：{result['file_name']}，"
            f"生成 {result['chunk_count']} 个知识片段"
        )

    print("=" * 50)
    print(f"知识库重建完成，共处理 {total_files} 个文件，生成 {total_chunks} 个知识片段")


def main():
    load_dotenv()

    print("开始重建 Chroma 向量库")
    remove_old_vector_db()
    rebuild_from_data_dir()


if __name__ == "__main__":
    main()