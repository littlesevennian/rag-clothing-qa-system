'''
知识库
'''

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

from config.config import config


class KnowledgeBaseService:
    def __init__(self):
        self.embedding = DashScopeEmbeddings(
            model=config.embedding_model_name
        )

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def load_text_file(self, file_name: str, file_content: bytes) -> str:
        suffix = Path(file_name).suffix.lower()

        if suffix not in config.supported_file_types:
            raise ValueError(f"暂不支持该文件类型：{suffix}")

        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            return file_content.decode("gbk")

    def build_documents(self, file_name: str, text: str) -> List[Document]:
        chunks = self.text_splitter.split_text(text)

        documents = []
        for index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_name,
                        "chunk_index": index,
                    },
                )
            )

        return documents

    def upload_file(self, file_name: str, file_content: bytes) -> dict:
        text = self.load_text_file(file_name, file_content)

        if not text.strip():
            raise ValueError("文件内容为空，无法入库")

        documents = self.build_documents(file_name, text)

        if not documents:
            raise ValueError("文件切分后没有生成有效文档片段")

        self.vector_store.add_documents(documents)

        return {
            "file_name": file_name,
            "chunk_count": len(documents),
        }