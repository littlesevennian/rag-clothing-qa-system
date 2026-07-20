from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from utils.history import get_history

from core.retriever import VectorStoresService
from langchain_community.embeddings import DashScopeEmbeddings
from config.config import config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi


def print_prompt(prompt):
    print("="*20 )
    print(prompt.to_string())
    print("="*20)

    return prompt


class RAGPipeline:
    def __init__(self):
        self.vector_service = VectorStoresService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name),
        )

        self.retriever = self.vector_service.get_retriever()
        self.chat_model = ChatTongyi(model=config.chat_model_name)

    # 1检索
    def retrieve(self, query: str):
        return self.retriever.invoke(query)

    # 2文档格式化
    def format_docs(self, docs):
        if not docs:
            return "无相关资料参考"

        formatted = []
        for i, d in enumerate(docs):
            formatted.append(
                f"[{i+1}] {d.page_content}\n来源：{d.metadata}"
        )

        return "\n\n".join(formatted)

    # 3prompt构建
    def build_prompt(self, context, query, history=None):
        return ChatPromptTemplate.from_messages([
            ("system",
            """
            你是一个严谨的知识库助手，只能基于给定资料回答。

            规则：
            1. 如果资料没有相关内容，必须回答“资料中未找到相关信息”
            2. 不允许编造答案
            3. 必须引用资料内容
            """),
            MessagesPlaceholder("history"),
            ("user", "{input}")
        ]).format_messages(
            context=context,
            input=query,
            history=history or []
        )

    # 4推理
    def run(self, query, history=None):
        docs = self.retrieve(query)
        context = self.format_docs(docs)
        prompt = self.build_prompt(context, query, history)

        result = self.chat_model.invoke(prompt)

        return {
            "answer": result.content,
            "context": context,
            "sources": [
                f"[{i+1}] {d.page_content[:80]}..."
                for i, d in enumerate(docs)
            ]
        }

if __name__ == '__main__':
    #session id 配置
    session_config = {
        "configurable":{
            "session_id":"user_001",
        }
    }
    res =RAGPipeline().chain.invoke({"input":"春天穿什么颜色的衣服"},session_config)
    print(res)