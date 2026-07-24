from core.retriever import VectorStoresService
from langchain_community.embeddings import DashScopeEmbeddings
from config.config import config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

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
    
    def format_sources(self, docs):
        sources: list[str] = []

        for i, d in enumerate(docs):
            metadata = d.metadata or {}

            source = (
                metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("filename")
                or "未知来源"
            )

            content = d.page_content.replace("\n", " ").strip()

            sources.append(
                f"[{i + 1}] 来源：{source}\n内容：{content[:120]}..."
            )

        return sources

    # 3prompt构建
    def build_prompt(self, context, query, history=None):
        return ChatPromptTemplate.from_messages([
            ("system",
            """
            你是一个严谨的服装知识库问答助手，只能基于给定资料回答。

            给定资料：
            {context}

            回答规则：
            1. 如果资料中没有相关内容，回答“资料中未找到相关信息”
            2. 不允许编造资料之外的信息
            3. 回答时要尽量引用资料中的依据
            4. 如果资料存在多个可能答案，需要说明判断依据
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
            "sources": self.format_sources(docs),
        }

if __name__ == "__main__":
    rag = RAGPipeline()
    res = rag.run("春天穿什么颜色的衣服")
    print(res)