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

    
    def get_doc_source(self, doc):
        metadata = doc.metadata or {}

        source = (
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("filename")
            or "未知来源"
        )

        chunk_index = metadata.get("chunk_index", "未知片段")

        return source, chunk_index

    # 2文档格式化
    def format_docs(self, docs):
        if not docs:
            return "无相关资料参考"

        formatted = []

        for i, doc in enumerate(docs):
            source, chunk_index = self.get_doc_source(doc)
            content = doc.page_content.strip()

            formatted.append(
                f"[资料{i + 1}]\n"
                f"来源文件：{source}\n"
                f"片段编号：{chunk_index}\n"
                f"内容：{content}"
            )

        return "\n\n".join(formatted)

    def format_sources(self, docs):
        if not docs:
            return []

        sources: list[str] = []

        for i, doc in enumerate(docs):
            source, chunk_index = self.get_doc_source(doc)
            content = doc.page_content.replace("\n", " ").strip()

            sources.append(
                f"[{i + 1}] 来源文件：{source}\n"
                f"片段编号：{chunk_index}\n"
                f"内容片段：{content[:120]}..."
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
    1. 必须优先基于“给定资料”回答用户问题。
    2. 如果给定资料中没有相关内容，必须回答“资料中未找到相关信息”。
    3. 不允许编造给定资料之外的事实、数据或建议。
    4. 回答中需要说明依据来自哪条资料，例如“根据资料[1]”。
    5. 如果多条资料存在不同建议，需要说明差异和判断依据。
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