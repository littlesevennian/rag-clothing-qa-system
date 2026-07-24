import streamlit as st
from utils.file_loader import KnowledgeBaseService

# ===== RAG核心（从rag.py导入）=====
from core.rag_pipeline import RAGPipeline


# ===== 上传模块 =====
@st.cache_resource
def load_kb_service():
    return KnowledgeBaseService()

def upload_app():
    st.title("知识库上传")

    uploaded_file = st.file_uploader(
        "上传知识库文件",
        type=["txt", "md"]
    )

    if uploaded_file is None:
        st.info("请上传 txt 或 md 格式的知识库文件")
        return

    if st.button("开始入库"):
        try:
            kb_service = load_kb_service()

            result = kb_service.upload_file(
                file_name=uploaded_file.name,
                file_content=uploaded_file.getvalue(),
            )

            st.success(
                f"入库成功：{result['file_name']}，共生成 {result['chunk_count']} 个知识片段"
            )

            st.cache_resource.clear()

        except Exception as e:
            st.error(f"知识库入库失败：{e}")

@st.cache_resource
def load_rag():
    return RAGPipeline()

def qa_app():
    st.title("RAG问答系统")

    question = st.text_input("请输入问题")

    if question:
        rag = load_rag()
        result = rag.run(question)

        st.markdown("## 🧠 回答")
        st.success(result["answer"])

        st.markdown("## 📚 参考来源")

        for s in result["sources"]:
            st.info(s)
# ===== 主入口 =====
def main():
    st.sidebar.title("功能选择")

    mode = st.sidebar.selectbox(
        "请选择模式",
        ["问答系统", "上传知识库"]
    )

    if mode == "问答系统":
        qa_app()

    elif mode == "上传知识库":
        upload_app()

if __name__ == "__main__":
    main()