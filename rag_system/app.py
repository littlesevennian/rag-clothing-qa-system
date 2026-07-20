import streamlit as st

# ===== RAG核心（从rag.py导入）=====
from core.rag_pipeline import RAGPipeline


# ===== 上传模块 =====
def upload_app():
    st.title("知识库上传")
    uploaded_file = st.file_uploader("上传文件")
    if uploaded_file:
        st.success("上传成功")

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