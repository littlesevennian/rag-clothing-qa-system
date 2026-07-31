import streamlit as st
from dotenv import load_dotenv

from core.rag_pipeline import RAGPipeline
from utils.file_loader import KnowledgeBaseService


load_dotenv()


st.set_page_config(
    page_title="Fashion RAG QA System",
    page_icon="👕",
    layout="wide",
)


@st.cache_resource
def load_rag():
    return RAGPipeline()


@st.cache_resource
def load_kb_service():
    return KnowledgeBaseService()


def render_sidebar():
    st.sidebar.title("Fashion RAG QA")
    st.sidebar.markdown(
        """
        基于 LangChain、Chroma、DashScope Embedding 和 Qwen 的服装领域 RAG 问答系统。
        """
    )

    mode = st.sidebar.radio(
        "功能选择",
        ["问答系统", "上传知识库"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **当前能力**

        - 服装知识库问答
        - txt / md 文件入库
        - Chroma 语义检索
        - Prompt context 注入
        - 参考来源展示
        """
    )

    return mode


def qa_app():
    st.title("服装垂直领域 RAG 知识库问答系统")

    st.markdown(
        """
        输入服装相关问题后，系统会先从本地 Chroma 向量数据库中检索相关知识片段，
        再将检索结果注入 Prompt，调用大模型生成基于资料的回答。
        """
    )

    with st.form("qa_form"):
        question = st.text_area(
            "请输入问题",
            placeholder="例如：我身高172，体重140斤，应该穿什么尺码？",
            height=100,
        )

        submitted = st.form_submit_button("开始提问")

    if not submitted:
        return

    if not question.strip():
        st.warning("请输入问题后再提交")
        return

    try:
        with st.spinner("正在检索知识库并生成回答..."):
            rag = load_rag()
            result = rag.run(question.strip())

        st.markdown("## 回答")
        st.success(result["answer"])

        st.markdown("## 参考来源")

        sources = result.get("sources", [])
        if sources:
            for source in sources:
                st.info(source)
        else:
            st.info("没有检索到可展示的参考来源")

        with st.expander("查看本次检索上下文"):
            st.code(result.get("context", "无上下文内容"), language="text")

    except Exception as e:
        st.error(f"问答失败：{e}")
        st.info("请检查 API Key、向量库是否已构建，以及网络连接是否正常。")


def upload_app():
    st.title("知识库上传")

    st.markdown(
        """
        支持上传 `.txt`、`.md` 格式的服装知识文档。上传后系统会自动完成文本读取、
        文档切分、Embedding 向量化和 Chroma 入库。
        """
    )

    uploaded_file = st.file_uploader(
        "请选择知识库文件",
        type=["txt", "md"],
    )

    if uploaded_file is None:
        st.info("请上传 txt 或 md 格式的知识库文件")
        return

    st.markdown("### 文件信息")
    st.write(f"文件名：{uploaded_file.name}")
    st.write(f"文件大小：{uploaded_file.size} bytes")

    if st.button("开始入库"):
        try:
            with st.spinner("正在切分文档并写入 Chroma 向量库..."):
                kb_service = load_kb_service()

                result = kb_service.upload_file(
                    file_name=uploaded_file.name,
                    file_content=uploaded_file.getvalue(),
                )

                st.cache_resource.clear()

            st.success(
                f"入库成功：{result['file_name']}，"
                f"共生成 {result['chunk_count']} 个知识片段"
            )

            st.info("缓存已清理，下一次问答会重新加载最新知识库。")

        except Exception as e:
            st.error(f"知识库入库失败：{e}")
            st.info("请检查文件格式、文件编码、API Key 和网络连接。")


def main():
    mode = render_sidebar()

    if mode == "问答系统":
        qa_app()
    elif mode == "上传知识库":
        upload_app()


if __name__ == "__main__":
    main()