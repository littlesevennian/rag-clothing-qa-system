# RAG知识库问答系统（服装领域）

## 项目简介
基于LangChain + Chroma + Qwen3-max构建的垂直领域RAG问答系统，用于服装知识（尺码/颜色/洗涤）问答。

## 技术栈
- LangChain
- Chroma
- DashScope / Qwen3-max
- Streamlit

## 系统架构

用户问题
   ↓
向量检索（Chroma）
   ↓
召回相关文档
   ↓
Prompt构建
   ↓
LLM生成回答
   ↓
返回答案 + 来源

## 核心能力
- 文档向量化
- 语义检索
- RAG增强生成
- 可解释输出（sources）
- 防幻觉机制

## 项目亮点
- 支持垂直领域知识问答
- 可追溯回答来源
- 防止大模型幻觉