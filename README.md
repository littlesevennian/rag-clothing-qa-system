# 服装垂直领域 RAG 知识库问答系统

## 项目简介

本项目是一个基于 LangChain、Chroma、DashScope Embedding 和 Qwen 大模型构建的服装垂直领域 RAG 知识库问答系统。

系统围绕服装尺码推荐、洗涤养护、颜色选择等知识场景，支持将本地知识文档切分并向量化存储到 Chroma 数据库中。用户输入问题后，系统会先进行语义检索，召回相关知识片段，再将检索结果注入 Prompt，调用大模型生成基于资料的回答，并展示参考来源。

## 技术栈

- Python
- Streamlit
- LangChain
- Chroma
- DashScope Embedding
- Qwen / ChatTongyi

## 核心功能

- 本地服装知识库管理
- 文档切分与向量化入库
- 基于 Chroma 的语义检索
- 基于 Prompt 的 RAG 增强生成
- 答案来源追踪
- Streamlit Web 交互页面

## RAG 执行流程

```text
用户问题
   ↓
Embedding 向量化
   ↓
Chroma 语义检索
   ↓
召回相关文档片段
   ↓
构建带上下文的 Prompt
   ↓
调用 Qwen 大模型
   ↓
返回答案和参考来源