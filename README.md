# Fashion RAG QA System

基于 **LangChain + Chroma + DashScope Embedding + Qwen** 构建的服装垂直领域 RAG 知识库问答系统。

项目支持将本地服装知识文档切分、向量化并写入 Chroma 向量数据库。用户提问后，系统会先从知识库中进行语义检索，再将召回内容注入 Prompt，调用大模型生成基于资料的回答，并展示参考来源。

---

## 项目特点

- 支持服装尺码、洗涤养护、颜色选择等垂直知识问答
- 基于 Chroma 实现本地向量数据库持久化
- 使用 DashScope Embedding 完成文本向量化
- 使用 LangChain Retriever 实现语义检索
- 将检索结果作为 context 注入 Prompt，降低大模型幻觉
- 支持 txt / md 文件上传入库
- 页面展示答案来源，包括来源文件、片段编号和内容片段
- 提供知识库重建脚本，便于项目 clone 后重新构建向量库

---

## 技术栈

| 模块 | 技术 |
|---|---|
| Web 页面 | Streamlit |
| RAG 框架 | LangChain |
| 向量数据库 | Chroma |
| Embedding | DashScope Embedding |
| 大语言模型 | Qwen / ChatTongyi |
| 文本切分 | RecursiveCharacterTextSplitter |
| 环境配置 | python-dotenv |

---

## 系统流程

```text
用户问题
  ↓
Embedding 向量化
  ↓
Chroma 语义检索
  ↓
召回相关 Document
  ↓
格式化为 context
  ↓
注入 Prompt
  ↓
Qwen 生成回答
  ↓
返回答案与参考来源
```

核心代码流程：

```python
docs = self.retrieve(query)
context = self.format_docs(docs)
prompt = self.build_prompt(context, query, history)
result = self.chat_model.invoke(prompt)
```

---

## 项目结构

```text
rag-clothing-qa-system/
├── app.py                      # Streamlit 页面入口
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── config.py               # 模型、向量库、切分参数配置
├── core/
│   ├── rag_pipeline.py         # RAG 主流程
│   └── retriever.py            # Chroma Retriever 封装
├── utils/
│   ├── file_loader.py          # 文档读取、切分、入库
│   └── history.py              # 对话历史预留模块
├── data/
│   ├── 尺码推荐.txt
│   ├── 洗涤养护.txt
│   └── 颜色选择.txt
└── scripts/
    └── rebuild_vector_db.py    # 从 data/ 重建向量库
```

---

## 核心模块

### `app.py`

负责 Streamlit 页面交互，包括：

- 问答页面
- 知识库上传页面
- 调用 RAGPipeline 获取回答
- 展示答案和参考来源
- 上传新知识后清理缓存，使新知识可被检索

### `core/rag_pipeline.py`

负责完整 RAG 问答流程：

- 检索相关文档
- 格式化 context
- 构造 Prompt
- 调用大模型
- 返回 answer / context / sources

其中：

```text
docs     ：Retriever 返回的 Document 列表
context  ：整理后注入 Prompt 的资料文本
sources  ：展示给用户的参考来源
```

### `utils/file_loader.py`

负责知识库入库：

- 检查文件类型
- 读取 txt / md 文件
- 按 chunk_size 和 chunk_overlap 切分文本
- 构造 LangChain Document
- 写入 Chroma 向量数据库

---

## 快速开始

## 检索效果评测

项目提供了一个简单的 Retriever 检索评测脚本，用于验证用户问题是否能够召回预期来源文件和关键词。

评测数据位于：

```text
tests/eval_questions.json
```

### 1. 克隆项目

```bash
git clone https://github.com/littlesevennian/rag-clothing-qa-system.git
cd rag-clothing-qa-system
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制环境变量示例文件：

```bash
copy .env.example .env
```

Linux / macOS 使用：

```bash
cp .env.example .env
```

然后在 `.env` 中填写 DashScope API Key：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

### 4. 构建向量库

项目不会上传本地 Chroma 向量库。首次运行前，需要根据 `data/` 目录重建知识库：

```bash
python scripts/rebuild_vector_db.py
```

### 5. 启动页面

```bash
streamlit run app.py
```

---

## 使用示例

用户提问：

```text
我身高172，体重140斤，穿什么码？
```

系统回答示例：

```text
根据资料[1]，身高170-176cm、体重130-150斤建议选择XL尺码。
您的身高172cm、体重140斤落在该范围内，因此建议选择XL。
如果喜欢宽松版型，可以选择2XL。
```

参考来源示例：

```text
[1] 来源文件：test_size.txt
片段编号：0
内容片段：身高170-176cm，体重130-150斤，建议选择XL尺码...
```

---

## 当前支持

- txt / md 知识库文件
- 本地 Chroma 向量库
- top_k 语义检索
- Prompt context 注入
- 来源文件与 chunk 编号展示
- Streamlit Web 页面
- 本地知识库上传入库
- data 目录重建向量库

---

## 不上传的文件

以下内容为本地运行生成文件或敏感文件，不应提交到 GitHub：

```text
.env
storage/chroma/
chroma_db/
__pycache__/
*.pyc
venv/
.venv/
```

其中：

- `.env` 保存个人 API Key
- `storage/chroma/` 和 `chroma_db/` 是本地向量数据库
- `__pycache__/` 和 `*.pyc` 是 Python 缓存文件
- `venv/`、`.venv/` 是本地虚拟环境

---

## 后续优化方向

- 增加检索相似度评分展示
- 增加低相关度拒答机制
- 增加问答评测集
- 增加文档 MD5 去重
- 支持 PDF / Word 文档解析
- 优化多轮对话历史
- 增加 Docker 部署支持

---

## 简历描述

基于 LangChain、Chroma、DashScope Embedding 和 Qwen 大模型构建服装垂直领域 RAG 知识库问答系统，实现本地文档切分入库、向量检索、Prompt 上下文注入、来源追踪和 Streamlit 可视化交互。

项目中完成了 RAG 主流程封装、知识库上传入库、Chroma 持久化检索、Prompt 防幻觉设计和向量库重建脚本，提高了系统的可复现性和答案可解释性。