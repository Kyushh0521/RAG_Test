# 企业创新战略预测系统 🚀

这是一个基于 **RAG（检索增强生成）** 技术的企业创新战略预测系统，基于 [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) 框架搭建。本项目主要用于在海量企业文献与专业语料中进行检索，并结合强大的生成模型，为企业战略发展、技术演进方向等提供问答预测。

用户通过交互式问答终端输入问题，系统自动检索知识库并生成回答。

## 项目结构

```
.
├── config/
│   └── my_config.yaml          # 全局配置文件（模型路径、检索参数、生成参数等）
├── scripts/
│   ├── build_corpus.py          # 从 Excel/CSV 构建统一知识库（corpus）
│   └── interactive_rag.py       # 交互式 RAG 问答控制台
├── excel/
│   ├── paper.csv                # 论文数据（标题 + 摘要）（均可替换为你提供的 csv 文件）
│   ├── achievement.xlsx         # 企业科技成果
│   ├── enterprise.xlsx          # 企业基础信息
│   └── expert.xlsx              # 专家信息
├── corpus/
│   └── corpus.jsonl             # 构建后的统一知识库
├── output/                      # 问答记录输出目录
└── README.md
```

## 环境准备

### 1. 安装依赖

确保你系统环境中已安装 Python (推荐 3.12)。在项目根目录下，执行以下命令安装相关的 Python 依赖包：

```bash
pip install -r requirements.txt
```

### 2. 准备模型

在 `config/my_config.yaml` 中配置模型路径。本项目需要两类模型：

| 角色 | 说明 | 配置字段 |
|------|------|----------|
| **检索模型 (Retriever)** | 用于将文档和问题编码为向量并检索相关文档 | `retrieval_method` |
| **生成模型 (Generator)** | 用于基于检索结果生成最终答案 | `generator_model` |

在 `model2path` 中将模型名称映射到本地路径：

```yaml
model2path:
  your_retriever: "/path/to/your/retriever"
  your_generator: "/path/to/your/generator"

model2pooling:
  your_retriever: "cls"  # 或 "mean"，取决于检索模型
```

如果你是初次运行且本地还没有这些模型，可以通过以下方式快速下载模型。推荐使用 `huggingface-cli`。

```bash
# 下载模型到指定路径 
huggingface-cli download model_name --resume-download --local-dir-use-symlinks False --local-dir /path/to/your/retriever
```

### 3. 配置 GPU

```yaml
gpu_id: "0"  # 使用的 GPU 编号
```

## 运行流程

### 第一步：构建知识库

将 `excel/` 目录下的 Excel/CSV 数据源统一转换为 `corpus.jsonl` 知识库文件。

编辑 `scripts/build_corpus.py` 中的 `FILE_CONFIGS` 列表，为每个数据源指定：

```python
FILE_CONFIGS = [
    {
        "input_path": "excel/paper.csv",       # 输入文件路径
        "prefix": "paper",                      # 文档 ID 前缀
        "title_col": "title",                   # 标题列名
        "content_cols": ["abstract"],           # 内容列名列表
    },
    {
        "input_path": "excel/achievement.xlsx",
        "prefix": "achievement",
        "title_col": "title",
        "content_cols": ["analyse_contect"],
    },
]
```

运行：

```bash
python scripts/build_corpus.py
```

生成的 `corpus/corpus.jsonl` 每行一条 JSON 记录，格式如下：

```json
{"id": "paper_001", "title": "论文标题", "contents": "摘要内容..."}
```

### 第二步：构建向量索引（首次运行前必须）

使用 FlashRAG 内置的索引构建工具，将 `corpus.jsonl` 编码为向量并生成 FAISS 索引：

```bash
python -m flashrag.pipeline.index_building \
    --corpus_path corpus/corpus.jsonl \
    --retrieval_method dense \
    --model_name_or_path /path/to/your/retriever \
    --save_dir corpus/ \
    --pooling cls \
    --gpu_id 0
```

| 参数 | 说明 |
|------|------|
| `--corpus_path` | 第一步生成的知识库文件路径 |
| `--retrieval_method` | 检索方式，使用 `dense` 表示稠密向量检索 |
| `--model_name_or_path` | 检索模型的本地路径 |
| `--save_dir` | 索引文件的输出目录，生成的 `.index` 文件将保存在此 |
| `--pooling` | 池化策略，`cls` 或 `mean`，取决于检索模型（需与 `config/my_config.yaml` 中 `model2pooling` 一致） |
| `--gpu_id` | 用于编码的 GPU 编号 |

构建完成后，将生成的索引文件路径填入 `config/my_config.yaml`：

```yaml
index_path: "corpus/your_corpus.index"
```

首次构建会较慢（需编码全部文档），后续运行将直接加载已有索引。

### 第三步：启动交互式问答

```bash
python scripts/interactive_rag.py --config_path config/my_config.yaml
```

启动后进入交互式问答终端：

```
============================================================
🚀 欢迎使用交互式 RAG 问答系统！
============================================================

⏳ 正在初始化 RAG 核心组件 (检索器、生成器及索引加载)，请耐心等待...
✅ 模型及组件加载完成！可以开始提问。

------------------------------------------------------------
👤 请输入您的问题 (输入 'quit' 或 'exit' 退出):
> 基于该企业的技术储备，最适合的未来创新方向是什么？

🔎 正在检索文档并生成回答...

🤖 [LLM 回答]:
根据该企业在XX领域的核心技术积累，建议重点布局......
答案为：B。
```

#### 使用须知

- 输入问题后自动检索知识库并生成回答
- 每轮对话的检索文档和生成结果实时写入 `output/interactive/<model>/<timestamp>/interactive_history.json`
- 输入 `quit` 或 `exit` 退出
- 按 `Ctrl+C` 可打断当前生成但不会退出终端

## 注意事项

1. **首次启动前**必须先完成知识库构建（第一步），否则系统会因找不到 `corpus.jsonl` 而报错。
2. **向量索引构建**是计算密集型操作，文档量大时可能需要数分钟到数十分钟，请耐心等待。
3. **GPU 显存**需要同时容纳检索模型和生成模型。如果显存不足，可尝试：
   - 降低 `gpu_memory_utilization`
   - 使用更小的生成模型
   - 将 `tensor_parallel_size` 设为大于 1 的值（需要多卡环境）
4. **Prompt 工程**：可通过修改 `interactive_rag.py` 中的 `system_prompt` 来调整模型回答风格。例如加入"请在回答末尾以'答案为：X'的格式给出结论"等约束。
