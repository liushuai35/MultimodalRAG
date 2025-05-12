多模态 RAG 系统初探


项目简介
本项目展示了一个结合 CLIP、Faiss 和 智谱 AI 的 检索增强生成 (RAG) 系统，能够处理文本和图像数据，支持索引、检索和生成文本、图像及多模态查询的响应。

✨ 关注我的分享渠道 ✨

小红书号: AnthroSeekTheX (Let's Seek The X!)
技术方案设计与思考 (飞书文档)

💡 温馨提示 💡

想快速获取代码的 AI 解析？试试将浏览器地址中的 github.com 替换为 deepwiki.com！（例如：https://github.com/user/repo → https://deepwiki.com/user/repo）。依赖 Deepwiki 服务。


🚀 主要特性

多模态索引：同时索引文本描述和关联图像。
向量嵌入：使用 Hugging Face 的 CLIP 模型 (openai/clip-vit-base-patch32) 为文本和图像生成统一向量表示。
高效检索：利用 Faiss (IndexIDMap2 + IndexFlatIP) 实现快速相似性搜索。
持久化存储：通过 SQLite 存储文档元数据，并将 Faiss 索引保存到磁盘。
上下文生成：结合检索到的信息，使用 智谱 AI (glm-4-flash) 生成精准回答。
灵活查询：支持纯文本、纯图像及文本+图像的多模态查询。
模块化设计：代码结构清晰，分为 MultimodalEncoder、Indexer、Retriever 和 Generator 类。

⚙️ 系统要求

Python 3.9 或更高版本
智谱 AI API 密钥（可从 智谱 AI 开放平台 获取）
必要的 Python 库（见 requirements.txt）

🛠️ 安装

克隆仓库：
git clone https://github.com/your-username/MultimodalRAG.git
cd MultimodalRAG


创建并激活虚拟环境：
# 使用 conda（推荐）
conda create -n multimodal_rag python=3.12 -y
conda activate multimodal_rag

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows


安装依赖：
pip install -r requirements.txt

注意：requirements.txt 包含 faiss-cpu。若需 GPU 支持，请在配置好 CUDA 后安装 faiss-gpu，并卸载 faiss-cpu。


🔑 配置

设置智谱 AI API 密钥：在项目根目录创建 .env 文件，添加以下内容：ZHIPUAI_API_KEY=your_api_key

确保脚本能加载 .env 文件（例如使用 python-dotenv 库）。

🚀 运行系统

准备数据（见下方 数据准备 部分）。
运行脚本：python yangrouchuan.py

脚本将自动：
加载数据和图像。
初始化编码器、索引器、检索器和生成器。
为文本、图像和平均向量构建索引。
执行示例查询并保存结果。
将索引和数据库持久化到磁盘。



📁 数据准备

data.json：一个 JSON 文件，包含文档对象列表，每个对象需包含：

name：唯一文档 ID（例如 Bandgap1）。
description：文档的文本描述。

示例 data.json：
[
  {
    "name": "Bandgap1",
    "description": "一个基础的带隙基准电路图，展示 BJT 晶体管和电阻，用于生成温度不敏感的参考电压。"
  },
  {
    "name": "PTAT_Current",
    "description": "该原理图展示如何使用两个不匹配的 BJT 生成与绝对温度成正比 (PTAT) 的电流。"
  }
]


images/ 目录：包含与 data.json 中 name 字段对应的图像文件（例如 Bandgap1.png、PTAT_Current.jpg）。文档若无图像则无需提供。


🏗️ 代码结构

MultimodalEncoder：使用 CLIP 将文本和图像编码为向量。
Indexer：管理 Faiss 索引和 SQLite 元数据存储。
Retriever：编码查询并检索相关文档。
Generator：与智谱 AI API 交互生成响应。
主脚本 (yangrouchuan.py)：协调整个 RAG 流程。

💡 注意事项与未来改进

性能：大规模数据可使用 faiss-gpu 或高级 Faiss 索引（如 IndexIVFFlat）。
多模态融合：探索比简单向量平均更复杂的融合策略。
图像理解：升级到多模态 LLM（例如 GLM-4V）以直接处理图像。
错误处理：增强健壮的错误管理和日志记录。
可扩展性：生产环境中可替换 SQLite 为向量数据库。
提示工程：优化生成器的 Prompt 以提升效果。
文本分块：对长文档进行分块处理。

📅 更新日志

2024.04.28：更新 CLIP 文本/图像处理、存储和检索方法。
2024.04.27：添加多种技术方案的初步实现。
2024.04.25：初始代码库。

🤝 如何贡献
欢迎任何形式的贡献！如有建议、发现 Bug 或想改进代码，请：

通过 Issues 报告问题或建议。
通过 Pull Requests 提交改进。

📄 许可证
本项目采用 MIT 许可证 开源。
