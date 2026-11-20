# RAG-product-instruction-Agent

基于 RAG（检索增强生成）的产品说明书智能问答 Agent。

## 项目简介

本系统通过检索增强生成技术，将产品说明书文档向量化存储，结合大语言模型实现针对产品使用场景的智能问答，帮助用户快速获取产品操作指引、故障排查等信息。

## 项目结构

```
RAG-product-instruction-Agent/
├── src/
│   ├── agent/        # Agent 核心逻辑（对话管理、工具调用）
│   ├── retrieval/    # 检索模块（向量检索、重排序）
│   ├── ingestion/    # 文档摄入模块（解析、分块、向量化）
│   └── api/          # API 接口层
├── configs/          # 配置文件
├── data/             # 产品说明书数据
├── tests/            # 测试用例
├── docs/             # 项目文档
└── README.md
```

## 技术栈

- **语言**: Python 3.13+
- **向量数据库**: FAISS / Chroma
- **Embedding**: 待定
- **LLM**: 待定
- **框架**: 待定

## 快速开始

```bash
# 安装依赖（待补充）
# pip install -r requirements.txt

# 准备数据：将产品说明书放入 data/ 目录

# 启动服务（待补充）
```

## 开发计划

- [ ] 文档摄入与解析
- [ ] 文本分块与向量化
- [ ] 检索模块
- [ ] Agent 对话逻辑
- [ ] API 接口
- [ ] 测试与评估
