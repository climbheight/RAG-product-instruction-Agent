# RAG-product-instruction-Agent

基于产品文档的大模型智能体 —— 2026 软件课程设计（华中科技大学 电子信息与通信学院 · 电信 2405）

用户输入产品使用问题（安装、连接、设置、故障处理），系统检索课程提供的手册，返回**相关文字 + 对应原图**，由本地轻量大模型组织成有依据的回答，通过对话界面呈现。

---

## 课程指定的技术路线

| 环节 | 选型 | 说明 |
| --- | --- | --- |
| PDF 解析 | **PyMuPDF** | 文本 + 图片提取，保留章节 / 型号 / 版本关联 |
| 切分 | 自实现 | 按章节段落，300~600 字符，重叠 50~100 字符（起步值，需实测调整） |
| 嵌入 | **BGE** `bge-small-zh-v1.5` | 文档与查询必须用同一模型 |
| 向量库 | **Chroma** / FAISS | 保留元数据；按型号检索并比较 Top-k |
| Agent 编排 | **LangGraph** | 节点 / 状态 / 条件边 |
| 本地推理 | **MiniMind** | 备选 Qwen3-0.6B、MiniCPM-1B |
| 界面 | **Gradio** | 输入框、产品选择、会话、等待态、图片放大 |

## 四项核心任务

1. **离线建库** —— 加载 → 切分 → 嵌入 → 存储。全部 16 份手册纳入知识库，保存"文本片段 ↔ 所属文档 ↔ 章节 ↔ 图片"的关系。
2. **Agent 链路** —— LangGraph 连接用户输入、检索、模型调用、结果返回；检索封装为可调用工具；无结果 / 工具失败要有明确提示。
3. **本地模型推理** —— MiniMind 权重与依赖匹配，CPU 或 GPU 加载，提示词工程约束"不编造手册外的参数"。
4. **图文对话界面** —— 答案 + 手册原图并排展示，无图时正常显示文字。

## 项目结构

```
RAG-product-instruction-Agent/
├── src/
│   ├── ingestion/    # PDF 解析、图片抽取、清洗、切分、嵌入、建库
│   ├── retrieval/    # 向量检索、Top-k 调参、图文关联回取
│   ├── agent/        # LangGraph 图：状态、节点、条件边
│   ├── llm/          # MiniMind 本地推理封装（待建）
│   └── api/          # 服务层
├── app/              # Gradio 界面（待建）
├── configs/          # 配置（模型路径、切分参数、Top-k 等）
├── data/
│   ├── raw_pdfs/     # 16 份课程手册（.gitignore，本地保留）
│   └── extracted/    # 解析产物：文本 / 图片（.gitignore）
├── docs/
│   ├── course/       # 课程 PPT、任务书、指导书（.gitignore）
│   └── ...           # 本项目文档
├── tests/
└── README.md
```

## 数据说明

课程提供 16 份产品手册，覆盖 12 个品牌：

| 类别 | 手册 |
| --- | --- |
| 手机 / 穿戴 | OPPO ColorOS 给长辈的玩机指南、华为 Pura70 Ultra (HarmonyOS 5)、华为 WATCH GT4 |
| 影像设备 | 大疆 Mini 4 Pro、Osmo Action 5 Pro、Osmo Pocket 3、DJI Mic 2 |
| 电脑 / 打印 | 联想 ThinkPad X1 Carbon Gen12、奔图 P3300 系列 |
| 家电 | 美的 MB-FB40E551 智能电饭煲、石头 G10S Pro |
| 网络 | TP-LINK WiFi6 无线路由器 |
| 汽车 | 问界 M9 智能座舱与车辆操作、辅助驾驶与泊车（章节摘录） |
| 机器人 / 3D 打印 | 宇树 G1 官方手册、创想三维 Ender3 V3 SE |

> `data/raw_pdfs/` 与 `docs/course/` 已加入 `.gitignore`，仅本地保留，不提交到仓库。

## 快速开始

```bash
# 1. 环境
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

# 2. 建库（读取 data/raw_pdfs/ → 写入 vectorstore/）
python -m src.ingestion.build_index

# 3. 启动
python -m app.main
```

## 开发计划

- [ ] 环境与依赖锁定（PyMuPDF / sentence-transformers / chromadb / langgraph / gradio）
- [ ] PDF 解析：文本层检测，扫描页转 OCR 兜底
- [ ] 图片抽取与章节关联（图注匹配）
- [ ] 切分策略与参数实验（300~600 / overlap 50~100）
- [ ] BGE 嵌入 + Chroma 建库，写入元数据
- [ ] 固定问题集验证检索召回与图文对应
- [ ] LangGraph 图搭建（retrieve → grade → generate → fallback）
- [ ] MiniMind 本地推理接入
- [ ] Gradio 界面（含图片放大、等待态、空输入提示）
- [ ] 端到端测试记录（设备、环境、响应时间）

## 交付节点

| 内容 | 命名 | 截止 |
| --- | --- | --- |
| 中期报告（每组一份） | 小组成员姓名.docx | 2026-11-30 |
| 课程设计报告（每人一份） | 个人姓名_学号.docx | 2027-01-06 |
| 源码 + 用户使用说明书（每组一份） | 小组成员姓名.rar | 2027-01-06 |
| 现场验收 | — | 2027-01-06 |

评分：现场验收 60% + 课程文档 40%。重点看 Agent 设计规范性、RAG 检索完整性、回答准确无幻觉、界面美观。
