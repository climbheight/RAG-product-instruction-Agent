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
│   ├── ingestion/        # PDF 解析、图片抽取、版面切分、切分、VLM 标注（9 个模块）
│   ├── retrieval/        # BGE 嵌入、Chroma 双索引、图文二次检索
│   ├── agent/            # LangGraph 图：状态、5 节点、2 条件边
│   ├── llm/              # 本地推理封装：LLMBase / HFLocalLLM / MockLLM
│   └── api/              # 服务层（预留）
├── app/
│   └── app.py            # Gradio 界面
├── configs/
│   └── manuals.json      # 16 份手册的 doc_id / 品牌 / 型号映射
├── data/
│   ├── raw_pdfs/         # 16 份课程手册（.gitignore，本地保留）
│   └── extracted/        # 解析产物：chunks.jsonl / figures.jsonl / images/（图片 .gitignore）
├── docs/
│   ├── course/           # 课程 PPT、任务书、指导书（.gitignore）
│   ├── annotation-requirements.md   # 图片标注规范
│   └── annotation-handover.md       # 标注任务交接说明
├── models/               # 本地模型权重（.gitignore）
├── scripts/              # 探测脚本与标注工具
├── tests/
├── vectorstore/          # Chroma 持久化（.gitignore）
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

# 2. 嵌入模型权重（bge-small-zh-v1.5，约 96MB）
#    放到 models/bge-small-zh-v1.5/，需含：
#    config.json  model.safetensors  tokenizer.json  vocab.txt
#    1_Pooling/config.json（必须含 "embedding_dimension": 512）
#    也可不下载，代码会回落到 HF 仓库名自动拉取

# 3. 建库（解析 → 切分 → 落盘）
python -m src.ingestion.build_index          # 全部 16 份，含 OCR 约 5 分钟
python -m src.ingestion.build_index --only roborock_g10s_pro   # 单份调试

# 4. 建向量索引（文本 + 图片双 collection，约 2 分钟）
python -m src.retrieval.build

# 5. 启动界面
python -m app.app                            # 打开 http://127.0.0.1:7860
```

### 接入本地推理模型（必做，否则只有占位输出）

当前 `models/` 下只有 BGE 嵌入模型，**没有生成模型**，链路会回落到 `MockLLM`。
接真实模型二选一：

```bash
# 方案 A：MiniMind2（课程首选，权重 208MB）
# 下载 jingyaogong/MiniMind2 的全部非 images/ 文件到 models/MiniMind2/

# 方案 B：Qwen3-0.6B（课程允许的备选）
# 下载 Qwen/Qwen3-0.6B 到 models/Qwen3-0.6B/
```

放好后 `load_llm()` 会自动扫描 `models/` 下第一个含 `config.json` 的非 bge 目录并加载；
也可显式指定：`load_llm("models/MiniMind2")`。

> 扫描页 OCR 需要 `rapidocr-onnxruntime`，首次运行会下载约 15MB 的模型。

## 当前进度

| 阶段 | 状态 | 产物 |
| --- | --- | --- |
| 手册解析 + 切分 | 完成 | 16 份 / 1242 页 / 1928 chunks |
| 图片抽取 | 完成 | 874 张（位图 / 矢量渲染 / 扫描页区块） |
| 扫描页版面切分 | 完成 | 宇树 G1、美的电饭煲共 61 个图区 |
| 图片去重 | 完成 | 46 组重复 / 67 张合并，874 → 807 |
| VLM 图片标注 | **完成** | **807 / 807（100%）** |
| BGE 嵌入 + 双索引 | 完成 | `manual_chunks` 1928 + `manual_figures` 785 |
| LangGraph 链路 | 完成 | 5 节点 + 2 条件边 |
| Gradio 界面 | 完成 | 问答 + 图片画廊 + 检索详情 |
| 本地推理模型 | **待接** | 当前用 MockLLM 占位 |

> **任务一与任务二的边界**：`src/retrieval/vectorstore.py` 里的
> `collect_candidates()` / `rerank_figures()` 是「答案生成后再检索图片」的实现，
> 已接入 LangGraph 的 `retrieve_images` 节点。

## 问答链路（LangGraph）

```
START → retrieve → grade ──不足──→ fallback → END
                    │ 足够
                    ↓
                 generate → retrieve_images → END
```

**哪些由程序控制、哪些由模型生成**（验收会问）：

| 环节 | 归属 | 说明 |
| --- | --- | --- |
| 文本检索 | 程序 | BGE + Chroma，top-6 |
| 召回判定 | 程序 | top-1 相似度 ≥ 0.56 |
| 候选池限定 | 程序 | 锁定 top-1 命中文档，杜绝跨产品配图 |
| 图片精排 | 程序 | 余弦相似度 ≥ 0.42，最多 2 张 |
| 兜底话术 | 程序 | 空输入 / 无结果 / 异常三种，各不同 |
| **自然语言答案** | **模型** | 仅此一处 |

**召回阈值 0.56 是实测标定的**，不是拍的：

| 类别 | top-1 分数 |
| --- | --- |
| 相关问题 | 0.596 ~ 0.778 |
| 无关问题 | 0.248 ~ 0.521 |

早先用 0.45 时，「量子计算机怎么搭建」会被判成召回充分、拿路由器手册作答——
典型的幻觉入口，已修正。

> ⚠️ **当前状态**：本地没有 MiniMind / Qwen3 权重，链路用 `MockLLM` 跑通，
> 它只验证节点流转与兜底，**回答质量无意义**。
> 接真实模型：把权重放到 `models/` 下（含 `config.json`），或 `load_llm("路径")`。

## 图片检索的设计要点

图片本身不能被 BGE 嵌入，所以每张图都要先变成一段文本（`search_text`）：

```
【产品名】章节 第N页插图。画面：<VLM描述> 说明：<邻近文本> 图中文字：<OCR/矢量取字>
```

四个必须踩对的点：

1. **本语料几乎没有图注**（实测 5 份正式手册 `^图\s*\d+-\d+` 命中 0 次），
   所以图文关联靠**空间邻近 + 阅读顺序 + 「如图/参见图」引用词**，不靠图注正则。
2. **矢量图要单独取字**：石头 G10S Pro、大疆 Mic2 全篇是矢量插图，
   `get_images()` 返回 0、OCR 也无从下手，必须用 `page.get_text(clip=bbox外扩)` 从文本层挖标签。
3. **套话必须清洗**：「配图仅供参考，请以产品实际为准」紧贴图片，
   空间邻近判据下必然胜出，不清掉会让所有图的向量高度相似、检索失效。
4. **候选池必须锁死单个产品**：实测「怎么把机器人搬起来」会同时命中石头和宇树，
   若把两个文档的图都放进池子，就会把宇树的充电图配给扫地机。

## 开发计划

**任务一 · 离线建库**
- [x] 环境与依赖锁定
- [x] PDF 解析：文本层检测，扫描页 OCR 兜底（55 页）
- [x] 图片抽取：位图 / 矢量渲染 / 扫描页版面切分
- [x] 图片去重：dHash + aHash + 像素差异三重判据，874 → 807
- [x] 切分策略（300~480 字符 / overlap 80，图与说明绑成原子单元）
- [x] BGE 嵌入 + Chroma 双索引，写入完整元数据
- [x] VLM 图片标注 **807 / 807（100%）**
- [x] 图文二次检索（单产品候选池 + 阈值触发）

**任务二 · Agent 链路**
- [x] LangGraph 图：retrieve → grade → generate → retrieve_images，含 fallback
- [x] 检索封装为可调用方法，节点间用 TypedDict 状态传递
- [x] 无结果 / 空输入 / 异常三种兜底话术
- [x] 召回阈值实测标定（0.56）
- [ ] **接入 MiniMind 真实推理**（当前 MockLLM 占位）

**任务四 · 界面**
- [x] Gradio：问题输入 + 产品下拉 + 答案 + 图片画廊
- [x] 检索详情折叠区（执行路径 / 命中片段 / 相似度 / 配图判定）
- [x] 空输入提示、等待态、无图时正常显示
- [ ] 图片点击放大（Gallery 已支持缩放，未做独立大图面板）

**收尾**
- [ ] 端到端测试记录（设备、环境、响应时间）
- [ ] 固定问题集回归（≥ 20 题，覆盖 16 个产品）
- [ ] 中期报告（2026-11-30）

## 已验证 / 未验证

诚实记录，避免把没测过的当已实现：

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 检索召回正确性 | ✅ 已验证 | 5 个相关问题 top-1 命中对应产品 |
| 无关问题不硬答 | ✅ 已验证 | 6 个无关问题全部走 fallback |
| 跨产品配图 | ✅ 已修复 | 候选池锁死 top-1 文档 |
| 链路节点流转 | ✅ 已验证 | 4 类输入的执行路径符合预期 |
| 界面构建 | ✅ 已验证 | 17 个产品选项，Blocks 构建成功 |
| **回答质量** | ❌ **未验证** | 无真实模型，MockLLM 输出无意义 |
| 图文配合准确度 | ❌ 未验证 | 需真实答案才能评估「用答案二次检索」的效果 |
| CPU 推理速度 | ❌ 未测 | 待接入模型后实测 |

## 交付节点

| 内容 | 命名 | 截止 |
| --- | --- | --- |
| 中期报告（每组一份） | 小组成员姓名.docx | 2026-11-30 |
| 课程设计报告（每人一份） | 个人姓名_学号.docx | 2027-01-06 |
| 源码 + 用户使用说明书（每组一份） | 小组成员姓名.rar | 2027-01-06 |
| 现场验收 | — | 2027-01-06 |

评分：现场验收 60% + 课程文档 40%。重点看 Agent 设计规范性、RAG 检索完整性、回答准确无幻觉、界面美观。
