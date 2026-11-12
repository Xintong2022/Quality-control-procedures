# 旅游 AI 质检平台 (Travel QA System)

本地部署的旅游顾问对话 / Bot 行程方案质检平台。围绕一份《需求测试题库》工作：每个案例自带「埋的坑」与「正确应对」作为评分底牌，平台据此对 Bot 产出的行程方案进行**避坑评测**。

> 纯本地运行，不依赖任何外部服务。仅在启用「真实 Bot 评测 / LLM 语义裁判」时才需要你自备一个 OpenAI 兼容的模型 API Key。

---

## 功能特性

- **双引擎评测**
  - **TREK（确定性硬逻辑）**：离线、可复现，评测结构化行程对合成知识库（375 城 / 航班·酒店·景点·租车）的匹配度（实体真实性、预算、城市约束、营业时间、时空可达等）。
  - **轻量裁判（软质量）**：LLM 盲审，补 TREK 抓不到的「避坑 + 主动服务」软质量分。
- **避坑评测 · 融合三层**（核心）：把题库「正确应对」拆成结构化「避坑断言」，三层融合打分——
  1. **红线零容忍**：致命项（安全 / 非法驾驶 / 证件 / 闭馆撞期）遗漏即一票否决；
  2. **8 维加权命中率**：证件合规 / 时间窗口 / 预订限流 / 交规地方法规 / 安全健康 / 地理动线 / 费用陷阱 / 人群适配；
  3. **LLM 语义裁判**：逐条判命中 / 遗漏 + 证据 + 置信度。
  - 输出：避坑得分、致命风险计数、8 维热力图、逐条命中 / 遗漏清单。
- **案例生成器**：基于题库词表 + 真实示例，调用模型合成全新对抗测试案例并入库。
- **管理员鉴权**：Flask session + pbkdf2 密码哈希，可设 / 改 / 删登录密码。
- **深色 / 浅色双主题**：设置页一键切换（高对比明快浅色主题）。
- **数据可视化**：仪表盘 KPI、评测报告、维度弱项热力图。

---

## 目录结构

```
travel_qa_system/
├── server.py              # Flask 后端主入口（端口 8501）
├── index.html             # 前端单页（由 server 直接托管）
├── trek_engine.py         # TREK 确定性评测引擎
├── pitfall_engine.py      # 避坑评测 · 融合三层
├── mock_trek_bot.py       # 可注入缺陷的 Mock Bot（演示/自检用）
├── build_assertions.py    # 把题库「正确应对」批量拆成避坑断言
├── import_bank.py         # 从 xlsx 题库导入 cases.json
├── app.py                 # 旧版 Streamlit 入口（已弃用，保留参考）
├── cases.json             # 案例库（埋的坑 / 正确应对 / 词表）
├── pitfall_vocab.json     # 坑类型词表（一级 + 二级）
├── settings.example.json  # 配置模板（复制为 settings.json 后填 Key）
├── static/                # 前端静态资源（logo 等）
├── trek_repo/             # TREK 评测基准（知识库 + 评分器）
├── requirements.txt
├── start.bat / start.sh   # 跨平台一键启动脚本
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动

- **Windows**：双击 `start.bat`
- **macOS / Linux**：`./start.sh`
- 或手动：

```bash
python server.py          # Windows
python3 server.py         # macOS / Linux
```

启动后访问 **http://localhost:8501**。

> 首次运行 `settings.json` 不存在时，平台处于**免登录开放态**（方便本地试用）。进入「设置」页可设置管理员密码；设密码后重启即需登录。

---

## 配置（可选）

复制模板并填入你的模型 Key：

```bash
cp settings.example.json settings.json
```

| 字段 | 说明 |
|---|---|
| `judge_api_key` | 裁判 / 生成模型 API Key（OpenAI 兼容） |
| `judge_base_url` | 模型服务地址，默认 SiliconFlow |
| `judge_model` | 裁判模型名 |
| `generator_model` | 生成模型名（留空则用 judge_model） |
| `bot_api` | 你的 Bot 结构化计划 API 地址（真实 Bot 评测用） |
| `sample_times` | 轻量裁判采样次数 |

> ⚠️ `settings.json` 含密钥，**已被 `.gitignore` 排除，不会进入版本库**。

---

## 使用说明

1. **评测控制台**：选案例 → 选数据源（TREK Gold 基准 / 内置 Mock Bot / 你的 Bot）→ 运行质检。Mock Bot 可注入缺陷验证引擎是否抓错。
2. **避坑评测**：选案例 → 选 `gold`（完美基线）或 `mock`（注入缺陷）→ 运行，查看避坑得分与逐条清单。真实 Bot 模式需配置 `bot_api` 与有效 Key。
3. **案例生成器**：基于词表合成新题，可在「词表」模态里预览 / 编辑坑类型词表。
4. **设置**：改登录密码、切换深色 / 浅色主题。

---

## 技术栈

Python 3.10+ · Flask · OpenAI SDK（兼容）· pandas / numpy · 原生 HTML/CSS/JS（单页）。

## 许可

应用层代码以 **MIT** 发布，见 `LICENSE`。`trek_repo/` 源自 TREK 评测基准，其许可见 `trek_repo/LICENSE`。
