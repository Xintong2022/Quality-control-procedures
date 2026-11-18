# Render 部署指南 — Aloka质量检测平台

仓库已含 `render.yaml` 蓝图，按下面步骤操作约 5 分钟拿到公网链接。

## 部署步骤

1. 打开 https://dashboard.render.com 并注册/登录（可用 GitHub 账号直接登录）。

2. 左侧 **Blueprints** → **New Blueprint Instance**（或 New + → Blueprint）。

3. 选择仓库 **Xintong2022/Quality-control-procedures**，点击 **Connect**。
   Render 会自动读取仓库里的 `render.yaml`，识别出服务 `aloka-qa-platform`。

4. 确认信息后点 **Apply / Deploy**。构建约 3-6 分钟（要装 Flask + pandas + Playwright 并下载 Chromium）。

5. 构建完成顶部会显示服务地址，形如：
   `https://aloka-qa-platform.onrender.com`
   这就是你的公网访问链接，手机/任何电脑都能打开。

## 首次访问

- 平台处于免登录开放态（仓库不含 settings.json）。**建议立即进入「设置」设置管理员密码**，否则任何拿到链接的人都能操作。

## 免费层限制（重要）

| 限制 | 说明 |
|---|---|
| 休眠 | 15 分钟无流量自动休眠，下次访问冷启动约 50 秒 |
| 内存 | 512MB，够 Web 操作与单窗口评测；**10 窗口并发 Bot 自动化跑不动** |
| 磁盘 | 非持久：重新部署后 runs.json / cases.json 等运行时数据重置（案例库在仓库内会保留初始 1000 条） |
| 流量 | 每月 100GB 免费 |

## Bot 自动化在云上能跑吗？

能启动，但受两个限制：
- Playwright Chromium 在 Linux 容器内运行（render.yaml 已加 `--with-deps` 装系统依赖）；
- 免费层内存小，并发窗口建议降到 **1-2 个**；要 10 窗口并发请升级 Starter（$7/月，2GB）。

## 更新版本

之后在本地 `git push` 到 main，Render 会**自动重新部署**，链接不变。

## 数据持久化（可选）

若想保留评测记录不因重新部署丢失，可在 Render 服务设置里挂载 Disk（付费功能），或改为把 runs.json 同步到外部存储。
