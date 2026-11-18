# -*- coding: utf-8 -*-
"""
bot_session.py — 用 Playwright CDP 驱动有头浏览器操作 bot 网页端

架构：
  1. launch(): 用独立 user-data-dir 启动有头 Chrome（带远程调试端口），
     打开 bot.alokatravel.cn。用户在这个窗口手动登录，登录态持久化到磁盘。
  2. 之后每次评测，通过 CDP 连到这个已登录的浏览器，填题 → 发送 → 等回答 → 抓文本。
  3. 不侵入 bot 服务端、不受 CSP/跨域限制——因为是真实浏览器自动化。

对外函数：
  - launch_browser(fresh=False)  启动浏览器（幂等）；fresh=True 时用全新无登录 profile
  - browser_status()       查询浏览器/会话/页面状态
  - probe_dom()            探测页面上的输入框/按钮/消息容器候选选择器
  - register_account()     自动完成手机号注册+资料填写，进入聊天页
  - launch_and_register()  全新启动 + 自动注册，一步到位
  - send_and_capture(question, cfg)   发一条题并抓回 bot 回答
  - close_browser()        关闭浏览器

配置（settings.json 可选，均有默认值）：
  - bot_url: bot 网站地址，默认 https://bot.alokatravel.cn/
  - bot_cdp_port: 调试端口，默认 9223
  - bot_user_data: 登录态目录，默认 <项目>/bot_profile
  - bot_input_selector / bot_send_selector / bot_answer_selector: 覆盖探测结果
"""

import os
import json
import time
import subprocess
import shutil
import threading

# 多 worker 并发：手机号生成、手机号文件读写的线程锁
_phone_lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BOT_URL = "https://bot.alokatravel.cn/"
DEFAULT_CDP_PORT = 9223
PROFILE_DIR = os.path.join(BASE_DIR, "bot_profile")
STATE_FILE = os.path.join(BASE_DIR, "bot_session.json")


def _settings():
    try:
        with open(os.path.join(BASE_DIR, "settings.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def make_worker(worker_id):
    """为自动模式创建一个独立 worker 配置：专属 CDP 端口 + 专属 profile 目录。
    worker_id 从 0 开始；端口 = 基础端口 + worker_id，避免多窗口互相抢占调试端口。
    """
    s = _settings()
    base_port = int(s.get("bot_cdp_port") or DEFAULT_CDP_PORT)
    port = base_port + 100 + int(worker_id)  # 9223 -> 9323+，与手动窗口(9223)隔离
    profile = os.path.join(BASE_DIR, f"bot_profile_w{int(worker_id)}_{int(time.time() * 1000)}")
    return {"id": int(worker_id), "cdp_port": port, "profile": profile,
            "state_file": os.path.join(BASE_DIR, f"bot_session_w{int(worker_id)}.json")}


def _cfg(worker=None):
    s = _settings()
    if worker:
        # 自动模式 worker：端口/profile/状态文件全部独立，不碰手动会话的全局 state
        return {
            "bot_url": s.get("bot_url") or DEFAULT_BOT_URL,
            "cdp_port": int(worker["cdp_port"]),
            "profile": worker["profile"],
            "input_sel": s.get("bot_input_selector") or "",
            "send_sel": s.get("bot_send_selector") or "",
            "answer_sel": s.get("bot_answer_selector") or "",
            "wait_ms": int(s.get("bot_wait_ms") or 60000),
        }
    st = _read_state()
    return {
        "bot_url": s.get("bot_url") or DEFAULT_BOT_URL,
        "cdp_port": int(s.get("bot_cdp_port") or DEFAULT_CDP_PORT),
        # fresh 注册使用唯一新目录并记入 active_profile；避免删除大目录触发系统批量删除拦截
        "profile": st.get("active_profile") or s.get("bot_user_data") or PROFILE_DIR,
        "input_sel": s.get("bot_input_selector") or "",
        "send_sel": s.get("bot_send_selector") or "",
        "answer_sel": s.get("bot_answer_selector") or "",
        "wait_ms": int(s.get("bot_wait_ms") or 60000),
    }


def _state_path(worker=None):
    return worker["state_file"] if worker else STATE_FILE


def _read_state(worker=None):
    try:
        with open(_state_path(worker), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_state(st, worker=None):
    with open(_state_path(worker), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def _find_chrome():
    """找本机 Chrome/Edge 可执行文件。"""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def browser_status(worker=None):
    """查询浏览器进程是否存活、调试端口是否可连。"""
    import urllib.request
    cfg = _cfg(worker)
    state = _read_state(worker)
    port_open = False
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{cfg['cdp_port']}/json/version", timeout=2)
        port_open = True
    except Exception:
        pass
    return {
        "running": port_open,
        "cdp_port": cfg["cdp_port"],
        "bot_url": cfg["bot_url"],
        "profile": cfg["profile"],
        "started_at": state.get("started_at"),
        "last_probe": state.get("last_probe"),
        "selectors": {
            "input": cfg["input_sel"],
            "send": cfg["send_sel"],
            "answer": cfg["answer_sel"],
        },
    }


def launch_browser(fresh=False, worker=None):
    """启动有头浏览器并打开 bot 网站（幂等）。

    fresh=True 时强制关闭已运行实例，用全新无登录态启动；用于"从注册开始"的场景。
    worker 不为 None 时使用其专属端口/profile/state（自动模式多窗口并发）。
    返回状态字典。
    """
    cfg = _cfg(worker)
    st = browser_status(worker)
    if fresh and st["running"]:
        close_browser(worker=worker)
        time.sleep(1)
        st = browser_status(worker)
    if st["running"]:
        return {"ok": True, "already_running": True, **st}

    chrome = _find_chrome()
    if not chrome:
        return {"ok": False, "error": "未找到 Chrome/Edge 浏览器"}

    if fresh and not worker:
        # 手动模式：绝不删除旧 profile 目录——大目录批量删除会触发系统安全拦截
        # （SAFE_DELETE_BULK_CONFIRM_REQUIRED）导致请求卡死。改用全新唯一目录天然无登录态。
        new_profile = os.path.join(BASE_DIR, "bot_profile_" + str(int(time.time() * 1000)))
        cfg["profile"] = new_profile
        state = _read_state()
        state["active_profile"] = new_profile
        state.pop("started_at", None)
        _write_state(state)
    # worker 模式的 profile 由 make_worker 生成，已是唯一新目录，无需再处理
    os.makedirs(cfg["profile"], exist_ok=True)
    # 关键：独立 user-data-dir + 远程调试端口。用户在此窗口登录后，登录态随目录持久化。
    args = [
        chrome,
        f"--remote-debugging-port={cfg['cdp_port']}",
        f"--user-data-dir={cfg['profile']}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-blink-features=AutomationControlled",
        "--start-maximized",
        (cfg["bot_url"].rstrip("/") + "/sign-up") if fresh else cfg["bot_url"],
    ]
    try:
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        return {"ok": False, "error": f"启动浏览器失败：{e}"}

    # 等调试端口就绪
    import urllib.request
    ready = False
    for _ in range(30):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{cfg['cdp_port']}/json/version", timeout=1)
            ready = True
            break
        except Exception:
            time.sleep(0.5)
    if not ready:
        return {"ok": False, "error": "浏览器已启动但调试端口未就绪，请稍后重试"}

    state = _read_state(worker)
    state["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["worker_id"] = worker["id"] if worker else None
    _write_state(state, worker)
    return {"ok": True, "already_running": False, **browser_status(worker)}


# ---------------------------------------------------------------------------
# 自动注册：随机手机号 → 验证码 → 资料 → 进入聊天页
# ---------------------------------------------------------------------------
PHONES_FILE = os.path.join(BASE_DIR, "bot_phones.json")
VERIFY_CODE = "261018"  # 统一验证码（测试环境固定）


def _load_phones():
    try:
        with open(PHONES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def gen_phone():
    """生成一个不重复的随机中国大陆手机号，并持久化到 bot_phones.json（线程安全）。"""
    import random
    with _phone_lock:
        used = set(str(x) for x in _load_phones())
        prefixes = ["130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
                    "150", "151", "152", "153", "155", "156", "157", "158", "159",
                    "176", "177", "178", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189"]
        for _ in range(500):
            phone = random.choice(prefixes) + "".join(random.choice("0123456789") for _ in range(8))
            if phone not in used:
                used.add(phone)
                try:
                    with open(PHONES_FILE, "w", encoding="utf-8") as f:
                        json.dump(sorted(used), f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                return phone
    raise RuntimeError("无法生成不重复的手机号")


def _rand_profile():
    import random
    nick = random.choice(["旅行", "风景", "逍遥", "云游", "自在", "探索", "漫游", "逐风", "星辰", "拾光"]) \
        + random.choice(["者", "客", "家", "人", "小分队", "同学"])
    nick += str(random.randint(10, 999))
    gender = random.choice(["女", "男", "暂不说"])
    year = random.randint(1985, 2000)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    birthday = f"{year:04d}-{month:02d}-{day:02d}"
    return {"nickname": nick, "gender": gender, "birthday": birthday}


def register_account(worker=None):
    """在全新未登录的 bot 页面走注册流程，进入聊天页。返回 {ok, phone, nickname, gender, birthday, ...}。

    前置：浏览器已用 fresh=True 启动，当前停在 /sign-up 未登录页。
    """
    if not browser_status(worker)["running"]:
        return {"ok": False, "error": "浏览器未启动"}
    pw = browser = None
    try:
        from playwright.sync_api import sync_playwright
        cfg = _cfg(worker)
        pw = sync_playwright().start()
        browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{cfg['cdp_port']}")
        ctx = browser.contexts[0] if browser.contexts else browser.new_context()
        page = None
        for p in ctx.pages:
            if "alokatravel" in (p.url or ""):
                page = p; break
        if page is None and ctx.pages:
            page = ctx.pages[0]
        if page is None:
            return {"ok": False, "error": "未找到 bot 页面"}
        # 强制回到全新注册页：清掉该域的本地存储/会话，确保是无登录初始态
        try:
            page.goto(cfg["bot_url"], wait_until="domcontentloaded", timeout=30000)
            ctx.clear_cookies()
            page.evaluate("() => { try{ localStorage.clear(); sessionStorage.clear(); }catch(e){} }")
            page.goto(cfg["bot_url"] + "sign-up", wait_until="domcontentloaded", timeout=30000)
        except Exception:
            if "sign-up" not in (page.url or ""):
                page.goto(cfg["bot_url"], wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)

        phone = gen_phone()
        prof = _rand_profile()
        steps = [phone]

        # 1) 手机号
        ph = page.locator("#phone").first
        ph.wait_for(state="visible", timeout=15000)
        ph.fill(phone)

        # 2) 获取验证码
        page.get_by_text("获取验证码", exact=True).last.click(timeout=10000)
        time.sleep(2)
        steps.append("sent_code")

        # 3) 验证码（此步页面只有一个可见 input）
        code_in = page.locator("input:visible").first
        code_in.wait_for(state="visible", timeout=10000)
        code_in.fill(VERIFY_CODE)
        steps.append("filled_code")

        # 4) 继续（点击后页面跳转、按钮会消失，容竞态：点击后只要昵称框出现即算成功）
        try:
            page.get_by_text("继续", exact=True).first.click(timeout=5000, no_wait_after=True)
        except Exception:
            pass  # 按钮可能因跳转而 detached，忽略
        steps.append("clicked_continue")

        # 5) 资料页：昵称 / 性别 / 生日（等昵称框出现即确认已到资料页）
        name_in = page.locator('input[id$="-name"]').first
        name_in.wait_for(state="visible", timeout=15000)
        name_in.fill(prof["nickname"])
        page.get_by_text(prof["gender"], exact=True).last.click(timeout=10000)
        bday = page.locator('input[type="date"]').first
        if bday.count():
            bday.fill(prof["birthday"])
        steps.append("filled_profile")

        # 6) 开始使用（点击后会跳转，容竞态）
        try:
            page.get_by_text("开始使用", exact=True).first.click(timeout=8000, no_wait_after=True)
        except Exception:
            pass
        # 等进入聊天页（出现 chat-message 输入框）
        page.wait_for_selector('textarea[name="chat-message"]', state="visible", timeout=30000)
        steps.append("entered_chat")

        state = _read_state(worker)
        state["registered"] = {"phone": phone, **prof, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        _write_state(state, worker)
        return {"ok": True, "phone": phone, "nickname": prof["nickname"],
                "gender": prof["gender"], "birthday": prof["birthday"],
                "url": page.url, "steps": steps}
    except Exception as e:
        return {"ok": False, "error": f"注册失败：{type(e).__name__}: {e}"}
    finally:
        try:
            if pw:
                pw.stop()
        except Exception:
            pass


def launch_and_register(worker=None):
    """一步到位：全新无登录启动 → 自动注册 → 进入聊天页。"""
    launched = launch_browser(fresh=True, worker=worker)
    if not launched.get("ok"):
        return launched
    reg = register_account(worker=worker)
    return {"ok": reg.get("ok"), "launch": launched, "worker": worker, **reg}


def _connect(worker=None):
    """通过 CDP 连到已登录的浏览器，返回 (playwright, context, page)。调用方需 stop。"""
    from playwright.sync_api import sync_playwright
    cfg = _cfg(worker)
    pw = sync_playwright().start()
    browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{cfg['cdp_port']}")
    ctx = browser.contexts[0] if browser.contexts else browser.new_context()
    # 选 bot 页面：优先 URL 含 bot 域名的
    page = None
    for p in ctx.pages:
        if "alokatravel" in (p.url or "") or "bot" in (p.url or ""):
            page = p
            break
    if page is None and ctx.pages:
        page = ctx.pages[0]
    if page is None:
        page = ctx.new_page()
        page.goto(cfg["bot_url"])
    return pw, browser, ctx, page


def probe_dom():
    """探测页面候选选择器：输入框、发送按钮、消息容器。返回结构化结果。"""
    if not browser_status()["running"]:
        return {"ok": False, "error": "浏览器未启动，请先启动并在窗口中登录 bot"}
    pw = browser = None
    try:
        pw, browser, ctx, page = _connect()
        info = {"ok": True, "url": page.url, "title": page.title()}

        # 输入框候选：textarea、contenteditable、input[type=text]
        inputs = page.eval_on_selector_all(
            "textarea, [contenteditable='true'], input[type='text'], input:not([type])",
            """els => els.map((e,i) => ({
                i, tag: e.tagName,
                ph: e.getAttribute('placeholder')||'',
                ce: e.getAttribute('contenteditable')||'',
                cls: (e.className||'').toString().slice(0,120),
                id: e.id||'',
                visible: !!(e.offsetWidth||e.offsetHeight||e.getClientRects().length)
            }))"""
        )
        info["inputs"] = inputs

        # 按钮候选
        buttons = page.eval_on_selector_all(
            "button, [role='button']",
            """els => els.map((e,i) => ({
                i, tag: e.tagName,
                txt: (e.innerText||'').trim().slice(0,40),
                title: e.getAttribute('title')||'',
                aria: e.getAttribute('aria-label')||'',
                cls: (e.className||'').toString().slice(0,120),
                visible: !!(e.offsetWidth||e.offsetHeight||e.getClientRects().length)
            })).filter(x=>x.visible).slice(0,40)"""
        )
        info["buttons"] = buttons

        state = _read_state()
        state["last_probe"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["probe_result"] = info
        _write_state(state)
        return info
    except Exception as e:
        return {"ok": False, "error": f"探测失败：{type(e).__name__}: {e}"}
    finally:
        # CDP 模式下只断开连接、不关闭浏览器（browser.close() 会关掉用户窗口）
        try:
            if pw:
                pw.stop()
        except Exception:
            pass


def _auto_allow_permission(page):
    """检测网页内的「授权访问」确认弹窗并自动点『始终允许』。

    bot 调用 web_fetch 等工具时会弹出确认卡片（如 "Review before web_fetch - xxx"，
    含 允许一次 / 始终允许 xxx / 拒绝 三个按钮）。不点它 bot 会一直等人工确认，
    导致抓答案超时。只匹配"始终允许"，绝不点"拒绝"。
    返回本次是否真的点到了按钮（bool）。
    """
    targets = ("始终允许", "Always allow", "always allow")
    for sel in ("button:visible", "[role='button']:visible", "a:visible"):
        try:
            btns = page.locator(sel)
            n = min(btns.count(), 80)
        except Exception:
            continue
        for i in range(n):
            try:
                txt = (btns.nth(i).inner_text() or "").strip()
            except Exception:
                continue
            if any(t in txt for t in targets):
                try:
                    btns.nth(i).click(timeout=2000, no_wait_after=True)
                    return True
                except Exception:
                    continue
    return False


def _auto_pick_choice(page, clicked=None):
    """检测 Bot 给出的 A/B/C 选择题按钮，自动选最新一组的第一项 A（替客户给明确答案）。

    bot 的追问选项渲染成按钮，文字形如 "A\\n换成..." / "A 2027年..."。
    点它之后 bot 会继续生成最终方案。只匹配形如 A/B/C 开头的短选项按钮。
    clicked: 可选的 set，记录已点过的选项指纹（该问题的 A 文本），避免重复点同一组。
    返回 (是否点到 bool, 本次点中指纹 str)。
    """
    import re
    try:
        btns = page.locator("button:visible, [role='button']:visible")
        n = min(btns.count(), 80)
        candidates = []
        for i in range(n):
            try:
                txt = (btns.nth(i).inner_text() or "").strip()
            except Exception:
                continue
            # A/B/C/D 开头（后接换行/空格/顿号/点），且长度适中（选项文字）
            if re.match(r"^[A-Da-d][\s\n\.、：:]+\S", txt) and len(txt) <= 120:
                candidates.append((i, txt))
        if not candidates:
            return False, ""
        # 关键：多轮追问时旧选项会残留在 DOM 里（已点过、点不动）。
        # 只认 DOM 顺序中「最后一组」选项——即 bot 最新抛出的问题，
        # 否则会一直点旧组的 A，卡在选项页无法推进。
        # 从末尾往前数，遇到 A 即到达本组第一个选项，收齐这一组 A~D。
        group = []
        for idx, txt in reversed(candidates):
            letter = txt.lstrip()[0] if txt.lstrip() else ""
            group.append((idx, txt))
            if letter in ("A", "a"):
                break
        group.reverse()
        a_idx, a_txt = group[0]
        # 指纹 = 本组全部选项文本拼接；同一组问题指纹相同，用它在调用方去重
        fingerprint = "|".join(t for _, t in group)
        if clicked is not None and fingerprint in clicked:
            return False, ""  # 这一组已经点过 A，等 bot 出新问题
        try:
            btns.nth(a_idx).scroll_into_view_if_needed(timeout=2000)
        except Exception:
            pass
        try:
            btns.nth(a_idx).click(timeout=3000, no_wait_after=True)
            if clicked is not None:
                clicked.add(fingerprint)
            return True, fingerprint
        except Exception:
            return False, ""
    except Exception:
        pass
    return False, ""


def _is_bot_processing(page):
    """检测 bot 是否仍在生成/处理回答（页面出现『处理中…』等状态）。

    实测：bot 处理期间会话列表/输入区会显示『处理中…』，回答文本可能短暂停顿。
    仅凭"文本连续几秒不变"会把半成品/空回答误判为已完成 → 必须叠加本检测：
    只要还在『处理中』，就绝不算回答完成。返回 bool。
    """
    try:
        js = r"""() => {
          // 1) 精确匹配页面自带的处理状态视觉标记（实测：侧边栏 text-status-working、
          //    会话头 text-shimmer/正在处理）。用 class 判定，避免回答正文偶含"处理中"误伤。
          const marks = document.querySelectorAll(
            '.text-status-working, .text-shimmer, [class*="status-working"], [class*="shimmer"]'
          );
          for (const e of marks) {
            const r = e.getBoundingClientRect();
            if (r.width && r.height) return true;         // 标记可见 = 正在处理
          }
          // 2) 兜底：薄叶子节点的文本恰好是短状态词（不含长段落）
          const kws = ['处理中','正在处理','生成中','正在生成','思考中','分析中','请稍候','正在输入','正在回复'];
          const els = document.querySelectorAll('button, span, div, p, [role="status"]');
          for (const e of els) {
            if (e.children.length > 1) continue;
            const r = e.getBoundingClientRect();
            if (!r.width || !r.height) continue;
            const t = (e.innerText || '').trim().replace(/[…\.。]+$/, '');
            if (!t || t.length > 12) continue;            // 只认很短的纯状态文本
            if (kws.some(k => t === k)) return true;
          }
          return false;
        }"""
        return bool(page.evaluate(js))
    except Exception:
        return False


def send_and_capture(question, cfg_override=None, worker=None):
    """发一条题目到 bot 聊天框，等待回答完成并抓回文本。

    返回 {ok, answer, elapsed, error, detail}
    """
    if not browser_status(worker)["running"]:
        return {"ok": False, "error": "浏览器未启动，请先启动并登录 bot"}
    cfg = _cfg(worker)
    if cfg_override:
        cfg.update({k: v for k, v in cfg_override.items() if v})
    in_sel = cfg["input_sel"]
    send_sel = cfg["send_sel"]
    ans_sel = cfg["answer_sel"]
    if not in_sel:
        return {"ok": False, "error": "未配置输入框选择器，请先探测并设置 bot_input_selector"}

    pw = browser = None
    try:
        pw, browser, ctx, page = _connect(worker)
        t0 = time.time()
        perms_clicked = 0

        # 0a. 发送前先清掉可能残留的授权弹窗，避免挡住输入框/发送按钮
        try:
            if _auto_allow_permission(page):
                perms_clicked += 1
        except Exception:
            pass

        # 0. 记录发送前的基线：回答节点数 + 最后一条文本。
        #    关键：必须先记基线，否则发送后旧回答仍在，会被误判为"已完成"抓回旧答案。
        base_count = 0
        base_last = ""
        if ans_sel:
            try:
                _nodes = page.locator(ans_sel)
                base_count = _nodes.count()
                base_last = _nodes.last.inner_text() if base_count else ""
            except Exception:
                pass

        # 1. 填题
        el = page.locator(in_sel).last
        el.wait_for(state="visible", timeout=10000)
        el.click()
        # contenteditable 用 keyboard 输入，textarea/input 用 fill
        tag = el.evaluate("e => e.tagName")
        ce = el.evaluate("e => e.getAttribute('contenteditable')")
        if ce == "true" or tag not in ("TEXTAREA", "INPUT"):
            page.keyboard.insert_text(question)
        else:
            el.fill(question)

        # 2. 发送
        if send_sel:
            page.locator(send_sel).last.click()
        else:
            page.keyboard.press("Enter")

        # 3. 等待回答完成：
        #    阶段A - 等"新回答出现"（节点数增加 或 最后一条文本变化）；
        #    阶段B - 新回答文本连续 N 次不变，视为生成完毕。
        wait_sec = cfg["wait_ms"] / 1000.0
        deadline = time.time() + wait_sec
        hard_deadline = time.time() + max(wait_sec * 4, 300)  # 绝对上限，防止处理中/授权反复顺延导致死循环
        last_text = ""
        stable = 0
        answer = ""
        started = False
        choices_clicked = 0
        last_choice_click = 0.0
        clicked_choice_fps = set()  # 已点过的选项组指纹，避免对同一组重复点 A
        processing_waits = 0
        if ans_sel:
            while time.time() < deadline and time.time() < hard_deadline:
                time.sleep(1.0)
                # bot 请求 web_fetch 时会弹「授权访问」卡片等人工确认，
                # 不点就一直卡住 → 每轮自动点「始终允许」放行。
                try:
                    if _auto_allow_permission(page):
                        perms_clicked += 1
                        deadline += 5  # 放行后 bot 要继续执行，给点缓冲
                except Exception:
                    pass
                # bot 给出 A/B/C 选择题按钮 → 自动选第一个答案（A），让它继续生成最终方案。
                # 注意：选项出现时页面往往仍显示「处理中…」，故选项点击必须排在「处理中」检测之前，
                # 否则会被 continue 跳过而永远不选，卡在选项页。
                # 点选项后 bot 还要约 1 分钟才出新方案，故重置稳定判定并延长等待；
                # 加 5 秒冷却避免同一批按钮被重复点击。
                now = time.time()
                if now - last_choice_click >= 5:
                    try:
                        picked, fp = _auto_pick_choice(page, clicked=clicked_choice_fps)
                        if picked:
                            last_choice_click = now
                            choices_clicked += 1
                            stable = 0
                            last_text = ""
                            deadline = max(deadline, now + wait_sec)
                            continue
                    except Exception:
                        pass
                # 关键：页面仍显示「处理中…」→ bot 还在生成，文本暂停只是假象。
                # 不判完成、清零稳定计数、顺延截止时间，直到处理结束才继续判定。
                try:
                    if _is_bot_processing(page):
                        processing_waits += 1
                        stable = 0
                        deadline = max(deadline, time.time() + 20)  # 至少再给 20 秒
                        continue
                except Exception:
                    pass
                try:
                    nodes = page.locator(ans_sel)
                    cnt = nodes.count()
                    cur = nodes.last.inner_text() if cnt else ""
                except Exception:
                    cnt, cur = 0, ""

                if not started:
                    # 阶段A：只认"新回答节点出现"这个硬指标。
                    if cnt > base_count:
                        started = True
                        last_text = cur
                        stable = 0
                    continue

                # 阶段B：新回答文本稳定判定
                if cur and cur == last_text:
                    stable += 1
                    if stable >= 3:  # 连续 3 秒不变，视为回答完成
                        answer = cur
                        break
                else:
                    stable = 0
                last_text = cur
            answer = answer or last_text
            # 若始终没有新回答，明确报错而不是返回旧答案
            if not started:
                return {"ok": False, "error": "等待超时：bot 未产生新回答（请确认已登录且发送成功）",
                        "elapsed": round(time.time() - t0, 1)}
        else:
            # 无回答选择器：等一个较长时间后抓取页面中最后一段较长文本
            time.sleep(min(cfg["wait_ms"] / 1000.0, 45))
            texts = page.eval_on_selector_all(
                "p, [class*='message'], [class*='markdown'], [role='listitem']",
                "els => els.map(e=>(e.innerText||'').trim()).filter(t=>t.length>20)"
            )
            answer = texts[-1] if texts else "(未抓取到回答，请配置 bot_answer_selector)"

        elapsed = round(time.time() - t0, 1)
        return {"ok": True, "answer": answer, "elapsed": elapsed,
                "url": page.url, "permissions_allowed": perms_clicked,
                "choices_clicked": choices_clicked, "processing_waits": processing_waits,
                "used_selectors": {"input": in_sel, "send": send_sel, "answer": ans_sel}}
    except Exception as e:
        return {"ok": False, "error": f"发题/抓取失败：{type(e).__name__}: {e}"}
    finally:
        # 只断开 CDP 连接，保持浏览器与登录态常驻
        try:
            if pw:
                pw.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 反问检测 + 自动追问补全
# ---------------------------------------------------------------------------
def _detect_followup(question, answer, settings):
    """判断 bot 的这条回答是不是"反问/索要信息"而非实质方案。

    返回 dict:
      is_question(bool), missing(str), supplement(str), reason(str), by(str)
    - is_question=true 表示这是反问，需要自动补全。
    - supplement 是建议发给 bot 的补全文本（用户口吻）。
    用裁判 LLM 判定（复用 judge_* 配置）；无 Key/失败时用规则兜底。
    """
    key = (settings or {}).get("judge_api_key") or ""
    if key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url=settings.get("judge_base_url") or "https://api.openai.com/v1")
            today_str = time.strftime("%Y-%m-%d")
            next_year = int(time.strftime("%Y")) + 1
            sys_prompt = (
                "你在旅游质检系统里负责判断 Bot 的回答是不是「反问/索要信息」，而不是真正的行程方案。\n"
                "判定口径：\n"
                "  - 若 Bot 主要在向用户提问、索要补充信息（如年份、出发地、人数、预算、偏好、具体日期），"
                "或表示需要用户先提供信息才能继续，而没有给出实质性行程内容 → is_question=true。\n"
                "  - 若 Bot 已给出实质行程/建议/方案，即使结尾附带确认性问题 → is_question=false。\n"
                "  - 极短的一句话反问（如『按哪个定』）→ is_question=true。\n"
                "当 is_question=true 时，请生成 supplement：以「真实客户」的口吻、用简短一句话把所有缺失信息补全，"
                "并明确要求 Bot 直接给出完整可执行方案、不要再反问。补全值要合理具体（年份、日期、出发地、人数、预算等），"
                "能从原始需求推断的就沿用，推断不出就用合理默认值。\n"
                f"【日期铁律·最高优先级】今天是 {today_str}。任何出行/出发日期都必须严格晚于今天，"
                f"绝不能补一个已经过去的日期。推断年份时：若原需求里的月日今年还没到就用今年；"
                f"若今年已经过了（例如今天是11月17日、需求却写11月8日），就顺延到明年同一天（{next_year}年同月同日）。\n"
                f"特别地，若 Bot 给出了 A/B/C 之类的选项让你选，必须直接替客户选定一个具体答案并说清楚"
                f"（例如『就选A，{next_year}年那个未来日期出发，玩9天』），且所选日期必须满足上面的日期铁律；"
                "绝不要把选项原样丢回去再问一遍，也绝不要说『你选吧/按你建议』；选定后明确要求 Bot 直接出完整方案。\n"
                "严格只输出 JSON："
                "{\"is_question\":<bool>,\"missing\":<str>,\"supplement\":<str>,\"reason\":<str>}"
            )
            user_prompt = (
                f"原始需求（发给 Bot 的题）：\n{question}\n\n"
                f"Bot 的回答：\n{answer}\n"
            )
            resp = client.chat.completions.create(
                model=settings.get("judge_model") or "gpt-4o",
                temperature=0,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
                timeout=60,
            )
            d = json.loads(resp.choices[0].message.content)
            return {
                "is_question": bool(d.get("is_question")),
                "missing": str(d.get("missing") or ""),
                "supplement": str(d.get("supplement") or ""),
                "reason": str(d.get("reason") or ""),
                "by": "llm",
            }
        except Exception as e:
            rule = _detect_followup_rule(answer)
            rule["reason"] = f"LLM判定失败({type(e).__name__})，已用规则兜底"
            return rule
    return _detect_followup_rule(answer)


def _detect_followup_rule(answer):
    """无裁判 Key 时的启发式兜底：短答案 + 问号/澄清词 → 视为反问。"""
    text = (answer or "").strip()
    q_marks = text.count("?") + text.count("？")
    clarify_kw = ["按哪个", "请提供", "请确认", "请问", "您希望", "请告知", "能否提供",
                  "需要知道", "先告诉", "方便的话", "多久", "从哪", "几个人", "预算多少"]
    has_kw = any(k in text for k in clarify_kw)
    short = len(text) <= 80
    is_q = (q_marks >= 1 and short) or (has_kw and short)
    # 日期必须选未来：绝对年份一律用明年（保证必然在未来），绝不补过去的日期
    today = time.strftime("%Y-%m-%d")
    next_year = int(time.strftime("%Y")) + 1
    supplement = (
        f"今天是{today}。如果上面是在问日期/年份，我选那个出发日期的选项，按 {next_year}年出发"
        f"（年份一定选 {next_year} 年这种未来年份，绝不选已经过去的日期）；"
        f"其余年份按 {next_year}年、出发地按国内主要城市、2位成人、预算按中等水平，"
        "所有选项我都直接替客户选定（不要再把选项丢回来问），"
        "请你直接给出完整可执行的行程方案，不要再反问。"
    )
    return {
        "is_question": bool(is_q),
        "missing": "年份/出发地/人数/预算等要素" if is_q else "",
        "supplement": supplement if is_q else "",
        "reason": f"规则兜底：长度{len(text)},问号{q_marks},澄清词{has_kw}",
        "by": "rule",
    }


def ask_with_followup(question, cfg_override=None, worker=None):
    """发题 → 抓答案；若检测到 bot 反问，自动发补全文本再抓一轮。

    返回 dict（在 send_and_capture 结果基础上扩展）:
      ok, answer, elapsed, rounds, followup_used, trace
      trace: [{round, answer, detection}, ...] 每轮的原始回答与检测结果
    """
    settings = _settings()
    max_rounds = int((cfg_override or {}).get("max_followup", 2) or 2)  # 含首轮，最多补1轮
    trace = []
    ans = ""
    last_res = None

    for rnd in range(1, max_rounds + 1):
        if rnd == 1:
            res = send_and_capture(question, cfg_override=cfg_override, worker=worker)
        else:
            # 用上一轮检测给出的补全文本追问
            supplement = trace[-1]["detection"].get("supplement") or _detect_followup_rule("")["supplement"]
            res = send_and_capture(supplement, cfg_override=cfg_override, worker=worker)
        last_res = res
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error"), "rounds": rnd,
                    "followup_used": rnd > 1, "trace": trace}

        ans = res.get("answer", "")
        detection = _detect_followup(question, ans, settings)
        trace.append({"round": rnd, "answer": ans, "detection": detection,
                      "sent": question if rnd == 1 else supplement})

        if not detection.get("is_question"):
            break  # 已给出实质回答，结束
        if rnd >= max_rounds:
            break  # 达到轮次上限，返回最后一轮答案

    return {
        "ok": True,
        "answer": ans,
        "elapsed": last_res.get("elapsed") if last_res else None,
        "url": last_res.get("url") if last_res else "",
        "rounds": len(trace),
        "followup_used": len(trace) > 1,
        "trace": trace,
        "used_selectors": last_res.get("used_selectors") if last_res else None,
    }


def _pid_on_port(port):
    """找出监听指定 TCP 端口的进程 PID（Windows 用 netstat，跨平台尽力而为）。"""
    pids = []
    try:
        out = subprocess.check_output(["netstat", "-ano", "-p", "TCP"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=10)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[-2].upper() == "LISTENING" and parts[1].endswith(f":{port}"):
                pids.append(int(parts[-1]))
    except Exception:
        pass
    return sorted(set(pids))


def close_browser(worker=None):
    """关闭浏览器（先 CDP 优雅关闭，兜底按端口杀进程）。"""
    if not browser_status(worker)["running"]:
        return {"ok": True, "msg": "浏览器未运行"}
    cfg = _cfg(worker)
    killed = []
    # 1) 优先 CDP 优雅关闭
    try:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{cfg['cdp_port']}")
        browser.close()
        pw.stop()
    except Exception:
        pass
    # 2) CDP 模式 connect_over_cdp 的 close() 常只断连不杀进程 → 等一下，仍活着就按端口杀
    import urllib.request
    time.sleep(1)
    still = False
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{cfg['cdp_port']}/json/version", timeout=2)
        still = True
    except Exception:
        pass
    if still:
        for pid in _pid_on_port(cfg["cdp_port"]):
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                else:
                    os.kill(pid, 9)
                killed.append(pid)
            except Exception:
                pass
    # 3) 清掉会话状态里的运行时间，避免下次误判
    try:
        st = _read_state(worker)
        st.pop("started_at", None)
        _write_state(st, worker)
    except Exception:
        pass
    final = browser_status(worker)
    if final["running"]:
        return {"ok": False, "error": "关闭失败：浏览器端口仍存活"}
    return {"ok": True, "msg": "已关闭", "killed_pids": killed}
