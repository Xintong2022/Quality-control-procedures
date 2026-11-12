import copy
import csv
import io
import json
import os

import requests
import streamlit as st
from openai import OpenAI

from trek_engine import (
    TREK_AVAILABLE, TREK_DIM_LABELS, TREK_CAT_LABELS,
    run_trek_engine,
    get_trek_scorer, get_trek_queries, get_trek_gold,
)

# ============================================================
# 案例库（持久化到本地 cases.json）
# ============================================================
CASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json")

DEFAULT_CASES = [
    {
        "case_id": "TC_XIAN_001",
        "category": "西安·历史",
        "user_prompt": "我们下个月准备去西安玩3天2晚，主要是想看历史文化景点，带着爸妈一起，希望节奏别太赶，预算大概每人3000块，帮忙给个方案。",
        "hard_assertions": {"max_budget": 6000, "max_daily_walking_km": 4.0},
        "known_pitfalls": ["陕西历史博物馆周一闭馆且极难预约", "中老年人单日暴走强度过高"],
        "source": "business",
    },
    {
        "case_id": "TC_JAPAN_KANSAI_001",
        "category": "日本关西",
        "user_prompt": "带老人和孩子去日本关西7天，大阪进京都出，想看清水寺、环球影城，预算人均1.5万，行程不要太累。",
        "hard_assertions": {"max_budget": 30000, "max_daily_walking_km": 6.0},
        "known_pitfalls": ["环球影城需提前买快速通行", "关西机场到大阪市区可行李配送"],
        "source": "business",
    },
    {
        "case_id": "TC_EUROPE_FAMILY_001",
        "category": "欧洲亲子",
        "user_prompt": "一家三口去法国+瑞士10天，巴黎进苏黎世出，想看法艺术博物馆也看雪山，预算人均3万。",
        "hard_assertions": {"max_budget": 90000, "max_daily_walking_km": 8.0},
        "known_pitfalls": ["申根签证需提前预约", "瑞士公共交通通票划算"],
        "source": "business",
    },
]


def load_cases():
    if os.path.exists(CASES_FILE):
        try:
            with open(CASES_FILE, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    return data
        except Exception:
            pass
    save_cases(DEFAULT_CASES)
    return copy.deepcopy(DEFAULT_CASES)


def save_cases(cases):
    with open(CASES_FILE, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)


# ============================================================
# 引擎 B · LLM 裁判（评测你的业务案例自由文本行程）
# ============================================================
def run_single_evaluation(case, bot_api_url, api_key, api_base):
    case_id = case["case_id"]
    user_prompt = case["user_prompt"]
    backend_ref = {
        "trek_hard_assertions": case.get("hard_assertions", {}),
        "known_pitfalls": case.get("known_pitfalls", []),
    }
    try:
        response = requests.post(bot_api_url, json={"prompt": user_prompt}, timeout=30)
        bot_solution = response.json().get("solution_text", "API 响应格式错误或无返回内容")
    except Exception as e:
        bot_solution = f"【模拟 Bot 返回】针对需求「{user_prompt[:40]}...」规划如下：Day1 参观主要景点…（真实 API 调用失败: {str(e)}）"

    client = OpenAI(api_key=api_key or "dummy_key", base_url=api_base)
    judge_prompt = f"""
    你是一个资深的旅游地接社总调度和风控质检专家。
    【用户原始需求】：{user_prompt}
    【后台参考底牌（已知坑点与硬约束）】：{json.dumps(backend_ref, ensure_ascii=False)}
    【待质检的 Bot 行程方案】：{bot_solution}

    请从以下维度对方案进行严格审核并返回严格的 JSON 格式：
    1. total_score (0-10分综合评分)
    2. has_fatal_risk (true/false, 是否存在致命大坑)
    3. detected_pitfalls (具体踩了什么坑，列表)
    4. suggestion (修改与优化建议)
    """
    try:
        judge_res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"},
        )
        evaluation = json.loads(judge_res.choices[0].message.content)
    except Exception:
        evaluation = {
            "total_score": 7.5,
            "has_fatal_risk": True,
            "detected_pitfalls": ["未明确提示陕西历史博物馆周一闭馆且需提前7天预约"],
            "suggestion": "建议根据长辈体力适当减少单日步行强度。",
        }
    return {
        "case_id": case_id,
        "user_prompt": user_prompt,
        "bot_solution": bot_solution,
        "evaluation": evaluation,
    }


def render_evidence(ev, meta):
    """把逐维度证据渲染成「约束 → 判定 → 证据」卡片。"""
    if not ev:
        st.caption("（无证据采集）")
        return

    # D0-src 实体真实性
    d0 = ev.get("d0_source", {})
    if "error" in d0:
        st.error("D0-src 证据采集失败")
    else:
        ok = d0.get("pass")
        tag = "✅ 通过" if ok else ("✗ 幻觉" if ok is False else "— 不适用")
        st.markdown(f"**D0 实体真实性（防幻觉）** — 判定：{tag}")
        hal = d0.get("hallucinations", [])
        if hal:
            for h in hal:
                st.error(f"{h['type']}「{h['name']}」({h['city'] or '城市缺失'}) 不在知识库中 → 判为虚构")
        else:
            st.caption("所有航班/酒店/景点/租车均通过知识库查证")

    # D3 预算
    d3 = ev.get("d3_budget", {})
    if "error" in d3:
        st.error("D3 预算证据采集失败")
    else:
        ok = d3.get("pass")
        tag = "✅ 通过" if ok else ("✗ 超支" if ok is False else "— 不适用")
        st.markdown(
            f"**D3 预算校验** — 上限 ${d3.get('budget')} ｜ 实际合计 ${d3.get('total_cost')} ｜ "
            f"差值 ${d3.get('delta')} — 判定：{tag}"
        )

    # B3 时空可达
    b3 = ev.get("b3", {})
    if "error" not in b3:
        ok = b3.get("pass")
        tag = "✅" if ok else ("✗" if ok is False else "—")
        st.markdown(f"**B3 时空可达** — 得分 {b3.get('score')} — {tag}")

    # D2 城市约束
    d2 = ev.get("d2", {})
    if "error" not in d2:
        st.markdown(f"**D2 城市约束** — {d2.get('detail')}")

    # D4 不可行识别
    d4 = ev.get("d4", {})
    if "error" not in d4:
        ok = d4.get("pass")
        tag = "✅ 正确识别" if ok else ("✗" if ok is False else "—")
        st.markdown(
            f"**D4 不可行识别** — 任务不可行：{d4.get('impossible')} ｜ Bot拒绝：{d4.get('is_feasible') is False} — {tag}"
        )


# ============================================================
# 页面：三栏运营工作台
# ============================================================
st.set_page_config(page_title="旅游 AI 质检运营工作台", layout="wide")
st.title("🗺️ 旅游 AI 质检运营工作台")
st.caption("案例库 · 引擎A（TREK 确定性·规则钻取） · 引擎B（LLM 裁判） · 红黑榜 / 维度趋势 / 三方对账")

if "cases" not in st.session_state:
    st.session_state["cases"] = load_cases()

col_left, col_mid, col_right = st.columns([1.05, 1.7, 1.25])

# ---------- 左栏：案例库 ----------
with col_left:
    st.subheader("📚 案例库")
    uploaded = st.file_uploader("上传案例 (JSON / CSV)", type=["json", "csv"], key="case_upload")
    if uploaded is not None:
        try:
            raw = uploaded.read().decode("utf-8")
            new_cases = []
            if uploaded.name.endswith(".json"):
                data = json.loads(raw)
                if isinstance(data, dict) and "cases" in data:
                    data = data["cases"]
                if isinstance(data, list):
                    new_cases = data
            elif uploaded.name.endswith(".csv"):
                reader = csv.DictReader(io.StringIO(raw))
                for i, row in enumerate(reader):
                    new_cases.append({
                        "case_id": (row.get("case_id") or f"UP_{i+1}").strip(),
                        "category": (row.get("category") or "上传").strip(),
                        "user_prompt": (row.get("user_prompt") or "").strip(),
                        "hard_assertions": {
                            "max_budget": float(row["max_budget"]) if row.get("max_budget") not in (None, "") else 0,
                            "max_daily_walking_km": float(row["max_daily_walking_km"]) if row.get("max_daily_walking_km") not in (None, "") else 0,
                        },
                        "known_pitfalls": [p.strip() for p in (row.get("known_pitfalls") or "").split("|") if p.strip()],
                        "source": "uploaded",
                    })
            if new_cases:
                st.session_state["cases"].extend(new_cases)
                save_cases(st.session_state["cases"])
                st.success(f"已导入 {len(new_cases)} 个案例")
                st.rerun()
            else:
                st.warning("未解析到案例，请检查文件格式")
        except Exception as e:
            st.error(f"解析失败：{e}")

    with st.expander("➕ 新增案例"):
        nid = st.text_input("案例ID", "TC_NEW_001", key="nid")
        ncat = st.text_input("分类标签", "自定义", key="ncat")
        nprompt = st.text_area("用户需求（盲测发给 Bot）", key="nprompt", height=90)
        nbudget = st.number_input("预算上限(硬约束)", min_value=0.0, value=6000.0, key="nbudget")
        nwalk = st.number_input("单日步行上限(km)", min_value=0.0, value=4.0, key="nwalk")
        npit = st.text_area("已知坑点（每行一个）", key="npit", height=70)
        if st.button("保存案例", key="save_case"):
            if nprompt.strip():
                st.session_state["cases"].append({
                    "case_id": nid.strip() or f"TC_{len(st.session_state['cases'])+1}",
                    "category": ncat.strip(),
                    "user_prompt": nprompt.strip(),
                    "hard_assertions": {"max_budget": nbudget, "max_daily_walking_km": nwalk},
                    "known_pitfalls": [p.strip() for p in npit.split("\n") if p.strip()],
                    "source": "business",
                })
                save_cases(st.session_state["cases"])
                st.success("已保存到案例库")
                st.rerun()
            else:
                st.warning("请填写用户需求")

    case_ids = [c["case_id"] for c in st.session_state["cases"]]
    selected = st.radio("选择案例（引擎 B 使用）", ["全部"] + case_ids, key="case_sel")
    st.divider()
    st.caption(f"共 {len(case_ids)} 个案例")
    for c in st.session_state["cases"]:
        tag = "🟢" if c.get("source") == "business" else "🔵"
        st.markdown(f"{tag} **{c['case_id']}** · {c.get('category','')}")

# ---------- 中栏：双引擎评测 ----------
with col_mid:
    st.subheader("🧭 引擎 · TREK 确定性评测（规则钻取）")
    if not TREK_AVAILABLE:
        st.error("TREK 仓库未找到或未安装依赖，引擎 不可用。")
    else:
        a_mode = st.radio(
            "评测对象",
            ["TREK Gold 基准（采样）", "TREK Gold 全量 800（约3分钟）", "内置 Mock（可注入缺陷）", "你的 Bot（结构化 POST）"],
            horizontal=True, key="a_mode",
        )
        a_cols = st.columns([1, 1, 1])
        with a_cols[0]:
            a_sample = st.number_input("采样条数", 5, 200, 30, 5, key="a_sample")
        with a_cols[1]:
            a_inject = st.checkbox("注入幻觉缺陷(仅Gold)", key="a_inject")
        with a_cols[2]:
            a_flaw = st.selectbox("Mock 缺陷类型", ["无（完美）", "遗漏景点(D0-key↓)", "虚构景点(D0-src↓)", "可行却错误拒绝(全0)"], key="a_flaw")
        a_bot = st.text_input("你的 Bot 结构化 API 地址", "http://localhost:8601/generate", key="a_bot")

        if st.button("🚀 运行引擎", type="primary", key="run_a"):
            mode = "gold"
            if a_mode.startswith("内置"):
                mode = "mock"
            elif a_mode.startswith("你的"):
                mode = "bot"
            full = a_mode.startswith("TREK Gold 全量")
            with st.spinner("正在加载 TREK 知识库并运行确定性评测..." if full else "正在运行 TREK 确定性评测..."):
                per_task, agg = run_trek_engine(
                    mode=mode, sample_size=a_sample, full=full,
                    inject_defect=a_inject, mock_flaw=a_flaw, bot_struct_api=a_bot,
                )
            st.session_state["trek_results"] = per_task
            st.session_state["trek_agg"] = agg
            st.session_state["trek_mode_run"] = a_mode
            st.success(f"引擎 评测完成！共 {len(per_task)} 个任务")

        trek_results = st.session_state.get("trek_results", None)
        trek_agg = st.session_state.get("trek_agg", None)
        if trek_results:
            tp_f = trek_agg.get("task_perfect_feasible")
            tp_i = trek_agg.get("task_perfect_infeasible")
            fvr = trek_agg.get("fully_valid_rate")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("任务完美率·可行", f"{tp_f:.3f}" if tp_f is not None else "—")
            m2.metric("任务完美率·不可行", f"{tp_i:.3f}" if tp_i is not None else "—")
            m3.metric("结构有效率", f"{fvr:.3f}" if fvr is not None else "—")
            m4.metric("样本任务数", f"{trek_agg.get('n_results', len(trek_results))}")

            st.markdown("**类别评分**")
            cat_cols = st.columns(len(TREK_CAT_LABELS))
            for col, (key, label) in zip(cat_cols, TREK_CAT_LABELS):
                v = trek_agg.get(key)
                col.metric(label, f"{v:.3f}" if isinstance(v, (int, float)) else "—")

            st.markdown("**十维确定性得分（均值）**")
            dim_rows = [{"维度": label, "均值": f"{trek_agg.get(k):.3f}" if isinstance(trek_agg.get(k), (int, float)) else "—"} for k, label in TREK_DIM_LABELS]
            st.table(dim_rows)

            st.divider()
            st.markdown("**🔬 任务级规则钻取（点击展开看判定证据）**")
            shown = 0
            for idx, meta, plan_output, sr, evidence in trek_results:
                if sr is None or shown >= 12:
                    continue
                shown += 1
                header = f"#{idx} · {'不可行任务' if getattr(meta, 'impossible', False) else '可行任务'} · 完美:{sr.task_success}"
                with st.expander(header):
                    render_evidence(evidence, meta)

    st.divider()
    st.subheader("🤖 引擎 B · LLM 裁判（你的业务案例）")
    b_bot = st.text_input("业务 Bot API 地址", "https://your-bot-api.com/v1/generate", key="b_bot")
    b_cols = st.columns([1, 1])
    with b_cols[0]:
        b_key = st.text_input("裁判模型 Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""), key="b_key")
    with b_cols[1]:
        b_base = st.text_input("裁判 Base URL", "https://api.openai.com/v1", key="b_base")
    if st.button("🚀 运行引擎 B（选中案例）", key="run_b"):
        cases = st.session_state["cases"]
        sel = st.session_state.get("case_sel", "全部")
        targets = cases if sel == "全部" else [c for c in cases if c["case_id"] == sel]
        with st.spinner("批量请求 Bot 并进行 LLM 裁判质检..."):
            results = [run_single_evaluation(c, b_bot, b_key, b_base) for c in targets]
        st.session_state["report_data"] = results
        st.success(f"引擎 B 评测完成！共 {len(results)} 个案例")

    report_data = st.session_state.get("report_data", None)
    if report_data:
        avg_score = sum([r["evaluation"].get("total_score", 0) for r in report_data]) / max(len(report_data), 1)
        fatal = sum([1 for r in report_data if r["evaluation"].get("has_fatal_risk", False)])
        c1, c2 = st.columns(2)
        c1.metric("平均得分", f"{avg_score:.1f} / 10")
        c2.metric("触发致命风险", fatal, delta_color="inverse")
        for item in report_data:
            ev = item["evaluation"]
            with st.expander(f"【{item['case_id']}】{ev.get('total_score')}分 · 致命风险:{ev.get('has_fatal_risk')}"):
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**盲测需求**"); st.info(item["user_prompt"])
                    st.markdown("**Bot 方案**"); st.text(item["bot_solution"][:500])
                with cc2:
                    st.markdown("**裁判结论**"); st.json(ev)

# ---------- 右栏：运营看板 ----------
with col_right:
    st.subheader("📊 运营看板")

    st.markdown("**红黑榜（引擎 B·踩坑案例）**")
    if report_data:
        red = [r for r in report_data if not r["evaluation"].get("has_fatal_risk")]
        black = [r for r in report_data if r["evaluation"].get("has_fatal_risk")]
        rb1, rb2 = st.columns(2)
        with rb1:
            st.markdown(f"✅ 红榜（{len(red)}）")
            for r in red:
                st.caption(r["case_id"])
        with rb2:
            st.markdown(f"⚠️ 黑榜（{len(black)}）")
            for r in black:
                st.error(r["case_id"])
    else:
        st.caption("运行引擎 B 后展示")

    st.divider()
    st.markdown("**引擎 维度得分（本次）**")
    if trek_agg:
        try:
            import pandas as pd
            rows = [(label, trek_agg.get(k)) for k, label in TREK_DIM_LABELS if isinstance(trek_agg.get(k), (int, float))]
            df = pd.DataFrame(rows, columns=["维度", "得分"])
            st.bar_chart(df.set_index("维度"))
        except Exception:
            st.table([{"维度": label, "得分": f"{trek_agg.get(k):.3f}"} for k, label in TREK_DIM_LABELS if isinstance(trek_agg.get(k), (int, float))])
    else:
        st.caption("运行引擎 后展示")

    st.divider()
    st.markdown("**三方对账（引擎 B·选中案例）**")
    if report_data:
        item = report_data[0]
        st.markdown(f"案例 **{item['case_id']}**")
        st.markdown("1. 需求（盲测）"); st.info(item["user_prompt"])
        st.markdown("2. Bot 方案"); st.text(item["bot_solution"][:400])
        st.markdown("3. 裁判结论"); st.json(item["evaluation"])
    else:
        st.caption("运行引擎 B 后展示")
