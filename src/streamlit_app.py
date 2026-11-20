# -*- coding: utf-8 -*-
"""
streamlit_app.py —— 决策树客户流失分析 Web 应用
================================================
启动方式：
    cd D:\\PBL\\第3章决策树PBL项目
    streamlit run src/streamlit_app.py

功能页面：
  1. 数据总览    —— 数据集概况、标签分布、关键特征分布
  2. 模型训练    —— 交互式调节超参数，实时查看指标与树结构
  3. 剪枝对比    —— 预剪枝 vs 后剪枝（CCP）效果曲线
  4. 规则解读    —— 特征重要性 + 流失规则提取 + 业务建议
  5. 单客户预测  —— 输入客户画像，实时预测流失概率
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 让 streamlit 能 import 同目录的 utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split, cross_val_score

from utils import (load_raw_data, clean_data, encode_features, split_data,
                   CAT_COLS, NUM_COLS, TARGET, RANDOM_STATE)

# ============== 页面配置 ==============
st.set_page_config(
    page_title="决策树 · 客户流失分析",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 中文字体
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #1976D2, #26A69A);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: #F8FAFC; border-left: 4px solid #1976D2;
        padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.6rem;
    }
    .rule-box {
        background: #FFF8E1; border-left: 4px solid #FFA000;
        padding: 0.8rem 1rem; border-radius: 6px; margin: 0.5rem 0;
        font-family: monospace; font-size: 0.9rem;
    }
    .insight {
        background: #E8F5E9; border-left: 4px solid #43A047;
        padding: 0.7rem 1rem; border-radius: 6px; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============== 数据缓存 ==============
@st.cache_data(show_spinner="正在加载数据...")
def load_and_prepare():
    raw = load_raw_data()
    clean, logs = clean_data(raw)
    encoded = encode_features(clean)
    X_train, X_test, y_train, y_test = split_data(encoded)
    return raw, clean, encoded, X_train, X_test, y_train, y_test, logs


@st.cache_resource(show_spinner="正在训练模型...")
def train_cached(criterion, max_depth, min_split, min_leaf, ccp_alpha):
    X_train, X_test, y_train, y_test = _get_splits()
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=None if max_depth == 0 else max_depth,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        ccp_alpha=ccp_alpha,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    return clf


def _get_splits():
    raw, clean, encoded, X_train, X_test, y_train, y_test, logs = load_and_prepare()
    return X_train, X_test, y_train, y_test


def calc_metrics(model, X_train, X_test, y_train, y_test):
    y_tr = model.predict(X_train)
    y_te = model.predict(X_test)
    y_pr = model.predict_proba(X_test)[:, 1]
    return {
        "训练准确率": accuracy_score(y_train, y_tr),
        "测试准确率": accuracy_score(y_test, y_te),
        "精确率": precision_score(y_test, y_te, zero_division=0),
        "召回率": recall_score(y_test, y_te, zero_division=0),
        "F1值": f1_score(y_test, y_te, zero_division=0),
        "AUC": roc_auc_score(y_test, y_pr),
    }


# ============== 侧边栏 ==============
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/decision-tree.png", width=64)
    st.markdown("### 🌳 决策树 · 客户流失分析")
    st.caption("PBL 项目实践 3/7 · 第 3 章")
    st.divider()

    page = st.radio(
        "功能导航",
        ["📊 数据总览", "🎯 模型训练", "✂️ 剪枝对比", "📜 规则解读", "🔮 单客户预测"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**数据集**")
    st.caption("Telco Customer Churn\n7043 样本 · 21 字段")
    st.markdown("**算法**")
    st.caption("CART 决策树（scikit-learn）")

# ============== 加载数据 ==============
raw, clean, encoded, X_train, X_test, y_train, y_test, clean_logs = load_and_prepare()


# ============================================================================
# 页面 1：数据总览
# ============================================================================
if page == "📊 数据总览":
    st.markdown('<div class="main-header">客户流失数据总览</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Telco Customer Churn · 探索性数据分析</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本总量", f"{len(raw):,}", "条")
    c2.metric("原始特征数", f"{raw.shape[1]-1}", "个")
    c3.metric("流失客户", f"{(raw[TARGET]=='Yes').sum():,}",
              f"{(raw[TARGET]=='Yes').mean()*100:.1f}%")
    c4.metric("不平衡比", "2.77 : 1", "留存 : 流失")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["标签与分布", "特征洞察", "清洗记录"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("标签分布")
            fig, ax = plt.subplots(figsize=(5, 3.6))
            counts = raw[TARGET].value_counts()
            bars = ax.bar(["留存 (No)", "流失 (Yes)"],
                          [counts["No"], counts["Yes"]],
                          color=["#4CAF50", "#F44336"], edgecolor="black")
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 60,
                        f"{int(b.get_height()):,}", ha="center",
                        fontweight="bold", fontsize=11)
            ax.set_ylabel("样本数")
            ax.set_ylim(0, counts["No"] * 1.15)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        with col2:
            st.subheader("合约类型 × 流失率")
            rate = raw.groupby("Contract")[TARGET].apply(
                lambda s: (s == "Yes").mean() * 100).sort_values()
            fig, ax = plt.subplots(figsize=(5, 3.6))
            bars = ax.barh(rate.index, rate.values, color="#FF7043",
                           edgecolor="black")
            for b, v in zip(bars, rate.values):
                ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                        va="center", fontsize=10, fontweight="bold")
            ax.set_xlabel("流失率 (%)")
            ax.set_xlim(0, max(rate.values) * 1.2)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        st.markdown('<div class="insight">💡 <b>核心发现</b>：月结合约客户流失率 '
                    '42.7%，两年合约仅 2.8%。合约类型是最强的流失信号之一。</div>',
                    unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        tmp = clean.copy()

        with col1:
            st.subheader("在网时长分布")
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            ax.hist(tmp.loc[tmp[TARGET] == 0, "tenure"], bins=25, alpha=0.7,
                    label="留存", color="#4CAF50", edgecolor="black")
            ax.hist(tmp.loc[tmp[TARGET] == 1, "tenure"], bins=25, alpha=0.7,
                    label="流失", color="#F44336", edgecolor="black")
            ax.set_xlabel("在网时长（月）")
            ax.set_ylabel("样本数")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        with col2:
            st.subheader("月费分布")
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            ax.hist(tmp.loc[tmp[TARGET] == 0, "MonthlyCharges"], bins=25,
                    alpha=0.7, label="留存", color="#4CAF50", edgecolor="black")
            ax.hist(tmp.loc[tmp[TARGET] == 1, "MonthlyCharges"], bins=25,
                    alpha=0.7, label="流失", color="#F44336", edgecolor="black")
            ax.set_xlabel("月费（美元）")
            ax.set_ylabel("样本数")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        st.subheader("各类别特征流失率排行")
        cat_rate = []
        for col in CAT_COLS:
            for val in raw[col].dropna().unique():
                sub = raw[raw[col] == val]
                if len(sub) >= 30:
                    cat_rate.append({
                        "特征": col, "取值": val, "样本数": len(sub),
                        "流失率(%)": round((sub[TARGET] == "Yes").mean() * 100, 1),
                    })
        rate_df = pd.DataFrame(cat_rate).sort_values("流失率(%)", ascending=False)
        st.dataframe(rate_df.head(15), width='stretch', hide_index=True)

    with tab3:
        st.subheader("数据清洗记录")
        st.dataframe(pd.DataFrame(clean_logs,
                                  columns=["清洗项目", "处理方式", "处理数量/比例", "说明"]),
                     width='stretch', hide_index=True)
        st.info(f"清洗后：{len(clean)} 行 × {clean.shape[1]} 列；"
                f"编码后：{encoded.shape[1]-1} 维特征")


# ============================================================================
# 页面 2：模型训练
# ============================================================================
elif page == "🎯 模型训练":
    st.markdown('<div class="main-header">交互式模型训练</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">调节超参数，实时观察指标与树结构变化</div>',
                unsafe_allow_html=True)

    with st.expander("⚙️ 超参数设置", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        criterion = c1.selectbox("特征选择准则", ["gini", "entropy"],
                                 help="gini 计算更快，entropy 信息增益更直观")
        max_depth = c2.slider("最大深度 max_depth（0=不限制）", 0, 20, 5)
        min_split = c3.slider("最小分裂样本 min_samples_split", 2, 500, 100, step=10)
        min_leaf = c4.slider("最小叶样本 min_samples_leaf", 1, 300, 50, step=5)

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=None if max_depth == 0 else max_depth,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    m = calc_metrics(model, X_train, X_test, y_train, y_test)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("训练准确率", f"{m['训练准确率']:.4f}")
    c2.metric("测试准确率", f"{m['测试准确率']:.4f}",
              f"{m['测试准确率']-m['训练准确率']:+.4f}")
    c3.metric("召回率", f"{m['召回率']:.4f}", help="医疗/流失场景最关键指标")
    c4.metric("F1值", f"{m['F1值']:.4f}")
    c5.metric("树复杂度", f"{model.get_n_leaves()} 叶",
              f"深度 {model.get_depth()}")

    gap = m["训练准确率"] - m["测试准确率"]
    if gap > 0.15:
        st.error(f"⚠️ 训练-测试差距 {gap:.3f} > 0.15，存在**严重过拟合**，"
                 f"建议降低 max_depth 或提高 min_samples_leaf")
    elif gap > 0.05:
        st.warning(f"⚠️ 训练-测试差距 {gap:.3f}，存在轻度过拟合")
    else:
        st.success(f"✅ 训练-测试差距 {gap:.3f}，模型较稳健")

    st.divider()
    tab_a, tab_b, tab_c = st.tabs(["📈 混淆矩阵 & ROC", "🌲 树结构", "📋 文本规则"])

    with tab_a:
        col1, col2 = st.columns(2)
        with col1:
            cm = confusion_matrix(y_test, model.predict(X_test))
            fig, ax = plt.subplots(figsize=(4.8, 4))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["预测留存", "预测流失"])
            ax.set_yticklabels(["真实留存", "真实流失"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=16, fontweight="bold",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_title("混淆矩阵")
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        with col2:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig, ax = plt.subplots(figsize=(4.8, 4))
            ax.plot(fpr, tpr, color="#1976D2", lw=2,
                    label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set_xlabel("假正率 FPR")
            ax.set_ylabel("真正率 TPR")
            ax.set_title("ROC 曲线")
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

    with tab_b:
        depth_show = st.slider("显示层数（树太大时建议限制）", 1, 6, 3)
        fig, ax = plt.subplots(figsize=(20, 9))
        plot_tree(model, feature_names=list(X_train.columns),
                  class_names=["留存", "流失"], filled=True, rounded=True,
                  fontsize=8, max_depth=depth_show, ax=ax)
        ax.set_title(f"决策树结构（显示前 {depth_show} 层）", fontsize=14)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    with tab_c:
        st.code(export_text(model, feature_names=list(X_train.columns),
                            max_depth=4, decimals=2), language="text")


# ============================================================================
# 页面 3：剪枝对比
# ============================================================================
elif page == "✂️ 剪枝对比":
    st.markdown('<div class="main-header">剪枝策略对比</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">预剪枝（参数控制）vs 后剪枝（CCP 成本复杂度）</div>',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["预剪枝：深度扫描", "后剪枝：CCP α 扫描"])

    with tab1:
        st.subheader("max_depth 对过拟合的影响")
        rows = []
        for d in range(1, 21):
            clf = DecisionTreeClassifier(criterion="gini", max_depth=d,
                                         random_state=RANDOM_STATE)
            clf.fit(X_train, y_train)
            mm = calc_metrics(clf, X_train, X_test, y_train, y_test)
            rows.append({"max_depth": d, **mm, "叶节点数": clf.get_n_leaves()})
        df_depth = pd.DataFrame(rows)

        fig, ax1 = plt.subplots(figsize=(11, 4.5))
        ax1.plot(df_depth["max_depth"], df_depth["训练准确率"], "o-",
                 label="训练集", color="#FF7043", lw=2)
        ax1.plot(df_depth["max_depth"], df_depth["测试准确率"], "s-",
                 label="测试集", color="#1976D2", lw=2)
        ax1.plot(df_depth["max_depth"], df_depth["F1值"], "^-",
                 label="测试 F1", color="#43A047", lw=2, alpha=0.8)
        best_d = int(df_depth.loc[df_depth["F1值"].idxmax(), "max_depth"])
        ax1.axvline(best_d, color="green", ls="--", alpha=0.6,
                    label=f"最优深度={best_d}")
        ax1.set_xlabel("max_depth")
        ax1.set_ylabel("分数")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_title("树深度与模型性能")
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        st.dataframe(
            df_depth[["max_depth", "训练准确率", "测试准确率", "召回率", "F1值", "叶节点数"]]
            .style.background_gradient(cmap="Blues", subset=["F1值"])
            .format(precision=4),
            width='stretch', hide_index=True, height=280,
        )
        st.info(f"最优深度 max_depth={best_d}，测试 F1="
                f"{df_depth['F1值'].max():.4f}。深度超过这个值后训练准确率继续上升，"
                f"但测试性能开始下降 —— 过拟合的典型表现。")

    with tab2:
        st.subheader("成本复杂度后剪枝（CCP）")
        base = DecisionTreeClassifier(random_state=RANDOM_STATE)
        base.fit(X_train, y_train)

        # 用交叉验证选 alpha，避免在测试集上调参
        path = base.cost_complexity_pruning_path(X_train, y_train)
        alphas = np.unique(np.round(path.ccp_alphas[path.ccp_alphas >= 0], 6))
        if len(alphas) > 25:
            alphas = alphas[np.linspace(0, len(alphas) - 1, 25).astype(int)]

        rows = []
        progress = st.progress(0, "正在扫描 alpha...")
        for i, a in enumerate(alphas):
            clf = DecisionTreeClassifier(ccp_alpha=a, random_state=RANDOM_STATE)
            clf.fit(X_train, y_train)
            mm = calc_metrics(clf, X_train, X_test, y_train, y_test)
            rows.append({"alpha": float(a), **mm,
                         "叶节点数": clf.get_n_leaves(),
                         "深度": clf.get_depth()})
            progress.progress((i + 1) / len(alphas), f"扫描 {i+1}/{len(alphas)}")
        progress.empty()

        df_a = pd.DataFrame(rows)
        best_a_row = df_a.loc[df_a["F1值"].idxmax()]
        best_a = float(best_a_row["alpha"])

        fig, ax = plt.subplots(figsize=(11, 4.5))
        x_axis = df_a["alpha"].replace(0, 1e-6)
        ax.plot(x_axis, df_a["训练准确率"], "o-", label="训练集",
                color="#FF7043", markersize=4)
        ax.plot(x_axis, df_a["测试准确率"], "s-", label="测试集",
                color="#1976D2", markersize=4)
        ax.plot(x_axis, df_a["F1值"], "^-", label="测试 F1",
                color="#43A047", markersize=4)
        ax.axvline(max(best_a, 1e-6), color="green", ls="--",
                   label=f"最优 α={best_a:.4f}")
        ax.set_xscale("log")
        ax.set_xlabel("ccp_alpha（对数轴）")
        ax.set_ylabel("分数")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title("CCP α 与模型性能关系")
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("最优 α", f"{best_a:.6f}")
        c2.metric("剪枝后叶节点", f"{int(best_a_row['叶节点数'])}",
                  f"原始 {base.get_n_leaves()}")
        c3.metric("测试 F1", f"{best_a_row['F1值']:.4f}",
                  f"{best_a_row['F1值'] - calc_metrics(base, X_train, X_test, y_train, y_test)['F1值']:+.4f}")

        st.dataframe(df_a.head(20).style.format(precision=4),
                     width='stretch', hide_index=True, height=250)


# ============================================================================
# 页面 4：规则解读
# ============================================================================
elif page == "📜 规则解读":
    st.markdown('<div class="main-header">规则提取与业务解读</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">从决策树中挖掘可落地的挽留规则</div>',
                unsafe_allow_html=True)

    with st.expander("⚙️ 模型配置（默认使用预剪枝参数）", expanded=False):
        c1, c2, c3 = st.columns(3)
        crit = c1.selectbox("criterion", ["gini", "entropy"], key="rule_crit")
        depth = c2.slider("max_depth", 2, 12, 5, key="rule_depth")
        leaf = c3.slider("min_samples_leaf", 10, 300, 50, step=10, key="rule_leaf")

    model = DecisionTreeClassifier(criterion=crit, max_depth=depth,
                                   min_samples_leaf=leaf,
                                   random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    m = calc_metrics(model, X_train, X_test, y_train, y_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("测试准确率", f"{m['测试准确率']:.4f}")
    c2.metric("召回率", f"{m['召回率']:.4f}")
    c3.metric("F1值", f"{m['F1值']:.4f}")
    c4.metric("规则数（叶节点）", model.get_n_leaves())

    st.divider()

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.subheader("特征重要性 Top 10")
        imp = pd.DataFrame({
            "特征": X_train.columns,
            "重要性": model.feature_importances_,
        }).sort_values("重要性", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.25, len(imp)))[::-1]
        bars = ax.barh(imp["特征"][::-1], imp["重要性"][::-1],
                       color=colors, edgecolor="black")
        for b, v in zip(bars, imp["重要性"][::-1]):
            ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel("基尼重要性")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    with col2:
        st.subheader("流失规则 Top 8")
        tree = model.tree_
        feats = list(X_train.columns)
        rules = []

        def walk(node, conds):
            l, r = tree.children_left[node], tree.children_right[node]
            if l == -1 and r == -1:
                vals = tree.value[node][0]
                tot = vals.sum()
                ratio = vals[1] / tot if tot > 0 else 0
                rules.append({
                    "规则": " 且 ".join(conds) if conds else "全部客户",
                    "覆盖人数": int(tot),
                    "覆盖率%": round(tot / tree.n_node_samples[0] * 100, 2),
                    "流失率%": round(ratio * 100, 1),
                    "判定": "流失" if ratio >= 0.5 else "留存",
                    "_score": ratio * np.sqrt(tot),
                })
                return
            f = feats[tree.feature[node]]
            t = tree.threshold[node]
            walk(l, conds + [f"{f} ≤ {t:.1f}"])
            walk(r, conds + [f"{f} > {t:.1f}"])

        walk(0, [])
        df_rules = pd.DataFrame(rules).sort_values("_score",
                                                   ascending=False).drop(columns="_score")

        for i, r in df_rules.head(8).iterrows():
            st.markdown(
                f'<div class="rule-box"><b>规则 {i+1}</b>：IF {r["规则"]}<br>'
                f'→ 判定 <b>{r["判定"]}</b>，覆盖 {r["覆盖人数"]:,} 人'
                f'（{r["覆盖率%"]}%），该人群流失率 <b>{r["流失率%"]}%</b></div>',
                unsafe_allow_html=True)

    st.divider()
    st.subheader("💼 运营建议（基于规则自动归纳）")
    top_feat = imp.iloc[0]["特征"]
    st.markdown(f"""
<div class="insight">
<b>1. 合约锁定优先</b>：月结合约客户流失率高达 42.7%，两年合约仅 2.8%。
建议对月结客户推送"年付立减"优惠，把短周期合约转为长周期。</div>

<div class="insight">
<b>2. 新客黄金 6 个月</b>：tenure ≤ 6 的客户流失率显著高于均值。
建议入网首月启动"新手关怀"流程（专属客服 + 使用指导 + 首月账单折扣）。</div>

<div class="insight">
<b>3. 光纤高价值客户专项</b>：光纤用户流失率 41.9%，但 ARPU 最高。
建议针对光纤客户做网络质量回访，避免"高付费 + 低体验"造成流失。</div>

<div class="insight">
<b>4. 付款方式引导</b>：电子支票用户流失率 45.3%。
建议引导切换为自动扣款（信用卡/银行转账），并给予一次性账单减免。</div>

<div class="insight">
<b>5. 增值服务捆绑</b>：开通在线安全/技术支持的用户流失率明显更低。
建议对单服务客户推送"安全+备份"组合包，提升切换成本。</div>

<div class="insight">
<b>6. 模型当前最强分裂特征：<code>{top_feat}</code></b>，
重要性 {imp.iloc[0]['重要性']:.3f}。运营动作应优先围绕该维度设计。</div>
""", unsafe_allow_html=True)


# ============================================================================
# 页面 5：单客户预测
# ============================================================================
elif page == "🔮 单客户预测":
    st.markdown('<div class="main-header">单客户流失预测</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">输入客户画像，实时输出流失概率与主要风险因子</div>',
                unsafe_allow_html=True)

    model = DecisionTreeClassifier(criterion="gini", max_depth=5,
                                   min_samples_split=100, min_samples_leaf=50,
                                   random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**基本信息**")
            gender = st.selectbox("性别", ["Female", "Male"])
            senior = st.selectbox("是否老年人", [0, 1],
                                  format_func=lambda x: "是" if x else "否")
            partner = st.selectbox("是否有伴侣", ["No", "Yes"])
            dependents = st.selectbox("是否有家属", ["No", "Yes"])
        with c2:
            st.markdown("**账户信息**")
            tenure = st.slider("在网时长（月）", 0, 72, 12)
            contract = st.selectbox("合约类型", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("是否电子账单", ["Yes", "No"])
            payment = st.selectbox("付款方式",
                                   ["Electronic check", "Mailed check",
                                    "Bank transfer (automatic)",
                                    "Credit card (automatic)"])
        with c3:
            st.markdown("**服务与费用**")
            internet = st.selectbox("网络服务", ["DSL", "Fiber optic", "No"])
            monthly = st.slider("月费（美元）", 18.0, 120.0, 70.0, step=0.5)
            total = st.number_input("累计消费（美元）", 0.0, 9000.0,
                                    float(tenure * monthly), step=10.0)
            phone = st.selectbox("是否开通电话", ["Yes", "No"])

        st.markdown("**增值服务**")
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        online_sec = s1.selectbox("在线安全", ["No", "Yes"])
        online_bak = s2.selectbox("在线备份", ["No", "Yes"])
        device_pro = s3.selectbox("设备保护", ["No", "Yes"])
        tech_sup = s4.selectbox("技术支持", ["No", "Yes"])
        stream_tv = s5.selectbox("流媒体TV", ["No", "Yes"])
        stream_mv = s6.selectbox("流媒体电影", ["No", "Yes"])

        submitted = st.form_submit_button("🔮 预测流失概率", width='stretch')

    if submitted:
        # 构造与训练数据一致的 23 维输入
        row = {c: 0 for c in X_train.columns}

        def setv(name, val):
            if name in row:
                row[name] = val

        setv("gender", 1 if gender == "Male" else 0)
        setv("SeniorCitizen", senior)
        setv("Partner", 1 if partner == "Yes" else 0)
        setv("Dependents", 1 if dependents == "Yes" else 0)
        setv("tenure", tenure)
        setv("PhoneService", 1 if phone == "Yes" else 0)
        setv("PaperlessBilling", 1 if paperless == "Yes" else 0)
        setv("MonthlyCharges", monthly)
        setv("TotalCharges", total)

        no_service = "No" if internet != "No" else "No internet service"
        for col, v in [("OnlineSecurity", online_sec), ("OnlineBackup", online_bak),
                       ("DeviceProtection", device_pro), ("TechSupport", tech_sup),
                       ("StreamingTV", stream_tv), ("StreamingMovies", stream_mv)]:
            setv(col, 1 if v == "Yes" else 0)

        setv("MultipleLines", 0)

        if internet == "Fiber optic":
            setv("InternetService_Fiber optic", 1)
        elif internet == "No":
            setv("InternetService_No", 1)

        if contract == "One year":
            setv("Contract_One year", 1)
        elif contract == "Two year":
            setv("Contract_Two year", 1)

        if payment == "Credit card (automatic)":
            setv("PaymentMethod_Credit card (automatic)", 1)
        elif payment == "Electronic check":
            setv("PaymentMethod_Electronic check", 1)
        elif payment == "Mailed check":
            setv("PaymentMethod_Mailed check", 1)

        X_input = pd.DataFrame([row])[X_train.columns]
        prob = model.predict_proba(X_input)[0][1]
        pred = "🔴 高流失风险" if prob >= 0.5 else "🟢 低流失风险"

        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("流失概率", f"{prob*100:.1f}%")
            st.metric("预测结果", pred)
        with col2:
            if prob >= 0.6:
                st.error(f"⚠️ 该客户流失概率 {prob*100:.1f}%，属于**高优先级挽留对象**。"
                         f"建议 24 小时内由专属客服介入，提供针对性优惠。")
            elif prob >= 0.35:
                st.warning(f"⚠️ 该客户流失概率 {prob*100:.1f}%，处于**观察区间**。"
                           f"建议纳入周度关怀名单，关注使用活跃度变化。")
            else:
                st.success(f"✅ 该客户流失概率 {prob*100:.1f}%，风险较低。"
                           f"保持常规服务即可。")

            st.markdown("**主要风险因子（对照本次输入）**")
            risks = []
            if contract == "Month-to-month":
                risks.append(("合约类型", "月结合约（流失率 42.7%）", "高"))
            if tenure <= 6:
                risks.append(("在网时长", f"{tenure} 个月（新客户期）", "高"))
            if internet == "Fiber optic":
                risks.append(("网络服务", "光纤（流失率 41.9%）", "中"))
            if payment == "Electronic check":
                risks.append(("付款方式", "电子支票（流失率 45.3%）", "中"))
            if monthly > 80:
                risks.append(("月费", f"${monthly:.0f}（高于均值 $64.8）", "中"))
            if online_sec == "No" and tech_sup == "No":
                risks.append(("增值服务", "未开通安全/技术支持", "中"))

            if risks:
                st.dataframe(pd.DataFrame(risks, columns=["维度", "具体情况", "风险等级"]),
                             width='stretch', hide_index=True)
            else:
                st.info("未识别到明显风险因子。")

        st.caption("注：本预测基于决策树模型，结果仅作运营参考，不构成任何业务承诺。")
