# -*- coding: utf-8 -*-
"""
exp01_exploration.py —— 数据加载与探索性分析
=============================================
对应报告 2.2 数据集说明、3.2.1 数据清洗

内容：
  1. 鸢尾花热身（3.2.4）：验证环境 + 熟悉 API
  2. Telco 数据探索：样本量、特征类型、标签分布、缺失值、类别统计
输出：
  results/01_data_summary.csv
  results/01_label_distribution.csv
  results/01_category_stats.csv
  results/01_missing_summary.csv
  images/01_data_exploration.png
  images/00_iris_warmup.png
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import (load_raw_data, clean_data, save_csv, setup_chinese_font,
                   IMG_DIR, CAT_COLS, NUM_COLS, TARGET)


# ============================================================================
# 一、鸢尾花热身
# ============================================================================
def iris_warmup():
    print("\n" + "=" * 60)
    print("【热身】鸢尾花数据集决策树快速入门")
    print("=" * 60)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )

    rows = []
    for criterion in ["gini", "entropy"]:
        for depth in [2, 3, None]:
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth,
                                         random_state=42)
            clf.fit(X_train, y_train)
            tr_acc = accuracy_score(y_train, clf.predict(X_train))
            te_acc = accuracy_score(y_test, clf.predict(X_test))
            rows.append({
                "criterion": criterion,
                "max_depth": depth if depth else "无限制",
                "训练集准确率": round(tr_acc, 4),
                "测试集准确率": round(te_acc, 4),
                "叶节点数": clf.get_n_leaves(),
                "树深度": clf.get_depth(),
            })
            print(f"  criterion={criterion:8s} max_depth={str(depth):6s} "
                  f"训练={tr_acc:.4f} 测试={te_acc:.4f} 叶={clf.get_n_leaves()}")

    save_csv(pd.DataFrame(rows), "00_iris_warmup.csv")

    # 可视化最优树
    plt = setup_chinese_font()
    best = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
    best.fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(best, feature_names=iris.feature_names,
              class_names=iris.target_names, filled=True, rounded=True,
              fontsize=9, ax=ax)
    ax.set_title(f"鸢尾花决策树（gini, max_depth=3）测试准确率="
                 f"{accuracy_score(y_test, best.predict(X_test)):.4f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "00_iris_warmup.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] images/00_iris_warmup.png")

    print("\n  思考：criterion 从 gini 换成 entropy，浅层树结果基本一致；"
          "max_depth=None 时训练准确率 100%，但测试集不再提升，说明进入过拟合区间。")
    return pd.DataFrame(rows)


# ============================================================================
# 二、Telco 数据探索
# ============================================================================
def telco_exploration():
    print("\n" + "=" * 60)
    print("【实验1】Telco 客户流失数据探索")
    print("=" * 60)

    raw = load_raw_data()
    print(f"\n原始形状：{raw.shape[0]} 行 × {raw.shape[1]} 列")

    # ---------- 1. 特征清单 ----------
    rows = []
    for col in raw.columns:
        s = raw[col]
        rows.append({
            "特征名称": col,
            "数据类型": str(s.dtype),
            "唯一值数": int(s.nunique()),
            "缺失值数": int(s.isnull().sum()),
            "示例值": str(s.iloc[0])[:30],
            "特征类型": ("标签" if col == TARGET else
                        "ID" if col == "customerID" else
                        "数值型" if pd.api.types.is_numeric_dtype(s) else
                        "类别型"),
        })
    df_summary = pd.DataFrame(rows)
    save_csv(df_summary, "01_data_summary.csv")
    print(f"  特征类型统计：数值 {sum(df_summary['特征类型']=='数值型')} 个，"
          f"类别 {sum(df_summary['特征类型']=='类别型')} 个")

    # ---------- 2. 标签分布 ----------
    counts = raw[TARGET].value_counts()
    label_df = pd.DataFrame({
        "类别": ["留存(No)", "流失(Yes)"],
        "样本数量": [int(counts["No"]), int(counts["Yes"])],
        "占比%": [round(counts["No"] / len(raw) * 100, 2),
                 round(counts["Yes"] / len(raw) * 100, 2)],
        "备注": ["多数类", "少数类（重点关注）"],
    })
    save_csv(label_df, "01_label_distribution.csv")
    print(f"\n  标签分布：留存 {counts['No']} ({counts['No']/len(raw)*100:.2f}%)，"
          f"流失 {counts['Yes']} ({counts['Yes']/len(raw)*100:.2f}%)")
    print(f"  不平衡比 ≈ {counts['No']/counts['Yes']:.2f}:1 —— 全猜留存可得 "
          f"{counts['No']/len(raw)*100:.2f}% 准确率，说明准确率不可单独作为评价指标")

    # ---------- 3. 缺失值 ----------
    miss = raw.isnull().sum()
    miss_df = pd.DataFrame({
        "特征": miss.index,
        "缺失值数": miss.values,
        "缺失率%": (miss.values / len(raw) * 100).round(2),
    })
    miss_df = miss_df[miss_df["缺失值数"] > 0]
    # TotalCharges 是 object，空格也算伪缺失
    total_charges_blank = (raw["TotalCharges"].astype(str).str.strip() == "").sum()
    extra = pd.DataFrame([{"特征": "TotalCharges(空白字符串)",
                           "缺失值数": int(total_charges_blank),
                           "缺失率%": round(total_charges_blank / len(raw) * 100, 2)}])
    miss_all = pd.concat([miss_df, extra], ignore_index=True) if len(miss_df) else extra
    save_csv(miss_all, "01_missing_summary.csv")
    print(f"\n  显式 NaN：{int(miss.sum())} 个；TotalCharges 空白字符串："
          f"{total_charges_blank} 个（11 条，均为新入网客户 tenure=0）")

    # ---------- 4. 类别特征透视 ----------
    cat_rows = []
    for col in CAT_COLS:
        for val, cnt in raw[col].value_counts().items():
            churn_rate = (raw.loc[raw[col] == val, TARGET] == "Yes").mean()
            cat_rows.append({
                "特征": col, "取值": val, "样本数": int(cnt),
                "占比%": round(cnt / len(raw) * 100, 2),
                "流失率%": round(churn_rate * 100, 2),
            })
    save_csv(pd.DataFrame(cat_rows), "01_category_stats.csv")

    # ---------- 5. 数值特征统计 ----------
    tmp = raw.copy()
    tmp["TotalCharges"] = pd.to_numeric(tmp["TotalCharges"], errors="coerce")
    num_rows = []
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        churn_yes = tmp.loc[tmp[TARGET] == "Yes", col]
        churn_no = tmp.loc[tmp[TARGET] == "No", col]
        num_rows.append({
            "特征": col,
            "最小值": round(tmp[col].min(), 2),
            "最大值": round(tmp[col].max(), 2),
            "均值": round(tmp[col].mean(), 2),
            "中位数": round(tmp[col].median(), 2),
            "流失组均值": round(churn_yes.mean(), 2),
            "留存组均值": round(churn_no.mean(), 2),
            "差异": round(churn_yes.mean() - churn_no.mean(), 2),
        })
    save_csv(pd.DataFrame(num_rows), "01_numeric_stats.csv")
    for r in num_rows:
        print(f"  {r['特征']:15s} 均值={r['均值']:8.2f}  "
              f"流失组={r['流失组均值']:8.2f} 留存组={r['留存组均值']:8.2f}")

    # ---------- 6. 可视化 ----------
    plt = setup_chinese_font()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 图1 标签分布
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(["留存(No)", "流失(Yes)"],
                  [counts["No"], counts["Yes"]],
                  color=["#4CAF50", "#F44336"], edgecolor="black")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 50,
                str(int(b.get_height())), ha="center", fontweight="bold")
    ax.set_ylabel("样本数")
    ax.set_title("标签分布（不平衡比 2.77:1）", fontsize=12)
    ax.set_ylim(0, counts["No"] * 1.15)

    # 图2 合约类型 vs 流失率
    ax = fig.add_subplot(gs[0, 1])
    contract_rate = raw.groupby("Contract")[TARGET].apply(
        lambda s: (s == "Yes").mean() * 100).sort_values()
    bars = ax.barh(contract_rate.index, contract_rate.values,
                   color="#FF7043", edgecolor="black")
    for b, v in zip(bars, contract_rate.values):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=9)
    ax.set_xlabel("流失率 (%)")
    ax.set_title("合约类型 × 流失率", fontsize=12)

    # 图3 在网时长分布
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(tmp.loc[tmp[TARGET] == "No", "tenure"], bins=25, alpha=0.65,
            label="留存", color="#4CAF50", edgecolor="black")
    ax.hist(tmp.loc[tmp[TARGET] == "Yes", "tenure"], bins=25, alpha=0.65,
            label="流失", color="#F44336", edgecolor="black")
    ax.set_xlabel("在网时长 tenure（月）")
    ax.set_ylabel("样本数")
    ax.set_title("在网时长分布（流失客户显著偏短）", fontsize=12)
    ax.legend()

    # 图4 月费分布
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(tmp.loc[tmp[TARGET] == "No", "MonthlyCharges"], bins=25, alpha=0.65,
            label="留存", color="#4CAF50", edgecolor="black")
    ax.hist(tmp.loc[tmp[TARGET] == "Yes", "MonthlyCharges"], bins=25, alpha=0.65,
            label="流失", color="#F44336", edgecolor="black")
    ax.set_xlabel("月费 MonthlyCharges（美元）")
    ax.set_ylabel("样本数")
    ax.set_title("月费分布（流失客户月费偏高）", fontsize=12)
    ax.legend()

    # 图5 网络服务 × 流失率
    ax = fig.add_subplot(gs[1, 1])
    svc_rate = raw.groupby("InternetService")[TARGET].apply(
        lambda s: (s == "Yes").mean() * 100).sort_values()
    bars = ax.barh(svc_rate.index, svc_rate.values, color="#AB47BC",
                   edgecolor="black")
    for b, v in zip(bars, svc_rate.values):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=9)
    ax.set_xlabel("流失率 (%)")
    ax.set_title("网络服务类型 × 流失率", fontsize=12)

    # 图6 付款方式 × 流失率
    ax = fig.add_subplot(gs[1, 2])
    pay_rate = raw.groupby("PaymentMethod")[TARGET].apply(
        lambda s: (s == "Yes").mean() * 100).sort_values()
    bars = ax.barh([p.replace(" (", "\n(") for p in pay_rate.index],
                   pay_rate.values, color="#42A5F5", edgecolor="black")
    for b, v in zip(bars, pay_rate.values):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=9)
    ax.set_xlabel("流失率 (%)")
    ax.set_title("付款方式 × 流失率", fontsize=12)

    fig.suptitle("Telco 客户流失数据集探索性分析", fontsize=15, fontweight="bold")
    plt.savefig(os.path.join(IMG_DIR, "01_data_exploration.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [SAVED] images/01_data_exploration.png")

    # ---------- 7. 业务洞察 ----------
    print("\n  【业务洞察】")
    print(f"  - 月结合约流失率 {cat_rate(raw,'Contract','Month-to-month'):.1f}%，"
          f"两年合约仅 {cat_rate(raw,'Contract','Two year'):.1f}%")
    print(f"  - 光纤用户流失率 {cat_rate(raw,'InternetService','Fiber optic'):.1f}%，"
          f"DSL 仅 {cat_rate(raw,'InternetService','DSL'):.1f}%")
    print(f"  - 电子支票付款流失率 {cat_rate(raw,'PaymentMethod','Electronic check'):.1f}%")

    return raw


def cat_rate(df, col, val):
    sub = df[df[col] == val]
    return (sub[TARGET] == "Yes").mean() * 100 if len(sub) else 0


if __name__ == "__main__":
    iris_warmup()
    telco_exploration()
    print("\n" + "=" * 60)
    print("exp01 完成")
    print("=" * 60)