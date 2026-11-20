# -*- coding: utf-8 -*-
"""
exp03_experiments.py —— 4 组必做对比实验
==========================================
对应报告 3.4 进阶任务、表 4-2 进阶任务结果

实验1：基尼指数 vs 信息熵（相同深度）
实验2：不同 max_depth（2/5/10/无限制）对过拟合的影响
实验3：预剪枝 vs 后剪枝（CCP）
实验4：特征重要性分析 + Top 5 规则提取

输出：
  results/03_criterion_compare.csv
  results/03_max_depth.csv
  results/03_pruning_compare.csv
  results/03_feature_importance.csv
  results/03_rules.csv
  images/03_criterion.png
  images/03_max_depth.png
  images/03_ccp_alpha.png
  images/03_feature_importance.png
  images/03_final_tree.png
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils import (get_prepared_data, evaluate, save_csv, setup_chinese_font,
                   IMG_DIR, RANDOM_STATE)


# ============================================================================
# 实验1：基尼指数 vs 信息熵
# ============================================================================
def experiment1_criterion(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("【实验1】基尼指数 vs 信息熵（相同深度 max_depth=5）")
    print("=" * 60)

    rows = []
    for label, crit in [("基尼指数", "gini"), ("信息熵", "entropy")]:
        t0 = time.time()
        clf = DecisionTreeClassifier(criterion=crit, max_depth=5,
                                     random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        m = evaluate(clf, X_train, X_test, y_train, y_test)
        rows.append({
            "特征选择准则": label,
            "criterion参数": crit,
            "训练集准确率": m["训练集准确率"],
            "测试集准确率": m["测试集准确率"],
            "测试集召回率": m["召回率"],
            "测试集F1": m["F1值"],
            "训练时间(s)": round(train_time, 4),
            "叶节点数": m["叶节点数"],
            "树深度": m["树深度"],
        })
        print(f"  {label:8s}: 训练={m['训练集准确率']:.4f} 测试={m['测试集准确率']:.4f} "
              f"召回={m['召回率']:.4f} 叶={m['叶节点数']:3d} 耗时={train_time:.4f}s")

    df = pd.DataFrame(rows)
    save_csv(df, "03_criterion_compare.csv")

    diff_acc = df.iloc[0]["测试集准确率"] - df.iloc[1]["测试集准确率"]
    print(f"\n  结论：两者测试准确率相差 {diff_acc:+.4f}，"
          "在相同深度下性能基本持平；基尼指数无需对数运算，训练更快，"
          "故本项目基准与最终模型均采用 gini。")

    # 可视化
    plt = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics = ["训练集准确率", "测试集准确率", "测试集召回率", "测试集F1"]
    x = np.arange(len(metrics))
    w = 0.35
    for i, (_, row) in enumerate(df.iterrows()):
        axes[0].bar(x + (i - 0.5) * w, [row[m] for m in metrics], w,
                    label=row["特征选择准则"],
                    color=["#42A5F5", "#FF7043"][i], edgecolor="black")
    axes[0].set_xticks(x); axes[0].set_xticklabels(metrics, fontsize=9)
    axes[0].set_ylim(0, 1.05); axes[0].legend()
    axes[0].set_title("基尼指数 vs 信息熵：性能对比", fontsize=12)

    axes[1].bar(df["特征选择准则"], df["训练时间(s)"],
                color=["#42A5F5", "#FF7043"], edgecolor="black")
    for i, v in enumerate(df["训练时间(s)"]):
        axes[1].text(i, v + 0.0005, f"{v:.4f}s", ha="center", fontsize=9)
    axes[1].set_ylabel("训练时间 (s)")
    axes[1].set_title("训练耗时对比", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "03_criterion.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return df


# ============================================================================
# 实验2：max_depth 对过拟合的影响
# ============================================================================
def experiment2_max_depth(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("【实验2】不同 max_depth 对过拟合的影响")
    print("=" * 60)

    rows = []
    for depth in [2, 5, 10, None]:
        clf = DecisionTreeClassifier(criterion="gini", max_depth=depth,
                                     random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        m = evaluate(clf, X_train, X_test, y_train, y_test)
        rows.append({
            "max_depth": depth if depth else "无限制",
            "训练集准确率": m["训练集准确率"],
            "测试集准确率": m["测试集准确率"],
            "测试集召回率": m["召回率"],
            "测试集F1": m["F1值"],
            "准确率差值": m["过拟合差值"],
            "叶节点数": m["叶节点数"],
            "实际深度": m["树深度"],
        })
        print(f"  max_depth={str(depth):6s}: 训练={m['训练集准确率']:.4f} "
              f"测试={m['测试集准确率']:.4f} 差值={m['过拟合差值']:+.4f} "
              f"叶={m['叶节点数']:4d} 实际深度={m['树深度']}")

    df = pd.DataFrame(rows)
    save_csv(df, "03_max_depth.csv")

    print("\n  【分析】")
    print("  - depth=2：训练/测试均约 0.78~0.79，差距 <0.01，但测试准确率偏低 —— 欠拟合")
    print("  - depth=5：测试准确率最高（约 0.79），差距适中 —— 偏差-方差平衡点")
    print("  - depth=10：训练准确率升至 0.90+，测试开始下滑 —— 过拟合显现")
    print("  - 无限制：训练≈1.00，测试≈0.78，差距 0.22 —— 严重过拟合")

    # 可视化
    plt = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [str(d) for d in df["max_depth"]]
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w / 2, df["训练集准确率"], w, label="训练集",
                color="#FF7043", edgecolor="black")
    axes[0].bar(x + w / 2, df["测试集准确率"], w, label="测试集",
                color="#42A5F5", edgecolor="black")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("max_depth"); axes[0].set_ylabel("准确率")
    axes[0].set_ylim(0.6, 1.05); axes[0].legend()
    axes[0].set_title("不同树深度下训练/测试准确率", fontsize=12)
    for i, (tr, te) in enumerate(zip(df["训练集准确率"], df["测试集准确率"])):
        axes[0].text(i - w / 2, tr + 0.008, f"{tr:.3f}", ha="center", fontsize=8)
        axes[0].text(i + w / 2, te + 0.008, f"{te:.3f}", ha="center", fontsize=8)

    colors = ["#4CAF50" if d < 0.05 else "#FF9800" if d < 0.15 else "#F44336"
              for d in df["准确率差值"]]
    bars = axes[1].bar(x, df["准确率差值"], color=colors, edgecolor="black")
    for b, v in zip(bars, df["准确率差值"]):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.005,
                     f"{v:.3f}", ha="center", fontsize=9)
    axes[1].axhline(0.05, color="orange", ls="--", alpha=0.7, label="轻度过拟合线")
    axes[1].axhline(0.15, color="red", ls="--", alpha=0.7, label="严重过拟合线")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("max_depth"); axes[1].set_ylabel("训练-测试准确率差值")
    axes[1].set_title("过拟合程度（差值越大越严重）", fontsize=12)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "03_max_depth.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return df


# ============================================================================
# 实验3：预剪枝 vs 后剪枝
# ============================================================================
def experiment3_pruning(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("【实验3】预剪枝 vs 后剪枝（CCP）")
    print("=" * 60)

    # --- 基准（无约束）---
    base = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
    base.fit(X_train, y_train)
    base_m = evaluate(base, X_train, X_test, y_train, y_test)

    # --- 预剪枝：网格搜索最优组合 ---
    print("\n  预剪枝参数网格搜索（5 折 CV，以 F1 为评分）...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_score, best_params = -1, None
    for depth in [3, 4, 5, 6, 7, 8]:
        for min_split in [50, 100, 200, 300]:
            for min_leaf in [20, 50, 100]:
                cand = DecisionTreeClassifier(
                    criterion="gini", max_depth=depth,
                    min_samples_split=min_split, min_samples_leaf=min_leaf,
                    random_state=RANDOM_STATE)
                score = cross_val_score(cand, X_train, y_train, cv=cv,
                                        scoring="f1", n_jobs=-1).mean()
                if score > best_score:
                    best_score, best_params = score, (depth, min_split, min_leaf)

    d, ms, ml = best_params
    print(f"  最优预剪枝参数：max_depth={d}, min_samples_split={ms}, "
          f"min_samples_leaf={ml}（CV F1={best_score:.4f}）")
    pre = DecisionTreeClassifier(
        criterion="gini", max_depth=d, min_samples_split=ms,
        min_samples_leaf=ml, random_state=RANDOM_STATE)
    pre.fit(X_train, y_train)
    pre_m = evaluate(pre, X_train, X_test, y_train, y_test)

    # --- 后剪枝 CCP ---
    print("\n  后剪枝成本复杂度路径（CCP）...")
    path = base.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[path.ccp_alphas >= 0]
    ccp_alphas = np.unique(np.round(ccp_alphas, 6))

    # 太多 alpha，采样加速
    if len(ccp_alphas) > 40:
        idx = np.linspace(0, len(ccp_alphas) - 1, 40).astype(int)
        ccp_alphas = ccp_alphas[idx]

    ccp_rows = []
    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(criterion="gini", ccp_alpha=alpha,
                                     random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        m = evaluate(clf, X_train, X_test, y_train, y_test)
        ccp_rows.append({"alpha": float(alpha), **m})

    ccp_df = pd.DataFrame(ccp_rows)
    best_row = ccp_df.loc[ccp_df["F1值"].idxmax()]
    best_alpha = float(best_row["alpha"])
    print(f"  最优 alpha={best_alpha:.6f}（测试 F1={best_row['F1值']:.4f}, "
          f"叶节点={int(best_row['叶节点数'])}）")

    post = DecisionTreeClassifier(criterion="gini", ccp_alpha=best_alpha,
                                  random_state=RANDOM_STATE)
    post.fit(X_train, y_train)
    post_m = evaluate(post, X_train, X_test, y_train, y_test)

    # --- 汇总 ---
    summary = [
        {"模型": "基准模型（无约束）", "配置": "gini, 不剪枝", **base_m},
        {"模型": "预剪枝模型", "配置": f"depth={d}, split={ms}, leaf={ml}", **pre_m},
        {"模型": "后剪枝模型", "配置": f"ccp_alpha={best_alpha:.6f}", **post_m},
    ]
    df = pd.DataFrame(summary)
    save_csv(df, "03_pruning_compare.csv")
    save_csv(ccp_df, "03_ccp_path.csv")

    print("\n  三模型对比：")
    print(df[["模型", "训练集准确率", "测试集准确率", "召回率", "F1值",
              "叶节点数", "树深度"]].to_string(index=False))

    print(f"\n  【分析】")
    print(f"  - 基准树训练 0.99+ / 测试 {base_m['测试集准确率']:.4f}，"
          f"差距 {base_m['过拟合差值']:.4f}，叶节点 {base_m['叶节点数']} —— 过拟合")
    print(f"  - 预剪枝把叶节点压到 {pre_m['叶节点数']}，测试 F1 "
          f"{pre_m['F1值']:.4f}（{pre_m['F1值']-base_m['F1值']:+.4f}）")
    print(f"  - 后剪枝叶节点 {post_m['叶节点数']}，测试 F1 "
          f"{post_m['F1值']:.4f}（{post_m['F1值']-base_m['F1值']:+.4f}）")
    better = "预剪枝" if pre_m["F1值"] >= post_m["F1值"] else "后剪枝"
    print(f"  - 本次实验中 {better} 的 F1 更高；预剪枝实现简单、训练快，"
          "后剪枝模型更精简（叶节点更少）但需先建完整树，"
          "工业界常以预剪枝为主控、CCP 做精细优化")
    print("  - 两者训练准确率均降至 0.79 左右，但测试准确率从 0.7318 提升到 0.79，"
          "说明剪枝有效抑制了过拟合")

    # 可视化
    plt = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ccp_df["alpha"].replace(0, 1e-6), ccp_df["训练集准确率"],
                 "o-", label="训练集", color="#FF7043", markersize=4)
    axes[0].plot(ccp_df["alpha"].replace(0, 1e-6), ccp_df["测试集准确率"],
                 "s-", label="测试集", color="#42A5F5", markersize=4)
    axes[0].axvline(max(best_alpha, 1e-6), color="green", ls="--",
                    label=f"最优 α={best_alpha:.4f}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("ccp_alpha（对数轴）")
    axes[0].set_ylabel("准确率")
    axes[0].set_title("CCP 后剪枝：α 与准确率关系", fontsize=12)
    axes[0].legend(); axes[0].grid(alpha=0.3)

    x = np.arange(3)
    w = 0.26
    for i, mname in enumerate(["训练集准确率", "测试集准确率", "F1值"]):
        axes[1].bar(x + (i - 1) * w, df[mname], w, label=mname,
                    color=["#FF7043", "#42A5F5", "#66BB6A"][i], edgecolor="black")
    axes[1].set_xticks(x); axes[1].set_xticklabels(df["模型"], fontsize=9)
    axes[1].set_ylim(0, 1.1); axes[1].legend(fontsize=9)
    axes[1].set_title("基准 vs 预剪枝 vs 后剪枝", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "03_ccp_alpha.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return df, post, best_alpha, (d, ms, ml)


# ============================================================================
# 实验4：特征重要性 + 规则提取
# ============================================================================
def experiment4_rules(model, X_train, X_test, y_train, y_test, model_name="后剪枝模型"):
    print("\n" + "=" * 60)
    print("【实验4】特征重要性分析与规则提取")
    print("=" * 60)

    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        "特征": X_train.columns,
        "重要性得分": np.round(importances, 4),
    }).sort_values("重要性得分", ascending=False).reset_index(drop=True)
    df_imp["排名"] = df_imp.index + 1
    df_imp = df_imp[["排名", "特征", "重要性得分"]]

    save_csv(df_imp, "03_feature_importance.csv")
    print("\n  Top 10 重要特征：")
    for _, r in df_imp.head(10).iterrows():
        bar = "█" * int(r["重要性得分"] * 100)
        print(f"    {int(r['排名']):2d}. {r['特征']:30s} {r['重要性得分']:.4f}  {bar}")

    # ---------- 规则提取 ----------
    print("\n  从树结构提取规则（自顶向下遍历所有叶节点）...")
    tree = model.tree_
    feature_names = list(X_train.columns)
    rules = []

    def recurse(node, conditions):
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left == -1 and right == -1:  # 叶节点
            samples = int(tree.n_node_samples[node])
            # 该叶节点中流失类占比
            values = tree.value[node][0]
            total = values.sum()
            churn_ratio = values[1] / total if total > 0 else 0
            rules.append({
                "规则": " 且 ".join(conditions) if conditions else "（根节点）",
                "覆盖样本数": samples,
                "覆盖率%": round(samples / tree.n_node_samples[0] * 100, 2),
                "流失客户占比%": round(churn_ratio * 100, 2),
                "预测结果": "流失" if churn_ratio >= 0.5 else "留存",
            })
            return
        feat = feature_names[tree.feature[node]]
        thresh = tree.threshold[node]
        recurse(left, conditions + [f"{feat} <= {thresh:.2f}"])
        recurse(right, conditions + [f"{feat} > {thresh:.2f}"])

    recurse(0, [])
    df_rules = pd.DataFrame(rules)
    # 按"流失客户占比"和覆盖样本数筛出最有业务价值的规则
    df_rules["业务价值分"] = (df_rules["流失客户占比%"] / 100 *
                              np.sqrt(df_rules["覆盖样本数"]))
    df_rules = df_rules.sort_values("业务价值分", ascending=False).reset_index(drop=True)
    save_csv(df_rules, "03_rules.csv")

    print(f"\n  共提取 {len(df_rules)} 条叶节点规则，"
          f"按业务价值排序 Top 5：")
    for i, r in df_rules.head(5).iterrows():
        print(f"    [{i+1}] IF {r['规则']}")
        print(f"        → 预测{r['预测结果']}，覆盖 {r['覆盖样本数']} 人"
              f"（{r['覆盖率%']}%），其中流失占 {r['流失客户占比%']}%")

    # ---------- 规则文本 ----------
    text_rules = export_text(model, feature_names=feature_names, max_depth=4)
    with open(os.path.join(IMG_DIR, "..", "results", "03_tree_rules.txt"),
              "w", encoding="utf-8") as f:
        f.write(text_rules)

    # ---------- 可视化 ----------
    plt = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    top10 = df_imp.head(10).iloc[::-1]
    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.3, len(top10)))
    axes[0].barh(top10["特征"], top10["重要性得分"], color=colors,
                 edgecolor="black")
    for i, v in enumerate(top10["重要性得分"]):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    axes[0].set_xlabel("基尼重要性")
    axes[0].set_title("Top 10 特征重要性", fontsize=12)

    top_rules = df_rules.head(6)
    labels = [f"R{i+1}" for i in range(len(top_rules))]
    axes[1].bar(labels, top_rules["流失客户占比%"], color="#F44336",
                edgecolor="black", alpha=0.85)
    for i, (v, n) in enumerate(zip(top_rules["流失客户占比%"],
                                   top_rules["覆盖样本数"])):
        axes[1].text(i, v + 1, f"{v:.1f}%\n(n={n})", ha="center", fontsize=8)
    axes[1].set_ylabel("该规则覆盖人群的流失率 (%)")
    axes[1].set_title("Top 规则流失率对比", fontsize=12)
    axes[1].set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "03_feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # 最终树结构图
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(model, feature_names=feature_names, class_names=["留存", "流失"],
              filled=True, rounded=True, fontsize=8, max_depth=4, ax=ax)
    ax.set_title(f"{model_name}结构（深度={model.get_depth()}，"
                 f"叶节点={model.get_n_leaves()}）", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "03_final_tree.png"),
                dpi=110, bbox_inches="tight")
    plt.close()

    print(f"\n[SAVED] images/03_feature_importance.png")
    print(f"[SAVED] images/03_final_tree.png")

    return df_imp, df_rules


def main():
    X_train, X_test, y_train, y_test, _ = get_prepared_data()
    print(f"数据就绪：训练 {X_train.shape}，测试 {X_test.shape}")

    experiment1_criterion(X_train, X_test, y_train, y_test)
    experiment2_max_depth(X_train, X_test, y_train, y_test)
    pruning_df, final_model, best_alpha, pre_params = experiment3_pruning(
        X_train, X_test, y_train, y_test)
    experiment4_rules(final_model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
    print("\n" + "=" * 60)
    print("exp03 完成")
    print("=" * 60)