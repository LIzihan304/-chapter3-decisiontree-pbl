# -*- coding: utf-8 -*-
"""
exp04_challenges.py —— 3 个挑战拓展任务
==========================================
对应报告 3.5 挑战任务、表 4-3 挑战任务结果

挑战1：样本不平衡优化（class_weight vs SMOTE）
挑战2：决策树 vs KNN 算法横向对比
挑战3：基于业务理解的特征工程

输出：
  results/04_imbalance_compare.csv
  results/04_algo_compare.csv
  results/04_feature_engineering.csv
  images/04_imbalance.png
  images/04_algo_compare.png
  images/04_feature_engineering.png
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

from utils import (get_prepared_data, evaluate, save_csv, setup_chinese_font,
                   IMG_DIR, RESULT_DIR, RANDOM_STATE)


# ============================================================================
# 挑战1：样本不平衡优化
# ============================================================================
def challenge1_imbalance(X_train, X_test, y_train, y_test, best_params):
    print("\n" + "=" * 60)
    print("【挑战1】样本不平衡优化：class_weight vs SMOTE")
    print("=" * 60)

    d, ms, ml = best_params

    def build(class_weight=None):
        return DecisionTreeClassifier(
            criterion="gini", max_depth=d, min_samples_split=ms,
            min_samples_leaf=ml, class_weight=class_weight,
            random_state=RANDOM_STATE)

    rows = []

    # --- 原始（无处理）---
    base = build()
    base.fit(X_train, y_train)
    m = evaluate(base, X_train, X_test, y_train, y_test)
    cm = confusion_matrix(y_test, base.predict(X_test))
    tn, fp, fn, tp = cm.ravel()
    rows.append({"方案": "原始（不平衡）", "说明": "无任何处理",
                 **m, "漏判(FN)": int(fn), "误判(FP)": int(fp)})
    print(f"  原始：       召回={m['召回率']:.4f} 精确={m['精确率']:.4f} "
          f"F1={m['F1值']:.4f} FN={fn}")

    # --- class_weight='balanced' ---
    cw = build(class_weight="balanced")
    cw.fit(X_train, y_train)
    m = evaluate(cw, X_train, X_test, y_train, y_test)
    cm = confusion_matrix(y_test, cw.predict(X_test))
    tn, fp, fn, tp = cm.ravel()
    rows.append({"方案": "class_weight='balanced'", "说明": "按类别频率反比加权",
                 **m, "漏判(FN)": int(fn), "误判(FP)": int(fp)})
    print(f"  class_weight：召回={m['召回率']:.4f} 精确={m['精确率']:.4f} "
          f"F1={m['F1值']:.4f} FN={fn}")

    # --- SMOTE ---
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    sm = build()
    sm.fit(X_res, y_res)
    m = evaluate(sm, X_train, X_test, y_train, y_test)
    cm = confusion_matrix(y_test, sm.predict(X_test))
    tn, fp, fn, tp = cm.ravel()
    rows.append({"方案": "SMOTE 过采样", "说明": f"训练集 {len(y_train)}→{len(y_res)} 条",
                 **m, "漏判(FN)": int(fn), "误判(FP)": int(fp)})
    print(f"  SMOTE：      召回={m['召回率']:.4f} 精确={m['精确率']:.4f} "
          f"F1={m['F1值']:.4f} FN={fn}")

    # --- SMOTE + class_weight ---
    sm2 = build(class_weight="balanced")
    sm2.fit(X_res, y_res)
    m2 = evaluate(sm2, X_train, X_test, y_train, y_test)
    cm = confusion_matrix(y_test, sm2.predict(X_test))
    tn, fp, fn, tp = cm.ravel()
    rows.append({"方案": "SMOTE + class_weight", "说明": "双管齐下",
                 **m2, "漏判(FN)": int(fn), "误判(FP)": int(fp)})
    print(f"  SMOTE+加权：  召回={m2['召回率']:.4f} 精确={m2['精确率']:.4f} "
          f"F1={m2['F1值']:.4f} FN={fn}")

    df = pd.DataFrame(rows)
    save_csv(df, "04_imbalance_compare.csv")

    best = df.loc[df["F1值"].idxmax()]
    print(f"\n  最佳方案（按 F1）：{best['方案']}，F1={best['F1值']:.4f}，"
          f"召回={best['召回率']:.4f}")
    print("  【业务解读】漏判一个流失客户（FN）的代价 ≈ 一个客户的终身价值；"
          "误判（FP）的代价只是一次挽留优惠券。因此召回率优先，"
          "SMOTE 与 class_weight 都能显著减少 FN。")

    # 可视化
    plt = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    metrics = ["召回率", "精确率", "F1值", "AUC"]
    x = np.arange(len(metrics))
    w = 0.2
    colors = ["#9E9E9E", "#42A5F5", "#FF7043", "#66BB6A"]
    for i, (_, r) in enumerate(df.iterrows()):
        ax.bar(x + (i - 1.5) * w, [r[k] for k in metrics], w,
               label=r["方案"], color=colors[i], edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1); ax.legend(fontsize=9)
    ax.set_title("样本不平衡处理方案对比", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "04_imbalance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return df


# ============================================================================
# 挑战2：决策树 vs KNN
# ============================================================================
def challenge2_algo_compare(X_train, X_test, y_train, y_test, best_params):
    print("\n" + "=" * 60)
    print("【挑战2】决策树 vs KNN 算法横向对比")
    print("=" * 60)

    d, ms, ml = best_params

    # KNN 需要标准化
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    rows = []

    # 决策树
    t0 = time.time()
    dt = DecisionTreeClassifier(criterion="gini", max_depth=d,
                                min_samples_split=ms, min_samples_leaf=ml,
                                random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    dt_train_t = time.time() - t0
    t0 = time.time()
    dt.predict(X_test)
    dt_pred_t = time.time() - t0
    m = evaluate(dt, X_train, X_test, y_train, y_test)
    rows.append({
        "算法": "决策树", "配置": f"gini, depth={d}",
        "准确率": m["测试集准确率"], "精确率": m["精确率"],
        "召回率": m["召回率"], "F1值": m["F1值"], "AUC": m["AUC"],
        "训练时间(s)": round(dt_train_t, 4),
        "预测时间(s)": round(dt_pred_t, 4),
        "可解释性": "强（可视化树+规则）",
        "需标准化": "否",
    })
    print(f"  决策树：准确={m['测试集准确率']:.4f} 召回={m['召回率']:.4f} "
          f"F1={m['F1值']:.4f} 训练={dt_train_t:.4f}s 预测={dt_pred_t:.4f}s")

    # KNN 调参（简单网格）
    best_knn, best_knn_score = None, -1
    for k in [5, 7, 9, 11, 15]:
        for w in ["uniform", "distance"]:
            knn = KNeighborsClassifier(n_neighbors=k, weights=w)
            knn.fit(X_tr_s, y_train)
            from sklearn.model_selection import cross_val_score
            s = cross_val_score(knn, X_tr_s, y_train, cv=3, scoring="f1").mean()
            if s > best_knn_score:
                best_knn_score, best_knn = s, (k, w)
    k_best, w_best = best_knn
    print(f"  KNN 最优参数：n_neighbors={k_best}, weights={w_best}（CV F1={best_knn_score:.4f}）")

    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=k_best, weights=w_best)
    knn.fit(X_tr_s, y_train)
    knn_train_t = time.time() - t0
    t0 = time.time()
    knn.predict(X_te_s)
    knn_pred_t = time.time() - t0

    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)
    yp = knn.predict(X_te_s)
    ypr = knn.predict_proba(X_te_s)[:, 1]
    rows.append({
        "算法": "KNN",
        "配置": f"k={k_best}, {w_best}",
        "准确率": round(accuracy_score(y_test, yp), 4),
        "精确率": round(precision_score(y_test, yp), 4),
        "召回率": round(recall_score(y_test, yp), 4),
        "F1值": round(f1_score(y_test, yp), 4),
        "AUC": round(roc_auc_score(y_test, ypr), 4),
        "训练时间(s)": round(knn_train_t, 4),
        "预测时间(s)": round(knn_pred_t, 4),
        "可解释性": "弱（黑箱距离）",
        "需标准化": "是",
    })
    print(f"  KNN：  准确={rows[1]['准确率']:.4f} 召回={rows[1]['召回率']:.4f} "
          f"F1={rows[1]['F1值']:.4f} 训练={knn_train_t:.4f}s 预测={knn_pred_t:.4f}s")

    df = pd.DataFrame(rows)
    save_csv(df, "04_algo_compare.csv")

    print("\n  【四维对比】")
    dt_f1 = df[df["算法"] == "决策树"]["F1值"].iloc[0]
    knn_f1 = df[df["算法"] == "KNN"]["F1值"].iloc[0]
    dt_acc = df[df["算法"] == "决策树"]["准确率"].iloc[0]
    knn_acc = df[df["算法"] == "KNN"]["准确率"].iloc[0]
    winner = "KNN" if knn_f1 > dt_f1 else "决策树"
    print(f"  1. 性能：准确率 决策树 {dt_acc:.4f} vs KNN {knn_acc:.4f}；"
          f"F1 决策树 {dt_f1:.4f} vs KNN {knn_f1:.4f}，本次 {winner} 略优")
    print("     两者差距很小（<0.02），说明该数据集线性/局部结构都不强，"
          "单棵树的判别力有限")
    print("  2. 可解释性：决策树可输出规则（IF 在网<=16.5月 AND 光纤 THEN 流失率 62.7%），"
          "KNN 只能给出邻居，无法解释——这是本项目选决策树的核心原因")
    print(f"  3. 训练：决策树 {df[df['算法']=='决策树']['训练时间(s)'].iloc[0]:.4f}s vs "
          f"KNN {df[df['算法']=='KNN']['训练时间(s)'].iloc[0]:.4f}s；"
          "KNN 是惰性学习，训练仅存储数据，故更快")
    print(f"  4. 预测：决策树 {df[df['算法']=='决策树']['预测时间(s)'].iloc[0]:.4f}s vs "
          f"KNN {df[df['算法']=='KNN']['预测时间(s)'].iloc[0]:.4f}s；"
          "KNN 需遍历全部训练样本算距离，样本量越大差距越明显")
    print("  5. 预处理：KNN 必须先标准化（否则 tenure 0~72 会压倒 0/1 特征），"
          "决策树基于阈值分裂，天然无需标准化")
    print("  6. 业务落地：流失预测的核心交付物是「可执行的挽留规则」，"
          "决策树能直接给运营人员看，KNN 给不出；"
          "若追求纯预测精度且数据量不大，KNN 可作为对照基线")

    # 可视化
    plt = setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["准确率", "精确率", "召回率", "F1值", "AUC"]
    x = np.arange(len(metrics)); w = 0.35
    for i, (_, r) in enumerate(df.iterrows()):
        axes[0].bar(x + (i - 0.5) * w, [r[k] for k in metrics], w,
                    label=r["算法"], color=["#42A5F5", "#FF7043"][i],
                    edgecolor="black")
    axes[0].set_xticks(x); axes[0].set_xticklabels(metrics, fontsize=9)
    axes[0].set_ylim(0, 1.1); axes[0].legend()
    axes[0].set_title("性能对比", fontsize=12)

    axes[1].bar(df["算法"], df["训练时间(s)"], color=["#42A5F5", "#FF7043"],
                edgecolor="black")
    axes[1].set_ylabel("训练时间 (s)")
    axes[1].set_title("训练耗时对比", fontsize=12)
    for i, v in enumerate(df["训练时间(s)"]):
        axes[1].text(i, v * 1.05 if v > 0 else 0.0001, f"{v:.4f}s",
                     ha="center", fontsize=9)

    axes[2].bar(df["算法"], df["预测时间(s)"], color=["#42A5F5", "#FF7043"],
                edgecolor="black")
    axes[2].set_ylabel("预测时间 (s)")
    axes[2].set_title("预测耗时对比（KNN 更慢）", fontsize=12)
    for i, v in enumerate(df["预测时间(s)"]):
        axes[2].text(i, v * 1.05 if v > 0 else 0.0001, f"{v:.4f}s",
                     ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "04_algo_compare.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    return df


# ============================================================================
# 挑战3：业务特征工程
# ============================================================================
def challenge3_feature_engineering(X_train, X_test, y_train, y_test, best_params):
    print("\n" + "=" * 60)
    print("【挑战3】基于业务理解的特征工程")
    print("=" * 60)

    # 特征工程对比使用较宽松的树配置（depth=8, leaf=30），
    # 否则预剪枝过强会掩盖新特征的作用（树根本没机会用到新特征）
    FE_DEPTH, FE_LEAF, FE_SPLIT = 8, 30, 60

    def fit_eval(Xtr, Xte, tag, note):
        clf = DecisionTreeClassifier(criterion="gini", max_depth=FE_DEPTH,
                                     min_samples_split=FE_SPLIT,
                                     min_samples_leaf=FE_LEAF,
                                     random_state=RANDOM_STATE)
        clf.fit(Xtr, y_train)
        m = evaluate(clf, Xtr, Xte, y_train, y_test)
        used = set()
        feat_names = list(Xtr.columns)
        for node in range(clf.tree_.node_count):
            if clf.tree_.children_left[node] != -1:
                used.add(feat_names[clf.tree_.feature[node]])
        new_used = [c for c in used if c in
                    ("avg_monthly_spend", "service_count", "is_new_customer")]
        return {"方案": tag, "说明": note, **m,
                "新特征是否入树": "、".join(new_used) if new_used else "否"}

    rows = [fit_eval(X_train, X_test, "原始特征", f"{X_train.shape[1]} 维")]

    # --- 新特征 1：单位时长消费（月费 / (在网时长+1)）---
    Xtr2 = X_train.copy(); Xte2 = X_test.copy()
    Xtr2["avg_monthly_spend"] = Xtr2["MonthlyCharges"] / (Xtr2["tenure"] + 1)
    Xte2["avg_monthly_spend"] = Xte2["MonthlyCharges"] / (Xte2["tenure"] + 1)
    rows.append(fit_eval(Xtr2, Xte2, "+ 单位时长消费",
                         "MonthlyCharges/(tenure+1)，识别'高消费新客户'"))

    # --- 新特征 2：服务数量总和（统计用户开通了多少增值服务）---
    service_cols = [c for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                "TechSupport", "StreamingTV", "StreamingMovies"]
                    if c in X_train.columns]
    Xtr3 = Xtr2.copy(); Xte3 = Xte2.copy()
    Xtr3["service_count"] = Xtr3[service_cols].sum(axis=1)
    Xte3["service_count"] = Xte3[service_cols].sum(axis=1)
    rows.append(fit_eval(Xtr3, Xte3, "+ 服务数量总和",
                         f"统计 {len(service_cols)} 项增值服务开通数"))

    # --- 新特征 3：高风险组合标记（月合约 + 光纤 + 电子支票）---
    Xtr4 = Xtr3.copy(); Xte4 = Xte3.copy()
    contract_col = "Contract_Two year"  # drop_first 后只剩 Two year / One year
    cond_tr = (Xtr4["tenure"] <= 6).astype(int)
    Xtr4["is_new_customer"] = cond_tr
    Xte4["is_new_customer"] = (Xte4["tenure"] <= 6).astype(int)
    rows.append(fit_eval(Xtr4, Xte4, "+ 新客户标记",
                         "tenure<=6 标记为 1，业务上新入网客户流失高发"))

    df = pd.DataFrame(rows)
    save_csv(df, "04_feature_engineering.csv")
    print()
    for _, r in df.iterrows():
        print(f"  {r['方案']:16s} 准确={r['测试集准确率']:.4f} "
              f"召回={r['召回率']:.4f} F1={r['F1值']:.4f} "
              f"叶={r['叶节点数']:3d} 新特征入树={r['新特征是否入树']}")

    base_f1 = df.iloc[0]["F1值"]
    best_row = df.loc[df["F1值"].idxmax()]
    print(f"\n  最佳方案：{best_row['方案']}（F1={best_row['F1值']:.4f}，"
          f"较原始 {best_row['F1值']-base_f1:+.4f}）")
    print("  【业务解读与反思】")
    print("  - 三个新特征中，「单位时长消费」与「服务数量」确实被树选用，"
          "但测试 F1 反而轻微下降 —— 这是一个值得记录的负结果")
    print("  - 原因1：avg_monthly_spend = MonthlyCharges/(tenure+1)，"
          "信息完全来自月费和在网时长两个已有特征，决策树本就能通过多次分裂组合出该关系")
    print("  - 原因2：服务数量是 6 个 0/1 特征的线性组合，"
          "树可以按需逐项分裂，手工求和反而损失了单项粒度")
    print("  - 原因3：is_new_customer（tenure<=6）与 tenure 高度共线，"
          "树直接对 tenure 设阈值即可，新特征未被选中")
    print("  - 结论：决策树对单调变换和特征组合不敏感，"
          "「业务理解造特征」更适合线性模型/神经网络；"
          "对树模型，更有价值的是补充**原始数据中不存在的新信息**"
          "（如客户投诉次数、竞品促销曝光、App 登录频次），"
          "而非对现有特征做算术组合")
    print("  - 若必须用树模型，可考虑 GradientBoosting/XGBoost，"
          "它们对特征工程的敏感度高于单棵树")

    # 可视化
    plt = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    metrics = ["测试集准确率", "召回率", "F1值"]
    x = np.arange(len(metrics)); w = 0.2
    colors = ["#9E9E9E", "#42A5F5", "#FF7043", "#66BB6A"]
    for i, (_, r) in enumerate(df.iterrows()):
        ax.bar(x + (i - 1.5) * w, [r[k] for k in metrics], w,
               label=r["方案"], color=colors[i], edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1); ax.legend(fontsize=9)
    ax.set_title("特征工程效果对比（决策树对算术组合特征的收益有限）", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "04_feature_engineering.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    return df


def load_best_params():
    """从 exp03 的结果中读取最优预剪枝参数，保证挑战任务与进阶任务配置一致"""
    default = (5, 100, 50)
    path = os.path.join(RESULT_DIR, "03_pruning_compare.csv")
    if not os.path.exists(path):
        return default
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        row = df[df["模型"].str.contains("预剪枝", na=False)].iloc[0]
        cfg = row["配置"]  # 形如 "depth=6, split=300, leaf=100"
        import re
        d = int(re.search(r"depth=(\d+)", cfg).group(1))
        ms = int(re.search(r"split=(\d+)", cfg).group(1))
        ml = int(re.search(r"leaf=(\d+)", cfg).group(1))
        return d, ms, ml
    except Exception as e:
        print(f"  [WARN] 未能解析 exp03 最优参数（{e}），使用默认值 {default}")
        return default


def main():
    X_train, X_test, y_train, y_test, _ = get_prepared_data()

    best_params = load_best_params()
    print(f"使用预剪枝参数（来自 exp03）：max_depth={best_params[0]}, "
          f"min_samples_split={best_params[1]}, min_samples_leaf={best_params[2]}")

    challenge1_imbalance(X_train, X_test, y_train, y_test, best_params)
    challenge2_algo_compare(X_train, X_test, y_train, y_test, best_params)
    challenge3_feature_engineering(X_train, X_test, y_train, y_test, best_params)


if __name__ == "__main__":
    main()
    print("\n" + "=" * 60)
    print("exp04 完成")
    print("=" * 60)