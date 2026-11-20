# -*- coding: utf-8 -*-
"""
exp02_baseline.py —— 数据预处理与基准模型
==========================================
对应报告 3.2 数据预处理、3.3 基础任务实现、表 4-1 基础任务结果

内容：
  1. 清洗记录导出（表 3-1）
  2. 编码后特征清单（表 3-3）
  3. 无约束基准决策树（criterion=gini，不剪枝）
  4. 训练/测试指标 + 复杂度 + 混淆矩阵
输出：
  results/02_cleaning_log.csv
  results/02_feature_list.csv
  results/02_split_info.csv
  results/02_baseline_metrics.csv
  results/02_confusion_matrix.csv
  images/02_baseline_tree.png
  images/02_confusion_matrix.png
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report

from utils import (load_raw_data, clean_data, encode_features, split_data,
                   evaluate, save_csv, setup_chinese_font, IMG_DIR,
                   RESULT_DIR, TARGET)


def main():
    print("=" * 60)
    print("【实验2】数据预处理与基准决策树")
    print("=" * 60)

    # ---------- 1. 清洗 ----------
    raw = load_raw_data()
    clean, logs = clean_data(raw)

    print(f"\n原始数据：{raw.shape[0]} 行 × {raw.shape[1]} 列")
    print(f"清洗后：  {clean.shape[0]} 行 × {clean.shape[1]} 列（已剔除 customerID）")

    log_df = pd.DataFrame(logs, columns=["清洗项目", "处理方式", "处理数量/比例", "说明"])
    save_csv(log_df, "02_cleaning_log.csv")
    print("\n清洗记录：")
    for _, r in log_df.iterrows():
        print(f"  [{r['清洗项目']}] {r['处理方式']} —— {r['处理数量/比例']}")

    # ---------- 2. 编码 ----------
    encoded = encode_features(clean)
    print(f"\n编码后特征数：{encoded.shape[1] - 1} 维")
    print(f"特征列表：{list(encoded.drop(columns=[TARGET]).columns)}")

    feat_rows = []
    for i, col in enumerate(encoded.drop(columns=[TARGET]).columns, 1):
        feat_rows.append({
            "序号": i,
            "编码后特征名": col,
            "数据类型": str(encoded[col].dtype),
            "取值范围": f"[{encoded[col].min()}, {encoded[col].max()}]",
            "均值": round(float(encoded[col].mean()), 4),
        })
    save_csv(pd.DataFrame(feat_rows), "02_feature_list.csv")

    # ---------- 3. 划分 ----------
    X_train, X_test, y_train, y_test = split_data(encoded)
    split_df = pd.DataFrame([
        {"项目": "训练集", "样本数": len(y_train),
         "留存(0)": int((y_train == 0).sum()), "流失(1)": int((y_train == 1).sum()),
         "流失率%": round(y_train.mean() * 100, 2)},
        {"项目": "测试集", "样本数": len(y_test),
         "留存(0)": int((y_test == 0).sum()), "流失(1)": int((y_test == 1).sum()),
         "流失率%": round(y_test.mean() * 100, 2)},
    ])
    save_csv(split_df, "02_split_info.csv")
    print(f"\n数据集划分（7:3 分层）：训练 {len(y_train)} 条，测试 {len(y_test)} 条")

    # ---------- 4. 基准模型：无约束 ----------
    print("\n" + "-" * 50)
    print("训练无约束基准决策树（criterion=gini, max_depth=None）")
    base = DecisionTreeClassifier(criterion="gini", random_state=42)
    base.fit(X_train, y_train)
    m = evaluate(base, X_train, X_test, y_train, y_test)

    print(f"  训练集准确率：{m['训练集准确率']:.4f}")
    print(f"  测试集准确率：{m['测试集准确率']:.4f}")
    print(f"  精确率：      {m['精确率']:.4f}")
    print(f"  召回率：      {m['召回率']:.4f}")
    print(f"  F1值：        {m['F1值']:.4f}")
    print(f"  AUC：         {m['AUC']:.4f}")
    print(f"  叶节点数：    {m['叶节点数']}")
    print(f"  树深度：      {m['树深度']}")
    print(f"  过拟合差值：  {m['过拟合差值']:.4f}")

    metrics_df = pd.DataFrame([{
        "模型": "基准模型（无约束 gini）",
        **m,
    }])
    save_csv(metrics_df, "02_baseline_metrics.csv")

    # ---------- 5. 混淆矩阵 ----------
    y_pred = base.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_df = pd.DataFrame([
        {"指标": "TN（留存判对）", "数值": int(tn)},
        {"指标": "FP（留存误判流失）", "数值": int(fp)},
        {"指标": "FN（流失漏判）", "数值": int(fn)},
        {"指标": "TP（流失判对）", "数值": int(tp)},
        {"指标": "召回率", "数值": round(tp / (tp + fn), 4)},
        {"指标": "精确率", "数值": round(tp / (tp + fp), 4)},
    ])
    save_csv(cm_df, "02_confusion_matrix.csv")

    print(f"\n混淆矩阵：TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  → 漏判 {fn} 个流失客户（占真实流失的 {fn/(fn+tp)*100:.1f}%），"
          f"这是业务最需要关注的损失")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=["留存", "流失"]))

    # ---------- 6. 可视化 ----------
    plt = setup_chinese_font()

    # 混淆矩阵
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测：留存", "预测：流失"])
    ax.set_yticklabels(["真实：留存", "真实：流失"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"基准模型混淆矩阵\n测试准确率={m['测试集准确率']:.4f}, "
                 f"召回率={m['召回率']:.4f}", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "02_confusion_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # 树结构（只画前 3 层，完整树太大）
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(base, feature_names=list(X_train.columns), class_names=["留存", "流失"],
              filled=True, rounded=True, fontsize=7, max_depth=3, ax=ax)
    ax.set_title(f"基准决策树（显示前 3 层，完整树深度={m['树深度']}，"
                 f"叶节点={m['叶节点数']}）", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "02_baseline_tree.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\n[SAVED] images/02_baseline_tree.png")
    print(f"[SAVED] images/02_confusion_matrix.png")

    print("\n【结论】基准树训练集准确率接近 1.0，测试集仅 "
          f"{m['测试集准确率']:.4f}，差值 {m['过拟合差值']:.4f}，"
          f"树深度 {m['树深度']}、叶节点 {m['叶节点数']} 个，"
          "属于典型过拟合，必须剪枝。")

    return m


if __name__ == "__main__":
    main()
    print("\n" + "=" * 60)
    print("exp02 完成")
    print("=" * 60)