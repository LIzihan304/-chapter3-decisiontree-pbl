# -*- coding: utf-8 -*-
"""
utils.py —— 决策树 PBL 项目共享工具模块
========================================
提供数据加载、清洗、编码、划分等全流程公共函数，
供各实验脚本与 Streamlit 应用复用。

数据集：Telco Customer Churn（电信客户流失）
  7043 条样本，21 列（1 个 ID + 19 个特征 + 1 个标签）
  标签 Churn：No=留存（0），Yes=流失（1）
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============== 路径配置 ==============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
IMG_DIR = os.path.join(BASE_DIR, "images")
RAW_PATH = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")

for _d in (DATA_DIR, RESULT_DIR, IMG_DIR):
    os.makedirs(_d, exist_ok=True)

# ============== 常量定义 ==============
TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.3

# 二分类映射（将 3 值类别压缩为 2 值，避免稀疏列）
BINARY_MAP_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# 类别特征清单（编码前）
CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


# ============================================================================
# 1. 数据加载
# ============================================================================
def load_raw_data():
    """加载原始 CSV"""
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"数据集不存在：{RAW_PATH}\n"
            "请从 Kaggle 下载 Telco Customer Churn，或从 "
            "https://github.com/IBM/telco-customer-churn-on-icp4d 获取。"
        )
    return pd.read_csv(RAW_PATH)


# ============================================================================
# 2. 数据清洗
# ============================================================================
def clean_data(df):
    """
    数据清洗：
      1. 剔除 customerID（唯一标识，无泛化价值；教科书中明确提示高基数特征会造成伪信息增益）
      2. TotalCharges 由 object 转 float（原始数据含空格字符串）
      3. TotalCharges 空值用 tenure × MonthlyCharges 填补（业务逻辑：消费总额≈月费×在网时长）
      4. 三值类别特征压缩为二值（No / Yes），减少稀疏分支
      5. 标签 Churn 映射为 0/1
    返回：清洗后的 df, 清洗记录 list
    """
    df = df.copy()
    logs = []

    # --- 2.1 剔除唯一标识 ---
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
        logs.append(("无效特征剔除", "删除 customerID",
                     f"{len(df)} 条",
                     "客户唯一标识，高基数且无泛化价值，计入会造成伪信息增益"))

    # --- 2.2 TotalCharges 类型转换 ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_before = df["TotalCharges"].isnull().sum()
    if n_before > 0:
        # 用 tenure * MonthlyCharges 填补（这类样本 tenure 均为 0）
        filled = df["tenure"] * df["MonthlyCharges"]
        df["TotalCharges"] = df["TotalCharges"].fillna(filled)
        logs.append(("缺失值处理", "按 tenure×MonthlyCharges 填补",
                     f"{n_before} 条 ({n_before/len(df)*100:.2f}%)",
                     "原始 TotalCharges 含空格字符串；该类样本 tenure=0，消费总额应按月费推算"))

    # --- 2.3 三值类别压缩为二值 ---
    compressed = []
    for col in BINARY_MAP_COLS:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No",
                                       "No phone service": "No"})
            compressed.append(col)
    if compressed:
        logs.append(("异常值处理", "三值类别压缩为二值",
                     f"{len(compressed)} 个特征",
                     "No internet service / No phone service 业务上等价于 No，"
                     "保留三值会产生稀疏分支"))

    # --- 2.4 标签映射 ---
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)

    # --- 2.5 重复样本检查 ---
    n_dup = df.duplicated().sum()
    logs.append(("重复样本处理", "保留" if n_dup == 0 else "删除",
                 f"{n_dup} 条",
                 "无重复样本" if n_dup == 0 else "完全重复的行已剔除"))
    if n_dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    return df, logs


# ============================================================================
# 3. 特征编码
# ============================================================================
def encode_features(df):
    """
    类别特征编码：采用「手动二值映射 + 独热编码」组合策略
      - 二值特征（Yes/No, Male/Female）→ 直接 0/1 映射，保留可解释性
      - 多值特征（Contract, PaymentMethod, InternetService）→ 独热编码
    决策树不需要标准化，但需要数值输入。
    """
    df = df.copy()

    # 二值映射
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
                   "PaperlessBilling"] + BINARY_MAP_COLS
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    # 多值独热
    multi_cols = [c for c in ["InternetService", "Contract", "PaymentMethod"]
                  if c in df.columns]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True, dtype=int)

    return df


# ============================================================================
# 4. 数据划分
# ============================================================================
def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    7:3 分层划分（stratify=y）
    返回 X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ============================================================================
# 5. 一键预处理
# ============================================================================
def get_prepared_data(test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """完整流程：加载 → 清洗 → 编码 → 划分"""
    raw = load_raw_data()
    clean, logs = clean_data(raw)
    encoded = encode_features(clean)
    X_train, X_test, y_train, y_test = split_data(encoded, test_size, random_state)
    return X_train, X_test, y_train, y_test, logs


# ============================================================================
# 6. 评估工具
# ============================================================================
def evaluate(model, X_train, X_test, y_train, y_test):
    """计算训练集与测试集的完整指标"""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)

    y_tr_pred = model.predict(X_train)
    y_te_pred = model.predict(X_test)
    y_te_prob = model.predict_proba(X_test)[:, 1]

    return {
        "训练集准确率": round(accuracy_score(y_train, y_tr_pred), 4),
        "测试集准确率": round(accuracy_score(y_test, y_te_pred), 4),
        "精确率": round(precision_score(y_test, y_te_pred, zero_division=0), 4),
        "召回率": round(recall_score(y_test, y_te_pred, zero_division=0), 4),
        "F1值": round(f1_score(y_test, y_te_pred, zero_division=0), 4),
        "AUC": round(roc_auc_score(y_test, y_te_prob), 4),
        "叶节点数": int(model.get_n_leaves()) if hasattr(model, "get_n_leaves") else -1,
        "树深度": int(model.get_depth()) if hasattr(model, "get_depth") else -1,
        "过拟合差值": round(
            accuracy_score(y_train, y_tr_pred) - accuracy_score(y_test, y_te_pred), 4
        ),
    }


def save_csv(df, name):
    """保存结果 CSV（UTF-8 BOM，兼容 Excel）"""
    path = os.path.join(RESULT_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [SAVED] {path}")
    return path


def setup_chinese_font():
    """matplotlib 中文字体"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt