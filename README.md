# marketing_model

# ===== 0. 必要ライブラリ =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font="IPAexGothic")  # 日本語対応（IPAexフォント）

# ===== 1. 基本情報の確認 =====
def basic_info(df):
    print("=== データ概要 ===")
    print(df.shape)
    print("\n=== カラムごとの型 ===")
    print(df.dtypes)
    print("\n=== 欠損値数 ===")
    print(df.isnull().sum())
    print("\n=== 数値変数の基本統計量 ===")
    display(df.describe())
    print("\n=== カテゴリ変数の基本統計量 ===")
    display(df.describe(include=["object", "category"]))

# ===== 2. 数値変数の分布とTARGET別比較 =====
def plot_numeric_distribution(df, num_col, target_col):
    plt.figure(figsize=(8,5))
    sns.kdeplot(data=df, x=num_col, hue=target_col, common_norm=False, fill=True)
    plt.title(f"{num_col} 分布（by {target_col}）")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.boxplot(x=target_col, y=num_col, data=df)
    plt.title(f"{num_col} vs {target_col}（箱ひげ図）")
    plt.show()

# ===== 3. カテゴリ変数のTARGET別割合比較 =====
def plot_categorical_rate(df, cat_col, target_col):
    rate_df = df.groupby(cat_col)[target_col].mean().reset_index()
    plt.figure(figsize=(8,5))
    sns.barplot(x=cat_col, y=target_col, data=rate_df, order=rate_df.sort_values(target_col, ascending=False)[cat_col])
    plt.xticks(rotation=45)
    plt.title(f"{cat_col}別 {target_col}率")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.countplot(x=cat_col, hue=target_col, data=df)
    plt.xticks(rotation=45)
    plt.title(f"{cat_col}別 件数（by {target_col}）")
    plt.show()

# ===== 4. クロス集計のヒートマップ =====
def plot_crosstab_heatmap(df, col1, col2, target_col):
    ct = pd.crosstab(df[col1], df[col2], values=df[target_col], aggfunc="mean")
    plt.figure(figsize=(10,6))
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"{col1}×{col2} 別 {target_col}率")
    plt.show()

# ===== 5. 相関ヒートマップ（数値変数のみ） =====
def plot_corr_heatmap(df, numeric_cols):
    plt.figure(figsize=(10,8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("数値変数の相関")
    plt.show()
     
