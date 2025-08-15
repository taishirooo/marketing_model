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


わかりました。
これまで出てきた変数をもとに、**特徴量設計（Feature Engineering）** の例をまとめて、明日のインターンですぐ試せる形にします。
目的は「Olive開設率（`TARGET_FRG`）に寄与しそうな説明変数を派生変数として追加する」ことです。

---

## 1. 元の変数一覧（これまで出てきたもの）

* **基本情報**

  * AGE（年齢）
  * GENDER（性別）
  * JOB\_TYPE（職種／欠損あり）
* **金融情報**

  * BAL1〜BAL5（各残高）
  * SMBC\_CARD\_COUNT（SMBCカード枚数）
  * OTHER\_ACCOUNT\_COUNT（他の金融口座数）
* **利用状況**

  * APP\_USE\_FREQ（アプリ利用頻度）
  * ATM\_USE\_FREQ（ATM利用頻度）
  * HOUSEHOLD\_HEAD（世帯主フラグ）
  * PAYROLL\_DEP, PENSION\_RECV（給与振込・年金受取）
  * WATER\_PAY, NHK\_PAY, PHONE\_PAY, INSURANCE\_PAY（公共料金など引落し）
* **住所情報**

  * PREFECTURE, CITY, BRANCH（都道府県、市町村、支店）

---

## 2. 特徴量設計例

### (A) 金融資産関連

```python
df["TOTAL_BAL"] = df[["BAL1","BAL2","BAL3","BAL4","BAL5"]].sum(axis=1)  # 総残高
df["AVG_BAL"] = df[["BAL1","BAL2","BAL3","BAL4","BAL5"]].mean(axis=1)   # 平均残高
df["MAX_BAL"] = df[["BAL1","BAL2","BAL3","BAL4","BAL5"]].max(axis=1)    # 最大残高
df["BAL_VARIANCE"] = df[["BAL1","BAL2","BAL3","BAL4","BAL5"]].var(axis=1)  # 資産の分散（分散投資傾向）
```

---

### (B) 利用習慣関連

```python
# SMBC関連サービス利用数
service_flags = ["PAYROLL_DEP","PENSION_RECV","WATER_PAY","NHK_PAY","PHONE_PAY","INSURANCE_PAY"]
df["SMBC_USE_COUNT"] = df[service_flags].sum(axis=1)

# 総取引活動スコア（アプリ利用頻度＋ATM利用頻度）
df["TOTAL_ACTIVITY"] = df["APP_USE_FREQ"] + df["ATM_USE_FREQ"]

# 他口座・カード合計
df["TOTAL_ACCOUNT_CARD"] = df["SMBC_CARD_COUNT"] + df["OTHER_ACCOUNT_COUNT"]
```

---

### (C) 年齢カテゴリ

```python
df["AGE_GROUP"] = pd.cut(df["AGE"],
                         bins=[0,29,39,49,59,69,100],
                         labels=["~29","30-39","40-49","50-59","60-69","70~"])
```

* カテゴリ変数として扱うことで、非線形な年齢効果を捉える

---

### (D) 地域関連

```python
# 支店単位の開設率（過去データベースから計算できる場合）
branch_rate = df.groupby("BRANCH")["TARGET_FRG"].mean()
df["BRANCH_TARGET_RATE"] = df["BRANCH"].map(branch_rate)

# 都道府県単位の平均残高
pref_bal = df.groupby("PREFECTURE")["TOTAL_BAL"].mean()
df["PREF_AVG_BAL"] = df["PREFECTURE"].map(pref_bal)
```

* 地域ごとの傾向をモデルに反映できる

---

### (E) 職業関連

```python
# 欠損を「不明」に統一
df["JOB_TYPE"] = df["JOB_TYPE"].fillna("不明")

# 職種ごとの開設率（Target Encoding）
job_rate = df.groupby("JOB_TYPE")["TARGET_FRG"].mean()
df["JOB_TYPE_RATE"] = df["JOB_TYPE"].map(job_rate)
```

* LightGBMでは`JOB_TYPE`を`category`型で直接渡すことも可能
* 他モデルでは`JOB_TYPE_RATE`を数値特徴量として使える

---

## 3. 特徴量設計テンプレコード（まとめ）

```python
def create_features(df):
    # 金融資産関連
    df["TOTAL_BAL"] = df[["BAL1","BAL2","BAL3","BAL4","BAL5"]].sum(axis=1)
    df["AVG_BAL"] = df[["BAL1","BAL2","BAL3","BAL4","BAL5"]].mean(axis=1)
    df["MAX_BAL"] = df[["BAL1","BAL2","BAL3","BAL4","BAL5"]].max(axis=1)
    df["BAL_VARIANCE"] = df[["BAL1","BAL2","BAL3","BAL4","BAL5"]].var(axis=1)

    # 利用習慣関連
    service_flags = ["PAYROLL_DEP","PENSION_RECV","WATER_PAY","NHK_PAY","PHONE_PAY","INSURANCE_PAY"]
    df["SMBC_USE_COUNT"] = df[service_flags].sum(axis=1)
    df["TOTAL_ACTIVITY"] = df["APP_USE_FREQ"] + df["ATM_USE_FREQ"]
    df["TOTAL_ACCOUNT_CARD"] = df["SMBC_CARD_COUNT"] + df["OTHER_ACCOUNT_COUNT"]

    # 年齢カテゴリ
    df["AGE_GROUP"] = pd.cut(df["AGE"],
                             bins=[0,29,39,49,59,69,100],
                             labels=["~29","30-39","40-49","50-59","60-69","70~"])

    # 地域関連
    branch_rate = df.groupby("BRANCH")["TARGET_FRG"].transform("mean")
    df["BRANCH_TARGET_RATE"] = branch_rate
    pref_bal = df.groupby("PREFECTURE")["TOTAL_BAL"].transform("mean")
    df["PREF_AVG_BAL"] = pref_bal

    # 職業関連
    df["JOB_TYPE"] = df["JOB_TYPE"].fillna("不明")
    job_rate = df.groupby("JOB_TYPE")["TARGET_FRG"].transform("mean")
    df["JOB_TYPE_RATE"] = job_rate

    return df
```

---

この関数を呼び出せば、EDAやモデリングにすぐ使える**派生変数が一括生成**できます。
もし希望なら、この関数を組み込んだ「**前処理＋特徴量作成＋LightGBM学習の一括ノート**」にまとめられます。
そうすれば明日のインターンで即分析に入れますが、その形にしますか？

了解です。
では、**欠損値補完用の汎用テンプレート** を作ります。
このテンプレは、数値変数・カテゴリ変数の両方に対応し、**平均値 / 中央値 / 最頻値** での補完方法を簡単に切り替えられます。

---

## 欠損値補完テンプレート

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def impute_missing_values(df, numeric_cols=None, categorical_cols=None,
                          num_strategy="mean", cat_strategy="most_frequent"):
    """
    欠損値を数値列とカテゴリ列で分けて補完する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        補完対象のデータフレーム
    numeric_cols : list
        数値型カラム名リスト
    categorical_cols : list
        カテゴリ型カラム名リスト
    num_strategy : str
        数値の補完方法 ("mean", "median", "most_frequent", "constant")
    cat_strategy : str
        カテゴリの補完方法 ("most_frequent", "constant")
    
    Returns
    -------
    pd.DataFrame : 補完後のデータフレーム
    """
    df_copy = df.copy()

    # ===== 数値列の補完 =====
    if numeric_cols:
        num_imputer = SimpleImputer(strategy=num_strategy)
        df_copy[numeric_cols] = num_imputer.fit_transform(df_copy[numeric_cols])

    # ===== カテゴリ列の補完 =====
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=cat_strategy)
        df_copy[categorical_cols] = cat_imputer.fit_transform(df_copy[categorical_cols])

    return df_copy
```

---

## 使用例

```python
# 数値・カテゴリ列の指定
numeric_cols = ["AGE", "BAL1", "BAL2", "BAL3", "BAL4", "BAL5", "SMBC_CARD_COUNT"]
categorical_cols = ["GENDER", "HOUSEHOLD_HEAD", "JOB_TYPE", "PREFECTURE"]

# 平均値で数値補完、最頻値でカテゴリ補完
df_filled = impute_missing_values(df, numeric_cols, categorical_cols,
                                  num_strategy="mean", cat_strategy="most_frequent")

# 中央値で数値補完
df_filled_median = impute_missing_values(df, numeric_cols, categorical_cols,
                                         num_strategy="median", cat_strategy="most_frequent")

# 特定値で補完（例：数値は0、カテゴリは"不明"）
df_filled_constant = impute_missing_values(df, numeric_cols, categorical_cols,
                                           num_strategy="constant", cat_strategy="constant")
```

---

## 補足

* **平均値補完**：外れ値に弱いが、分布の中心を維持できる
* **中央値補完**：外れ値に強い（スキューのある変数に有効）
* **最頻値補完**：カテゴリや2値変数で有効
* **定数補完**：分析上、欠損を明示的に残したい場合に有効（例："不明"カテゴリ）

---

この関数を使えば、**変数リストと戦略を指定するだけで欠損補完が一括**でできます。
もし希望なら、この欠損補完を **EDAテンプレートの前処理ステップ** に組み込んだ「前処理付きEDA版」にできます。
そうするとインターン中、データ読み込み後すぐに欠損補完→EDA→仮説検証まで1本で流せますが、それにしますか？

