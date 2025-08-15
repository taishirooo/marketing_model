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


私なら、今回の仮説検証の流れを踏まえて、
**「単変量で開設率に差が出た特徴」＋「仮説を数値化した派生変数」** を意識してロジスティック回帰の説明変数に組み込みます。
特にロジスティック回帰は線形モデルなので、変数の作り方次第で予測力と解釈力が大きく変わります。

---

## 1. 仮説と特徴量の対応表

| 仮説              | 特徴量（説明変数）例                                      | 理由                         |
| --------------- | ----------------------------------------------- | -------------------------- |
| ① アプリ利用頻度×若年層   | `APP_USE_FREQ`、`AGE`、`APP_USE_FREQ × (AGE<=30)` | 若年層×高頻度利用者が高開設率なら交互作用項で捉える |
| ② SMBC利用頻度が高い顧客 | `SMBC_USE_COUNT`（給与振込、年金、公共料金などの合計フラグ数）         | サービス利用度が高い人はOlive開設意欲が高い   |
| ③ 残高が多い富裕層      | `TOTAL_BAL`、`log(TOTAL_BAL+1)`                  | 総残高をスケール調整してモデルに入れる        |
| ④ 特典活用層         | `TOTAL_SERVICE_COUNT`（引落＋受取＋アプリ＋ATMなどの合計）       | 多機能利用度を代理指標化               |
| ⑤ 投資家・ビジネスマン    | `JOB_TYPE`（カテゴリ型）＋`INVEST_BAL`（投資口座残高）          | 職種と投資傾向を反映                 |
| ⑥ 三井住友カード保有数    | `SMBC_CARD_COUNT`、`CARD_COUNT_BIN`（枚数カテゴリ）      | 枚数と開設率の非線形関係を捉える           |
| ⑦ ATMサービス利用者    | `ATM_USE_FREQ`                                  | 利用頻度と開設意欲の関連性              |

---

## 2. 実際に追加したい説明変数

### (A) 金融資産関連

```python
df["TOTAL_BAL"] = df[["BAL1","BAL2","BAL3","BAL4","BAL5"]].sum(axis=1)
df["LOG_TOTAL_BAL"] = np.log1p(df["TOTAL_BAL"])  # スケーリングして非線形対応
df["INVEST_BAL"] = df["BAL4"] + df["BAL5"]       # 投資口座系残高の合計
```

---

### (B) 利用習慣関連

```python
service_flags = ["PAYROLL_DEP","PENSION_RECV","WATER_PAY","NHK_PAY","PHONE_PAY","INSURANCE_PAY"]
df["SMBC_USE_COUNT"] = df[service_flags].sum(axis=1)
df["TOTAL_SERVICE_COUNT"] = df["SMBC_USE_COUNT"] + df["APP_USE_FREQ"] + df["ATM_USE_FREQ"]
```

---

### (C) 属性カテゴリ化

```python
df["AGE_GROUP"] = pd.cut(df["AGE"],
                         bins=[0,29,39,49,59,69,100],
                         labels=["~29","30-39","40-49","50-59","60-69","70~"])
df["CARD_COUNT_BIN"] = pd.cut(df["SMBC_CARD_COUNT"], bins=[-1,1,2,3,10], labels=["0-1","2","3","4+"])
df["JOB_TYPE"] = df["JOB_TYPE"].fillna("不明")
```

---

### (D) 交互作用（仮説①）

```python
df["YOUNG_APP_ACTIVE"] = ((df["AGE"] <= 30) & (df["APP_USE_FREQ"] >= df["APP_USE_FREQ"].median())).astype(int)
```

* 若年層かつアプリ利用頻度高い層を直接1/0で表現

---

### (E) 地域効果（仮説からの裏付け）

```python
branch_rate = df.groupby("BRANCH")["TARGET_FRG"].transform("mean")
df["BRANCH_TARGET_RATE"] = branch_rate
```

* 支店ごとの過去開設率を数値化（集計は学習データのみで行う）

---

## 3. モデルに入れる際のポイント

* ロジスティック回帰は **線形＋単調な関係を想定** するので、金額や頻度は `log1p()` やビニングで変換して非線形性に対応
* カテゴリ変数は One-Hot または `category` 型＋エンコーディング
* 交互作用は `×` ではなく、新しい列として事前に作る
* 欠損値は適切に補完（職種は"不明"、金額系は0など）

---

## 4. 推奨する説明変数セット（例）

```python
explanatory_vars = [
    # 金融資産
    "LOG_TOTAL_BAL", "INVEST_BAL",
    # 利用習慣
    "SMBC_USE_COUNT", "TOTAL_SERVICE_COUNT",
    # 属性
    "AGE", "AGE_GROUP", "GENDER", "JOB_TYPE",
    "CARD_COUNT_BIN", "SMBC_CARD_COUNT",
    # 利用頻度
    "APP_USE_FREQ", "ATM_USE_FREQ",
    # 交互作用
    "YOUNG_APP_ACTIVE",
    # 地域
    "BRANCH_TARGET_RATE"
]
```

---

もしよければ、この特徴量生成をまとめて行い、そのままロジスティック回帰に投入できる **前処理＋学習の一括コード** にして渡せます。
そうすれば明日のインターンでデータを流すだけで検証→モデル構築まで一気にできますが、その形にしますか？

いいね、ロジスティック回帰に進みましょう。
下は **前処理 → 学習 → 評価 → 係数の解釈 → 予測の保存** まで一発で回るテンプレです。
（`numeric_cols` / `categorical_cols` はあなたの環境のリストをそのまま使ってください）

```python
# === 0) ライブラリ ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer

# === 1) 入力（あなたの df_train / 列名を使用） ===
target_col = "TARGET_FRG"
# numeric_cols = [...]
# categorical_cols = [...]

X = df_train[numeric_cols + categorical_cols].copy()
y = df_train[target_col].astype(int).copy()

# === 2) 前処理: 欠損補完 + スケーリング + One-Hot ===
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # 平均なら "mean"
    ("scaler",  StandardScaler())
])

# 高カードinalityに強い one-hot（頻度の低いカテゴリはまとめる）
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ],
    remainder="drop"
)

# === 3) モデル本体（不均衡対応の重み付け） ===
logreg = LogisticRegression(
    penalty="l2",            # L1にすると特徴選択にも効く：'l1', solver='liblinear' or saga
    class_weight="balanced", # 不均衡に強くする
    max_iter=2000,
    n_jobs=-1,
    solver="lbfgs"           # 高次元でも安定
)

# === 4) パイプライン ===
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", logreg)
])

# === 5) ホールドアウト評価 ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_val)[:, 1]
pred  = (proba >= 0.5).astype(int)

metrics = {
    "accuracy": accuracy_score(y_val, pred),
    "precision": precision_score(y_val, pred),
    "recall": recall_score(y_val, pred),
    "f1": f1_score(y_val, pred),
    "roc_auc": roc_auc_score(y_val, proba),
    "pr_auc": average_precision_score(y_val, proba),
}
print({k: round(v, 4) for k, v in metrics.items()})
print("\nConfusion matrix:\n", confusion_matrix(y_val, pred))
print("\nReport:\n", classification_report(y_val, pred, digits=4))

# === 6) 5Fold CV（AUCの安定度チェック） ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"\nCV ROC-AUC: mean={cv_auc.mean():.4f}, std={cv_auc.std():.4f}, folds={cv_auc}")

# === 7) 係数の解釈（オッズ比）
#  前処理後の特徴名を抽出 → 係数をオッズ比に変換
pipe.fit(X_train, y_train)
# ColumnTransformer経由の特徴名
ohe = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["ohe"]
cat_features = ohe.get_feature_names_out(categorical_cols)
num_features = np.array(numeric_cols)
feature_names = np.concatenate([num_features, cat_features])

coefs = pipe.named_steps["clf"].coef_.ravel()
odds_ratio = np.exp(coefs)
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs,
    "odds_ratio": odds_ratio
}).sort_values("odds_ratio", ascending=False)
print("\nTop positive (odds_ratio):")
print(coef_df.head(15))
print("\nTop negative (odds_ratio):")
print(coef_df.tail(15))

# === 8) アンサンブル用にバリデーション予測を保存 ===
oof_df = pd.DataFrame({
    "y_true": y_val.reset_index(drop=True),
    "lr_pred_proba": pd.Series(proba).reset_index(drop=True),
    "lr_pred_label": pd.Series(pred).reset_index(drop=True),
})
oof_df.to_csv("logreg_val_predictions.csv", index=False)

# パイプラインをそのまま保存（joblib推奨）
import joblib
joblib.dump(pipe, "logreg_pipeline.joblib")
```

### 使い方・ポイント

* **不均衡対応**：`class_weight="balanced"` を指定済み。必要なら閾値0.5を調整して F1 / Recall を改善してもOK。
* **高次元One-Hot**：`min_frequency=20` でレアカテゴリを自動集約（列爆発を抑制）。
* **係数の解釈**：`odds_ratio = exp(coef)`。1より大きければ開設確率を押し上げる方向、小さければ下げる方向。
* **アンサンブル準備**：OOF（検証）予測をCSVに保存。後で LightGBM / XGBoost のOOFと横持ちしてブレンディング可能。

> さらに精度を上げたい場合
>
> * 罰則を **L1（Lasso）** に変更してスパース化（特徴選択）
> * `solver="saga"` にして大規模One-Hotでも高速化
> * グリッド/ベイズ最適化で `C`（正則化強度）を調整

必要なら、このロジスティックとLightGBM/XGBoost/CatBoostの **共通OOF作成スクリプト** も用意できます。

