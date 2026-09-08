# Full EDA Process — Complete Reference

Every step, check, and operation in a thorough exploratory data analysis, with
the code that implements it. This is the source material for the EDA MCP
server's tool design: each section maps to something the server must be able
to do.

**Legend:** 🔍 analysis (read-only) · ✏️ mutation (write) · 📊 visualisation

---

## 0. Setup

```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
```

---

## 1. Loading 🔍

### 1.1 Formats

```python
df = pd.read_csv("data.csv")
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df = pd.read_parquet("data.parquet")
df = pd.read_json("data.json", lines=True)
df = pd.read_sql("SELECT * FROM t", con)
df = pd.read_csv("data.tsv", sep="\t")
df = pd.read_clipboard()
```

### 1.2 Reading awkward files

```python
# Encoding problems
df = pd.read_csv(f, encoding="utf-8")        # try first
df = pd.read_csv(f, encoding="latin-1")      # fallback
df = pd.read_csv(f, encoding="cp1252")       # Windows exports

# Detect it instead of guessing
import chardet
enc = chardet.detect(open(f, "rb").read(100_000))["encoding"]

# Unknown delimiter
df = pd.read_csv(f, sep=None, engine="python")   # sniff

# Messy structure
df = pd.read_csv(f, skiprows=3)                  # junk header rows
df = pd.read_csv(f, header=None, names=cols)     # no header
df = pd.read_csv(f, skipfooter=2, engine="python")
df = pd.read_csv(f, thousands=",", decimal=".")
df = pd.read_csv(f, na_values=["", "NA", "N/A", "null", "-", "?", "missing"])
df = pd.read_csv(f, true_values=["yes","Y"], false_values=["no","N"])
df = pd.read_csv(f, parse_dates=["date"], dayfirst=True)
df = pd.read_csv(f, dtype={"zip": str})          # preserve leading zeros
df = pd.read_csv(f, usecols=["a","b"])           # subset for speed
df = pd.read_csv(f, nrows=1000)                  # peek
```

### 1.3 Large files

```python
# Chunked
for chunk in pd.read_csv(f, chunksize=100_000):
    process(chunk)

# Memory-efficient dtypes on load
df = pd.read_csv(f, dtype={"category_col": "category"}, engine="pyarrow")
```

---

## 2. First look 🔍

```python
df.shape                    # (rows, cols)
df.head(10); df.tail(10); df.sample(10)
df.info(memory_usage="deep")
df.dtypes
df.columns.tolist()
df.index
len(df); df.size
df.memory_usage(deep=True).sum() / 1024**2      # MB
df.describe()                                    # numeric
df.describe(include="object")                    # categorical
df.describe(include="all")
df.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
```

**Questions to answer here:** What does one row represent? What is the grain?
Is there a natural key? Which column is the target?

---

## 3. Types 🔍 ✏️

### 3.1 Classify every column

```python
numeric  = df.select_dtypes(include=np.number).columns.tolist()
cats     = df.select_dtypes(include=["object","category"]).columns.tolist()
dates    = df.select_dtypes(include="datetime").columns.tolist()
bools    = df.select_dtypes(include="bool").columns.tolist()
```

### 3.2 Detect wrong types

```python
# Numbers stored as text
for c in cats:
    converted = pd.to_numeric(df[c], errors="coerce")
    if converted.notna().mean() > 0.9:
        print(f"{c}: {converted.notna().mean():.0%} numeric — likely mistyped")

# Dates stored as text
for c in cats:
    parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
    if parsed.notna().mean() > 0.9:
        print(f"{c}: parses as datetime")

# Mixed types in one column
for c in cats:
    kinds = df[c].dropna().map(type).value_counts()
    if len(kinds) > 1:
        print(f"{c}: mixed types {dict(kinds)}")

# Identifier columns (unique per row)
for c in df.columns:
    if df[c].nunique() == len(df):
        print(f"{c}: unique per row — identifier, not a feature")

# Constant / near-constant
for c in df.columns:
    top = df[c].value_counts(normalize=True, dropna=False).iloc[0]
    if top > 0.99:
        print(f"{c}: {top:.1%} single value — no signal")
```

### 3.3 Convert ✏️

```python
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["d"] = pd.to_datetime(df["d"], errors="coerce", format="%Y-%m-%d")
df["c"] = df["c"].astype("category")
df["i"] = df["i"].astype("Int64")          # nullable integer
df["b"] = df["b"].astype(bool)
df["s"] = df["s"].astype("string")         # nullable string

# Downcast to save memory
df["n"] = pd.to_numeric(df["n"], downcast="integer")
df["f"] = pd.to_numeric(df["f"], downcast="float")
```

---

## 4. Missing data 🔍 ✏️

### 4.1 Detect

```python
df.isna().sum()
df.isna().mean().sort_values(ascending=False)      # proportion
df.isna().sum().sum()                              # total
df.isna().any(axis=1).sum()                        # rows with any NA
df[df.isna().any(axis=1)]                          # inspect them
df.notna().sum()

# Disguised missing values
for c in cats:
    suspicious = df[c].isin(["", " ", "NA", "N/A", "null", "None",
                             "-", "?", "unknown", "missing", "nan"])
    if suspicious.any():
        print(f"{c}: {suspicious.sum()} disguised missing")

# Sentinel values in numerics
for c in numeric:
    for sentinel in (-1, -999, 9999, 0):
        n = (df[c] == sentinel).sum()
        if n > len(df) * 0.05:
            print(f"{c}: {n} occurrences of {sentinel} — possible sentinel")
```

### 4.2 Understand the pattern

```python
# Is missingness related to other columns? (MAR vs MCAR)
for c in df.columns[df.isna().any()]:
    flag = df[c].isna()
    for other in numeric:
        if other == c: continue
        t, p = stats.ttest_ind(df.loc[flag, other].dropna(),
                               df.loc[~flag, other].dropna(),
                               equal_var=False)
        if p < 0.01:
            print(f"{c} missingness relates to {other} (p={p:.4f}) — MAR, not MCAR")

# Do columns go missing together?
df.isna().corr()

# Missingness by group
df.groupby("segment").apply(lambda g: g.isna().mean())
```

📊 `sns.heatmap(df.isna(), cbar=False)` — the missingness matrix.

### 4.3 Handle ✏️

```python
# Drop
df.dropna()                                  # any NA
df.dropna(how="all")                         # entirely empty rows
df.dropna(subset=["important"])              # only where key col missing
df.dropna(thresh=int(0.8*df.shape[1]))       # keep rows ≥80% complete
df.dropna(axis=1, thresh=int(0.6*len(df)))   # drop sparse columns

# Impute
df["x"].fillna(df["x"].mean())
df["x"].fillna(df["x"].median())             # safer with outliers
df["c"].fillna(df["c"].mode()[0])
df["x"].fillna(0)
df["x"].ffill(); df["x"].bfill()             # time series
df["x"].interpolate(method="linear")
df["x"].fillna(df.groupby("g")["x"].transform("median"))   # group-wise

# Model-based
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
df[numeric] = KNNImputer(n_neighbors=5).fit_transform(df[numeric])
df[numeric] = IterativeImputer(random_state=0).fit_transform(df[numeric])

# Flag instead of impute — preferred when missingness is informative
df["x_was_missing"] = df["x"].isna().astype(int)
df["x"] = df["x"].fillna(df["x"].median())
```

**Rule of thumb:** >60% missing → drop or flag, don't impute. Missing not at
random → always add a flag.

---

## 5. Duplicates 🔍 ✏️

```python
df.duplicated().sum()
df[df.duplicated(keep=False)].sort_values(by=cols)     # see all copies
df.duplicated(subset=["id"]).sum()                     # key duplicates
df.drop_duplicates()                                   # ✏️
df.drop_duplicates(subset=["id"], keep="last")         # ✏️

# Near-duplicates (fuzzy)
from difflib import SequenceMatcher
# or: rapidfuzz, recordlinkage for scale
```

---

## 6. Univariate — numeric 🔍

```python
s = df["x"]
s.mean(); s.median(); s.mode()
s.std(); s.var(); s.min(); s.max()
s.quantile([.25,.5,.75])
s.max() - s.min()                             # range
s.quantile(.75) - s.quantile(.25)             # IQR
s.skew()                                      # >1 right, <-1 left
s.kurtosis()                                  # >3 heavy tails
stats.variation(s.dropna())                   # coefficient of variation
s.nunique(); s.value_counts()
(s == 0).sum(); (s < 0).sum()                 # zeros, negatives
s.is_monotonic_increasing
```

### Normality

```python
stats.shapiro(s.sample(min(5000, len(s))))    # n < 5000
stats.normaltest(s.dropna())                  # D'Agostino
stats.anderson(s.dropna())
stats.jarque_bera(s.dropna())
```

📊 histogram, KDE, boxplot, violin, Q-Q plot (`stats.probplot`), ECDF.

---

## 7. Univariate — categorical 🔍

```python
s = df["c"]
s.value_counts()
s.value_counts(normalize=True)
s.value_counts(dropna=False)
s.nunique()
s.nunique() / len(s)                          # cardinality ratio
s.mode()

# Rare categories
rare = s.value_counts(normalize=True)
rare[rare < 0.01].index.tolist()

# Inconsistent encoding — the classic USA/usa/U.S.A. problem
norm = s.astype(str).str.strip().str.lower()
collisions = norm.value_counts()[norm.value_counts() != s.value_counts().reindex(norm.unique()).fillna(0)]
s.str.strip().str.lower().nunique() < s.nunique()     # True = inconsistent

# High cardinality warning
if s.nunique() > 50:
    print("high cardinality — one-hot will explode; consider target/frequency encoding")
```

📊 bar chart, count plot, pie (sparingly), treemap.

---

## 8. Univariate — datetime 🔍

```python
s = df["date"]
s.min(); s.max(); s.max() - s.min()           # range
s.dt.year.value_counts().sort_index()
s.dt.month.value_counts().sort_index()
s.dt.dayofweek.value_counts()
s.dt.hour.value_counts()
s.is_monotonic_increasing                     # sorted?
s.diff().value_counts()                       # granularity / regularity
s.diff().max()                                # largest gap

# Sanity
(s > pd.Timestamp.now()).sum()                # future dates
(s < pd.Timestamp("1900-01-01")).sum()        # impossible
s.dt.date.duplicated().sum()

# Gaps in a supposedly complete series
full = pd.date_range(s.min(), s.max(), freq="D")
missing_days = full.difference(s.dt.normalize().unique())
```

📊 line plot over time, seasonal decomposition, gap chart.

---

## 9. Univariate — text 🔍

```python
s = df["text"]
s.str.len().describe()
s.str.split().str.len().describe()             # word count
s.str.isupper().sum(); s.str.islower().sum()
s.str.contains(r"\d").sum()
s.str.strip().ne(s).sum()                      # stray whitespace
s.str.match(r"^[\w.]+@[\w.]+$").sum()          # emails
s.str.match(r"^https?://").sum()               # URLs
s.str.extract(r"(\d{4}-\d{2}-\d{2})")          # embedded dates
s.value_counts().head(20)
```

---

## 10. Outliers 🔍 ✏️

```python
s = df["x"]

# IQR
q1, q3 = s.quantile([.25,.75]); iqr = q3 - q1
lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
outliers = s[(s < lo) | (s > hi)]

# Z-score
z = np.abs(stats.zscore(s.dropna()))
outliers = s.dropna()[z > 3]

# Modified z-score (robust — use with skewed data)
med = s.median(); mad = stats.median_abs_deviation(s.dropna())
mz = 0.6745 * (s - med) / mad
outliers = s[np.abs(mz) > 3.5]

# Percentile
lo, hi = s.quantile([.01,.99])

# Multivariate
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
flags = IsolationForest(contamination=0.05, random_state=0).fit_predict(df[numeric])
flags = LocalOutlierFactor(n_neighbors=20).fit_predict(df[numeric])

# Mahalanobis distance
```

Handling ✏️:
```python
df = df[(s >= lo) & (s <= hi)]                # remove
df["x"] = s.clip(lo, hi)                      # winsorise
df["x"] = np.log1p(s)                         # transform
df["x_outlier"] = ((s < lo) | (s > hi)).astype(int)   # flag
```

**Always ask whether an outlier is an error or a real extreme value.**

---

## 11. Bivariate — numeric ↔ numeric 🔍

```python
df[numeric].corr()                             # Pearson (linear)
df[numeric].corr(method="spearman")            # monotonic, rank-based
df[numeric].corr(method="kendall")
stats.pearsonr(df.x, df.y)                     # with p-value
stats.spearmanr(df.x, df.y)

# Strong pairs only — never dump the whole matrix
c = df[numeric].corr().abs()
pairs = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack()
pairs[pairs > 0.7].sort_values(ascending=False)

df.x.cov(df.y)
```

📊 scatter, hexbin, pairplot, correlation heatmap, regression plot.

---

## 12. Bivariate — categorical ↔ numeric 🔍

```python
df.groupby("c")["x"].agg(["count","mean","median","std","min","max"])
df.groupby("c")["x"].describe()

# Significance
groups = [g["x"].dropna() for _, g in df.groupby("c")]
stats.f_oneway(*groups)                        # ANOVA
stats.kruskal(*groups)                         # non-parametric
stats.ttest_ind(a, b, equal_var=False)         # two groups
stats.mannwhitneyu(a, b)

# Effect size (matters more than the p-value)
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*a.var() + (nb-1)*b.var()) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled

# Correlation ratio (eta squared)
```

📊 grouped boxplot, violin, bar with error bars, strip/swarm.

---

## 13. Bivariate — categorical ↔ categorical 🔍

```python
pd.crosstab(df.a, df.b)
pd.crosstab(df.a, df.b, normalize="index")
pd.crosstab(df.a, df.b, margins=True)

chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(df.a, df.b))

# Cramér's V — effect size for categorical association
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
```

📊 stacked bar, grouped bar, mosaic plot, heatmap of the crosstab.

---

## 14. Multivariate 🔍

```python
# Collinearity — VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[numeric].dropna()
vif = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
# VIF > 10 → serious multicollinearity

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(X)
p = PCA().fit(Xs)
p.explained_variance_ratio_.cumsum()

# Clustering structure
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

📊 pairplot, PCA scatter, parallel coordinates, andrews curves, clustermap.

---

## 15. Target analysis 🔍

```python
y = df["target"]

# Classification
y.value_counts()
y.value_counts(normalize=True)
imbalance = y.value_counts(normalize=True).max()
if imbalance > 0.9:
    print(f"severe imbalance: {imbalance:.1%} majority class")

# Regression
y.describe(); y.skew()
np.log1p(y).skew()                             # would a transform help?

# Feature → target strength
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
mi = mutual_info_classif(X, y)

# LEAKAGE — a feature that is suspiciously predictive
for c in numeric:
    r = abs(df[c].corr(y))
    if r > 0.95:
        print(f"{c}: r={r:.3f} with target — probable leakage")

# Categorical leakage: does one category map perfectly to one class?
for c in cats:
    purity = df.groupby(c)["target"].nunique()
    if (purity == 1).mean() > 0.9:
        print(f"{c}: categories map 1:1 to target — leakage")
```

---

## 16. Distributions and transforms ✏️

```python
np.log1p(s)                                    # right skew, zeros ok
np.sqrt(s)                                     # mild right skew
np.square(s)                                   # left skew
stats.boxcox(s[s>0])                           # positive only
stats.yeojohnson(s)                            # handles zero/negative
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
PowerTransformer(method="yeo-johnson").fit_transform(X)
QuantileTransformer(output_distribution="normal").fit_transform(X)
```

Check improvement: `s.skew()` before vs after.

---

## 17. Visualisation catalogue 📊

| Purpose | Plot |
|---|---|
| Numeric distribution | histogram, KDE, boxplot, violin, ECDF |
| Normality | Q-Q plot |
| Categorical frequency | bar, count plot |
| Numeric ↔ numeric | scatter, hexbin, regplot |
| Cat ↔ numeric | grouped box, violin, bar+CI |
| Cat ↔ cat | stacked bar, mosaic, crosstab heatmap |
| Correlation | heatmap, clustermap |
| All pairs | pairplot, scatter matrix |
| Missingness | `sns.heatmap(df.isna())`, missingno matrix/bar/dendrogram |
| Time | line, seasonal decomposition, lag plot, autocorrelation |
| Outliers | boxplot, scatter with flags |
| High dimensions | PCA scatter, t-SNE, UMAP, parallel coordinates |

```python
fig, axes = plt.subplots(nrows, ncols, figsize=(15,10))
for ax, c in zip(axes.flat, numeric):
    df[c].hist(ax=ax, bins=30); ax.set_title(c)
plt.tight_layout(); plt.savefig("dist.png", dpi=120, bbox_inches="tight")
```

---

## 18. Cleaning operations ✏️

```python
# Column hygiene
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.rename(columns={"old":"new"})
df = df.drop(columns=["unused"])

# String hygiene
df["c"] = df["c"].str.strip().str.lower()
df["c"] = df["c"].str.replace(r"\s+", " ", regex=True)
df["c"] = df["c"].replace({"u.s.a.":"usa", "united states":"usa"})
df["c"] = df["c"].str.normalize("NFKD")

# Value repair
df["age"] = df["age"].where(df["age"].between(0, 120))
df = df[df["price"] > 0]
df["pct"] = df["pct"].clip(0, 100)

# Structural
df = df.reset_index(drop=True)
df = df.sort_values(["date","id"])
df = df.set_index("date")
```

---

## 19. Feature engineering ✏️

### Encoding

```python
pd.get_dummies(df, columns=["c"], drop_first=True)          # one-hot
df["c"].map({"low":0,"med":1,"high":2})                     # ordinal
df["c"].map(df["c"].value_counts(normalize=True))           # frequency
df.groupby("c")["target"].transform("mean")                 # target (leaky — use CV folds)
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
import category_encoders as ce                              # binary, hashing, WOE
```

### Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
StandardScaler()      # mean 0, sd 1
MinMaxScaler()        # [0,1]
RobustScaler()        # median/IQR — outlier resistant
```

### Binning

```python
pd.cut(df.age, bins=[0,18,35,60,120], labels=["child","young","adult","senior"])
pd.qcut(df.income, q=4, labels=["Q1","Q2","Q3","Q4"])       # equal frequency
```

### Dates

```python
d = df["date"].dt
df["year"], df["month"], df["day"] = d.year, d.month, d.day
df["dow"], df["week"], df["quarter"] = d.dayofweek, d.isocalendar().week, d.quarter
df["is_weekend"] = d.dayofweek >= 5
df["days_since"] = (pd.Timestamp.now() - df["date"]).dt.days
df["month_sin"] = np.sin(2*np.pi*d.month/12)                # cyclical
df["month_cos"] = np.cos(2*np.pi*d.month/12)
```

### Derived

```python
df["ratio"] = df.a / df.b.replace(0, np.nan)
df["total"] = df[["a","b","c"]].sum(axis=1)
df["a_x_b"] = df.a * df.b                                   # interaction
from sklearn.preprocessing import PolynomialFeatures

# Time series
df["lag_1"] = df.x.shift(1)
df["roll_7"] = df.x.rolling(7).mean()
df["expanding"] = df.x.expanding().mean()
df["pct_change"] = df.x.pct_change()
df["cumsum"] = df.x.cumsum()

# Group aggregates
df["mean_by_g"] = df.groupby("g")["x"].transform("mean")
df["rank_in_g"] = df.groupby("g")["x"].rank()
```

---

## 20. Reshaping ✏️

```python
df.pivot_table(index="a", columns="b", values="x", aggfunc="mean")
df.melt(id_vars=["id"], value_vars=["x","y"])
pd.merge(a, b, on="id", how="left", indicator=True)      # check merge quality
pd.concat([a, b], axis=0, ignore_index=True)
df.groupby("g").agg({"x":["mean","sum"], "y":"count"})
df.stack(); df.unstack()
df.transpose()
df.explode("list_col")
```

Post-merge validation:
```python
merged["_merge"].value_counts()      # left_only rows = failed joins
assert len(merged) == len(a)         # no unexpected fan-out
```

---

## 21. Time series specifics 🔍

```python
df = df.set_index("date").sort_index()
df.resample("M").mean()
df.asfreq("D")                                  # expose gaps
df.x.rolling(30).mean()
from statsmodels.tsa.seasonal import seasonal_decompose, STL
seasonal_decompose(df.x, model="additive", period=12).plot()
from statsmodels.tsa.stattools import adfuller, acf, pacf
adfuller(df.x.dropna())                         # stationarity
pd.plotting.autocorrelation_plot(df.x)
```

---

## 22. Sanity checks 🔍

```python
assert df["id"].is_unique
assert df["age"].between(0,120).all()
assert (df["end"] >= df["start"]).all()
assert df.groupby("id").size().max() == 1
assert df["pct"].between(0,100).all()
assert abs(df["parts"].sum() - df["total"].iloc[0]) < 1e-6

# Business rules
assert (df["discount"] <= df["price"]).all()
assert df["quantity"].ge(0).all()
```

---

## 23. Reporting and export

```python
# Automated profiling
from ydata_profiling import ProfileReport
ProfileReport(df, explorative=True).to_file("report.html")
import sweetviz; sweetviz.analyze(df).show_html("sv.html")
from dtale import show; show(df)
import autoviz

# Export
df.to_csv("clean.csv", index=False)
df.to_parquet("clean.parquet", compression="snappy")
df.to_excel("clean.xlsx", sheet_name="data", index=False)
df.to_json("clean.json", orient="records", lines=True)
```

---

## 24. Mapping to server tools

| Reference section | Tool |
|---|---|
| 1, 2 | `load_dataset` |
| 2, 3, 4.1, 5 | `profile_dataset` |
| 6, 7, 8, 9, 10 | `analyze_column` |
| 3.2, 4.1–4.2, 5, 7, 10, 22 | `find_issues` |
| 11, 12, 13, 14 | `check_relationships` |
| 12, 13 | `compare_groups` |
| 15 | `analyze_target` *(candidate new tool)* |
| 17 | `plot` |
| 4.3, 5, 10, 18 | `clean_data` |
| 16, 19 | `transform_data` |
| 20, 21 | `reshape_data` |
| 23 | `export_dataset`, `generate_code` |

### Gaps this reference exposes in `speckit.md`

1. **Target analysis deserves its own tool** — imbalance, feature strength,
   and leakage detection (§15) drive every modelling decision and are
   currently buried in `find_issues`.
2. **Visualisation is under-specified** — §17 lists ~20 plot types against a
   single `plot` tool.
3. **Sanity checks (§22) have no home** — user-supplied business rules are a
   distinct capability worth a `validate_rules` tool.
4. **No report generation** — §23 is a real end-state; `generate_report`
   producing an HTML/Markdown summary is missing.
5. **Time series (§21) is unaddressed** in the current spec.
