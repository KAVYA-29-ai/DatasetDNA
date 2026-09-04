# 🧬 DatasetDNA

### One command. Know your dataset before you train.

[![PyPI](https://img.shields.io/pypi/v/datasetdna?logo=pypi)](https://pypi.org/project/datasetdna/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/github/license/KAVYA-29-ai/DatasetDNA)](https://github.com/KAVYA-29-ai/DatasetDNA/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-137%20passed-success)](https://github.com/KAVYA-29-ai/DatasetDNA)
[![Release](https://img.shields.io/badge/release-1.1.0-success)](https://pypi.org/project/datasetdna/1.1.0/)

DatasetDNA is an automated dataset health profiler for machine learning. It analyzes your CSV, detects real data-quality problems, separates them from statistical signals, identifies targets, calculates a **0–100 health score**, and provides actionable recommendations.

> 🛡️ **Your data stays on your machine.** No cloud upload. No notebook required. No configuration required.

---

## ⚡ Quick Start

```bash
pip install datasetdna
datasetdna your_data.csv
```

```bash
datasetdna data.csv --target churn
datasetdna data.csv --html
```

---

## 🎯 The Core Idea

### Not every unusual pattern is a data-quality problem.

DatasetDNA makes a strict distinction between **data quality** and **statistical signals**.

| 🧬 Data Quality | 📊 Statistical Signals |
|---|---|
| **Affects Health Score** | **Does not affect Health Score** |
| Missing values | Strong correlations |
| Duplicate rows | Skewness |
| Target imbalance | High cardinality |
| Impossible values | Statistical outliers |

For example, `petal_length ↔ petal_width = 0.96` is not automatically a problem. A statistical pattern can simply describe a real property of the data.

> **A statistical pattern is not automatically a data-quality defect.**

---

<details>
<summary>🔍 What DatasetDNA Checks</summary>

### 📋 Overview
Row count, column count, memory usage, missing cells, and duplicate rows.

### 🧱 Schema & Semantic Types
Numeric, categorical, boolean, date, ID-like, and empty columns.

### ⚠️ Missing Values
Detects normal and disguised missing values such as `NaN`, `N/A`, `null`, `NULL`, `?`, `-`, and `unknown`.

### 🔁 Duplicates
Detects duplicate rows and their proportion.

### 🔢 Cardinality
Measures unique values and identifies unusually high-cardinality columns.

### 📈 Numerical Analysis
Mean, median, standard deviation, minimum, maximum, and skewness.

### 🏷️ Categorical Analysis
Category counts, percentages, associations, and binary variables.

### 📌 Outliers
Statistical outliers with domain-aware rules where appropriate.

### 🔗 Correlations
Strong numerical relationships using Pearson correlation.

### 🎯 Target Analysis
Classification and regression detection, class counts, distributions, and imbalance ratios for classification.

### 🧩 Mixed-Type Detection
Identifies columns containing multiple underlying Python value types, such as numbers mixed with strings.

### 🏷️ Category Consistency
Detects multiple representations of the same category, such as `Male`, `male`, and `M`.

</details>

---

<details>
<summary>🧠 Semantic Type Detection</summary>

DatasetDNA goes beyond `df.describe()` and tries to understand what columns represent.

```text
customer_id    → ID
age            → Numeric
income         → Numeric
city           → Categorical
joined_at      → Date
churn          → Target
```

It also handles messy values such as `$55,000`, `1,200`, `N/A`, `null`, `?`, and `-`.

ID-like columns can be excluded from statistical analysis where they would otherwise create misleading relationships.

</details>

---

<details>
<summary>❤️ Health Score</summary>

Every dataset receives a **0–100 Health Score**.

```text
100 ───────────────── Excellent
 80 ────────────────── Good
 60 ────────────────── Fair
 40 ────────────────── Poor
  0 ────────────────── Critical
```

### Score-affecting issues
- Missing values
- Duplicate rows
- Target class imbalance
- Impossible/domain-invalid values

### Informational signals
- Correlation
- Skewness
- Cardinality
- Statistical outliers

Mixed-type and category-consistency findings are surfaced as recommendations and do not directly penalize the health score.

The score focuses on genuine data-quality concerns rather than penalizing valid statistical structure.

</details>

---

## 🖥️ Example Output

```bash
datasetdna examples/real.csv
```

```text
╭────────────── 🧬 Dataset Health ──────────────╮
│ 🟠 Health Score                               │
│                                               │
│ 70 / 100                                      │
│ Fair                                          │
│                                               │
│ 6 quality issue(s) · 14 statistical signal(s) │
╰───────────────────────────────────────────────╯

🧬 DATA QUALITY
 MEDIUM  Column 'age' contains 10.0% missing values.
 MEDIUM  Dataset contains 5.0% duplicate rows.
 HIGH    Age-like column 'age' contains values outside 0-120.

📊 STATISTICAL SIGNALS
 🔎 HIGH    Strong correlation detected between age and salary.
 🔎 MEDIUM  Column 'customer_id' has very high cardinality.
```

---

<details>
<summary>🎯 Target Detection</summary>

Explicit target:

```bash
datasetdna data.csv --target churn
```

Automatic target detection checks:

```text
target → label → churn
```

For classification, DatasetDNA reports class count, distribution, percentages, and imbalance ratio.

For regression, continuous targets are identified as regression targets and classification-specific metrics are not applied.

</details>

---

<details>
<summary>💡 Recommendations</summary>

DatasetDNA generates recommendations from detected dataset conditions.

```text
🟡 MEDIUM

Target 'churn' has an imbalance ratio of 4:1.

Recommendation:
Use stratified splitting and monitor
class-aware evaluation metrics.
```

Recommendations also flag mixed Python value types and inconsistent categorical representations so they can be standardized before model training.

Recommendations guide investigation rather than blindly applying one universal fix.

</details>

---

## 🐍 Python API

Profile a pandas DataFrame directly in memory:

```python
import pandas as pd
import datasetdna as dna

df = pd.DataFrame({
    "age": [20, 21, 22, 23],
    "income": [100, 200, 300, 400],
    "churn": [0, 0, 1, 1],
})

result = dna.profile(df)
```

Specify a target:

```python
result = dna.profile(df, target="churn")
```

Generate an HTML report:

```python
result = dna.profile(
    df,
    target="churn",
    html=True,
    output="datasetdna_report.html",
)
```

Access results:

```python
print(result["health"])
print(result["target"])
print(result["recommendations"])
```

The result contains:

```text
overview
schema
mixed_types
category_consistency
missing
duplicates
cardinality
numerical
categorical
outliers
correlations
target
health
recommendations
```

The CLI and Python API use the same underlying profiling engine.

---

## 🏗️ Architecture

```text
                         DatasetDNA
                              │
                 ┌────────────┴────────────┐
                 │                         │
                CLI                   Python API
                 │                         │
                 └────────────┬────────────┘
                              ▼
                       Profiling Engine
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
          Profilers      Health Score    Recommendations
              │
              ├── Overview
              ├── Schema
              ├── Mixed Types
              ├── Category Consistency
              ├── Missing
              ├── Duplicates
              ├── Cardinality
              ├── Numerical
              ├── Categorical
              ├── Outliers
              ├── Correlations
              └── Target
                              │
                              ▼
                    Console / HTML Report
```

---

## ⚡ Installation

### From PyPI

```bash
pip install datasetdna
```

To install the current published release explicitly:

```bash
pip install datasetdna==1.1.0
```

### From Source

```bash
git clone https://github.com/KAVYA-29-ai/DatasetDNA.git
cd DatasetDNA
pip install -e .
```

Requires **Python 3.10+**.

---

## 🧪 Testing

Automated coverage includes profilers, semantic types, target detection, health scoring, recommendations, CLI behavior, HTML reporting, Python API behavior, mixed-type detection, category consistency, and edge cases.

```text
137 tests passed
```

Run the suite:

```bash
pytest -q
```

---

## 🤔 Why DatasetDNA?

Most ML failures aren't model failures. They're data failures:

```text
❌ Hidden missing values
❌ Duplicate rows
❌ Numeric values stored as strings
❌ Impossible values
❌ ID columns treated as features
❌ Severe target imbalance
❌ Mixed Python value types
❌ Inconsistent category representations
❌ Statistical patterns mistaken for data problems
```

DatasetDNA is designed as a **pre-flight check for machine learning datasets**.

```text
Raw Dataset
     ↓
 DatasetDNA
     ↓
 Understand
     ↓
   Fix
     ↓
 Validate
     ↓
  Train
```

---

## 🗺️ Roadmap

### ✅ Phase 1 — Core Profiler
- [x] Semantic type detection
- [x] Missingness analysis
- [x] Duplicate detection
- [x] Cardinality analysis
- [x] Numerical analysis
- [x] Categorical analysis
- [x] Outlier detection
- [x] Correlation analysis
- [x] Target analysis
- [x] Two-tier health scoring

### ✅ Phase 2 — Better UX
- [x] Actionable recommendations
- [x] Expanded validation
- [x] Automatic target detection
- [x] Large-file handling
- [x] Deeper test coverage
- [x] Mixed-type detection
- [x] Category consistency detection

### ✅ Phase 3 — HTML Dashboard
- [x] Interactive Plotly charts
- [x] Exportable HTML reports
- [x] Dataset health visualization
- [x] Target visualization
- [x] Category consistency section

### ✅ Phase 4 — Public Release
- [x] PyPI package
- [x] CLI
- [x] Python API
- [x] Stable `1.0.0` release
- [x] `1.1.0` release

### 🔮 Phase 5 — Ecosystem
- [ ] Expanded integrations
- [ ] More edge-case coverage
- [ ] API extensions
- [ ] ML pipeline integrations

---

## 🤝 Contributing

Issues and pull requests are welcome.

If DatasetDNA crashes, detects the wrong type, misses a real issue, or produces unexpected output, please open an issue with:

- Dataset shape
- Column names
- Relevant data types
- Expected behavior
- Actual behavior

**Please don't upload private or sensitive dataset contents.**

---

## 📄 License

DatasetDNA is released under the **MIT License**.

Copyright © Kavya Rajput

---

<p align="center">

### 🧬 DatasetDNA

**Profile better. Fix earlier. Train smarter.**

</p>
