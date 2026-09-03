# 🧬 DatasetDNA

**One command. Know if your dataset is safe to train on.**

DatasetDNA profiles any CSV and produces a clear, actionable health report — catching real data-quality problems (missing values, duplicates, invalid ranges) while separating them from statistical signals that need human judgment, not automatic penalties (correlation, skew, cardinality).

```bash
pip install datasetdna
datasetdna your_data.csv
```

That's it. No config, no notebook, no cloud upload — your data never leaves your machine.

---

## The problem with most profiling tools

Most auto-EDA tools treat every statistical pattern as a "problem" and dock your data a few points for it. That sounds fine — until you run it on the Iris dataset, one of the most famous, cleanest datasets in all of ML, and it tells you `petal_length ↔ petal_width` correlation of `0.96` is a health issue.

It isn't. That correlation is **expected botany**, not a data-quality defect. A tool that can't tell the difference between "this is broken" and "this is just how the world works" isn't trustworthy enough to run before training a real model.

DatasetDNA draws a hard line between the two:

| 🧬 Data Quality | 📊 Statistical Signals |
|---|---|
| **Affects your Health Score.** Objectively bad, in any context. | **Never affects your score.** Informational — could be fine, could be worth a look. |
| Missing values | Strong correlations |
| Duplicate rows | Skewness |
| Target class imbalance | High cardinality |
| Impossible values (age = 150, age = -5) | Outliers |

This distinction is the core design decision behind the whole tool — not an afterthought.

---

## What it checks

```
CSV
 ↓
Loader + Cleaning        → handles $1,200 / "1,234" / N/A / null / "?" / "-" as one CSV, cleanly
 ↓
Semantic Type Detection  → tells an ID column from a real feature, a date from a category
 ↓
┌─────────────────────────────┐
│ Overview        Numerical   │
│ Schema          Categorical │
│ Missingness     Outliers    │
│ Duplicates      Correlations│
│ Cardinality     Target      │
└─────────────────────────────┘
 ↓
Health Scoring   → 0–100, driven only by real quality issues
 ↓
Console Report   → color-coded, readable, no PhD required
```

**Semantic Type Detection** is what separates this from a plain `df.describe()` script — it automatically:
- Cleans numbers written as text (`"$55,000"`, `"1,200"`) into real numeric columns
- Recognizes disguised missing values (`N/A`, `null`, `?`, `-`, `unknown`) as actual missing data, not valid strings
- Flags ID-like columns (customer IDs, emails) and excludes them from correlation/association analysis, where they'd otherwise create meaningless "100% correlated" noise
- Validates domain-plausible ranges (e.g., an `age` column with values of `-5` or `150` gets flagged as a real error, not just an "outlier")

---

## Example output

```bash
$ datasetdna examples/real.csv

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
 HIGH    Age-like column 'age' contains values outside 0-120. (range: -5.0 to 150.0)

📊 STATISTICAL SIGNALS
 🔎 HIGH    Strong correlation detected between age and salary. (Pearson: 0.949)
 🔎 MEDIUM  Column 'customer_id' has very high cardinality. (100.0% unique)
```

Full breakdown tables for schema, missingness, cardinality, numerical stats, outliers, correlations, and categorical associations follow below the summary.

---

## Installation

```bash
pip install datasetdna
```

Or from source:

```bash
git clone https://github.com/KAVYA-29-ai/DatasetDNA
cd DatasetDNA
pip install -e .
```

Requires Python 3.10+.

## Usage

```bash
# Basic profile
datasetdna data.csv

# Include target-column analysis (class imbalance, target relationships)
datasetdna data.csv --target churn
```

---

## Why this exists

Most ML failures aren't model failures — they're data failures nobody caught before training started: a leaked ID column, silent duplicates, a numeric column secretly stored as `"$1,200"` strings, a target with 95/5 imbalance nobody noticed until the model just predicted the majority class every time.

DatasetDNA is meant to be the first command you run against any new CSV, before you write a single line of `train_test_split`.

---

## Roadmap

- [x] **Phase 1 — Core profiler**: semantic types, missingness, duplicates, cardinality, numerical/categorical analysis, outliers, correlations, target analysis, two-tier health scoring
- [x] **Phase 2 — Better UX**: actionable recommendations per issue, expanded validation, deeper test coverage
- [x] **Phase 3 — HTML dashboard**: interactive Plotly charts, exportable shareable reports
- [ ] **Phase 4 — PyPI**: stable public release

---

## Contributing

Issues and PRs welcome. If you've found a dataset that breaks DatasetDNA (crashes, wrong classification, missed issue) — that's exactly the kind of bug report that helps most. Open an issue with the CSV shape (not necessarily the data itself) and what you expected to see.

## License

MIT © Kavya Rajput
