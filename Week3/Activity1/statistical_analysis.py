"""
Week 3 - Activity 1: Initial Statistical Analysis
Statistical analysis of Sample_dataset.csv with detailed metric explanations.

Run from anywhere:  python statistical_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Resolve the CSV relative to this script so the program runs from any directory.
# The dataset ships in Week3/ while this script lives in Week3/Activity1/, so
# check alongside the script first, then the parent folder.
SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_PATHS = (
    SCRIPT_DIR / 'Sample_dataset.csv',
    SCRIPT_DIR.parent / 'Sample_dataset.csv',
)
DATA_FILE = next((p for p in CANDIDATE_PATHS if p.is_file()), None)
if DATA_FILE is None:
    searched = '\n  '.join(str(p) for p in CANDIDATE_PATHS)
    raise SystemExit(f"Sample_dataset.csv not found. Searched:\n  {searched}")

# Fixed analysis date keeps tenure figures reproducible across re-runs.
# Using date.today() would silently change the results every day.
ANALYSIS_DATE = pd.Timestamp('2026-08-16')

# Below this sample size, shape statistics are too unstable to report honestly
SMALL_SAMPLE_THRESHOLD = 20

df = pd.read_csv(DATA_FILE)

print("=" * 80)
print("STATISTICAL ANALYSIS REPORT - Week 3 Activity 1")
print("=" * 80)

# ============================================================================
# 1. DATA QUALITY OVERVIEW
# ============================================================================
print("\n1. DATA QUALITY OVERVIEW")
print("-" * 80)

print(f"\nSource file: {DATA_FILE.name}")
print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn Names: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print("\n  -> Note: Age and Salary load as 'object' (text), not numbers, because")
print("     a single non-numeric entry forces the whole column to string type.")
print("     Arithmetic on these columns fails until they are cleaned.")

print("\n\nMISSING VALUES ANALYSIS (raw file, before cleaning):")
print("-" * 40)
missing_df = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': df.isnull().sum().values,
    'Missing %': (df.isnull().sum() / len(df) * 100).values,
})
print(missing_df.to_string(index=False))

# ============================================================================
# 2. DATA CLEANING
# ============================================================================
print("\n\n2. DATA CLEANING")
print("-" * 80)

df_clean = df.copy()

# --- 2a. Numeric type coercion -------------------------------------------
# Spelled-out numbers appear in both Age and Salary. They are handled through
# one shared lookup so the two columns are treated consistently -- an earlier
# version converted "thirty-eight" but silently dropped "sixty five thousand".
WORD_NUMBERS = {
    'thirty-eight': 38.0,
    'thirty eight': 38.0,
    'sixty five thousand': 65000.0,
}

conversion_log = []


def to_number(val, column):
    """Coerce a cell to float. Returns NaN when the value cannot be trusted."""
    if pd.isna(val):
        return np.nan

    if not isinstance(val, str):
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    text = val.strip()
    if text == '':
        return np.nan

    # Spelled-out number, e.g. "thirty-eight"
    if text.lower() in WORD_NUMBERS:
        number = WORD_NUMBERS[text.lower()]
        conversion_log.append(f"{column}: '{text}' -> {number:g} (spelled-out number)")
        return number

    # Thousands separators and stray quotes, e.g. "30,000"
    stripped = text.replace(',', '').replace('"', '')
    try:
        number = float(stripped)
    except ValueError:
        conversion_log.append(f"{column}: '{text}' -> NaN (unrecognised, not guessed)")
        return np.nan

    if stripped != text:
        conversion_log.append(f"{column}: '{text}' -> {number:g} (removed separators)")
    return number


for column in ('Age', 'Salary', 'Net worth'):
    df_clean[column] = df_clean[column].apply(to_number, column=column)

print("\nNumeric coercion (Age, Salary, Net worth):")
for entry in conversion_log:
    print(f"  {entry}")
print("\n  Missing values remaining after coercion:")
for column in ('Age', 'Salary', 'Net worth'):
    print(f"    {column}: {df_clean[column].isna().sum()}")

# --- 2b. Country code standardisation ------------------------------------
COUNTRY_ALIASES = {'AU': 'AUS'}

print("\nStandardising Country codes...")
before_countries = sorted(df_clean['Country'].dropna().unique())
df_clean['Country'] = df_clean['Country'].replace(COUNTRY_ALIASES)
after_countries = sorted(df_clean['Country'].dropna().unique())
print(f"  Before: {before_countries}")
print(f"  After:  {after_countries}")
print("  Merged 'AU' into 'AUS' -- same country, inconsistent coding")

# --- 2c. Date parsing -----------------------------------------------------
# Explicit formats only. A permissive parser reads "2019-13-01" as YYYY-DD-MM
# and silently invents 13 Jan 2019 rather than rejecting month 13.
ACCEPTED_DATE_FORMATS = ('%d/%m/%Y', '%Y-%m-%d')


def parse_join_date(val):
    if pd.isna(val) or str(val).strip() == '':
        return pd.NaT
    for fmt in ACCEPTED_DATE_FORMATS:
        parsed = pd.to_datetime(str(val).strip(), format=fmt, errors='coerce')
        if pd.notna(parsed):
            return parsed
    return pd.NaT


print("\nParsing Join Date...")
raw_dates = df_clean['Join Date']
df_clean['Join Date'] = raw_dates.apply(parse_join_date)

blank_dates = int(raw_dates.isna().sum())
rejected_dates = raw_dates[df_clean['Join Date'].isna() & raw_dates.notna()]

print(f"  Accepted formats: {', '.join(ACCEPTED_DATE_FORMATS)}")
print(f"  Parsed successfully: {df_clean['Join Date'].notna().sum()}/{len(df_clean)}")
print(f"  Blank in source: {blank_dates}")
if len(rejected_dates) > 0:
    print(f"  REJECTED as invalid: {list(rejected_dates.values)}")
    print("    -> '2019-13-01' has month 13, which does not exist")

df_clean['Tenure (years)'] = (
    (ANALYSIS_DATE - df_clean['Join Date']).dt.days / 365.25
).round(2)
print(f"  Derived 'Tenure (years)' relative to fixed analysis date {ANALYSIS_DATE.date()}")

# --- 2d. Deduplication ----------------------------------------------------
print("\nDeduplicating records by ID...")
ids = df_clean.loc[df_clean['ID'].notna(), 'ID']
dup_ids = ids[ids.duplicated(keep=False)].unique()

if len(dup_ids) > 0:
    print(f"  Duplicate ID(s): {[int(i) for i in dup_ids]}")

    # A merge is only safe when the duplicate rows do not disagree. Report any
    # column where two rows both hold a value and those values differ.
    for dup_id in dup_ids:
        rows = df_clean[df_clean['ID'] == dup_id]
        print(f"    ID {int(dup_id)}: {len(rows)} rows")
        for column in df_clean.columns:
            present = rows[column].dropna()
            if len(present) > 1 and present.nunique() > 1:
                print(f"      CONFLICT in '{column}': {list(present.values)} -- keeping first")
        complementary = [
            c for c in df_clean.columns if rows[c].isna().sum() == len(rows) - 1
        ]
        if complementary:
            print(f"      Complementary (one row supplies each): {complementary}")

    rows_before = len(df_clean)
    no_id = df_clean[df_clean['ID'].isna()]          # cannot group without a key
    with_id = df_clean[df_clean['ID'].notna()]
    merged = with_id.groupby('ID', as_index=False, sort=False).first()  # first non-null
    df_clean = pd.concat([merged, no_id], ignore_index=True)
    print(f"  Rows: {rows_before} -> {len(df_clean)}")
else:
    print("  No duplicate IDs found")

# ============================================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n\n3. DESCRIPTIVE STATISTICS - NUMERICAL COLUMNS")
print("-" * 80)

numerical_cols = ['Age', 'Salary', 'Net worth', 'Tenure (years)']
print("\n" + df_clean[numerical_cols].describe().to_string())

print("\n\nDETAILED METRIC EXPLANATIONS:")
print("-" * 80)

for col in numerical_cols:
    valid = df_clean[col].dropna()
    n = len(valid)

    print(f"\n{col.upper()}  (n = {n} valid observations)")

    if n == 0:
        print("  No valid data -- all values missing. No statistics computable.")
        continue

    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)

    print(f"  Count      : {n}")
    print("    -> Non-missing data points. Every statistic below rests on these only.")
    print(f"  Mean       : {valid.mean():.2f}")
    print("    -> Arithmetic average. Pulled toward extreme values.")
    print(f"  Median     : {valid.median():.2f}")
    print("    -> Middle value when sorted. Unaffected by extremes.")
    print(f"  Std Dev    : {valid.std():.2f}")
    print("    -> Typical distance from the mean (sample std, ddof=1).")
    print(f"  Variance   : {valid.var():.2f}")
    print("    -> Std Dev squared. Same information, squared units.")
    print(f"  Min / Max  : {valid.min():.2f} / {valid.max():.2f}")
    print(f"  Range      : {valid.max() - valid.min():.2f}")
    print("    -> Max minus Min. Simple but driven entirely by two points.")
    print(f"  Q1 (25%)   : {q1:.2f}")
    print("    -> A quarter of values fall below this.")
    print(f"  Q3 (75%)   : {q3:.2f}")
    print("    -> Three quarters of values fall below this.")
    print(f"  IQR        : {q3 - q1:.2f}")
    print("    -> Spread of the middle 50%. Basis of the outlier rule in section 6.")

    # Shape statistics need a minimum sample to be defined at all
    if n >= 3:
        skew = float(valid.skew())
        print(f"  Skewness   : {skew:.3f}")
        direction = ("right-skewed (tail of high values)" if skew > 0.5
                     else "left-skewed (tail of low values)" if skew < -0.5
                     else "roughly symmetric")
        print(f"    -> {direction}. Guide: <-0.5 left, -0.5..0.5 symmetric, >0.5 right.")
    else:
        print("  Skewness   : not computable (needs n >= 3)")

    if n >= 4:
        kurt = float(valid.kurtosis())
        print(f"  Kurtosis   : {kurt:.3f}")
        tails = "heavier tails than normal" if kurt > 0 else "lighter tails than normal"
        print(f"    -> {tails}. Excess kurtosis; 0 matches a normal distribution.")
    else:
        print("  Kurtosis   : not computable (needs n >= 4)")

    if n < SMALL_SAMPLE_THRESHOLD:
        print(f"    !! n = {n} is below {SMALL_SAMPLE_THRESHOLD}: skewness and kurtosis")
        print("       are highly unstable here. Treat them as descriptive only.")

# ============================================================================
# 4. CATEGORICAL ANALYSIS
# ============================================================================
print("\n\n4. CATEGORICAL ANALYSIS")
print("-" * 80)

print("\nCOUNTRY DISTRIBUTION:")
print(f"  Distinct countries (excluding missing): {df_clean['Country'].nunique()}")
counts = df_clean['Country'].value_counts(dropna=False)
pcts = counts / len(df_clean) * 100
print("\n" + pd.DataFrame({'Count': counts, 'Percent': pcts.round(1)}).to_string())
print("\n  -> Frequency of each category. The dominant category indicates where")
print("     the sample is concentrated, and therefore where findings apply.")

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================
print("\n\n5. CORRELATION ANALYSIS")
print("-" * 80)

correlation_matrix = df_clean[numerical_cols].corr()
print("\nPearson Correlation Matrix:")
print(correlation_matrix.to_string())

# Thresholds used for both the guide and the labels below, so they cannot drift
# apart. An earlier version printed a guide saying 0.7+ was "Very Strong" while
# labelling those same values merely "Strong".
CORRELATION_BANDS = (
    (0.7, "Very Strong"),
    (0.5, "Strong"),
    (0.3, "Moderate"),
    (0.0, "Weak"),
)


def describe_correlation(r):
    if pd.isna(r):
        return "Not computable"
    for threshold, label in CORRELATION_BANDS:
        if abs(r) >= threshold:
            return f"{label} {'positive' if r >= 0 else 'negative'}"
    return "Weak"


print("\n\nINTERPRETATION GUIDE (|r| thresholds):")
print("-" * 40)
for threshold, label in CORRELATION_BANDS:
    print(f"  |r| >= {threshold:.1f}: {label}")
print("  Sign: positive = move together; negative = move oppositely.")
print("  r measures LINEAR association only, and never proves causation.")

print("\n\nKey Relationships:")
for i, col1 in enumerate(numerical_cols):
    for col2 in numerical_cols[i + 1:]:
        r = correlation_matrix.loc[col1, col2]
        # Pandas uses pairwise deletion: each pair uses rows where both are present
        pair_n = int((df_clean[col1].notna() & df_clean[col2].notna()).sum())
        print(f"\n{col1} vs {col2}: r = {r:.3f}  (n = {pair_n} complete pairs)")
        print(f"  -> {describe_correlation(r)}", end="")
        if pd.notna(r):
            print(f". r^2 = {r ** 2:.2f}, so about {r ** 2 * 100:.0f}% of the variation"
                  f" in one tracks the other.")
        else:
            print(".")
        if pair_n < 10:
            print(f"     !! Only {pair_n} pairs -- a single record could change this materially.")

# ============================================================================
# 6. OUTLIER DETECTION
# ============================================================================
print("\n\n6. OUTLIER DETECTION (IQR Method)")
print("-" * 80)
print("\nRule: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are flagged as outliers.")
print("Chosen over the mean +/- 3*SD rule because quartiles are not themselves")
print("distorted by the very extremes being searched for.\n")

for col in numerical_cols:
    valid = df_clean[col].dropna()
    print(f"\n{col}:")

    if len(valid) < 4:
        print(f"  Skipped -- only {len(valid)} values, quartiles are not meaningful.")
        continue

    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = valid[(valid < lower) | (valid > upper)]

    print(f"  Acceptable range: [{lower:.2f}, {upper:.2f}]")
    print(f"  Outliers found  : {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Values          : {list(outliers.values)}")
        print("  -> Investigate before removing: outliers can be data-entry errors")
        print("     or genuine extreme cases that matter.")
    else:
        print("  -> All values fall within the expected spread.")

# ============================================================================
# 7. SUMMARY INSIGHTS
# ============================================================================
print("\n\n7. KEY INSIGHTS & SUMMARY")
print("-" * 80)

print("\nData Quality:")
print(f"  - Unique records after deduplication: {len(df_clean)} (from {len(df)} raw rows)")
for col in numerical_cols:
    print(f"  - Usable {col}: {df_clean[col].notna().sum()}/{len(df_clean)}")

age_valid = df_clean['Age'].dropna()
if len(age_valid) > 0:
    print("\nAge Demographics:")
    print(f"  - Average age: {age_valid.mean():.1f} years "
          f"(range {age_valid.min():.0f}-{age_valid.max():.0f})")

salary_valid = df_clean['Salary'].dropna()
if len(salary_valid) > 0:
    cv = salary_valid.std() / salary_valid.mean() * 100
    print("\nSalary Insights:")
    print(f"  - Average salary: ${salary_valid.mean():,.2f}")
    print(f"  - Range: ${salary_valid.min():,.2f} to ${salary_valid.max():,.2f}")
    print(f"  - Coefficient of variation: {cv:.1f}% "
          f"({'tight' if cv < 15 else 'wide'} spread relative to the mean)")

tenure_valid = df_clean['Tenure (years)'].dropna()
if len(tenure_valid) > 0:
    excluded = len(df_clean) - len(tenure_valid)
    print("\nTenure Insights:")
    print(f"  - Average tenure: {tenure_valid.mean():.2f} years "
          f"(range {tenure_valid.min():.2f}-{tenure_valid.max():.2f})")
    print(f"  - Excludes {excluded} record(s) without a usable join date")

print("\nGeographic Distribution:")
country_dist = df_clean['Country'].value_counts()
if len(country_dist) > 0:
    top_country, top_count = country_dist.index[0], country_dist.values[0]
    print(f"  - {top_country}: {top_count}/{len(df_clean)} records "
          f"({top_count / len(df_clean) * 100:.0f}%) -- primary location")
    print(f"  - Spans {df_clean['Country'].nunique()} distinct countries")
    print(f"  - Missing country: {df_clean['Country'].isna().sum()} record(s)")

print("\nPrincipal Caveat:")
print(f"  With {len(df_clean)} records and missing values throughout, every figure")
print("  above is descriptive of THIS sample. None of it supports inference about")
print("  a wider population, and no correlation here establishes causation.")

print("\n" + "=" * 80)
print("END OF REPORT")
print("=" * 80)
