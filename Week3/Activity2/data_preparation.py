"""
Week 3 - Activity 2: Reusable data preparation

The cleaning logic developed in Activity 1 (statistical_analysis.py), extracted
into importable functions so Activity 2 builds on exactly the same cleaned data
rather than a second, drifting copy of the rules.

Nothing here prints at import time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Fixed analysis date keeps tenure reproducible across re-runs
ANALYSIS_DATE = pd.Timestamp('2026-08-16')

# Spelled-out numbers found in Age and Salary. Both columns share one table so
# the two are treated consistently.
WORD_NUMBERS = {
    'thirty-eight': 38.0,
    'thirty eight': 38.0,
    'sixty five thousand': 65000.0,
}

COUNTRY_ALIASES = {'AU': 'AUS'}

# Explicit formats only. A permissive parser reads "2019-13-01" as YYYY-DD-MM
# and silently invents 13 Jan 2019 rather than rejecting month 13.
ACCEPTED_DATE_FORMATS = ('%d/%m/%Y', '%Y-%m-%d')

NUMERIC_COLUMNS = ('Age', 'Salary', 'Net worth')


def find_dataset(start_dir=None):
    """Locate Sample_dataset.csv near this script (Activity2/ or Week3/)."""
    base = Path(start_dir) if start_dir else Path(__file__).resolve().parent
    candidates = (base / 'Sample_dataset.csv', base.parent / 'Sample_dataset.csv')
    for path in candidates:
        if path.is_file():
            return path
    searched = '\n  '.join(str(p) for p in candidates)
    raise FileNotFoundError(f"Sample_dataset.csv not found. Searched:\n  {searched}")


def to_number(val, column, log=None):
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

    if text.lower() in WORD_NUMBERS:
        number = WORD_NUMBERS[text.lower()]
        if log is not None:
            log.append(f"{column}: '{text}' -> {number:g} (spelled-out number)")
        return number

    stripped = text.replace(',', '').replace('"', '')
    try:
        number = float(stripped)
    except ValueError:
        if log is not None:
            log.append(f"{column}: '{text}' -> NaN (unrecognised, not guessed)")
        return np.nan

    if stripped != text and log is not None:
        log.append(f"{column}: '{text}' -> {number:g} (removed separators)")
    return number


def parse_join_date(val):
    """Parse a join date against known formats only; anything else is NaT."""
    if pd.isna(val) or str(val).strip() == '':
        return pd.NaT
    for fmt in ACCEPTED_DATE_FORMATS:
        parsed = pd.to_datetime(str(val).strip(), format=fmt, errors='coerce')
        if pd.notna(parsed):
            return parsed
    return pd.NaT


def clean_dataset(csv_path=None):
    """Run the full Activity 1 cleaning pipeline.

    Returns (df_clean, report) where report records what the cleaning did.
    """
    path = Path(csv_path) if csv_path else find_dataset()
    raw = pd.read_csv(path)
    df = raw.copy()

    report = {
        'source': path,
        'raw_rows': len(raw),
        'conversions': [],
        'rejected_dates': [],
        'duplicate_ids': [],
        'conflicts': [],
    }

    # 1. Numeric coercion
    for column in NUMERIC_COLUMNS:
        df[column] = df[column].apply(
            to_number, column=column, log=report['conversions']
        )

    # 2. Country codes
    report['countries_before'] = sorted(df['Country'].dropna().unique())
    df['Country'] = df['Country'].replace(COUNTRY_ALIASES)
    report['countries_after'] = sorted(df['Country'].dropna().unique())

    # 3. Dates -> tenure
    raw_dates = df['Join Date']
    df['Join Date'] = raw_dates.apply(parse_join_date)
    rejected = raw_dates[df['Join Date'].isna() & raw_dates.notna()]
    report['rejected_dates'] = list(rejected.values)
    df['Tenure (years)'] = (
        (ANALYSIS_DATE - df['Join Date']).dt.days / 365.25
    ).round(2)

    # 4. Deduplicate by ID, merging complementary fields
    ids = df.loc[df['ID'].notna(), 'ID']
    dup_ids = ids[ids.duplicated(keep=False)].unique()
    report['duplicate_ids'] = [int(i) for i in dup_ids]

    if len(dup_ids) > 0:
        for dup_id in dup_ids:
            rows = df[df['ID'] == dup_id]
            for column in df.columns:
                present = rows[column].dropna()
                if len(present) > 1 and present.nunique() > 1:
                    report['conflicts'].append(
                        f"ID {int(dup_id)} '{column}': {list(present.values)}"
                    )
        no_id = df[df['ID'].isna()]
        with_id = df[df['ID'].notna()]
        merged = with_id.groupby('ID', as_index=False, sort=False).first()
        df = pd.concat([merged, no_id], ignore_index=True)

    report['clean_rows'] = len(df)
    return df, report
