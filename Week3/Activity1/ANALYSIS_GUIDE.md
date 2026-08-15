# Week 3 - Activity 1: Initial Statistical Analysis
## Comprehensive Guide to Statistical Metrics

---

## Executive Summary

This statistical analysis examines a dataset of individuals with information about their age, salary, net worth, country, and join dates. The raw file contains 10 rows, which reduce to **9 unique people** after merging a duplicated record. Key findings:

- **Average Age**: 30.8 years (Range: 22-40)
- **Average Salary**: $62,625.00 (Range: $55,000-$72,000)
- **Average Net Worth**: $38,571.43 (Range: $22,000-$60,000)
- **Average Tenure**: 6.52 years (Range: 5.06-8.21)
- **Primary Location**: New Zealand (5 of 9 records, 56%)

The strongest relationships are **tenure vs net worth (r = 0.84)** and **age vs tenure (r = 0.72)**. Salary correlates strongly with age (r = 0.63) but is essentially unrelated to both net worth (r = 0.15) and tenure (r = 0.01).

> **Read the sample sizes before the coefficients.** Every correlation here rests
> on 5–8 complete pairs. `r = 0.84` sounds authoritative; it is computed from
> **five people**. These figures describe this sample and support no inference
> about any wider population.

---

## Part 1: Data Quality Overview

### Missing Data Analysis

| Column | Missing Count | Missing % | Interpretation |
|--------|---------------|-----------|---|
| ID | 1 | 10% | One record lacks an ID |
| Name | 1 | 10% | One person's name is missing |
| Age | 2 | 20% | Two age values missing (including one non-numeric "thirty-eight") |
| Net worth | 3 | 30% | Three net worth values missing |
| Country | 1 | 10% | One country not specified |
| Salary | 2 | 20% | Two salary values missing (one blank, one non-numeric) |
| Join Date | 1 | 10% | One join date missing |

**Why This Matters**: Missing data can bias results. In this dataset, 30% missing net worth values suggest caution when interpreting wealth metrics. Strategies include:
- **Deletion**: Remove incomplete records (reduces sample size)
- **Imputation**: Fill with mean/median (adds assumptions)
- **Analysis-specific**: Only use records with complete data for specific analyses

### Data Type Issues Encountered
- "thirty-eight" stored as text in Age column → Converted to 38
- "sixty five thousand" stored as text in Salary → Converted to 65000
- "30,000" with commas in Net worth → Removed commas before conversion
- Invalid date "2019-13-01" (month 13 doesn't exist) → Rejected as missing

### Three Structural Fixes Applied

**1. Duplicate record merged.** ID `2` (Bob) appeared on two rows with
complementary data — one had Age and Net worth but no Salary, the other had
Salary but neither Age nor Net worth. These were merged into a single complete
record by taking the first non-null value per column. Row count: **10 → 9**.
Without this, Bob would be double-counted in the country distribution.

**2. Country codes standardised.** `AU` and `AUS` both denote Australia but were
counted as separate categories, inflating the country count to 3. After mapping
`AU → AUS`, the dataset spans **2 countries**, not 3.

**3. Join Date parsed into tenure.** Dates are parsed against an explicit format
list (`%d/%m/%Y`, `%Y-%m-%d`) rather than a permissive parser.

> **Why explicit formats matter:** a permissive parser reads `2019-13-01` as
> YYYY-DD-MM and silently returns 13 January 2019 — inventing a plausible date
> from invalid input. The first version of this script did exactly that, and
> reported 9/10 dates parsed with zero problems flagged. Restricting to known
> formats correctly rejects it, giving 8/10 parsed with the bad value reported.

A derived `Tenure (years)` column measures time from join date to a fixed
analysis date of **2026-08-16**. The date is hard-coded rather than using
`today()` so the numbers are reproducible on re-run.

---

## Part 2: Descriptive Statistics Explained

### **Mean** (Average)
**Formula**: Sum of all values ÷ Number of values

**Example - Age**: (25 + 29 + 30 + 35 + 38 + 40 + 27 + 22) ÷ 8 = 30.75 years

**How to Interpret**: 
- Represents the "center" of your data
- Sensitive to extreme values (outliers)
- If mean > median, data is right-skewed (has large outliers)

**Age Mean = 30.75**: On average, people in this dataset are just over 30 years old

---

### **Median** (Middle Value)
**Method**: Sort all values, pick the middle one (or average of two middle if even number)

**Example - Age**: Sorted ages: 22, 25, 27, 29, 30, 35, 38, 40 → Median = (29 + 30) ÷ 2 = 29.5

**How to Interpret**:
- More resistant to outliers than mean
- Useful for skewed distributions
- If median < mean, there are extreme high values pulling the average up

**Age Median = 29.5**: Half the group is younger than 29.5, half is older

---

### **Standard Deviation (Std Dev)**
**Formula**: √[Sum of (each value - mean)² ÷ (n-1)]

**Example Logic**: 
- Measures how spread out data is from the average
- Higher std dev = more variability = less predictable

**Age Std Dev = 6.36**: On average, people differ from the mean age by about 6.4 years

**Salary Std Dev = 5655.28**: Salaries vary by about $5,655 from the average ($62,625)

**Comparing spread across different units**: Age std dev (6.36 years) and salary
std dev ($5,655) cannot be compared directly — different units. Use the
**coefficient of variation** (CV = std dev ÷ mean × 100):
- Salary CV = 5655 ÷ 62625 = **9.0%** → tight clustering around the mean
- Net worth CV = 14164 ÷ 38571 = **36.7%** → four times more dispersed

So despite salary having the larger raw standard deviation, **net worth is by far
the more unequal variable**. Raw std dev alone would have told you the opposite.

**How to Interpret**:
- Small std dev → Data clustered around mean → More consistent
- Large std dev → Data spread out → More diversity
- Use the "68-95-99.7 rule" with normal distributions:
  - 68% of data within 1 std dev of mean
  - 95% within 2 std devs
  - 99.7% within 3 std devs

---

### **Variance**
**Formula**: (Standard Deviation)²

**Age Variance = 40.50**: This is 6.36²

**How to Interpret**: 
- Harder to interpret directly (different units than original data)
- Mainly used in statistical calculations
- Larger variance = more dispersed data

---

### **Minimum & Maximum**
**How to Interpret**: Shows the range of your data

**Age Min = 22, Max = 40**: Youngest is 22, oldest is 40

**Salary Min = $55,000, Max = $72,000**: Lowest salary is $55k, highest is $72k

**Use for**: Identifying extreme values, checking data entry errors

---

### **Range**
**Formula**: Maximum - Minimum

**Age Range = 40 - 22 = 18 years**

**How to Interpret**: 
- Simple measure of spread
- Very sensitive to outliers
- Useful for quick overview, but limited statistical value

---

### **Percentiles / Quartiles**

**25th Percentile (Q1)**: 25% of data falls below this value

**Age Q1 = 26.5**: 25% of people are younger than 26.5 years old

**50th Percentile (Q2/Median)**: 50% of data falls below (already covered above)

**75th Percentile (Q3)**: 75% of data falls below this value

**Age Q3 = 35.75**: 75% of people are younger than 35.75 years old

**How to Interpret**: Useful for understanding distribution across groups
- Bottom 25%, Middle 50%, Top 25%
- Better than mean/median for seeing overall shape

---

### **Interquartile Range (IQR)**
**Formula**: Q3 - Q1

**Age IQR = 35.75 - 26.5 = 9.25 years**

**How to Interpret**:
- Range where middle 50% of data falls
- Ignores extreme 25% on each end
- More robust than range to outliers
- Used for outlier detection

**Rule for Outliers**: Any value outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] is considered an outlier

**Age Outlier Range**: [26.5 - 1.5×9.25, 35.75 + 1.5×9.25] = [12.62, 49.62]
- Since all ages fall within this range, **no age outliers detected**

---

### **Skewness**
**What it measures**: How symmetric (or asymmetric) is your distribution?

**Ranges**:
- **-1 to -0.5**: Left-skewed (tail on left, median > mean)
- **-0.5 to 0.5**: Approximately symmetric
- **0.5 to 1**: Right-skewed (tail on right, mean > median)
- **> 1 or < -1**: Highly skewed

**Age Skewness = 0.244** (slightly right-skewed)
- Slightly more data clustered on the left, with tail on right
- Mean (30.75) > Median (29.5), confirming right skew

**Salary Skewness = 0.637** (moderately right-skewed)
- Some people earn notably more, pulling average up

**Net Worth Skewness = 0.646** (moderately right-skewed)
- Some individuals have significantly higher wealth

**How to Interpret**: 
- Skewed data may violate assumptions of some statistical tests
- Consider using median instead of mean for highly skewed data

---

### **Kurtosis**
**What it measures**: How heavy are the tails? Are there extreme outliers?

**Ranges**:
- **Negative**: Lighter tails (fewer outliers, flatter distribution)
- **Near 0**: Similar to normal distribution
- **Positive**: Heavier tails (more extreme outliers)

**Age Kurtosis = -1.227** (negative)
- Lighter tails, fewer extreme ages
- Distribution is flatter than normal

**Salary Kurtosis = -0.681** (negative)
- Fewer extreme salaries
- More uniform distribution across salary range

**How to Interpret**: 
- Helps identify how unusual extreme values are
- Important for risk assessment (more kurtosis = more risk of extreme events)

---

## Part 3: Correlation Analysis

### **Correlation Coefficient (Pearson's r)**

**What it measures**: How strongly two variables move together

**Range**: -1.0 to +1.0

| Range | Interpretation |
|-------|---|
| 0.7 to 1.0 | Very strong positive: As X increases, Y strongly increases |
| 0.5 to 0.7 | Strong positive: As X increases, Y generally increases |
| 0.3 to 0.5 | Moderate positive: Some tendency for Y to increase with X |
| 0.0 to 0.3 | Weak positive: Little relationship, slight upward tendency |
| 0.0 to -0.3 | Weak negative: Little relationship, slight downward tendency |
| -0.3 to -0.5 | Moderate negative: Some tendency for Y to decrease as X increases |
| -0.5 to -0.7 | Strong negative: As X increases, Y generally decreases |
| -0.7 to -1.0 | Very strong negative: As X increases, Y strongly decreases |
| 0 | No correlation: No linear relationship |

### **Key Findings from Dataset**:

Full matrix after cleaning:

|                | Age | Salary | Net worth | Tenure |
|----------------|-----|--------|-----------|--------|
| **Age**        | 1.000 | 0.628 | 0.484 | 0.717 |
| **Salary**     | 0.628 | 1.000 | 0.153 | 0.011 |
| **Net worth**  | 0.484 | 0.153 | 1.000 | 0.839 |
| **Tenure**     | 0.717 | 0.011 | 0.839 | 1.000 |

**Pairwise deletion matters here.** `pandas.corr()` computes each cell from the
rows where *both* variables are present, so every coefficient has a different
sample size:

| Pair | r | Complete pairs |
|---|---|---|
| Net worth vs Tenure | 0.839 | **5** |
| Age vs Tenure | 0.717 | 6 |
| Age vs Salary | 0.628 | 8 |
| Age vs Net worth | 0.484 | 7 |
| Salary vs Net worth | 0.153 | 7 |
| Salary vs Tenure | 0.011 | 6 |

**Net Worth vs Tenure: r = 0.839** (Very Strong Positive) — *strongest in dataset*
- Interpretation: The longer someone has been with the organisation, the more wealth they have accumulated
- Intuitive: wealth builds over time through sustained saving
- R² = 0.70: tenure tracks roughly 70% of net worth variation
- **Built on 5 pairs.** This is the least trustworthy figure in the report despite being the largest

**Age vs Tenure: r = 0.717** (Very Strong Positive)
- Interpretation: Older individuals joined earlier
- Expected, but it creates **multicollinearity** — age and tenure carry overlapping information, so including both in a regression model would be problematic

**Age vs Salary: r = 0.628** (Strong Positive)
- Interpretation: Older individuals tend to earn higher salaries
- Correlation ≠ causation; experience and job level likely drive this
- R² = 0.39: Age tracks about 39% of salary variation
- Largest sample of any pair (8), so relatively the most stable result here

**Age vs Net Worth: r = 0.484** (Moderate Positive)
- Interpretation: Older people tend to have higher net worth
- Weaker than the tenure–net worth link, suggesting time-in-role matters more than age alone

**Salary vs Net Worth: r = 0.153** (Weak Positive)
- Interpretation: Current income barely predicts accumulated wealth
- Net worth depends on savings history, investments, and inheritance — not just present salary
- Two people on identical salaries can have vastly different net worth

**Salary vs Tenure: r = 0.011** (Essentially Zero)
- Interpretation: No detectable relationship between how long someone has been here and what they earn

> ### A worked lesson in how a data bug fabricates a finding
>
> An earlier version of this analysis reported **r = −0.312** for this pair and
> offered a tidy explanation: newer hires being recruited at higher market rates,
> long-tenured staff sitting in flatter pay bands. It read plausibly.
>
> It was an artefact. The salary cleaner silently dropped
> `"sixty five thousand"` to `NaN` instead of converting it to `65000`, removing
> one person from every salary calculation. Restoring that single value moved the
> coefficient from −0.312 to **0.011** — the negative relationship vanished
> entirely, and with it the explanation built on top of it.
>
> Two lessons: a cleaning bug that loses one row out of nine can invent a
> finding, and a coefficient computed from six pairs is fragile enough that the
> correct response to *any* of these numbers is to check `n` first and resist
> explaining a pattern before confirming it is real.

---

## Part 4: Categorical Analysis

### **Country Distribution**

After standardising `AU → AUS` and merging the duplicate record:

| Country | Count | Percentage |
|---------|-------|-----------|
| NZ | 5 | 56% |
| AUS | 3 | 33% |
| Missing | 1 | 11% |

**How to Interpret**:
- Majority of records (56%) are from New Zealand
- This is a **geographic bias** — results may not represent a global population
- One record lacks country data

**Before cleaning** this table read NZ 6 / AUS 2 / AU 1 / missing 1 across
"3 countries" — two separate errors compounding: Bob counted twice under NZ, and
Australia split across two spellings. Both are now resolved.

---

## Part 5: Key Insights & Business Implications

### **1. Age & Salary Relationship**
- **Finding**: Strong positive correlation (r = 0.628, n = 8)
- **Implication**: Seniority appears to drive compensation
- **Recommendation**: Compare salary growth patterns by age cohort

### **2. Wealth Is Far More Unequal Than Income**
- **Finding**: Net worth CV = 36.7% vs salary CV = 9.0%
- **Implication**: Salaries cluster tightly; accumulated wealth does not. People on
  comparable pay hold very different net worth
- **Recommendation**: Investigate savings behaviour and time-in-role, not pay bands

### **3. Missing Data Pattern**
- **Finding**: Net worth has highest missing rate (30%)
- **Implication**: Wealth information harder to obtain than salary/age
- **Recommendation**: Collect more complete net worth data in future surveys

### **4. Geographic Concentration**
- **Finding**: 56% from New Zealand (5 of 9)
- **Implication**: Findings may reflect NZ market conditions
- **Recommendation**: Expand data collection for global perspective

### **5. No Salary-Wealth Correlation**
- **Finding**: r = 0.03 between salary and net worth
- **Implication**: Income doesn't strongly determine wealth
- **Recommendation**: Investigate savings rates, investment returns, inheritance effects

---

## Part 6: Statistical Considerations

### **Assumptions & Limitations**

1. **Small Sample Size (n=10)**: 
   - Results may not generalize to larger population
   - Outliers have more influence
   - Correlations less stable

2. **Missing Data**:
   - Reduces effective sample size
   - May introduce bias if data missing non-randomly

3. **Normal Distribution Assumption**:
   - Skewness values (0.24-0.65) suggest slight right-skew
   - Some statistical tests assume normality; violated slightly here

4. **Linear Relationships**:
   - Correlation measures only linear relationships
   - May miss curved or non-linear patterns

### **Outlier Detection Results**

Using **Interquartile Range (IQR) Method**:
- **Age**: No outliers detected
  - Range: [12.62, 49.62] contains all ages
  
- **Salary**: No outliers detected
  - Range: [$47,250, $77,250] contains all salaries
  
- **Net Worth**: No outliers detected
  - Range: [$1,250, $75,250] contains all net worth values

**Interpretation**: No extreme unusual values; data appears consistent

---

## Part 7: Recommendations for Further Analysis

### **Data Quality Improvements**
1. Standardize country codes (AUS vs AU)
2. Validate dates against reasonable ranges
3. Confirm non-numeric values (e.g., "thirty-eight" for 38)
4. Collect missing values where possible

### **Statistical Enhancements**
1. Increase sample size for more robust results
2. Perform regression analysis to predict salary/net worth
3. Segment analysis by country to control for geographic differences
4. Investigate causation (not just correlation) for age-salary relationship

### **Data Visualization**
1. Scatter plot: Age vs Salary (shows moderate positive relationship)
2. Distribution plots: Histograms for age, salary, net worth
3. Box plots: Identify spread and outliers
4. Heatmap: Correlation matrix for all numeric variables

---

## Technical Notes

**Data Cleaning Applied**:
- Converted text numbers ("thirty-eight") to numeric
- Removed currency symbols and commas
- Handled empty strings and NaN values
- Standardized date formats where possible

**Statistical Methods Used**:
- Pearson correlation (assumes linear relationships)
- IQR method for outlier detection
- Descriptive statistics (mean, median, std dev, quartiles)

**Software**: Python 3 with pandas, numpy libraries

---

## Conclusion

This dataset provides insights into age-salary relationships and wealth distribution, but is limited by small size and missing data. The moderate age-salary correlation suggests that experience (proxied by age) influences compensation. The weak salary-wealth correlation indicates that wealth accumulation depends on factors beyond current income. Future analyses should focus on data quality improvements and larger sample collection.
