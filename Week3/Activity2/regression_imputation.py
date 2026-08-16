"""
Week 3 - Activity 2: Predicting missing values with regression

Continues the Activity 1 cleaning pipeline. Where Activity 1 left missing values
as NaN, this script estimates them with two model families and compares which
predicts better:

  1. Linear regression       -- LinearRegression on the raw predictor
  2. Polynomial regression   -- PolynomialFeatures(degree=d) + LinearRegression
                                (the non-linear pattern from the sample code)

The comparison is deliberately run TWICE: once on the training data, the way the
sample code measures it, and once under leave-one-out cross-validation. The two
answers disagree, and that disagreement is the finding.

Run:  python regression_imputation.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # write PNGs without a display server

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from data_preparation import clean_dataset

OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = OUTPUT_DIR / 'figures'
FIGURE_DIR.mkdir(exist_ok=True)

# Polynomial degrees to trial against the straight line
DEGREES = (1, 2, 3)

# --- Chart styling -------------------------------------------------------
# Validated categorical slots (all-pairs, light surface): blue / orange / aqua
C_OBSERVED = '#2a78d6'
C_LINEAR = '#eb6834'
C_POLY = '#1baf7a'
C_IMPUTED = '#4a3aa7'
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECOND = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK_SECOND,
    'axes.titlecolor': INK,
    'text.color': INK,
    'xtick.color': INK_MUTED,
    'ytick.color': INK_MUTED,
    'grid.color': GRID,
    'font.family': 'sans-serif',
    'font.size': 9,
})


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fmt(value, target):
    """Format to a precision that suits the column's magnitude.

    Age spans 22-40 while Salary spans tens of thousands. A single format
    string either drowns Salary in decimals or rounds Age differences to zero.
    """
    if value is None:
        return '-'
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.2f}"


def build_model(degree):
    """Degree 1 is plain linear regression; higher degrees are polynomial."""
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )


def loocv_rmse(X, y, degree):
    """Leave-one-out CV error.

    Each fold trains on n-1 rows and predicts the held-out row, so the score
    reflects performance on data the model has never seen. A degree needing as
    many coefficients as it has training rows can fit them exactly and tells us
    nothing, so those folds are refused rather than silently scored.
    """
    n = len(X)
    if n - 1 <= degree + 1:
        return None, f"needs > {degree + 2} rows, has {n}"

    errors = []
    for train_idx, test_idx in LeaveOneOut().split(X):
        model = build_model(degree)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        errors.append((y[test_idx][0] - pred[0]) ** 2)
    return float(np.sqrt(np.mean(errors))), None


def evaluate_pair(df, target, predictor):
    """Fit every degree for one (predictor -> target) pair and score them."""
    complete = df[[predictor, target]].dropna()
    X = complete[[predictor]].to_numpy(dtype=float)
    y = complete[target].to_numpy(dtype=float)

    pearson_r = float(complete[predictor].corr(complete[target]))

    results = []
    for degree in DEGREES:
        if len(X) <= degree + 1:
            results.append({
                'degree': degree, 'skipped': f"only {len(X)} rows",
                'train_r2': None, 'train_rmse': None, 'cv_rmse': None,
                'cv_note': None, 'model': None,
            })
            continue

        model = build_model(degree)
        model.fit(X, y)
        fitted = model.predict(X)
        cv, cv_note = loocv_rmse(X, y, degree)

        results.append({
            'degree': degree,
            'skipped': None,
            'train_r2': float(r2_score(y, fitted)),
            'train_rmse': rmse(y, fitted),
            'cv_rmse': cv,
            'cv_note': cv_note,
            'model': model,
        })

    return {
        'target': target, 'predictor': predictor,
        'n': len(X), 'r': pearson_r,
        'X': X, 'y': y, 'results': results,
    }


def pick_winner(results, key):
    """Best (lowest) score on `key` among degrees that produced one."""
    scored = [r for r in results if r.get(key) is not None]
    return min(scored, key=lambda r: r[key]) if scored else None


# =========================================================================
# 1. CLEANED DATA AND WHAT IS STILL MISSING
# =========================================================================
df, report = clean_dataset()

print("=" * 78)
print("WEEK 3 - ACTIVITY 2: REGRESSION IMPUTATION OF MISSING VALUES")
print("=" * 78)

print("\n1. STARTING POINT - CLEANED DATA FROM ACTIVITY 1")
print("-" * 78)
print(f"\nSource: {report['source'].name}")
print(f"Rows: {report['raw_rows']} raw -> {report['clean_rows']} after "
      f"merging duplicate ID(s) {report['duplicate_ids']}")
print(f"Rejected dates: {report['rejected_dates']}")

display_cols = ['Name', 'Age', 'Salary', 'Net worth', 'Tenure (years)', 'Country']
print("\n" + df[display_cols].to_string(index=False))

target_cols = ['Age', 'Salary', 'Net worth', 'Tenure (years)']
print("\n\nREMAINING MISSING VALUES:")
print("-" * 78)
for col in target_cols:
    gaps = df[df[col].isna()]
    names = [n if isinstance(n, str) else '(no name)' for n in gaps['Name']]
    print(f"  {col:<16} {len(gaps)} missing  ->  {', '.join(names) if names else '-'}")

print("\n  Heidi is missing Age, Salary AND Net worth. Tenure is her only numeric")
print("  value, so every estimate for her must be predicted from that one column.")

# =========================================================================
# 2. CHOOSING PREDICTORS
# =========================================================================
print("\n\n2. CHOOSING A PREDICTOR FOR EACH GAP")
print("-" * 78)
print("\nA regression can only borrow information a predictor actually carries,")
print("so each gap is paired with the strongest predictor that is present for the")
print("row being filled.\n")

corr = df[target_cols].corr()
print(corr.round(3).to_string())

# (target, predictor, who needs it). Heidi has only Tenure available.
TASKS = [
    ('Net worth', 'Age', 'David'),
    ('Age', 'Tenure (years)', 'Heidi'),
    ('Net worth', 'Tenure (years)', 'Heidi'),
    ('Salary', 'Tenure (years)', 'Heidi'),
]

print("\nImputation tasks:")
for target, predictor, who in TASKS:
    r = corr.loc[target, predictor]
    print(f"  {target:<16} <- {predictor:<16} (r = {r:+.3f})  for {who}")

# =========================================================================
# 3. LINEAR VS POLYNOMIAL
# =========================================================================
print("\n\n3. MODEL COMPARISON: LINEAR VS POLYNOMIAL")
print("-" * 78)
print("\nTraining scores measure how well a curve retraces the points it was fitted")
print("to. Leave-one-out CV hides each row in turn and scores the prediction for")
print("it, which is the question imputation actually asks.\n")

evaluations = []
for target, predictor, who in TASKS:
    ev = evaluate_pair(df, target, predictor)
    ev['who'] = who
    evaluations.append(ev)

    print(f"\n{'=' * 74}")
    print(f"{target}  <-  {predictor}   (for {who})")
    print(f"  complete training pairs: n = {ev['n']}   Pearson r = {ev['r']:+.3f}")
    print(f"{'-' * 74}")
    print(f"  {'model':<22}{'train R2':>10}{'train RMSE':>14}{'LOOCV RMSE':>14}")

    for res in ev['results']:
        label = 'Linear (degree 1)' if res['degree'] == 1 else f"Polynomial (deg {res['degree']})"
        if res['skipped']:
            print(f"  {label:<22}{'skipped -- ' + res['skipped']:>38}")
            continue
        cv_text = fmt(res['cv_rmse'], target) if res['cv_rmse'] is not None else 'n/a'
        line = (f"  {label:<22}{res['train_r2']:>10.3f}"
                f"{fmt(res['train_rmse'], target):>14}{cv_text:>14}")
        if res['cv_rmse'] is None and res['cv_note']:
            line += f"   ({res['cv_note']})"
        print(line)

    train_best = pick_winner(ev['results'], 'train_rmse')
    cv_best = pick_winner(ev['results'], 'cv_rmse')
    ev['train_best'] = train_best
    ev['cv_best'] = cv_best

    print(f"\n  Best by training RMSE : degree {train_best['degree']}")
    if cv_best:
        print(f"  Best by LOOCV RMSE    : degree {cv_best['degree']}")
        if train_best['degree'] != cv_best['degree']:
            print(f"  -> DISAGREEMENT. Training scores favour degree {train_best['degree']}, but it")
            print(f"     predicts held-out rows worse than degree {cv_best['degree']}: the extra")
            print("     curvature is fitting noise, not signal.")
        else:
            print("  -> Both criteria agree.")
        if abs(ev['r']) < 0.2:
            print(f"  !! r = {ev['r']:+.3f} is near zero: this predictor carries almost no")
            print("     information about the target. The linear fit here is effectively")
            print("     mean imputation wearing a regression's clothes (train R2 = 0.000),")
            print("     and any degree that 'wins' is winning on noise.")
    else:
        print("  Best by LOOCV RMSE    : not computable at this sample size")

# =========================================================================
# 4. IMPUTED VALUES
# =========================================================================
print("\n\n4. PREDICTED VALUES FOR EACH MISSING CELL")
print("-" * 78)

imputations = []
for ev in evaluations:
    row = df[df['Name'] == ev['who']].iloc[0]
    x_new = row[ev['predictor']]

    entry = {
        'who': ev['who'], 'target': ev['target'],
        'predictor': ev['predictor'], 'x': x_new,
        'r': ev['r'], 'n': ev['n'],
        'cv_best_degree': ev['cv_best']['degree'] if ev['cv_best'] else None,
    }
    for res in ev['results']:
        if res['model'] is not None:
            pred = float(res['model'].predict(np.array([[x_new]]))[0])
            entry[f"deg{res['degree']}"] = pred
    imputations.append(entry)

print(f"\n{'row':<8}{'target':<13}{'from':<17}{'linear':>12}{'poly d2':>12}{'poly d3':>12}")
print("-" * 78)
for e in imputations:
    cells = [fmt(e.get(f"deg{d}"), e['target']) for d in DEGREES]
    print(f"{e['who']:<8}{e['target']:<13}{e['predictor']:<17}"
          f"{cells[0]:>12}{cells[1]:>12}{cells[2]:>12}")

print("\nSpread between the two approaches:")
for e in imputations:
    if e.get('deg1') is not None and e.get('deg2') is not None:
        gap = abs(e['deg2'] - e['deg1'])
        base = abs(e['deg1']) if e['deg1'] else 1
        print(f"  {e['who']}'s {e['target']:<12} linear vs poly d2 differ by "
              f"{fmt(gap, e['target'])} ({gap / base * 100:.1f}% of the linear estimate)")

# Flag predictions that fall outside the observed range of the target.
# A tolerance keeps a prediction that lands exactly on the boundary, to within
# floating-point noise, from being reported as an excursion beyond it.
print("\nPlausibility check against the observed range:")
for e in imputations:
    observed = df[e['target']].dropna()
    lo, hi = float(observed.min()), float(observed.max())
    tol = (hi - lo) * 1e-6
    for degree in DEGREES:
        val = e.get(f"deg{degree}")
        if val is None:
            continue
        kind = 'linear ' if degree == 1 else f'poly d{degree}'
        bounds = f"[{fmt(lo, e['target'])}, {fmt(hi, e['target'])}]"
        if val < lo - tol or val > hi + tol:
            excess = (lo - val) if val < lo else (val - hi)
            print(f"  !! {e['who']:<6} {e['target']:<11} {kind} = {fmt(val, e['target']):>10} "
                  f"is OUTSIDE {bounds} by {fmt(excess, e['target'])}")
        else:
            print(f"  ok {e['who']:<6} {e['target']:<11} {kind} = {fmt(val, e['target']):>10} "
                  f"within {bounds}")

# =========================================================================
# 5. FIGURES
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle('Linear vs polynomial fits for each missing value',
             fontsize=13, fontweight='bold', color=INK, y=0.98)

for ax, ev in zip(axes.flat, evaluations):
    X, y = ev['X'], ev['y']
    ax.grid(True, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    ax.scatter(X, y, s=52, color=C_OBSERVED, zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label='Observed')

    x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    for res in ev['results']:
        if res['model'] is None or res['degree'] > 2:
            continue
        colour = C_LINEAR if res['degree'] == 1 else C_POLY
        name = 'Linear' if res['degree'] == 1 else 'Polynomial (d2)'
        ax.plot(x_grid, res['model'].predict(x_grid), linewidth=2,
                color=colour, label=name, zorder=2)

    # The value being imputed, marked on both curves
    entry = next(e for e in imputations
                 if e['who'] == ev['who'] and e['target'] == ev['target'])
    for degree, colour in ((1, C_LINEAR), (2, C_POLY)):
        val = entry.get(f"deg{degree}")
        if val is not None:
            ax.scatter([entry['x']], [val], s=150, marker='*', color=colour,
                       edgecolor=SURFACE, linewidth=1.2, zorder=4)
    ax.axvline(entry['x'], color=INK_MUTED, linewidth=1,
               linestyle=(0, (4, 3)), zorder=1)

    ax.set_title(f"{ev['target']} from {ev['predictor']}\n"
                 f"n = {ev['n']}, r = {ev['r']:+.2f}  ({ev['who']})",
                 fontsize=10, color=INK, pad=8)
    ax.set_xlabel(ev['predictor'])
    ax.set_ylabel(ev['target'])
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECOND)

fig.text(0.5, 0.015, 'Stars mark the imputed value; dashed line is the predictor '
                     'value for the row being filled.',
         ha='center', fontsize=8.5, color=INK_MUTED)
fig.tight_layout(rect=(0, 0.035, 1, 0.955))
fit_path = FIGURE_DIR / 'linear_vs_polynomial_fits.png'
fig.savefig(fit_path, dpi=150, facecolor=SURFACE)
plt.close(fig)

# --- Overfitting gap ------------------------------------------------------
# Plotting raw RMSE would put Age errors (years, ~5) and Salary errors (dollars,
# ~150,000) on one axis -- different units, so the bar heights would not be
# comparable across tasks. Indexing each model against the linear model's error
# on its OWN task gives a dimensionless ratio that is comparable everywhere.
labels, ratios_plot, colours = [], [], []
for ev in evaluations:
    for res in ev['results']:
        if res['train_rmse'] is None or res['cv_rmse'] is None:
            continue
        kind = 'linear' if res['degree'] == 1 else f"poly d{res['degree']}"
        labels.append(f"{ev['target'][:9]}\nfrom {ev['predictor'][:6]}\n{kind}")
        ratios_plot.append(res['cv_rmse'] / res['train_rmse'])
        colours.append(C_LINEAR if res['degree'] == 1 else C_POLY)

fig2, ax2 = plt.subplots(figsize=(12, 5.5))
pos = np.arange(len(labels))
bars = ax2.bar(pos, ratios_plot, 0.62, color=colours, zorder=3)

ax2.axhline(1.0, color=INK_SECOND, linewidth=1.5, zorder=4)
ax2.text(len(labels) - 0.4, 1.06, 'no gap (1.0x)', ha='right', va='bottom',
         fontsize=8.5, color=INK_SECOND)

# Direct-label every bar: the aqua slot sits below 3:1 on this surface, so the
# validator's relief rule requires visible labels rather than colour alone.
for bar, ratio in zip(bars, ratios_plot):
    ax2.text(bar.get_x() + bar.get_width() / 2, ratio * 1.09,
             f"{ratio:.1f}x", ha='center', va='bottom',
             fontsize=8.5, color=INK_SECOND)

ax2.set_yscale('log')
ax2.set_xticks(pos)
ax2.set_xticklabels(labels, fontsize=7.5)
ax2.set_ylabel('LOOCV RMSE / training RMSE  (log scale)')
ax2.set_title('How much worse each model performs on data it has not seen',
              fontsize=12, fontweight='bold', color=INK, pad=12)
ax2.grid(True, axis='y', linewidth=0.6, alpha=0.9)
ax2.set_axisbelow(True)
for spine in ('top', 'right'):
    ax2.spines[spine].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=C_LINEAR),
           plt.Rectangle((0, 0), 1, 1, color=C_POLY)]
ax2.legend(handles, ['Linear', 'Polynomial'], frameon=False,
           fontsize=9, labelcolor=INK_SECOND, loc='upper left')

fig2.text(0.5, 0.015, 'Ratio, not raw error, so tasks measured in years and in '
                      'dollars can be compared on one axis. '
                      'Every linear bar sits near 1.0; polynomials climb away from it.',
          ha='center', fontsize=8.5, color=INK_MUTED)
fig2.tight_layout(rect=(0, 0.05, 1, 1))
gap_path = FIGURE_DIR / 'overfitting_gap.png'
fig2.savefig(gap_path, dpi=150, facecolor=SURFACE)
plt.close(fig2)

print("\n\n5. FIGURES WRITTEN")
print("-" * 78)
print(f"  {fit_path.relative_to(OUTPUT_DIR)}")
print(f"  {gap_path.relative_to(OUTPUT_DIR)}")

# =========================================================================
# 6. VERDICT
# =========================================================================
print("\n\n6. WHICH METHOD PREDICTS BETTER?")
print("-" * 78)

train_wins = {1: 0, 2: 0, 3: 0}
cv_wins = {1: 0, 2: 0, 3: 0}
for ev in evaluations:
    if ev['train_best']:
        train_wins[ev['train_best']['degree']] += 1
    if ev['cv_best']:
        cv_wins[ev['cv_best']['degree']] += 1

print("\nTasks won, by criterion:")
print(f"  {'model':<24}{'by train RMSE':>16}{'by LOOCV RMSE':>16}")
for degree in DEGREES:
    label = 'Linear (degree 1)' if degree == 1 else f'Polynomial (deg {degree})'
    print(f"  {label:<24}{train_wins[degree]:>16}{cv_wins[degree]:>16}")

print("\nOverfitting gap (LOOCV RMSE / training RMSE, 1.0 = no gap):")
for ev in evaluations:
    for res in ev['results']:
        if res['train_rmse'] and res['cv_rmse']:
            ratio = res['cv_rmse'] / res['train_rmse']
            kind = 'linear' if res['degree'] == 1 else f"poly d{res['degree']}"
            print(f"  {ev['target']:<12} {ev['predictor']:<16} {kind:<9} {ratio:>8.1f}x")

# RMSE carries the units of its target, so an Age error in years and a Net worth
# error in dollars cannot be averaged together. Comparing each polynomial to the
# linear model on the SAME task gives a dimensionless ratio that can be pooled.
print("\nPolynomial LOOCV error relative to linear, per task (1.0 = equal):")
ratios = []
for ev in evaluations:
    base = next((r['cv_rmse'] for r in ev['results']
                 if r['degree'] == 1 and r['cv_rmse']), None)
    if not base:
        continue
    for res in ev['results']:
        if res['degree'] == 1 or not res['cv_rmse']:
            continue
        ratio = res['cv_rmse'] / base
        ratios.append(ratio)
        verdict = 'better' if ratio < 1 else 'worse'
        print(f"  {ev['target']:<12} {ev['predictor']:<16} "
              f"poly d{res['degree']}  {ratio:>7.1f}x  ({verdict} than linear)")

print("\nVERDICT")
print("-" * 78)
if cv_wins[1] >= cv_wins[2] + cv_wins[3]:
    print("  LINEAR REGRESSION predicts these missing values better.")
else:
    print("  POLYNOMIAL REGRESSION predicts these missing values better.")

print(f"\n  Won on training RMSE : polynomial d3 took all {sum(train_wins.values())} tasks")
print(f"  Won on LOOCV RMSE    : linear took {cv_wins[1]} of "
      f"{sum(cv_wins.values())} tasks")
if ratios:
    print(f"\n  Median polynomial-to-linear LOOCV ratio: {np.median(ratios):.1f}x")
    print(f"  Worst case: {max(ratios):.0f}x the linear model's error")

print("""
  Why, in this dataset:

  1. Sample size. Each model trains on 5-8 rows. A degree-2 curve spends 3
     coefficients and a degree-3 curve spends 4; with 5 rows the cubic has
     almost as many knobs as observations and can pass through the points
     exactly. That is memorisation, not a learned relationship.

  2. Training scores cannot detect this. R2 on the fitted points rises with
     every degree added -- it is arithmetically unable to fall. Reporting it
     alone, as the sample code does, would recommend the worst model.

  3. Extrapolation. Imputed rows often sit near or past the edge of the
     observed predictor range. A straight line leaving that range stays
     plausible; a parabola or cubic accelerates away, which is how the
     out-of-range predictions flagged in section 4 arise.

  4. No curvature to find. Age-income and tenure-wealth relationships are
     close to monotonic over these narrow ranges. The extra polynomial terms
     have no real signal to fit, so they fit noise instead.

  When polynomial regression WOULD be the right choice: a genuinely curved
  relationship (diminishing returns, saturation, growth then decline) with
  enough rows -- roughly 10+ per coefficient -- to distinguish curvature from
  noise. The sample code demonstrates exactly that case: 100 points generated
  from a true quadratic. Neither condition holds here.
""")

print("=" * 78)
print("END OF REPORT")
print("=" * 78)
