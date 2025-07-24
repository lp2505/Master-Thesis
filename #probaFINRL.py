#probaFINRL

import numpy as np
import pandas as pd
from scipy.stats import genextreme, logistic, norm, gumbel_r, wilcoxon

# === Fonctions ===

def empirical_probability(returns, sp_return):
    returns = np.array(returns)
    n = len(returns)
    count = np.sum(returns > sp_return)
    proba = count / n
    std_error = np.sqrt(proba * (1 - proba) / n)
    ci_low = max(0, proba - 1.96 * std_error)
    ci_high = min(1, proba + 1.96 * std_error)
    return proba, (ci_low, ci_high)

def bootstrap_probability_cdf(distribution, params, sp_return, n_iter=1000, perturb_pct=0.05):
    shape, loc, scale = params
    probas = []
    for _ in range(n_iter):
        pert_scale = scale * np.random.normal(1, perturb_pct)
        pert_loc = loc * np.random.normal(1, perturb_pct)
        pert_shape = shape * np.random.normal(1, perturb_pct) if shape is not None else None

        try:
            if pert_shape is not None:
                cdf_val = distribution.cdf(sp_return, pert_shape, loc=pert_loc, scale=pert_scale)
            else:
                cdf_val = distribution.cdf(sp_return, loc=pert_loc, scale=pert_scale)
            probas.append(1 - cdf_val)
        except:
            continue

    if len(probas) == 0:
        return None, (None, None)
    
    return np.mean(probas), (np.percentile(probas, 2.5), np.percentile(probas, 97.5))

def wilcoxon_one_sided(returns, sp_return):
    diff = np.array(returns) - sp_return
    if len(diff[diff > 0]) == 0:
        return None, 1.0
    try:
        stat, p_two_sided = wilcoxon(diff)
        return stat, p_two_sided / 2 if stat > 0 else 1 - p_two_sided / 2
    except:
        return None, None

# === Données ===

sp500_returns = {
    2012: 0.1341, 2013: 0.296, 2014: 0.1139, 2015: -0.0073,
    2016: 0.0954, 2017: 0.1942, 2018: -0.0624, 2019: 0.2888,
    2020: 0.1626, 2021: 0.2689, 2022: -0.1944, 2023: 0.2423,
    2024: 0.2331  # adapte selon tes valeurs
}

distribution_map = {
    "Normal": norm,
    "Logistic": logistic,
    "Gumbel": gumbel_r,
    "GEV": genextreme,
    "LogNormal": None  # Non utilisé ici
}

# === Chargement ===

df_returns = pd.read_csv("all_simulated_returns.csv")
df_fits = pd.read_csv("resultats_aggreges_10000.csv")

results = []

# === Analyse année par année ===

for year in sorted(sp500_returns.keys()):
    sp_return = sp500_returns[year]
    returns = df_returns[df_returns["year"] == year]["return"].values

    if len(returns) < 100:
        continue

    emp_proba, emp_ci = empirical_probability(returns, sp_return)
    w_stat, w_pval = wilcoxon_one_sided(returns, sp_return)

    fit_row = df_fits[df_fits["Year"] == year]
    if fit_row.empty:
        continue

    dist_name = fit_row["PDF"].values[0]
    dist = distribution_map.get(dist_name)
    shape = fit_row["Shape"].values[0]
    loc = fit_row["Location"].values[0]
    scale = fit_row["Scale"].values[0]

    if dist is None or np.isnan(scale):
        boot_proba, boot_ci = None, (None, None)
    else:
        boot_proba, boot_ci = bootstrap_probability_cdf(dist, (shape, loc, scale), sp_return)

    results.append({
        "Year": year,
        "S&P 500 Return": sp_return,
        "Empirical Proba": emp_proba,
        "Empirical CI Low": emp_ci[0],
        "Empirical CI High": emp_ci[1],
        "Wilcoxon Stat": w_stat,
        "Wilcoxon P-Value": w_pval,
        "Bootstrap Proba": boot_proba,
        "Bootstrap CI Low": boot_ci[0],
        "Bootstrap CI High": boot_ci[1]
    })

# === Export ===

df_results = pd.DataFrame(results)
df_results.to_csv("test_probabilite_vs_sp500.csv", index=False)
print("✅ Résultats exportés dans test_probabilite_vs_sp500.csv")
