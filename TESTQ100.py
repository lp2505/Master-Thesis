import numpy as np
import pandas as pd
from scipy.stats import genextreme, logistic, norm, gumbel_r, wilcoxon
import pickle

# Fonctions de test

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
        perturbed_scale = scale * np.random.normal(1, perturb_pct)
        perturbed_loc = loc * np.random.normal(1, perturb_pct)
        perturbed_shape = shape * np.random.normal(1, perturb_pct) if shape is not None else None

        try:
            if perturbed_shape is not None:
                cdf_val = distribution.cdf(sp_return, perturbed_shape, loc=perturbed_loc, scale=perturbed_scale)
            else:
                cdf_val = distribution.cdf(sp_return, loc=perturbed_loc, scale=perturbed_scale)
            probas.append(1 - cdf_val)
        except:
            continue

    if len(probas) == 0:
        return None, (None, None)
    
    proba_mean = np.mean(probas)
    ci_low = np.percentile(probas, 2.5)
    ci_high = np.percentile(probas, 97.5)
    return proba_mean, (ci_low, ci_high)

def wilcoxon_one_sided(returns, sp_return):
    diff = np.array(returns) - sp_return
    if len(diff[diff > 0]) == 0:
        return None, 1.0  # Pas de surperformance observée
    try:
        stat, p_two_sided = wilcoxon(diff)
        if stat > 0:
            p_one_sided = p_two_sided / 2
        else:
            p_one_sided = 1 - p_two_sided / 2
        return stat, p_one_sided
    except Exception:
        return None, None

# Chargement des données

df_portfolios = pd.read_parquet("portefeuilles_q100_eodhd.parquet")  # <== adapter à q=3, q=30, etc.
with open("prices_by_year_eodhd.pkl", "rb") as f:
    prices_by_year = pickle.load(f)

df_fits = pd.read_csv("resultats_portefeuillesq100.csv")

sp500_returns = {
    2024: 0.2331,
    2023: 0.2423,
    2022: -0.1944,
    2021: 0.2689,
    2020: 0.1626,
    2019: 0.2888,
    2018: -0.0624,
    2017: 0.1942,
    2016: 0.0954,
    2015: -0.0073,
    2014: 0.1139,
    2013: 0.296,
    2012: 0.1341
}

distribution_map = {
    "Normal": norm,
    "Logistic": logistic,
    "Gumbel": gumbel_r,
    "Generalized Extreme Value": genextreme
}

# Calcul par année

results = []

for year in range(2012, 2025):
    sp_return = sp500_returns[year]
    yearly_portfolios = df_portfolios[df_portfolios["year"] == year]
    returns = []

    for _, row in yearly_portfolios.iterrows():
        tickers = row["tickers"]
        weights = row["weights"]
        valid = True
        weighted_returns = []

        for i, ticker in enumerate(tickers):
            try:
                open_price, close_price = prices_by_year[year][ticker]
                if open_price == 0:
                    valid = False
                    break
                r = (close_price - open_price) / open_price
                weighted_returns.append(weights[i] * r)
            except:
                valid = False
                break

        if valid:
            returns.append(sum(weighted_returns))

    if len(returns) < 100:
        continue

    emp_proba, emp_ci = empirical_probability(returns, sp_return)
    w_stat, w_pval = wilcoxon_one_sided(returns, sp_return)

    fit_row = df_fits[df_fits["Year"] == year].iloc[0]
    dist_name = fit_row["PDF"]
    if dist_name not in distribution_map:
        boot_proba, boot_ci = None, (None, None)
    else:
        dist = distribution_map[dist_name]
        shape = fit_row["Shape"]
        loc = fit_row["Location"]
        scale = fit_row["Scale"]
        params = (shape, loc, scale)
        boot_proba, boot_ci = bootstrap_probability_cdf(dist, params, sp_return)

    results.append({
        "Year": year,
        "Empirical Proba": emp_proba,
        "Empirical CI low": emp_ci[0],
        "Empirical CI high": emp_ci[1],
        "Wilcoxon Stat": w_stat,
        "Wilcoxon p-val": w_pval,
        "Bootstrap Proba": boot_proba,
        "Bootstrap CI low": boot_ci[0],
        "Bootstrap CI high": boot_ci[1]
    })

# Export CSV
df_tests = pd.DataFrame(results)
df_tests.to_csv("tests_statistiques_completsq100.csv", index=False)
print("✅ Résultats exportés dans tests_statistiques_completsq100.csv")
