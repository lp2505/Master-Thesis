import pandas as pd
import pickle
import numpy as np
from scipy.stats import genextreme, logistic, norm, gumbel_r
import warnings

warnings.filterwarnings("ignore")

# Charger les données
df_portfolios = pd.read_parquet("portefeuilles_q30_eodhd.parquet")

with open("prices_by_year_eodhd.pkl", "rb") as f:
    prices_by_year = pickle.load(f)

results = []

for year in range(2012, 2025):
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

    if len(returns) >= 100:
        data = np.array(returns)
        mean_return = np.mean(data)
        aic_scores = {}

        # Distributions à tester
        distributions = {
            "Normal": norm,
            "Logistic": logistic,
            "Gumbel": gumbel_r,
            "Generalized Extreme Value": genextreme
        }

        for name, dist in distributions.items():
            try:
                params = dist.fit(data)
                loglike = np.sum(dist.logpdf(data, *params))
                k = len(params)
                aic = 2 * k - 2 * loglike
                aic_scores[name] = (aic, params)
            except Exception as e:
                aic_scores[name] = (np.inf, str(e))

        best_model = min(aic_scores.items(), key=lambda x: x[1][0])
        best_name, (best_aic, best_params) = best_model

        shape = round(best_params[0], 4) if len(best_params) == 3 else ""
        scale = round(best_params[-1], 4)
        loc = round(best_params[-2], 4)

        results.append({
            "Year": year,
            "Sample size": len(data),
            "Shape": shape,
            "Scale": scale,
            "Location": loc,
            "Mean return": round(mean_return, 4),
            "AIC": round(best_aic, 2),
            "PDF": best_name
        })

# Export final
df_result = pd.DataFrame(results)
df_result.to_csv("resultats_portefeuillesq30.csv", index=False)
print("✅ Résultats sauvegardés dans resultats_portefeuillesq30.csv")

with open("resultats_portefeuilles_latexq30.txt", "w") as f:
    for row in results:
        shape = row["Shape"] if row["Shape"] != "" else " "
        line = f"{row['Year']} & {row['Sample size']} & {shape} & {row['Scale']} & {row['Location']} & {row['AIC']} & {row['PDF']} \\\\"
        f.write(line + "\n")

print("✅ Table LaTeX générée dans resultats_portefeuilles_latexq30.txt")

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 5))
plt.hist(df_result["Mean return"], bins=10, edgecolor='black', alpha=0.7)
plt.title("Average annual return distribution (2012-2024) q=30")
plt.xlabel("Return")
plt.ylabel("Frequence")
plt.grid(True)
plt.tight_layout()
plt.savefig("histogramme_rendements_moyensq30.png")
plt.close()

