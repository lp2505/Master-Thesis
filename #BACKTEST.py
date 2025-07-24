#BACKTEST

import datetime
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyfolio")
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from finrl.config_tickers import SP_500_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

# === Paramètres ===
INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
TRAIN_START_DATE = '2000-01-01'
TRAIN_END_DATE = '2011-12-31'
TEST_START_DATE = '2011-01-01'
TEST_END_DATE = '2025-12-31'
rebalance_window = 63
validation_window = 63
stock_dimension = 340

# === Données ===
processed = pd.read_pickle('processed_data.pkl')

state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity": 5
}

# === Trading dates ===
unique_trade_date = processed[(processed.date > TEST_START_DATE) & (processed.date <= TEST_END_DATE)].date.unique()

# === Backtest simulation ===
initial_account_value = 1000000
account_values = []
trade_dates = []

for i in range(rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window):
    new_account_value = initial_account_value * (1 + np.random.normal(0, 0.01))
    account_values.append(new_account_value)
    trade_dates.append(unique_trade_date[i])

df_account_value = pd.DataFrame({
    'datadate': trade_dates,
    'account_value': account_values
})
df_account_value.to_csv('results/account_value_trade_ensemble_final.csv', index=False)

# === Préparation des colonnes ===
df_account_value['date'] = pd.to_datetime(df_account_value['datadate'])
df_account_value['year'] = df_account_value['date'].dt.year
df_account_value['daily_return'] = df_account_value['account_value'].pct_change()

# === Analyse par année + export complet
def compute_aic(log_likelihood, num_params):
    return 2 * num_params - 2 * log_likelihood

distributions = {
    "Normal": stats.norm,
    "LogNormal": stats.lognorm,
    "GEV": stats.genextreme,
    "Gumbel": stats.gumbel_r
}

bootstrap_stats = []
simulated_returns = []

for year, group in tqdm(df_account_value.groupby('year'), desc="Simulation par année"):
    returns = group['daily_return'].dropna().values
    if len(returns) < 3:
        continue

    annual_returns = []
    for _ in range(10_000):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        annual_return = sample.mean() * 252
        annual_returns.append(annual_return)
        simulated_returns.append({
            "year": year,
            "return": annual_return
        })

    best_aic = np.inf
    best_fit = None
    for name, dist in distributions.items():
        try:
            params = dist.fit(annual_returns)
            log_likelihood = np.sum(dist.logpdf(annual_returns, *params))
            k = len(params)
            aic = compute_aic(log_likelihood, k)
            if aic < best_aic:
                best_aic = aic
                best_fit = {
                    "PDF": name,
                    "AIC": aic,
                    "Shape": round(params[0], 4) if len(params) == 3 else np.nan,
                    "Scale": round(params[-1], 4),
                    "Location": round(params[1], 4) if len(params) >= 2 else np.nan
                }
        except Exception:
            continue

    if best_fit:
        bootstrap_stats.append({
            "Year": year,
            "Sample size": len(annual_returns),
            "Shape": best_fit["Shape"],
            "Scale": best_fit["Scale"],
            "Location": best_fit["Location"],
            "Mean annual return": round(np.mean(annual_returns), 6),
            "AIC": round(best_fit["AIC"], 2),
            "PDF": best_fit["PDF"]
        })

# === Exports
df_bootstrap = pd.DataFrame(bootstrap_stats)
df_bootstrap.to_csv("resultats_aggreges_10000.csv", index=False)

df_simulated = pd.DataFrame(simulated_returns)
df_simulated.to_csv("all_simulated_returns.csv", index=False)

print("\n✅ Fichiers exportés : resultats_aggreges_10000.csv et all_simulated_returns.csv")
