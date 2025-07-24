from eodhd import APIClient
import pandas as pd
import requests 
import time
import json
import random
from datetime import datetime
import pyarrow
import pickle

API_TOKEN = "6851e2af6b99c0.07953759"
INDEX_TICKER = 'GSPC.INDX'
BASE_URL = 'https://eodhd.com/api/mp/unicornbay/spglobal/comp/GSPC.INDX?fmt=json&api_token=6851e2af6b99c0.07953759'

response = requests.get(BASE_URL)
print("Status code:", response.status_code)

# Sauvegarde dans un fichier local
with open("sp500_historical_components.json", "w", encoding="utf-8") as f:
    f.write(response.text)

# Charger le JSON
with open("sp500_historical_components.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

historical_components = raw.get("HistoricalTickerComponents", {})
components_by_year = {year: set() for year in range(2012, 2025)}

for item in historical_components.values():
    ticker = item["Code"]
    start_raw = item.get("StartDate")
    if not start_raw:
        continue

    start = datetime.strptime(start_raw, "%Y-%m-%d")
    end_raw = item.get("EndDate")
    end = datetime.strptime(end_raw, "%Y-%m-%d") if end_raw else datetime(2100, 1, 1)

    for year in components_by_year:
        start_of_year = datetime(year, 1, 1)
        end_of_year = datetime(year, 12, 31)
        if start <= end_of_year and end >= start_of_year:
            components_by_year[year].add(ticker)

components_by_year = {year: sorted(list(tickers)) for year, tickers in components_by_year.items()}

for year in range(2012, 2025):
    print(f"{year}: {len(components_by_year[year])} tickers")

all_portfolios = []

for year, tickers in components_by_year.items():
    if len(tickers) < 100:
        print(f"⛔ {year} ignoré (moins de 100 tickers)")
        continue

    for _ in range(10_000):
        selected = random.sample(tickers, 100)
        weights = [random.random() for _ in range(100)]
        total = sum(weights)
        weights = [w / total for w in weights]
        all_portfolios.append({
            'year': year,
            'tickers': selected,
            'weights': weights
        })

df_q100 = pd.DataFrame(all_portfolios)
df_q100.to_parquet("portefeuilles_q100_eodhd.parquet", index=False)
print("✅ Portefeuilles de 100 actions générés et exportés.")

# Afficher un portefeuille spécifique (par exemple le premier dans la liste)
#portfolio = df_q100.iloc[0]  # Cela sélectionne le premier portefeuille
#print("Portefeuille sélectionné :")
#print(f"Année : {portfolio['year']}")
#print(f"Tickers : {portfolio['tickers']}")
#print(f"Poids : {portfolio['weights']}")

