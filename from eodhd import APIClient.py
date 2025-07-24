from eodhd import APIClient
import pandas as pd
import requests 
import time
import json
import random
from datetime import datetime
import pyarrow
import pickle

API_TOKEN = "6851e2af6b99c0.07953759"  # Your actual EODHD API key
INDEX_TICKER = 'GSPC.INDX'
BASE_URL = 'https://eodhd.com/api/mp/unicornbay/spglobal/comp/GSPC.INDX?fmt=json&api_token=6851e2af6b99c0.07953759'



response = requests.get(BASE_URL)
print("Status code:", response.status_code)

# Sauvegarde dans un fichier local
with open("sp500_historical_components.json", "w", encoding="utf-8") as f:
    f.write(response.text)

#print("✅ JSON sauvegardé dans sp500_historical_components.json")

# Charger le JSON
with open("sp500_historical_components.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

# Extraire uniquement la partie "HistoricalTickerComponents"
historical_components = raw.get("HistoricalTickerComponents", {})

# Initialiser le dictionnaire annuel
components_by_year = {year: set() for year in range(2012, 2025)}

# Parcourir les composants indexés (clé = "0", "1", etc.)
for item in historical_components.values():
    ticker = item["Code"]

    start_raw = item.get("StartDate")
    if not start_raw:
        continue  # ⛔ ignorer les lignes sans StartDate

    start = datetime.strptime(start_raw, "%Y-%m-%d")

    end_raw = item.get("EndDate")
    end = datetime.strptime(end_raw, "%Y-%m-%d") if end_raw else datetime(2100, 1, 1)

    for year in components_by_year:
        start_of_year = datetime(year, 1, 1)
        end_of_year = datetime(year, 12, 31)

        if start <= end_of_year and end >= start_of_year:
            components_by_year[year].add(ticker)

# Nettoyage : tri et conversion
components_by_year = {year: sorted(list(tickers)) for year, tickers in components_by_year.items()}

# Aperçu
for year in range(2012, 2025):
    print(f"{year}: {len(components_by_year[year])} tickers")

# Export Excel
#pd.DataFrame(dict([(k, pd.Series(v)) for k, v in components_by_year.items()])).to_excel("sp500_by_year.xlsx", index=False)


# ✅ components_by_year doit être défini (cf. étape précédente)
# Exemple attendu : components_by_year[2012] = ['AAPL', 'MSFT', 'GOOGL', ...]

all_portfolios = []

for year, tickers in components_by_year.items():
    if len(tickers) < 3:
        print(f"⛔ {year} ignoré (moins de 3 tickers)")
        continue

    for _ in range(10_000):
        selected = random.sample(tickers, 3)
        weights = [random.random() for _ in range(3)]
        total = sum(weights)
        weights = [w / total for w in weights]  # Normaliser pour que ça fasse 1

        all_portfolios.append({
            'year': year,
            'tickers': selected,
            'weights': weights
        })

# ✅ Convertir en DataFrame
df_q3 = pd.DataFrame(all_portfolios)

# Export optionnel
df_q3.to_parquet("portefeuilles_q3_eodhd.parquet", index=False)
print("✅ Portefeuilles générés et exportés.")

#Prix

# Template URL for fetching EOD data
URL_TEMPLATE = "https://eodhd.com/api/eod/{ticker}.US?from={start}&to={end}&period=d&api_token=6851e2af6b99c0.07953759&fmt=json"

# Load generated portfolios
df_q3 = pd.read_parquet("portefeuilles_q3_eodhd.parquet")

# Initialize dictionaries
prices_by_year = {}
ignored_tickers_by_year = {}

# Process each year
for year in range(2012, 2025):
    tickers = set()
    for tickers_list in df_q3[df_q3["year"] == year]["tickers"]:
        tickers.update(tickers_list)

    prices_by_year[year] = {}
    ignored_tickers_by_year[year] = []

    for ticker in tickers:
        url = URL_TEMPLATE.format(ticker=ticker, start=f"{year}-01-01", end=f"{year}-12-31")
        try:
            response = requests.get(url)
            data = response.json()

            if isinstance(data, list) and len(data) >= 2:
                open_price = float(data[-1].get("open", 0))
                close_price = float(data[0].get("close", 0))

                if open_price > 0 and close_price > 0:
                    prices_by_year[year][ticker] = (open_price, close_price)
                else:
                    ignored_tickers_by_year[year].append(ticker)
            else:
                ignored_tickers_by_year[year].append(ticker)

        except Exception as e:
            ignored_tickers_by_year[year].append(ticker)

        time.sleep(0.7)  # Respect API rate limits
    

    print(f"✅ {year} : {len(prices_by_year[year])} downloaded, {len(ignored_tickers_by_year[year])} ignored")
    #print(URL_TEMPLATE.format(ticker="MMM", start="2012-01-01", end="2012-12-31"))


# Save results
with open("prices_by_year_eodhd.pkl", "wb") as f:
    pickle.dump(prices_by_year, f)
with open("ignored_tickers_eodhd.pkl", "wb") as f:
    pickle.dump(ignored_tickers_by_year, f)

