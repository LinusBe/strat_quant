"""
Beispiel-Notebook für das Trading Strategies Tester Framework

Dieses Notebook demonstriert die Verwendung des Frameworks anhand eines einfachen Beispiels
mit der Moving Average Crossover Strategie.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Füge das Projektverzeichnis zum Pfad hinzu
sys.path.append('..')

# Importiere die benötigten Module
from src.data_providers.yahoo_finance import YahooFinanceProvider
from src.backtest_engine.backtest import Backtest
from strategies.moving_average_crossover import MovingAverageCrossover

# 1. Daten laden
print("Daten werden geladen...")
data_provider = YahooFinanceProvider()
symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2022-12-31"

data = data_provider.get_data(symbol, start_date=start_date, end_date=end_date)
print(f"Daten für {symbol} von {start_date} bis {end_date} geladen.")
print(f"Datenpunkte: {len(data)}")
print(data.head())

# 2. Strategie initialisieren
print("\nStrategie wird initialisiert...")
short_window = 20
long_window = 50
strategy = MovingAverageCrossover(short_window=short_window, long_window=long_window)
print(f"Moving Average Crossover Strategie initialisiert mit short_window={short_window}, long_window={long_window}")

# 3. Backtest durchführen
print("\nBacktest wird durchgeführt...")
initial_capital = 100000.0
commission = 0.001  # 0.1%
slippage = 0.0005   # 0.05%

backtest = Backtest(
    data=data,
    strategy=strategy,
    initial_capital=initial_capital,
    commission=commission,
    slippage=slippage
)

results = backtest.run()
print("Backtest abgeschlossen.")

# 4. Ergebnisse analysieren
print("\nErgebnisse werden analysiert...")
metrics = results.get_metrics()

print("\nPerformance-Metriken:")
print(f"- Gesamtrendite: {metrics['total_return']*100:.2f}%")
print(f"- Jährliche Rendite: {metrics['annual_return']*100:.2f}%")
print(f"- Jährliche Volatilität: {metrics['annual_volatility']*100:.2f}%")
print(f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"- Maximaler Drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"- Win-Rate: {metrics['win_rate']*100:.2f}%")
print(f"- Anzahl der Trades: {metrics['num_trades']}")
print(f"- Endkapital: ${metrics['final_equity']:.2f}")

# 5. Visualisierung
print("\nErgebnisse werden visualisiert...")
fig = results.plot_performance()
plt.tight_layout()
plt.show()

# 6. Zusammenfassung
print("\nZusammenfassung:")
print(results.print_summary())

# 7. Speichern der Ergebnisse
print("\nErgebnisse werden gespeichert...")
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
result_files = results.save_results(output_dir=output_dir)

print("\nErgebnisse gespeichert in:")
for key, file_path in result_files.items():
    print(f"- {key}: {file_path}")

print("\nNotebook abgeschlossen.")
