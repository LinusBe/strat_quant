# Trading Strategies Tester

Ein umfassendes Framework zum Testen verschiedener Trading-Strategien, von Faktormodellen bis zu quantitativen Modellen, mit Anbindung an Yahoo Finance.

## Überblick

Dieses Repository bietet eine flexible Plattform zum Entwickeln, Testen und Analysieren von Trading-Strategien. Es enthält:

- **Daten-Provider**: Einfache Schnittstelle zu Yahoo Finance für historische Marktdaten
- **Strategie-Interface**: Standardisierte Schnittstelle für die Implementierung verschiedener Strategietypen
- **Backtest-Engine**: Robustes Framework zum Backtesten von Strategien mit historischen Daten
- **Performance-Metriken**: Umfassende Kennzahlen zur Bewertung der Strategie-Performance
- **Beispielstrategien**: Verschiedene implementierte Strategien als Referenz und Ausgangspunkt

## Projektstruktur

```
trading-strategies-tester/
├── data/                      # Ordner für heruntergeladene Daten
├── src/                       # Quellcode
│   ├── data_providers/        # Datenquellen (Yahoo Finance)
│   ├── backtest_engine/       # Backtesting-Framework
│   ├── performance_metrics/   # Kennzahlen zur Performance-Bewertung
│   └── utils/                 # Hilfsfunktionen
├── strategies/                # Hier kommen die Strategien rein
│   ├── factor_models/         # Faktormodelle
│   ├── quant_models/          # Quantitative Modelle
│   └── example_strategy.py    # Beispielstrategie
├── notebooks/                 # Jupyter Notebooks für Analysen
├── tests/                     # Unit-Tests
└── README.md                  # Dokumentation
```

## Installation

```bash
# Repository klonen
git clone https://github.com/username/trading-strategies-tester.git
cd trading-strategies-tester

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Schnellstart

```python
from src.data_providers.yahoo_finance import YahooFinanceProvider
from src.backtest_engine.backtest import Backtest
from strategies.example_strategy import MovingAverageCrossover

# Daten laden
data_provider = YahooFinanceProvider()
data = data_provider.get_data("AAPL", start_date="2020-01-01", end_date="2021-01-01")

# Strategie initialisieren
strategy = MovingAverageCrossover(short_window=20, long_window=50)

# Backtest durchführen
backtest = Backtest(data, strategy)
results = backtest.run()

# Ergebnisse analysieren
results.plot_performance()
print(results.get_metrics())
```

## Eigene Strategien entwickeln

Um eine eigene Strategie zu implementieren, erstellen Sie eine neue Klasse, die von der `Strategy`-Basisklasse erbt:

```python
from src.strategy_interface.base_strategy import Strategy

class MyCustomStrategy(Strategy):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
        
    def generate_signals(self, data):
        # Implementieren Sie Ihre Signallogik hier
        return signals
        
    def calculate_positions(self, data, signals):
        # Berechnen Sie die Positionsgrößen basierend auf den Signalen
        return positions
```

## Verfügbare Beispielstrategien

- **Moving Average Crossover**: Klassische Strategie basierend auf dem Kreuzen von gleitenden Durchschnitten
- **RSI (Relative Strength Index)**: Überkauft/überverkauft Momentum-Oszillator
- **Momentum**: Strategie basierend auf Preismomentum
- **Mean Reversion**: Strategie, die auf die Rückkehr zum Mittelwert setzt
- **Faktormodell**: Multi-Faktor-Modell für Aktienauswahl
- **Machine Learning**: ML-basierte Prognosemodelle

## Lizenz

MIT
