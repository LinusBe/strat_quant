# Dokumentation: Trading Strategies Tester

## Inhaltsverzeichnis
1. [Einführung](#einführung)
2. [Installation](#installation)
3. [Projektstruktur](#projektstruktur)
4. [Schnellstart](#schnellstart)
5. [Daten-Provider](#daten-provider)
6. [Strategie-Interface](#strategie-interface)
7. [Backtest-Engine](#backtest-engine)
8. [Performance-Metriken](#performance-metriken)
9. [Beispielstrategien](#beispielstrategien)
10. [Eigene Strategien entwickeln](#eigene-strategien-entwickeln)
11. [Fortgeschrittene Funktionen](#fortgeschrittene-funktionen)

## Einführung

Das Trading Strategies Tester Framework bietet eine umfassende Plattform zum Entwickeln, Testen und Analysieren von Trading-Strategien. Es ermöglicht Benutzern, verschiedene Arten von Strategien zu implementieren, von einfachen technischen Indikatoren bis hin zu komplexen Faktormodellen und Machine Learning-Ansätzen.

Das Framework besteht aus mehreren Hauptkomponenten:
- **Daten-Provider**: Einfache Schnittstelle zu Yahoo Finance für historische Marktdaten
- **Strategie-Interface**: Standardisierte Schnittstelle für die Implementierung verschiedener Strategietypen
- **Backtest-Engine**: Robustes Framework zum Backtesten von Strategien mit historischen Daten
- **Performance-Metriken**: Umfassende Kennzahlen zur Bewertung der Strategie-Performance
- **Beispielstrategien**: Verschiedene implementierte Strategien als Referenz und Ausgangspunkt

## Installation

### Voraussetzungen
- Python 3.6 oder höher
- pip (Python-Paketmanager)

### Installation der Abhängigkeiten

```bash
# Repository klonen
git clone https://github.com/username/trading-strategies-tester.git
cd trading-strategies-tester

# Abhängigkeiten installieren
pip install -r requirements.txt
```

Die Hauptabhängigkeiten umfassen:
- pandas: Für Datenverarbeitung und -analyse
- numpy: Für numerische Berechnungen
- matplotlib: Für Datenvisualisierung
- scikit-learn: Für Machine Learning-Algorithmen
- seaborn: Für erweiterte Visualisierungen

## Projektstruktur

```
trading-strategies-tester/
├── data/                      # Ordner für heruntergeladene Daten
├── src/                       # Quellcode
│   ├── data_providers/        # Datenquellen (Yahoo Finance)
│   ├── strategy_interface/    # Strategie-Interface
│   ├── backtest_engine/       # Backtesting-Framework
│   ├── performance_metrics/   # Kennzahlen zur Performance-Bewertung
│   └── utils/                 # Hilfsfunktionen
├── strategies/                # Implementierte Strategien
│   ├── factor_models/         # Faktormodelle
│   ├── quant_models/          # Quantitative Modelle
│   ├── moving_average_crossover.py
│   ├── rsi_strategy.py
│   ├── momentum_strategy.py
│   └── mean_reversion_strategy.py
├── notebooks/                 # Jupyter Notebooks für Analysen
├── tests/                     # Unit-Tests
└── README.md                  # Projektübersicht
```

## Schnellstart

Hier ist ein einfaches Beispiel, wie Sie eine Strategie testen können:

```python
from src.data_providers.yahoo_finance import YahooFinanceProvider
from src.backtest_engine.backtest import Backtest
from strategies.moving_average_crossover import MovingAverageCrossover

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
print(results.print_summary())
```

## Daten-Provider

Der `YahooFinanceProvider` bietet eine einfache Schnittstelle zum Abrufen historischer Marktdaten von Yahoo Finance.

### Hauptfunktionen

```python
# Einzelne Aktie laden
data = data_provider.get_data("AAPL", start_date="2020-01-01", end_date="2021-01-01")

# Mehrere Aktien laden
symbols = ["AAPL", "MSFT", "GOOGL"]
data_dict = data_provider.get_multiple_symbols(symbols, start_date="2020-01-01", end_date="2021-01-01")

# Marktdaten für mehrere Aktien in einem Panel organisieren
market_data = data_provider.get_market_data(symbols, start_date="2020-01-01", end_date="2021-01-01")
```

### Daten-Caching

Der Provider unterstützt automatisches Caching von Daten, um wiederholte API-Anfragen zu vermeiden:

```python
# Mit Caching (Standard)
data = data_provider.get_data("AAPL", start_date="2020-01-01", end_date="2021-01-01", use_cache=True)

# Ohne Caching (erzwingt neue Daten)
data = data_provider.get_data("AAPL", start_date="2020-01-01", end_date="2021-01-01", use_cache=False)
```

## Strategie-Interface

Das Strategie-Interface definiert die Struktur, die alle Trading-Strategien implementieren müssen.

### Basis-Strategie-Klasse

Die `Strategy`-Klasse ist eine abstrakte Basisklasse mit zwei Hauptmethoden, die implementiert werden müssen:

```python
from src.strategy_interface.base_strategy import Strategy

class MyStrategy(Strategy):
    def __init__(self, param1, param2, name=None):
        super().__init__(name=name)
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data):
        # Implementieren Sie Ihre Signallogik hier
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        # ... Signalberechnung ...
        return signals
    
    def calculate_positions(self, data, signals):
        # Berechnen Sie die Positionsgrößen basierend auf den Signalen
        positions = signals.copy()
        # ... Positionsberechnung ...
        return positions
```

### Positionsgrößenberechner

Das Framework bietet verschiedene Positionsgrößenberechner:

```python
from src.strategy_interface.base_strategy import FixedPositionSizer, PercentagePositionSizer, VolatilityPositionSizer

# Feste Positionsgröße
position_sizer = FixedPositionSizer(position_size=1.0)

# Prozentuale Positionsgröße
position_sizer = PercentagePositionSizer(percentage=0.1)

# Volatilitätsbasierte Positionsgröße
position_sizer = VolatilityPositionSizer(target_volatility=0.01, lookback_period=20)
```

## Backtest-Engine

Die Backtest-Engine führt Backtests für Trading-Strategien durch und berechnet Performance-Metriken.

### Einfacher Backtest

```python
from src.backtest_engine.backtest import Backtest

# Backtest mit Standardparametern
backtest = Backtest(data, strategy)
results = backtest.run()

# Backtest mit angepassten Parametern
backtest = Backtest(
    data=data,
    strategy=strategy,
    initial_capital=100000.0,
    commission=0.001,  # 0.1%
    slippage=0.0005    # 0.05%
)
results = backtest.run()
```

### Backtest-Ergebnisse

Die `BacktestResult`-Klasse bietet Methoden zur Analyse der Backtest-Ergebnisse:

```python
# Performance-Metriken abrufen
metrics = results.get_metrics()

# Performance visualisieren
results.plot_performance()

# Zusammenfassung ausgeben
summary = results.print_summary()

# Ergebnisse speichern
result_files = results.save_results(output_dir="results")
```

### Ereignisbasierter Backtest

Für komplexere Szenarien bietet das Framework auch einen ereignisbasierten Backtest:

```python
from src.backtest_engine.backtest import EventDrivenBacktest

# Ereignisbasierter Backtest (fortgeschrittene Verwendung)
backtest = EventDrivenBacktest(
    data_handler=data_handler,
    strategy=strategy,
    portfolio=portfolio,
    execution_handler=execution_handler
)
results = backtest.run()
```

## Performance-Metriken

Das Framework bietet umfassende Performance-Metriken zur Bewertung von Trading-Strategien.

### Grundlegende Metriken

```python
from src.performance_metrics.metrics import PerformanceMetrics

# Performance-Metriken berechnen
metrics = PerformanceMetrics(returns=results.portfolio['Net_Strategy_Returns'])

# Grundlegende Metriken abrufen
total_return = metrics.total_return()
annual_return = metrics.annual_return()
annual_volatility = metrics.annual_volatility()
sharpe_ratio = metrics.sharpe_ratio()
max_drawdown = metrics.max_drawdown()

# Zusammenfassung aller Metriken
summary = metrics.summary()
```

### Visualisierungen

```python
# Kumulative Renditen visualisieren
metrics.plot_cumulative_returns()

# Drawdown visualisieren
metrics.plot_drawdown()

# Monatliche Renditen als Heatmap
metrics.plot_monthly_returns_heatmap()

# Rollierende Sharpe Ratio
metrics.plot_rolling_sharpe()

# Renditeverteilung
metrics.plot_return_distribution()

# Performance-Dashboard mit mehreren Visualisierungen
metrics.plot_performance_dashboard()
```

### Risikometriken

```python
from src.performance_metrics.metrics import RiskAnalysis

# Risikoanalyse durchführen
risk = RiskAnalysis(returns=results.portfolio['Net_Strategy_Returns'])

# Risikometriken abrufen
downside_deviation = risk.downside_deviation()
skewness = risk.skewness()
kurtosis = risk.kurtosis()
tail_ratio = risk.tail_ratio()
max_drawdown_duration = risk.maximum_drawdown_duration()
ulcer_index = risk.ulcer_index()

# Zusammenfassung aller Risikometriken
risk_summary = risk.summary()
```

## Beispielstrategien

Das Framework enthält mehrere Beispielstrategien, die als Referenz und Ausgangspunkt dienen können.

### Moving Average Crossover

```python
from strategies.moving_average_crossover import MovingAverageCrossover

# Strategie initialisieren
strategy = MovingAverageCrossover(short_window=20, long_window=50)
```

Diese Strategie generiert Kauf-Signale, wenn der kurzfristige gleitende Durchschnitt den langfristigen gleitenden Durchschnitt von unten nach oben kreuzt, und Verkauf-Signale, wenn der kurzfristige gleitende Durchschnitt den langfristigen gleitenden Durchschnitt von oben nach unten kreuzt.

### RSI (Relative Strength Index)

```python
from strategies.rsi_strategy import RSIStrategy

# Strategie initialisieren
strategy = RSIStrategy(window=14, oversold=30, overbought=70)
```

Diese Strategie generiert Kauf-Signale, wenn der RSI unter einen überverkauften Schwellenwert fällt, und Verkauf-Signale, wenn der RSI über einen überkauften Schwellenwert steigt.

### Momentum

```python
from strategies.momentum_strategy import MomentumStrategy

# Strategie initialisieren
strategy = MomentumStrategy(lookback_period=20, threshold=0.0)
```

Diese Strategie kauft Vermögenswerte mit positivem Momentum und verkauft solche mit negativem Momentum, basierend auf der Annahme, dass Preisbewegungen dazu neigen, in die gleiche Richtung weiterzugehen.

### Mean Reversion

```python
from strategies.mean_reversion_strategy import MeanReversionStrategy

# Strategie initialisieren
strategy = MeanReversionStrategy(window=20, std_dev=2.0)
```

Diese Strategie kauft, wenn der Preis signifikant unter seinen gleitenden Durchschnitt fällt, und verkauft, wenn er signifikant darüber steigt, basierend auf der Annahme, dass Preise tendenziell zu ihrem Mittelwert zurückkehren.

### Faktormodell

```python
from strategies.factor_models.factor_model_strategy import FactorModelStrategy

# Strategie initialisieren
strategy = FactorModelStrategy(
    momentum_window=60,
    value_weight=0.25,
    momentum_weight=0.25,
    size_weight=0.25,
    quality_weight=0.25,
    top_pct=0.2
)
```

Diese Strategie verwendet mehrere Faktoren (Momentum, Value, Größe, Qualität), um Aktien zu bewerten und zu selektieren.

### Machine Learning

```python
from strategies.quant_models.ml_strategy import MLStrategy

# Strategie initialisieren
strategy = MLStrategy(lookback_period=10, train_size=0.7, n_estimators=100)
```

Diese Strategie verwendet einen Random Forest Classifier, um Preisbewegungen vorherzusagen und entsprechende Handelssignale zu generieren.

## Eigene Strategien entwickeln

Um eine eigene Strategie zu entwickeln, müssen Sie eine Klasse erstellen, die von der `Strategy`-Basisklasse erbt und die Methoden `generate_signals` und `calculate_positions` implementiert.

### Beispiel: Einfache Breakout-Strategie

```python
from src.strategy_interface.base_strategy import Strategy
import pandas as pd
import numpy as np

class BreakoutStrategy(Strategy):
    def __init__(self, window=20, name=None):
        if name is None:
            name = f"Breakout_{window}"
        super().__init__(name=name)
        self.window = window
    
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        
        # Berechne höchsten Hoch und niedrigsten Tief im Fenster
        signals['highest_high'] = data['High'].rolling(window=self.window).max()
        signals['lowest_low'] = data['Low'].rolling(window=self.window).min()
        
        # Kaufsignal, wenn Preis über höchstem Hoch
        signals.loc[data['Close'] > signals['highest_high'].shift(1), 'signal'] = 1.0
        
        # Verkaufssignal, wenn Preis unter niedrigstem Tief
        signals.loc[data['Close'] < signals['lowest_low'].shift(1), 'signal'] = -1.0
        
        # Berechne Positionsänderungen
        signals['position'] = signals['signal']
        signals['position'] = signals['position'].diff().fillna(signals['position'].iloc[0])
        signals.loc[signals['position'] == 0, 'position'] = np.nan
        signals['position'] = signals['position'].fillna(method='ffill').fillna(0)
        
        return signals
    
    def calculate_positions(self, data, signals):
        positions = signals.copy()
        positions = positions.fillna(0.0)
        return positions
```

### Strategie testen

```python
# Daten laden
data_provider = YahooFinanceProvider()
data = data_provider.get_data("AAPL", start_date="2020-01-01", end_date="2021-01-01")

# Strategie initialisieren
strategy = BreakoutStrategy(window=20)

# Backtest durchführen
backtest = Backtest(data, strategy)
results = backtest.run()

# Ergebnisse analysieren
results.plot_performance()
print(results.print_summary())
```

## Fortgeschrittene Funktionen

### Mehrere Vermögenswerte

Um Strategien mit mehreren Vermögenswerten zu testen, können Sie den `get_multiple_symbols` oder `get_market_data` Methoden des `YahooFinanceProvider` verwenden:

```python
# Mehrere Aktien laden
symbols = ["AAPL", "MSFT", "GOOGL"]
data_dict = data_provider.get_multiple_symbols(symbols, start_date="2020-01-01", end_date="2021-01-01")

# Marktdaten für mehrere Aktien in einem Panel organisieren
market_data = data_provider.get_market_data(symbols, start_date="2020-01-01", end_date="2021-01-01")
```

### Portfoliooptimierung

Für Portfoliooptimierung können Sie die Positionsgrößenberechnung in Ihrer Strategie anpassen:

```python
def calculate_positions(self, data, signals):
    positions = signals.copy()
    
    # Gleichgewichte die Positionen
    num_assets = len([col for col in positions.columns if positions.iloc[-1][col] > 0])
    
    if num_assets > 0:
        weight_per_asset = 1.0 / num_assets
        
        for col in positions.columns:
            if positions.iloc[-1][col] > 0:
                positions[col] = weight_per_asset
    
    return positions
```

### Ereignisbasiertes Backtesting

Für komplexere Szenarien, wie Intraday-Trading oder Order-Ausführungsmodellierung, können Sie das ereignisbasierte Backtesting-Framework verwenden:

```python
from src.backtest_engine.backtest import EventDrivenBacktest

# Ereignisbasierter Backtest (fortgeschrittene Verwendung)
backtest = EventDrivenBacktest(
    data_handler=data_handler,
    strategy=strategy,
    portfolio=portfolio,
    execution_handler=execution_handler
)
results = backtest.run()
```

### Parallelisierung von Backtests

Für die Optimierung von Strategieparametern können Sie mehrere Backtests parallel ausführen:

```python
import concurrent.futures

def run_backtest(params):
    short_window, long_window = params
    strategy = MovingAverageCrossover(short_window=short_window, long_window=long_window)
    backtest = Backtest(data, strategy)
    results = backtest.run()
    return {
        'params': params,
        'metrics': results.get_metrics()
    }

# Parameter-Grid
param_grid = [
    (5, 20), (10, 30), (15, 40), (20, 50), (25, 60)
]

# Parallele Ausführung
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(run_backtest, param_grid))

# Ergebnisse analysieren
for result in results:
    params = result['params']
    metrics = result['metrics']
    print(f"Params: {params}, Sharpe: {metrics['sharpe_ratio']:.2f}, Return: {metrics['total_return']*100:.2f}%")
```
