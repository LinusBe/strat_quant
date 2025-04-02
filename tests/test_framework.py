"""
Test-Skript für das Trading Strategies Tester Framework

Dieses Skript testet die grundlegenden Funktionen des Frameworks, einschließlich
Daten-Provider, Strategie-Interface, Backtest-Engine und Beispielstrategien.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Füge das Projektverzeichnis zum Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiere die benötigten Module
from src.data_providers.yahoo_finance import YahooFinanceProvider
from src.backtest_engine.backtest import Backtest
from strategies.moving_average_crossover import MovingAverageCrossover
from strategies.rsi_strategy import RSIStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy

# Erstelle Ausgabeverzeichnis für Testergebnisse
output_dir = os.path.join(os.path.dirname(__file__), 'test_results')
os.makedirs(output_dir, exist_ok=True)

def test_data_provider():
    """
    Testet den Yahoo Finance Daten-Provider.
    """
    print("=== Test: Yahoo Finance Daten-Provider ===")
    
    # Initialisiere den Daten-Provider
    data_provider = YahooFinanceProvider()
    
    # Teste das Abrufen von Daten für ein einzelnes Symbol
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    
    print(f"Lade Daten für {symbol} von {start_date} bis {end_date}...")
    data = data_provider.get_data(symbol, start_date=start_date, end_date=end_date)
    
    # Überprüfe, ob Daten erfolgreich abgerufen wurden
    if data is not None and not data.empty:
        print(f"Daten erfolgreich abgerufen. Shape: {data.shape}")
        print(f"Erster Tag: {data.index[0].strftime('%Y-%m-%d')}")
        print(f"Letzter Tag: {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Spalten: {data.columns.tolist()}")
        print("Beispieldaten:")
        print(data.head())
        
        # Speichere die Daten für spätere Tests
        return data
    else:
        print("Fehler beim Abrufen der Daten.")
        return None

def test_moving_average_crossover_strategy(data):
    """
    Testet die Moving Average Crossover Strategie.
    """
    print("\n=== Test: Moving Average Crossover Strategie ===")
    
    # Initialisiere die Strategie
    short_window = 20
    long_window = 50
    strategy = MovingAverageCrossover(short_window=short_window, long_window=long_window)
    
    print(f"Strategie initialisiert: {strategy.name}")
    print(f"Parameter: short_window={short_window}, long_window={long_window}")
    
    # Führe die Strategie aus
    signals = strategy.generate_signals(data)
    
    # Überprüfe die generierten Signale
    if signals is not None and not signals.empty:
        print(f"Signale erfolgreich generiert. Shape: {signals.shape}")
        print(f"Spalten: {signals.columns.tolist()}")
        
        # Zähle die Anzahl der Kauf- und Verkaufssignale
        buy_signals = signals[signals['position'] > 0]
        sell_signals = signals[signals['position'] < 0]
        
        print(f"Anzahl der Kaufsignale: {len(buy_signals)}")
        print(f"Anzahl der Verkaufssignale: {len(sell_signals)}")
        
        # Führe einen Backtest durch
        return test_backtest(data, strategy)
    else:
        print("Fehler beim Generieren der Signale.")
        return None

def test_rsi_strategy(data):
    """
    Testet die RSI Strategie.
    """
    print("\n=== Test: RSI Strategie ===")
    
    # Initialisiere die Strategie
    window = 14
    oversold = 30
    overbought = 70
    strategy = RSIStrategy(window=window, oversold=oversold, overbought=overbought)
    
    print(f"Strategie initialisiert: {strategy.name}")
    print(f"Parameter: window={window}, oversold={oversold}, overbought={overbought}")
    
    # Führe die Strategie aus
    signals = strategy.generate_signals(data)
    
    # Überprüfe die generierten Signale
    if signals is not None and not signals.empty:
        print(f"Signale erfolgreich generiert. Shape: {signals.shape}")
        print(f"Spalten: {signals.columns.tolist()}")
        
        # Zähle die Anzahl der Kauf- und Verkaufssignale
        buy_signals = signals[signals['position'] > 0]
        sell_signals = signals[signals['position'] < 0]
        
        print(f"Anzahl der Kaufsignale: {len(buy_signals)}")
        print(f"Anzahl der Verkaufssignale: {len(sell_signals)}")
        
        # Führe einen Backtest durch
        return test_backtest(data, strategy)
    else:
        print("Fehler beim Generieren der Signale.")
        return None

def test_momentum_strategy(data):
    """
    Testet die Momentum Strategie.
    """
    print("\n=== Test: Momentum Strategie ===")
    
    # Initialisiere die Strategie
    lookback_period = 20
    threshold = 0.0
    strategy = MomentumStrategy(lookback_period=lookback_period, threshold=threshold)
    
    print(f"Strategie initialisiert: {strategy.name}")
    print(f"Parameter: lookback_period={lookback_period}, threshold={threshold}")
    
    # Führe die Strategie aus
    signals = strategy.generate_signals(data)
    
    # Überprüfe die generierten Signale
    if signals is not None and not signals.empty:
        print(f"Signale erfolgreich generiert. Shape: {signals.shape}")
        print(f"Spalten: {signals.columns.tolist()}")
        
        # Zähle die Anzahl der Kauf- und Verkaufssignale
        buy_signals = signals[signals['position'] > 0]
        sell_signals = signals[signals['position'] < 0]
        
        print(f"Anzahl der Kaufsignale: {len(buy_signals)}")
        print(f"Anzahl der Verkaufssignale: {len(sell_signals)}")
        
        # Führe einen Backtest durch
        return test_backtest(data, strategy)
    else:
        print("Fehler beim Generieren der Signale.")
        return None

def test_mean_reversion_strategy(data):
    """
    Testet die Mean Reversion Strategie.
    """
    print("\n=== Test: Mean Reversion Strategie ===")
    
    # Initialisiere die Strategie
    window = 20
    std_dev = 2.0
    strategy = MeanReversionStrategy(window=window, std_dev=std_dev)
    
    print(f"Strategie initialisiert: {strategy.name}")
    print(f"Parameter: window={window}, std_dev={std_dev}")
    
    # Führe die Strategie aus
    signals = strategy.generate_signals(data)
    
    # Überprüfe die generierten Signale
    if signals is not None and not signals.empty:
        print(f"Signale erfolgreich generiert. Shape: {signals.shape}")
        print(f"Spalten: {signals.columns.tolist()}")
        
        # Zähle die Anzahl der Kauf- und Verkaufssignale
        buy_signals = signals[signals['position'] > 0]
        sell_signals = signals[signals['position'] < 0]
        
        print(f"Anzahl der Kaufsignale: {len(buy_signals)}")
        print(f"Anzahl der Verkaufssignale: {len(sell_signals)}")
        
        # Führe einen Backtest durch
        return test_backtest(data, strategy)
    else:
        print("Fehler beim Generieren der Signale.")
        return None

def test_backtest(data, strategy):
    """
    Testet die Backtest-Engine mit einer gegebenen Strategie.
    """
    print(f"\n--- Backtest für {strategy.name} ---")
    
    # Initialisiere den Backtest
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
    
    print(f"Backtest initialisiert mit:")
    print(f"- Anfangskapital: ${initial_capital:.2f}")
    print(f"- Kommission: {commission*100:.2f}%")
    print(f"- Slippage: {slippage*100:.2f}%")
    
    # Führe den Backtest durch
    results = backtest.run()
    
    # Überprüfe die Backtest-Ergebnisse
    if results is not None:
        print("Backtest erfolgreich durchgeführt.")
        
        # Zeige Performance-Metriken
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
        
        # Speichere die Ergebnisse
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_files = results.save_results(output_dir=output_dir)
        
        print("\nErgebnisse gespeichert in:")
        for key, file_path in result_files.items():
            print(f"- {key}: {file_path}")
        
        return results
    else:
        print("Fehler beim Durchführen des Backtests.")
        return None

def run_all_tests():
    """
    Führt alle Tests aus.
    """
    print("=== Trading Strategies Tester - Testlauf ===")
    print(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ausgabeverzeichnis: {output_dir}")
    print("=" * 50)
    
    # Teste den Daten-Provider
    data = test_data_provider()
    
    if data is not None:
        # Teste die Strategien
        ma_results = test_moving_average_crossover_strategy(data)
        rsi_results = test_rsi_strategy(data)
        momentum_results = test_momentum_strategy(data)
        mean_reversion_results = test_mean_reversion_strategy(data)
        
        # Vergleiche die Strategien
        if all([ma_results, rsi_results, momentum_results, mean_reversion_results]):
            print("\n=== Strategievergleich ===")
            
            strategies = {
                "Moving Average Crossover": ma_results,
                "RSI": rsi_results,
                "Momentum": momentum_results,
                "Mean Reversion": mean_reversion_results
            }
            
            # Erstelle eine Tabelle mit den wichtigsten Metriken
            comparison = pd.DataFrame(columns=[
                "Strategie", "Gesamtrendite (%)", "Jährliche Rendite (%)",
                "Sharpe Ratio", "Max Drawdown (%)", "Win-Rate (%)", "Anzahl Trades"
            ])
            
            for name, result in strategies.items():
                metrics = result.get_metrics()
                comparison = comparison.append({
                    "Strategie": name,
                    "Gesamtrendite (%)": metrics['total_return'] * 100,
                    "Jährliche Rendite (%)": metrics['annual_return'] * 100,
                    "Sharpe Ratio": metrics['sharpe_ratio'],
                    "Max Drawdown (%)": metrics['max_drawdown'] * 100,
                    "Win-Rate (%)": metrics['win_rate'] * 100,
                    "Anzahl Trades": metrics['num_trades']
                }, ignore_index=True)
            
            # Sortiere nach Sharpe Ratio
            comparison = comparison.sort_values("Sharpe Ratio", ascending=False)
            
            print(comparison.to_string(index=False))
            
            # Speichere den Vergleich
            comparison_file = os.path.join(output_dir, f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            comparison.to_csv(comparison_file, index=False)
            print(f"\nVergleich gespeichert in: {comparison_file}")
            
            # Erstelle einen Plot mit den kumulativen Renditen
            plt.figure(figsize=(12, 6))
            
            for name, result in strategies.items():
                plt.plot(result.portfolio['Cumulative_Net_Strategy_Returns'], label=name)
            
            plt.title("Vergleich der kumulativen Renditen")
            plt.xlabel("Datum")
            plt.ylabel("Kumulative Rendite")
            plt.legend()
            plt.grid(True)
            
            # Speichere den Plot
            plot_file = os.path.join(output_dir, f"strategy_comparison_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_file)
            print(f"Vergleichsplot gespeichert in: {plot_file}")
    
    print("\n=== Testlauf abgeschlossen ===")

if __name__ == "__main__":
    run_all_tests()
