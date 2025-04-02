"""
Backtest-Engine Modul

Dieses Modul implementiert ein Framework zum Backtesten von Trading-Strategien
mit historischen Marktdaten.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pathlib import Path


class Backtest:
    """
    Eine Klasse zum Backtesten von Trading-Strategien.
    
    Diese Klasse führt Backtests für Trading-Strategien durch und
    berechnet Performance-Metriken.
    """
    
    def __init__(self, data, strategy, initial_capital=100000.0, 
                 commission=0.001, slippage=0.0):
        """
        Initialisiert den Backtest.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            strategy (Strategy): Die zu testende Trading-Strategie.
            initial_capital (float, optional): Anfangskapital für den Backtest.
                Standardmäßig 100000.0.
            commission (float, optional): Kommissionsrate pro Trade.
                Standardmäßig 0.001 (0.1%).
            slippage (float, optional): Slippage-Rate pro Trade.
                Standardmäßig 0.0 (0%).
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.positions = None
        self.portfolio = None
        self.trades = []
        
    def run(self):
        """
        Führt den Backtest durch.
        
        Returns:
            BacktestResult: Ergebnisobjekt mit Performance-Metriken.
        """
        # Strategie ausführen, um Positionen zu erhalten
        self.positions = self.strategy.run_strategy(self.data)
        
        # Portfolio-Performance berechnen
        self.portfolio = self._calculate_portfolio_performance()
        
        # Trades identifizieren
        self._identify_trades()
        
        # Ergebnisobjekt erstellen
        result = BacktestResult(
            self.data,
            self.strategy,
            self.positions,
            self.portfolio,
            self.trades,
            self.initial_capital,
            self.commission,
            self.slippage
        )
        
        return result
    
    def _calculate_portfolio_performance(self):
        """
        Berechnet die Portfolio-Performance basierend auf den Positionen.
        
        Returns:
            pandas.DataFrame: DataFrame mit Portfolio-Performance-Daten.
        """
        # Erstelle einen DataFrame für die Portfolio-Performance
        portfolio = pd.DataFrame(index=self.data.index)
        
        # Füge Schlusskurse hinzu
        portfolio['Close'] = self.data['Close']
        
        # Berechne Renditen
        portfolio['Returns'] = portfolio['Close'].pct_change()
        
        # Füge Positionen hinzu
        portfolio['Position'] = self.positions['position']
        
        # Berechne Strategie-Renditen (Position * Rendite)
        portfolio['Strategy_Returns'] = portfolio['Position'].shift(1) * portfolio['Returns']
        
        # Berücksichtige Transaktionskosten
        portfolio['Trade'] = portfolio['Position'].diff().abs()
        portfolio['Commission'] = portfolio['Trade'] * portfolio['Close'] * self.commission
        
        # Berücksichtige Slippage
        portfolio['Slippage'] = portfolio['Trade'] * portfolio['Close'] * self.slippage
        
        # Berechne Netto-Strategie-Renditen
        portfolio['Net_Strategy_Returns'] = portfolio['Strategy_Returns'] - \
                                           (portfolio['Commission'] + portfolio['Slippage']) / \
                                           (portfolio['Close'] * portfolio['Position'].shift(1))
        
        # Berechne kumulierte Renditen
        portfolio['Cumulative_Returns'] = (1 + portfolio['Returns']).cumprod()
        portfolio['Cumulative_Strategy_Returns'] = (1 + portfolio['Strategy_Returns']).cumprod()
        portfolio['Cumulative_Net_Strategy_Returns'] = (1 + portfolio['Net_Strategy_Returns']).cumprod()
        
        # Berechne Equity
        portfolio['Equity'] = self.initial_capital * portfolio['Cumulative_Net_Strategy_Returns']
        
        return portfolio
    
    def _identify_trades(self):
        """
        Identifiziert Trades basierend auf Positionsänderungen.
        """
        position_changes = self.positions['position'].diff()
        
        # Finde Zeitpunkte, an denen sich die Position ändert
        trade_points = position_changes[position_changes != 0]
        
        for date, position_change in trade_points.items():
            # Bestimme Handelsrichtung
            direction = 'BUY' if position_change > 0 else 'SELL'
            
            # Bestimme Handelsvolumen
            volume = abs(position_change)
            
            # Bestimme Preis
            price = self.data.loc[date, 'Close']
            
            # Berechne Kommission
            commission = volume * price * self.commission
            
            # Berechne Slippage
            slippage = volume * price * self.slippage
            
            # Erstelle Trade-Objekt
            trade = {
                'date': date,
                'direction': direction,
                'volume': volume,
                'price': price,
                'commission': commission,
                'slippage': slippage,
                'total_cost': volume * price + commission + slippage
            }
            
            self.trades.append(trade)


class BacktestResult:
    """
    Eine Klasse zur Speicherung und Analyse von Backtest-Ergebnissen.
    
    Diese Klasse bietet Methoden zur Berechnung von Performance-Metriken
    und zur Visualisierung der Backtest-Ergebnisse.
    """
    
    def __init__(self, data, strategy, positions, portfolio, trades, 
                 initial_capital, commission, slippage):
        """
        Initialisiert das BacktestResult-Objekt.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            strategy (Strategy): Die getestete Trading-Strategie.
            positions (pandas.DataFrame): Berechnete Positionen.
            portfolio (pandas.DataFrame): Portfolio-Performance-Daten.
            trades (list): Liste der durchgeführten Trades.
            initial_capital (float): Anfangskapital für den Backtest.
            commission (float): Kommissionsrate pro Trade.
            slippage (float): Slippage-Rate pro Trade.
        """
        self.data = data
        self.strategy = strategy
        self.positions = positions
        self.portfolio = portfolio
        self.trades = trades
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Berechne Performance-Metriken
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self):
        """
        Berechnet Performance-Metriken für den Backtest.
        
        Returns:
            dict: Dictionary mit Performance-Metriken.
        """
        # Extrahiere relevante Daten
        returns = self.portfolio['Returns']
        strategy_returns = self.portfolio['Strategy_Returns']
        net_strategy_returns = self.portfolio['Net_Strategy_Returns']
        equity = self.portfolio['Equity']
        
        # Berechne Gesamtrendite
        total_return = equity.iloc[-1] / self.initial_capital - 1
        
        # Berechne annualisierte Rendite
        days = (self.portfolio.index[-1] - self.portfolio.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # Berechne Volatilität
        daily_std = net_strategy_returns.std()
        annual_std = daily_std * np.sqrt(252)
        
        # Berechne Sharpe Ratio
        risk_free_rate = 0.0  # Annahme: Risikofreier Zinssatz ist 0
        sharpe_ratio = (annual_return - risk_free_rate) / annual_std if annual_std != 0 else 0
        
        # Berechne Drawdown
        cumulative_returns = self.portfolio['Cumulative_Net_Strategy_Returns']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Berechne Win-Rate
        if len(self.trades) > 0:
            # Berechne Gewinne/Verluste für jeden Trade
            trade_returns = []
            current_position = 0
            current_entry_price = 0
            
            for trade in self.trades:
                if trade['direction'] == 'BUY':
                    # Einstieg oder Erhöhung der Position
                    new_position = current_position + trade['volume']
                    # Gewichteter Durchschnittspreis
                    current_entry_price = (current_position * current_entry_price + 
                                          trade['volume'] * trade['price']) / new_position
                    current_position = new_position
                else:  # 'SELL'
                    # Ausstieg oder Reduzierung der Position
                    if current_position > 0:
                        # Berechne Gewinn/Verlust
                        trade_return = (trade['price'] - current_entry_price) / current_entry_price
                        trade_returns.append(trade_return)
                    
                    current_position -= trade['volume']
                    if current_position <= 0:
                        current_position = 0
                        current_entry_price = 0
            
            # Berechne Win-Rate
            winning_trades = sum(1 for r in trade_returns if r > 0)
            win_rate = winning_trades / len(trade_returns) if trade_returns else 0
        else:
            win_rate = 0
        
        # Erstelle Metrics-Dictionary
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'final_equity': equity.iloc[-1],
        }
        
        return metrics
    
    def get_metrics(self):
        """
        Gibt die berechneten Performance-Metriken zurück.
        
        Returns:
            dict: Dictionary mit Performance-Metriken.
        """
        return self.metrics
    
    def plot_performance(self, figsize=(12, 8), save_path=None):
        """
        Visualisiert die Performance des Backtests.
        
        Args:
            figsize (tuple, optional): Größe der Abbildung. Standardmäßig (12, 8).
            save_path (str, optional): Pfad zum Speichern der Abbildung.
                Wenn nicht angegeben, wird die Abbildung nicht gespeichert.
                
        Returns:
            matplotlib.figure.Figure: Die erstellte Abbildung.
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Kursverlauf und Positionen
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='Kurs')
        ax1.set_ylabel('Kurs')
        ax1.set_title(f'Backtest: {self.strategy.name}')
        
        # Füge Kauf- und Verkaufssignale hinzu
        buy_signals = self.positions[self.positions['position'] > self.positions['position'].shift(1)]
        sell_signals = self.positions[self.positions['position'] < self.positions['position'].shift(1)]
        
        ax1.scatter(buy_signals.index, self.data.loc[buy_signals.index, 'Close'], 
                   marker='^', color='g', label='Kauf')
        ax1.scatter(sell_signals.index, self.data.loc[sell_signals.index, 'Close'], 
                   marker='v', color='r', label='Verkauf')
        
        ax1.legend()
        
        # Plot 2: Equity-Kurve
        ax2 = axes[1]
        ax2.plot(self.portfolio.index, self.portfolio['Equity'], label='Equity')
        ax2.set_ylabel('Kapital')
        ax2.legend()
        
        # Plot 3: Drawdown
        ax3 = axes[2]
        cumulative_returns = self.portfolio['Cumulative_Net_Strategy_Returns']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1) * 100  # In Prozent
        
        ax3.fill_between(self.portfolio.index, drawdown, 0, color='r', alpha=0.3)
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Datum')
        
        plt.tight_layout()
        
        # Speichere die Abbildung, wenn ein Pfad angegeben ist
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def print_summary(self):
        """
        Gibt eine Zusammenfassung des Backtests aus.
        
        Returns:
            str: Zusammenfassung des Backtests.
        """
        metrics = self.metrics
        
        summary = f"""
        ===== Backtest-Zusammenfassung: {self.strategy.name} =====
        
        Zeitraum: {self.portfolio.index[0].strftime('%Y-%m-%d')} bis {self.portfolio.index[-1].strftime('%Y-%m-%d')}
        Anfangskapital: {self.initial_capital:.2f}
        Endkapital: {metrics['final_equity']:.2f}
        
        Performance-Metriken:
        - Gesamtrendite: {metrics['total_return'] * 100:.2f}%
        - Jährliche Rendite: {metrics['annual_return'] * 100:.2f}%
        - Jährliche Volatilität: {metrics['annual_volatility'] * 100:.2f}%
        - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        - Maximaler Drawdown: {metrics['max_drawdown'] * 100:.2f}%
        - Win-Rate: {metrics['win_rate'] * 100:.2f}%
        - Anzahl der Trades: {metrics['num_trades']}
        
        Transaktionskosten:
        - Kommission: {self.commission * 100:.2f}%
        - Slippage: {self.slippage * 100:.2f}%
        """
        
        return summary
    
    def save_results(self, output_dir):
        """
        Speichert die Backtest-Ergebnisse in Dateien.
        
        Args:
            output_dir (str): Verzeichnis zum Speichern der Ergebnisse.
            
        Returns:
            dict: Dictionary mit Pfaden zu den gespeicherten Dateien.
        """
        # Stelle sicher, dass das Ausgabeverzeichnis existiert
        os.makedirs(output_dir, exist_ok=True)
        
        # Generiere einen Zeitstempel für die Dateinamen
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Speichere Portfolio-Daten
        portfolio_file = os.path.join(output_dir, f'portfolio_{self.strategy.name}_{timestamp}.csv')
        self.portfolio.to_csv(portfolio_file)
        
        # Speichere Trades
        trades_file = os.path.join(output_dir, f'trades_{self.strategy.name}_{timestamp}.csv')
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.to_csv(trades_file, index=False)
        
        # Speichere Performance-Plot
        plot_file = os.path.join(output_dir, f'performance_{self.strategy.name}_{timestamp}.png')
        self.plot_performance(save_path=plot_file)
        
        # Speichere Zusammenfassung
        summary_file = os.path.join(output_dir, f'summary_{self.strategy.name}_{timestamp}.txt')
        with open(summary_file, 'w') as f:
            f.write(self.print_summary())
        
        # Erstelle Dictionary mit Pfaden
        result_files = {
            'portfolio': portfolio_file,
            'trades': trades_file,
            'plot': plot_file,
            'summary': summary_file
        }
        
        return result_files


class EventDrivenBacktest:
    """
    Eine ereignisbasierte Backtest-Engine für komplexere Szenarien.
    
    Diese Klasse implementiert einen ereignisbasierten Backtest, der
    realistischere Simulationen von Marktbedingungen ermöglicht.
    """
    
    def __init__(self, data_handler, strategy, portfolio, execution_handler):
        """
        Initialisiert den ereignisbasierten Backtest.
        
        Args:
            data_handler: Handler für Marktdaten.
            strategy: Die zu testende Trading-Strategie.
            portfolio: Portfolio-Manager.
            execution_handler: Handler für die Orderausführung.
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.events = []
    
    def run(self):
        """
        Führt den ereignisbasierten Backtest durch.
        
        Returns:
            BacktestResult: Ergebnisobjekt mit Performance-Metriken.
        """
        # Implementierung des ereignisbasierten Backtests
        # (Vereinfachte Version für dieses Beispiel)
        
        # Initialisiere Daten
        self.data_handler.initialize()
        
        # Hauptschleife des Backtests
        while self.data_handler.has_more_data():
            # Aktualisiere Marktdaten
            self.data_handler.update()
            
            # Generiere Signale
            self.strategy.generate_signals(self.data_handler.get_latest_data())
            
            # Aktualisiere Portfolio
            self.portfolio.update(self.data_handler.get_latest_data())
            
            # Führe Orders aus
            self.execution_handler.execute_orders(self.portfolio.get_orders())
        
        # Erstelle Ergebnisobjekt
        result = BacktestResult(
            self.data_handler.get_all_data(),
            self.strategy,
            self.portfolio.get_positions(),
            self.portfolio.get_performance(),
            self.portfolio.get_trades(),
            self.portfolio.get_initial_capital(),
            self.execution_handler.get_commission_rate(),
            self.execution_handler.get_slippage_rate()
        )
        
        return result
