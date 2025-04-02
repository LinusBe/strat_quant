"""
Performance-Metriken Modul

Dieses Modul implementiert verschiedene Performance-Metriken zur Bewertung
von Trading-Strategien.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


class PerformanceMetrics:
    """
    Eine Klasse zur Berechnung und Visualisierung von Performance-Metriken
    für Trading-Strategien.
    """
    
    def __init__(self, returns, benchmark_returns=None, risk_free_rate=0.0):
        """
        Initialisiert das PerformanceMetrics-Objekt.
        
        Args:
            returns (pandas.Series): Zeitreihe der Strategie-Renditen.
            benchmark_returns (pandas.Series, optional): Zeitreihe der Benchmark-Renditen.
            risk_free_rate (float, optional): Risikofreier Zinssatz (annualisiert).
                Standardmäßig 0.0.
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        # Berechne täglichen risikofreien Zinssatz
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        # Berechne kumulative Renditen
        self.cumulative_returns = self._calculate_cumulative_returns(returns)
        
        if benchmark_returns is not None:
            self.cumulative_benchmark_returns = self._calculate_cumulative_returns(benchmark_returns)
    
    def _calculate_cumulative_returns(self, returns):
        """
        Berechnet kumulative Renditen aus einer Zeitreihe von Renditen.
        
        Args:
            returns (pandas.Series): Zeitreihe der Renditen.
            
        Returns:
            pandas.Series: Zeitreihe der kumulativen Renditen.
        """
        return (1 + returns).cumprod() - 1
    
    def total_return(self):
        """
        Berechnet die Gesamtrendite der Strategie.
        
        Returns:
            float: Gesamtrendite.
        """
        return self.cumulative_returns.iloc[-1]
    
    def annual_return(self):
        """
        Berechnet die annualisierte Rendite der Strategie.
        
        Returns:
            float: Annualisierte Rendite.
        """
        total_return = self.total_return()
        days = len(self.returns)
        return (1 + total_return) ** (252 / days) - 1
    
    def annual_volatility(self):
        """
        Berechnet die annualisierte Volatilität der Strategie.
        
        Returns:
            float: Annualisierte Volatilität.
        """
        return self.returns.std() * np.sqrt(252)
    
    def sharpe_ratio(self):
        """
        Berechnet das Sharpe Ratio der Strategie.
        
        Returns:
            float: Sharpe Ratio.
        """
        excess_returns = self.returns - self.daily_risk_free_rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def sortino_ratio(self):
        """
        Berechnet das Sortino Ratio der Strategie.
        
        Returns:
            float: Sortino Ratio.
        """
        excess_returns = self.returns - self.daily_risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std()
        
        if downside_deviation == 0:
            return np.nan
        
        return excess_returns.mean() / downside_deviation * np.sqrt(252)
    
    def max_drawdown(self):
        """
        Berechnet den maximalen Drawdown der Strategie.
        
        Returns:
            float: Maximaler Drawdown.
        """
        cumulative_returns = self.cumulative_returns
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        return drawdown.min()
    
    def calmar_ratio(self):
        """
        Berechnet das Calmar Ratio der Strategie.
        
        Returns:
            float: Calmar Ratio.
        """
        max_dd = self.max_drawdown()
        
        if max_dd == 0:
            return np.nan
        
        return self.annual_return() / abs(max_dd)
    
    def omega_ratio(self, threshold=0.0):
        """
        Berechnet das Omega Ratio der Strategie.
        
        Args:
            threshold (float, optional): Rendite-Schwellenwert.
                Standardmäßig 0.0.
                
        Returns:
            float: Omega Ratio.
        """
        excess_returns = self.returns - threshold
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns < 0].sum())
        
        if negative_returns == 0:
            return np.inf
        
        return positive_returns / negative_returns
    
    def information_ratio(self):
        """
        Berechnet das Information Ratio der Strategie.
        
        Returns:
            float: Information Ratio.
        """
        if self.benchmark_returns is None:
            return np.nan
        
        active_returns = self.returns - self.benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        
        if tracking_error == 0:
            return np.nan
        
        return active_returns.mean() * 252 / tracking_error
    
    def beta(self):
        """
        Berechnet das Beta der Strategie.
        
        Returns:
            float: Beta.
        """
        if self.benchmark_returns is None:
            return np.nan
        
        covariance = np.cov(self.returns, self.benchmark_returns)[0, 1]
        benchmark_variance = self.benchmark_returns.var()
        
        if benchmark_variance == 0:
            return np.nan
        
        return covariance / benchmark_variance
    
    def alpha(self):
        """
        Berechnet das Alpha der Strategie.
        
        Returns:
            float: Alpha.
        """
        if self.benchmark_returns is None:
            return np.nan
        
        beta = self.beta()
        return self.annual_return() - (self.risk_free_rate + beta * (self.benchmark_returns.mean() * 252 - self.risk_free_rate))
    
    def value_at_risk(self, confidence=0.05):
        """
        Berechnet den Value at Risk (VaR) der Strategie.
        
        Args:
            confidence (float, optional): Konfidenzniveau.
                Standardmäßig 0.05 (95% Konfidenz).
                
        Returns:
            float: Value at Risk.
        """
        return np.percentile(self.returns, confidence * 100)
    
    def conditional_value_at_risk(self, confidence=0.05):
        """
        Berechnet den Conditional Value at Risk (CVaR) der Strategie.
        
        Args:
            confidence (float, optional): Konfidenzniveau.
                Standardmäßig 0.05 (95% Konfidenz).
                
        Returns:
            float: Conditional Value at Risk.
        """
        var = self.value_at_risk(confidence)
        return self.returns[self.returns <= var].mean()
    
    def drawdown_periods(self):
        """
        Identifiziert Drawdown-Perioden.
        
        Returns:
            pandas.DataFrame: DataFrame mit Drawdown-Perioden.
        """
        cumulative_returns = self.cumulative_returns
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        
        # Identifiziere Beginn und Ende von Drawdown-Perioden
        is_drawdown = drawdown < 0
        
        # Finde Übergänge
        starts = is_drawdown.astype(int).diff()
        starts = starts[starts == 1].index
        ends = is_drawdown.astype(int).diff()
        ends = ends[ends == -1].index
        
        # Wenn der letzte Drawdown noch nicht beendet ist
        if len(starts) > len(ends):
            ends = ends.append(pd.Index([drawdown.index[-1]]))
        
        # Erstelle DataFrame mit Drawdown-Perioden
        periods = []
        
        for i in range(len(starts)):
            start_date = starts[i]
            end_date = ends[i]
            
            # Finde den tiefsten Punkt im Drawdown
            lowest_point = drawdown[start_date:end_date].idxmin()
            max_drawdown = drawdown.loc[lowest_point]
            
            # Berechne die Dauer in Tagen
            duration = (end_date - start_date).days
            
            periods.append({
                'start': start_date,
                'end': end_date,
                'lowest_point': lowest_point,
                'max_drawdown': max_drawdown,
                'duration': duration
            })
        
        return pd.DataFrame(periods)
    
    def monthly_returns(self):
        """
        Berechnet monatliche Renditen.
        
        Returns:
            pandas.DataFrame: DataFrame mit monatlichen Renditen.
        """
        return self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    def annual_returns(self):
        """
        Berechnet jährliche Renditen.
        
        Returns:
            pandas.DataFrame: DataFrame mit jährlichen Renditen.
        """
        return self.returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    
    def rolling_sharpe(self, window=252):
        """
        Berechnet rollierende Sharpe Ratios.
        
        Args:
            window (int, optional): Fenstergröße in Tagen.
                Standardmäßig 252 (1 Jahr).
                
        Returns:
            pandas.Series: Zeitreihe der rollierenden Sharpe Ratios.
        """
        excess_returns = self.returns - self.daily_risk_free_rate
        return excess_returns.rolling(window=window).mean() / excess_returns.rolling(window=window).std() * np.sqrt(252)
    
    def rolling_sortino(self, window=252):
        """
        Berechnet rollierende Sortino Ratios.
        
        Args:
            window (int, optional): Fenstergröße in Tagen.
                Standardmäßig 252 (1 Jahr).
                
        Returns:
            pandas.Series: Zeitreihe der rollierenden Sortino Ratios.
        """
        excess_returns = self.returns - self.daily_risk_free_rate
        
        def downside_std(x):
            downside_returns = x[x < 0]
            return downside_returns.std() if len(downside_returns) > 0 else 0
        
        rolling_downside_std = excess_returns.rolling(window=window).apply(downside_std)
        rolling_mean = excess_returns.rolling(window=window).mean()
        
        return np.where(rolling_downside_std == 0, np.nan, rolling_mean / rolling_downside_std * np.sqrt(252))
    
    def plot_cumulative_returns(self, figsize=(12, 6), title=None):
        """
        Visualisiert kumulative Renditen.
        
        Args:
            figsize (tuple, optional): Größe der Abbildung.
                Standardmäßig (12, 6).
            title (str, optional): Titel der Abbildung.
                
        Returns:
            matplotlib.figure.Figure: Die erstellte Abbildung.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotte Strategie-Renditen
        ax.plot(self.cumulative_returns, label='Strategie')
        
        # Plotte Benchmark-Renditen, falls vorhanden
        if self.benchmark_returns is not None:
            ax.plot(self.cumulative_benchmark_returns, label='Benchmark')
        
        # Setze Titel und Beschriftungen
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Kumulative Renditen')
        
        ax.set_ylabel('Kumulative Rendite')
        ax.set_xlabel('Datum')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_drawdown(self, figsize=(12, 6), title=None):
        """
        Visualisiert Drawdowns.
        
        Args:
            figsize (tuple, optional): Größe der Abbildung.
                Standardmäßig (12, 6).
            title (str, optional): Titel der Abbildung.
                
        Returns:
            matplotlib.figure.Figure: Die erstellte Abbildung.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Berechne Drawdown
        cumulative_returns = self.cumulative_returns
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1) * 100  # In Prozent
        
        # Plotte Drawdown
        ax.fill_between(drawdown.index, drawdown, 0, color='r', alpha=0.3)
        
        # Setze Titel und Beschriftungen
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Drawdown')
        
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Datum')
        ax.grid(True)
        
        return fig
    
    def plot_monthly_returns_heatmap(self, figsize=(12, 8), title=None):
        """
        Visualisiert monatliche Renditen als Heatmap.
        
        Args:
            figsize (tuple, optional): Größe der Abbildung.
                Standardmäßig (12, 8).
            title (str, optional): Titel der Abbildung.
                
        Returns:
            matplotlib.figure.Figure: Die erstellte Abbildung.
        """
        # Berechne monatliche Renditen
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Erstelle eine Pivot-Tabelle mit Jahren als Zeilen und Monaten als Spalten
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_returns_table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        
        # Erstelle Heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(monthly_returns_table * 100, annot=True, fmt='.2f', cmap='RdYlGn',
                   linewidths=1, ax=ax, cbar_kws={'label': 'Rendite (%)'})
        
        # Setze Titel und Beschriftungen
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Monatliche Renditen (%)')
        
        ax.set_ylabel('Jahr')
        ax.set_xlabel('Monat')
        
        return fig
    
    def plot_rolling_sharpe(self, window=252, figsize=(12, 6), title=None):
        """
        Visualisiert rollierende Sharpe Ratios.
        
        Args:
            window (int, optional): Fenstergröße in Tagen.
                Standardmäßig 252 (1 Jahr).
            figsize (tuple, optional): Größe der Abbildung.
                Standardmäßig (12, 6).
            title (str, optional): Titel der Abbildung.
                
        Returns:
            matplotlib.figure.Figure: Die erstellte Abbildung.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Berechne rollierende Sharpe Ratios
        rolling_sharpe = self.rolling_sharpe(window=window)
        
        # Plotte rollierende Sharpe Ratios
        ax.plot(rolling_sharpe)
        
        # Füge horizontale Linie bei 0 hinzu
        ax.axhline(y=0, color='r', linestyle='--')
        
        # Setze Titel und Beschriftungen
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Rollierende Sharpe Ratio ({window} Tage)')
        
        ax.set_ylabel('Sharpe Ratio')
        ax.set_xlabel('Datum')
        ax.grid(True)
        
        return fig
    
    def plot_return_distribution(self, figsize=(12, 6), title=None):
        """
        Visualisiert die Verteilung der Renditen.
        
        Args:
            figsize (tuple, optional): Größe der Abbildung.
                Standardmäßig (12, 6).
            title (str, optional): Titel der Abbildung.
                
        Returns:
            matplotlib.figure.Figure: Die erstellte Abbildung.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotte Histogramm der Renditen
        sns.histplot(self.returns * 100, kde=True, ax=ax)
        
        # Füge vertikale Linie bei 0 hinzu
        ax.axvline(x=0, color='r', linestyle='--')
        
        # Setze Titel und Beschriftungen
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Verteilung der Renditen')
        
        ax.set_ylabel('Häufigkeit')
        ax.set_xlabel('Rendite (%)')
        
        return fig
    
    def summary(self):
        """
        Erstellt eine Zusammenfassung der Performance-Metriken.
        
        Returns:
            pandas.DataFrame: DataFrame mit Performance-Metriken.
        """
        metrics = {
            'Gesamtrendite': self.total_return(),
            'Jährliche Rendite': self.annual_return(),
            'Jährliche Volatilität': self.annual_volatility(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Maximaler Drawdown': self.max_drawdown(),
            'Calmar Ratio': self.calmar_ratio(),
            'Omega Ratio': self.omega_ratio(),
            'Value at Risk (95%)': self.value_at_risk(),
            'Conditional Value at Risk (95%)': self.conditional_value_at_risk()
        }
        
        # Füge Benchmark-bezogene Metriken hinzu, falls vorhanden
        if self.benchmark_returns is not None:
            metrics.update({
                'Alpha': self.alpha(),
                'Beta': self.beta(),
                'Information Ratio': self.information_ratio()
            })
        
        # Erstelle DataFrame
        metrics_df = pd.DataFrame({
            'Metrik': list(metrics.keys()),
            'Wert': list(metrics.values())
        })
        
        return metrics_df
    
    def plot_performance_dashboard(self, figsize=(15, 10)):
        """
        Erstellt ein Performance-Dashboard mit mehreren Visualisierungen.
        
        Args:
            figsize (tuple, optional): Größe der Abbildung.
                Standardmäßig (15, 10).
                
        Returns:
            matplotlib.figure.Figure: Die erstellte Abbildung.
        """
        fig = plt.figure(figsize=figsize)
        
        # Erstelle Grid für Subplots
        gs = fig.add_gridspec(3, 2)
        
        # Plot 1: Kumulative Renditen
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.cumulative_returns, label='Strategie')
        if self.benchmark_returns is not None:
            ax1.plot(self.cumulative_benchmark_returns, label='Benchmark')
        ax1.set_title('Kumulative Renditen')
        ax1.set_ylabel('Kumulative Rendite')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        cumulative_returns = self.cumulative_returns
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1) * 100  # In Prozent
        ax2.fill_between(drawdown.index, drawdown, 0, color='r', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)
        
        # Plot 3: Rollierende Sharpe Ratio
        ax3 = fig.add_subplot(gs[1, 1])
        rolling_sharpe = self.rolling_sharpe(window=252)
        ax3.plot(rolling_sharpe)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_title('Rollierende Sharpe Ratio (252 Tage)')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True)
        
        # Plot 4: Renditeverteilung
        ax4 = fig.add_subplot(gs[2, 0])
        sns.histplot(self.returns * 100, kde=True, ax=ax4)
        ax4.axvline(x=0, color='r', linestyle='--')
        ax4.set_title('Verteilung der Renditen')
        ax4.set_ylabel('Häufigkeit')
        ax4.set_xlabel('Rendite (%)')
        
        # Plot 5: Performance-Metriken
        ax5 = fig.add_subplot(gs[2, 1])
        metrics = self.summary()
        ax5.axis('off')
        
        # Erstelle Tabelle
        table_data = []
        for _, row in metrics.iterrows():
            if isinstance(row['Wert'], float):
                if row['Metrik'] in ['Gesamtrendite', 'Jährliche Rendite', 'Jährliche Volatilität', 'Maximaler Drawdown']:
                    value = f"{row['Wert'] * 100:.2f}%"
                else:
                    value = f"{row['Wert']:.2f}"
            else:
                value = str(row['Wert'])
            
            table_data.append([row['Metrik'], value])
        
        table = ax5.table(cellText=table_data, colLabels=['Metrik', 'Wert'],
                         cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.tight_layout()
        
        return fig


class RiskAnalysis:
    """
    Eine Klasse zur Analyse von Risikometriken für Trading-Strategien.
    """
    
    def __init__(self, returns, benchmark_returns=None):
        """
        Initialisiert das RiskAnalysis-Objekt.
        
        Args:
            returns (pandas.Series): Zeitreihe der Strategie-Renditen.
            benchmark_returns (pandas.Series, optional): Zeitreihe der Benchmark-Renditen.
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
    
    def downside_deviation(self, threshold=0.0):
        """
        Berechnet die Downside-Deviation der Strategie.
        
        Args:
            threshold (float, optional): Rendite-Schwellenwert.
                Standardmäßig 0.0.
                
        Returns:
            float: Downside-Deviation.
        """
        downside_returns = self.returns[self.returns < threshold]
        return downside_returns.std() * np.sqrt(252)
    
    def semi_variance(self, threshold=0.0):
        """
        Berechnet die Semi-Varianz der Strategie.
        
        Args:
            threshold (float, optional): Rendite-Schwellenwert.
                Standardmäßig 0.0.
                
        Returns:
            float: Semi-Varianz.
        """
        downside_returns = self.returns[self.returns < threshold]
        return downside_returns.var() * 252
    
    def skewness(self):
        """
        Berechnet die Schiefe der Renditeverteilung.
        
        Returns:
            float: Schiefe.
        """
        return stats.skew(self.returns)
    
    def kurtosis(self):
        """
        Berechnet die Kurtosis der Renditeverteilung.
        
        Returns:
            float: Kurtosis.
        """
        return stats.kurtosis(self.returns)
    
    def jarque_bera_test(self):
        """
        Führt den Jarque-Bera-Test auf Normalität durch.
        
        Returns:
            tuple: (Teststatistik, p-Wert)
        """
        return stats.jarque_bera(self.returns)
    
    def tail_ratio(self, quantile=0.05):
        """
        Berechnet das Tail Ratio der Renditeverteilung.
        
        Args:
            quantile (float, optional): Quantil für die Tail-Definition.
                Standardmäßig 0.05.
                
        Returns:
            float: Tail Ratio.
        """
        upper_tail = np.abs(np.percentile(self.returns, 100 - quantile * 100))
        lower_tail = np.abs(np.percentile(self.returns, quantile * 100))
        
        if lower_tail == 0:
            return np.nan
        
        return upper_tail / lower_tail
    
    def cvar_cdar_ratio(self, confidence=0.05):
        """
        Berechnet das CVaR/CDaR Ratio.
        
        Args:
            confidence (float, optional): Konfidenzniveau.
                Standardmäßig 0.05 (95% Konfidenz).
                
        Returns:
            float: CVaR/CDaR Ratio.
        """
        # Berechne CVaR
        var = np.percentile(self.returns, confidence * 100)
        cvar = self.returns[self.returns <= var].mean()
        
        # Berechne CDaR
        cumulative_returns = (1 + self.returns).cumprod() - 1
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        
        sorted_drawdowns = np.sort(drawdown)
        cdar_index = int(np.ceil(confidence * len(sorted_drawdowns))) - 1
        cdar = sorted_drawdowns[cdar_index]
        
        if cdar == 0:
            return np.nan
        
        return cvar / cdar
    
    def maximum_drawdown_duration(self):
        """
        Berechnet die maximale Drawdown-Dauer in Tagen.
        
        Returns:
            int: Maximale Drawdown-Dauer in Tagen.
        """
        cumulative_returns = (1 + self.returns).cumprod() - 1
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        
        # Identifiziere Beginn und Ende von Drawdown-Perioden
        is_drawdown = drawdown < 0
        
        # Finde Übergänge
        starts = is_drawdown.astype(int).diff()
        starts = starts[starts == 1].index
        ends = is_drawdown.astype(int).diff()
        ends = ends[ends == -1].index
        
        # Wenn der letzte Drawdown noch nicht beendet ist
        if len(starts) > len(ends):
            ends = ends.append(pd.Index([drawdown.index[-1]]))
        
        # Berechne Dauer für jede Drawdown-Periode
        durations = [(ends[i] - starts[i]).days for i in range(len(starts))]
        
        if not durations:
            return 0
        
        return max(durations)
    
    def ulcer_index(self):
        """
        Berechnet den Ulcer Index der Strategie.
        
        Returns:
            float: Ulcer Index.
        """
        cumulative_returns = (1 + self.returns).cumprod() - 1
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1) * 100  # In Prozent
        
        return np.sqrt(np.mean(drawdown ** 2))
    
    def pain_index(self):
        """
        Berechnet den Pain Index der Strategie.
        
        Returns:
            float: Pain Index.
        """
        cumulative_returns = (1 + self.returns).cumprod() - 1
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1) * 100  # In Prozent
        
        return np.abs(drawdown).mean()
    
    def pain_ratio(self, risk_free_rate=0.0):
        """
        Berechnet das Pain Ratio der Strategie.
        
        Args:
            risk_free_rate (float, optional): Risikofreier Zinssatz (annualisiert).
                Standardmäßig 0.0.
                
        Returns:
            float: Pain Ratio.
        """
        # Berechne annualisierte Rendite
        total_return = (1 + self.returns).prod() - 1
        days = len(self.returns)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # Berechne Pain Index
        pain_index = self.pain_index()
        
        if pain_index == 0:
            return np.nan
        
        return (annual_return - risk_free_rate) / pain_index
    
    def summary(self):
        """
        Erstellt eine Zusammenfassung der Risikometriken.
        
        Returns:
            pandas.DataFrame: DataFrame mit Risikometriken.
        """
        metrics = {
            'Downside-Deviation': self.downside_deviation(),
            'Semi-Varianz': self.semi_variance(),
            'Schiefe': self.skewness(),
            'Kurtosis': self.kurtosis(),
            'Tail Ratio': self.tail_ratio(),
            'Maximale Drawdown-Dauer': self.maximum_drawdown_duration(),
            'Ulcer Index': self.ulcer_index(),
            'Pain Index': self.pain_index(),
            'Pain Ratio': self.pain_ratio()
        }
        
        # Erstelle DataFrame
        metrics_df = pd.DataFrame({
            'Metrik': list(metrics.keys()),
            'Wert': list(metrics.values())
        })
        
        return metrics_df
