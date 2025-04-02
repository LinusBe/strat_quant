"""
Faktormodell Strategie

Diese Strategie implementiert ein Multi-Faktor-Modell für die Aktienauswahl
basierend auf verschiedenen Faktoren wie Momentum, Value, Größe und Qualität.
"""

import pandas as pd
import numpy as np
from src.strategy_interface.base_strategy import Strategy


class FactorModelStrategy(Strategy):
    """
    Faktormodell Strategie.
    
    Diese Strategie verwendet mehrere Faktoren, um Aktien zu bewerten und
    zu selektieren. Sie kombiniert verschiedene Faktoren mit unterschiedlichen
    Gewichtungen, um ein Gesamtranking zu erstellen.
    """
    
    def __init__(self, momentum_window=60, value_weight=0.25, momentum_weight=0.25, 
                 size_weight=0.25, quality_weight=0.25, top_pct=0.2, name=None):
        """
        Initialisiert die Faktormodell Strategie.
        
        Args:
            momentum_window (int, optional): Zeitraum für die Berechnung des Momentum-Faktors.
                Standardmäßig 60 Tage.
            value_weight (float, optional): Gewichtung des Value-Faktors.
                Standardmäßig 0.25 (25%).
            momentum_weight (float, optional): Gewichtung des Momentum-Faktors.
                Standardmäßig 0.25 (25%).
            size_weight (float, optional): Gewichtung des Größen-Faktors.
                Standardmäßig 0.25 (25%).
            quality_weight (float, optional): Gewichtung des Qualitäts-Faktors.
                Standardmäßig 0.25 (25%).
            top_pct (float, optional): Prozentsatz der Top-Aktien, die gekauft werden sollen.
                Standardmäßig 0.2 (20%).
            name (str, optional): Name der Strategie. Wenn nicht angegeben,
                wird ein Name basierend auf den Parametern generiert.
        """
        if name is None:
            name = f"FactorModel_{momentum_window}_{value_weight}_{momentum_weight}_{size_weight}_{quality_weight}"
            
        super().__init__(name=name)
        
        self.momentum_window = momentum_window
        self.value_weight = value_weight
        self.momentum_weight = momentum_weight
        self.size_weight = size_weight
        self.quality_weight = quality_weight
        self.top_pct = top_pct
    
    def calculate_momentum_factor(self, data):
        """
        Berechnet den Momentum-Faktor.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            float: Momentum-Faktor.
        """
        # Berechne Rendite über den Momentum-Zeitraum
        return data['Close'].pct_change(periods=self.momentum_window).iloc[-1]
    
    def calculate_value_factor(self, data):
        """
        Berechnet den Value-Faktor (vereinfachte Version).
        
        In einer realen Implementierung würde dieser Faktor auf fundamentalen
        Daten wie P/E-Ratio, P/B-Ratio, etc. basieren.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            float: Value-Faktor.
        """
        # Vereinfachte Version: Verwende das Verhältnis von aktuellem Preis zum 52-Wochen-Hoch
        # Ein niedrigeres Verhältnis deutet auf einen besseren Value hin
        high_52_week = data['High'].rolling(window=252, min_periods=1).max().iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # Invertiere das Verhältnis, damit höhere Werte besseren Value darstellen
        return 1 - (current_price / high_52_week)
    
    def calculate_size_factor(self, data):
        """
        Berechnet den Größen-Faktor (vereinfachte Version).
        
        In einer realen Implementierung würde dieser Faktor auf der Marktkapitalisierung basieren.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            float: Größen-Faktor.
        """
        # Vereinfachte Version: Verwende das durchschnittliche Handelsvolumen als Proxy für die Größe
        # Invertiere den Wert, damit kleinere Unternehmen höhere Werte erhalten
        avg_volume = data['Volume'].rolling(window=20, min_periods=1).mean().iloc[-1]
        
        # Normalisiere und invertiere
        return -np.log(avg_volume) if avg_volume > 0 else 0
    
    def calculate_quality_factor(self, data):
        """
        Berechnet den Qualitäts-Faktor (vereinfachte Version).
        
        In einer realen Implementierung würde dieser Faktor auf fundamentalen
        Daten wie ROE, Gewinnstabilität, etc. basieren.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            float: Qualitäts-Faktor.
        """
        # Vereinfachte Version: Verwende die Volatilität als Proxy für Qualität
        # Niedrigere Volatilität deutet auf höhere Qualität hin
        volatility = data['Close'].pct_change().rolling(window=20, min_periods=1).std().iloc[-1]
        
        # Invertiere die Volatilität, damit höhere Werte bessere Qualität darstellen
        return -volatility if not np.isnan(volatility) else 0
    
    def generate_signals(self, data_dict):
        """
        Generiert Trading-Signale basierend auf dem Faktormodell.
        
        Diese Methode erwartet ein Dictionary mit DataFrames für mehrere Aktien.
        
        Args:
            data_dict (dict): Dictionary mit historischen Marktdaten für mehrere Aktien.
                Format: {'SYMBOL': pandas.DataFrame, ...}
            
        Returns:
            pandas.DataFrame: DataFrame mit generierten Signalen.
        """
        # Erstelle einen DataFrame für die Faktoren
        factor_data = pd.DataFrame(columns=['Symbol', 'Momentum', 'Value', 'Size', 'Quality', 'Total'])
        
        # Berechne Faktoren für jede Aktie
        for symbol, data in data_dict.items():
            momentum = self.calculate_momentum_factor(data)
            value = self.calculate_value_factor(data)
            size = self.calculate_size_factor(data)
            quality = self.calculate_quality_factor(data)
            
            # Füge Daten zum DataFrame hinzu
            factor_data = factor_data.append({
                'Symbol': symbol,
                'Momentum': momentum,
                'Value': value,
                'Size': size,
                'Quality': quality,
                'Total': 0.0  # Wird später berechnet
            }, ignore_index=True)
        
        # Normalisiere Faktoren (Z-Score)
        for factor in ['Momentum', 'Value', 'Size', 'Quality']:
            mean = factor_data[factor].mean()
            std = factor_data[factor].std()
            
            if std > 0:
                factor_data[factor] = (factor_data[factor] - mean) / std
            else:
                factor_data[factor] = 0.0
        
        # Berechne gewichteten Gesamtscore
        factor_data['Total'] = (
            self.momentum_weight * factor_data['Momentum'] +
            self.value_weight * factor_data['Value'] +
            self.size_weight * factor_data['Size'] +
            self.quality_weight * factor_data['Quality']
        )
        
        # Sortiere nach Gesamtscore
        factor_data = factor_data.sort_values('Total', ascending=False)
        
        # Bestimme die Top-Aktien
        num_stocks = len(factor_data)
        num_top_stocks = max(1, int(num_stocks * self.top_pct))
        
        top_stocks = factor_data.head(num_top_stocks)['Symbol'].tolist()
        
        # Erstelle Signale
        # Verwende den ersten DataFrame, um die Datumsindizes zu erhalten
        first_symbol = list(data_dict.keys())[0]
        dates = data_dict[first_symbol].index
        
        signals = pd.DataFrame(index=dates)
        signals['signal'] = 0.0
        
        # Setze Signale für den letzten Tag
        for symbol in data_dict.keys():
            if symbol in top_stocks:
                signals.loc[signals.index[-1], symbol] = 1.0
            else:
                signals.loc[signals.index[-1], symbol] = -1.0
        
        # Fülle frühere Daten mit NaN
        signals = signals.fillna(method='ffill')
        
        # Berechne Positionsänderungen
        signals['position'] = signals['signal']
        
        return signals
    
    def calculate_positions(self, data, signals):
        """
        Berechnet die Positionsgrößen basierend auf den generierten Signalen.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            signals (pandas.DataFrame): Generierte Trading-Signale.
            
        Returns:
            pandas.DataFrame: DataFrame mit berechneten Positionsgrößen.
        """
        # Kopiere die Signale
        positions = signals.copy()
        
        # Fülle NaN-Werte mit 0 auf
        positions = positions.fillna(0.0)
        
        # Gleichgewichte die Positionen
        num_top_stocks = sum(1 for col in positions.columns if positions.iloc[-1][col] > 0)
        
        if num_top_stocks > 0:
            weight_per_stock = 1.0 / num_top_stocks
            
            for col in positions.columns:
                if positions.iloc[-1][col] > 0:
                    positions[col] = weight_per_stock
        
        return positions
