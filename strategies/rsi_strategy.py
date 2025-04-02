"""
Relative Strength Index (RSI) Strategie

Diese Strategie generiert Kauf- und Verkaufssignale basierend auf dem
Relative Strength Index (RSI) Indikator.
"""

import pandas as pd
import numpy as np
from src.strategy_interface.base_strategy import Strategy


class RSIStrategy(Strategy):
    """
    Relative Strength Index (RSI) Strategie.
    
    Diese Strategie generiert Kauf-Signale, wenn der RSI unter einen
    überverkauften Schwellenwert fällt, und Verkauf-Signale, wenn der
    RSI über einen überkauften Schwellenwert steigt.
    """
    
    def __init__(self, window=14, oversold=30, overbought=70, name=None):
        """
        Initialisiert die RSI Strategie.
        
        Args:
            window (int, optional): Länge des RSI-Fensters.
                Standardmäßig 14 Tage.
            oversold (int, optional): Überverkauft-Schwellenwert.
                Standardmäßig 30.
            overbought (int, optional): Überkauft-Schwellenwert.
                Standardmäßig 70.
            name (str, optional): Name der Strategie. Wenn nicht angegeben,
                wird ein Name basierend auf den Parametern generiert.
        """
        if name is None:
            name = f"RSI_{window}_{oversold}_{overbought}"
            
        super().__init__(name=name)
        
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, data):
        """
        Berechnet den Relative Strength Index (RSI).
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            pandas.Series: RSI-Werte.
        """
        # Berechne tägliche Preisänderungen
        delta = data['Close'].diff()
        
        # Separiere positive und negative Preisänderungen
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Berechne den durchschnittlichen Gewinn und Verlust
        avg_gain = gain.rolling(window=self.window, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.window, min_periods=1).mean()
        
        # Berechne den Relative Strength (RS)
        rs = avg_gain / avg_loss
        
        # Berechne den RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data):
        """
        Generiert Trading-Signale basierend auf dem RSI.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            pandas.DataFrame: DataFrame mit generierten Signalen.
        """
        # Erstelle einen DataFrame für die Signale
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        
        # Berechne RSI
        signals['rsi'] = self.calculate_rsi(data)
        
        # Generiere Signale basierend auf RSI-Schwellenwerten
        signals['signal'] = 0.0
        
        # Kaufsignal, wenn RSI unter überverkauft-Schwellenwert fällt
        signals.loc[signals['rsi'] < self.oversold, 'signal'] = 1.0
        
        # Verkaufssignal, wenn RSI über überkauft-Schwellenwert steigt
        signals.loc[signals['rsi'] > self.overbought, 'signal'] = -1.0
        
        # Berechne Positionsänderungen
        signals['position'] = signals['signal']
        
        # Entferne aufeinanderfolgende gleiche Signale
        signals['position'] = signals['position'].diff().fillna(signals['position'].iloc[0])
        signals.loc[signals['position'] == 0, 'position'] = np.nan
        signals['position'] = signals['position'].fillna(method='ffill').fillna(0)
        
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
        
        return positions
