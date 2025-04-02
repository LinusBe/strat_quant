"""
Moving Average Crossover Strategie

Diese Strategie generiert Kauf- und Verkaufssignale basierend auf dem Kreuzen
von zwei gleitenden Durchschnitten unterschiedlicher Länge.
"""

import pandas as pd
import numpy as np
from src.strategy_interface.base_strategy import Strategy


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover Strategie.
    
    Diese Strategie generiert Kauf-Signale, wenn der kurzfristige gleitende
    Durchschnitt den langfristigen gleitenden Durchschnitt von unten nach oben
    kreuzt, und Verkauf-Signale, wenn der kurzfristige gleitende Durchschnitt
    den langfristigen gleitenden Durchschnitt von oben nach unten kreuzt.
    """
    
    def __init__(self, short_window=20, long_window=50, name=None):
        """
        Initialisiert die Moving Average Crossover Strategie.
        
        Args:
            short_window (int, optional): Länge des kurzfristigen gleitenden
                Durchschnitts. Standardmäßig 20 Tage.
            long_window (int, optional): Länge des langfristigen gleitenden
                Durchschnitts. Standardmäßig 50 Tage.
            name (str, optional): Name der Strategie. Wenn nicht angegeben,
                wird ein Name basierend auf den Parametern generiert.
        """
        if name is None:
            name = f"MA_Crossover_{short_window}_{long_window}"
            
        super().__init__(name=name)
        
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        """
        Generiert Trading-Signale basierend auf dem Kreuzen der gleitenden Durchschnitte.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            pandas.DataFrame: DataFrame mit generierten Signalen.
        """
        # Erstelle einen DataFrame für die Signale
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        
        # Erstelle Spalte für den kurzfristigen gleitenden Durchschnitt
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window,
                                                     min_periods=1).mean()
        
        # Erstelle Spalte für den langfristigen gleitenden Durchschnitt
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window,
                                                    min_periods=1).mean()
        
        # Erstelle Signale
        # 1.0 = Kaufen, 0.0 = Halten, -1.0 = Verkaufen
        signals['signal'][self.long_window:] = np.where(
            signals['short_mavg'][self.long_window:] > signals['long_mavg'][self.long_window:],
            1.0, -1.0)
        
        # Generiere Handelssignale
        signals['position'] = signals['signal'].diff()
        
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
