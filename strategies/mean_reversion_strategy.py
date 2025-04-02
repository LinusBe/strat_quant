"""
Mean Reversion Strategie

Diese Strategie basiert auf dem Konzept der Rückkehr zum Mittelwert und
generiert Kauf- und Verkaufssignale basierend auf Abweichungen vom
gleitenden Durchschnitt.
"""

import pandas as pd
import numpy as np
from src.strategy_interface.base_strategy import Strategy


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategie.
    
    Diese Strategie kauft, wenn der Preis signifikant unter seinen gleitenden
    Durchschnitt fällt, und verkauft, wenn er signifikant darüber steigt,
    basierend auf der Annahme, dass Preise tendenziell zu ihrem Mittelwert
    zurückkehren.
    """
    
    def __init__(self, window=20, std_dev=2.0, name=None):
        """
        Initialisiert die Mean Reversion Strategie.
        
        Args:
            window (int, optional): Länge des Fensters für den gleitenden
                Durchschnitt. Standardmäßig 20 Tage.
            std_dev (float, optional): Anzahl der Standardabweichungen für
                die Signalgenerierung. Standardmäßig 2.0.
            name (str, optional): Name der Strategie. Wenn nicht angegeben,
                wird ein Name basierend auf den Parametern generiert.
        """
        if name is None:
            name = f"MeanReversion_{window}_{std_dev}"
            
        super().__init__(name=name)
        
        self.window = window
        self.std_dev = std_dev
    
    def generate_signals(self, data):
        """
        Generiert Trading-Signale basierend auf Abweichungen vom gleitenden Durchschnitt.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            pandas.DataFrame: DataFrame mit generierten Signalen.
        """
        # Erstelle einen DataFrame für die Signale
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        
        # Berechne gleitenden Durchschnitt
        signals['sma'] = data['Close'].rolling(window=self.window, min_periods=1).mean()
        
        # Berechne Standardabweichung
        signals['std'] = data['Close'].rolling(window=self.window, min_periods=1).std()
        
        # Berechne oberes und unteres Band
        signals['upper_band'] = signals['sma'] + (signals['std'] * self.std_dev)
        signals['lower_band'] = signals['sma'] - (signals['std'] * self.std_dev)
        
        # Generiere Signale
        # Kaufsignal, wenn Preis unter das untere Band fällt
        signals.loc[data['Close'] < signals['lower_band'], 'signal'] = 1.0
        
        # Verkaufssignal, wenn Preis über das obere Band steigt
        signals.loc[data['Close'] > signals['upper_band'], 'signal'] = -1.0
        
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
