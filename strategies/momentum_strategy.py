"""
Momentum Strategie

Diese Strategie generiert Kauf- und Verkaufssignale basierend auf dem
Preismomentum über einen bestimmten Zeitraum.
"""

import pandas as pd
import numpy as np
from src.strategy_interface.base_strategy import Strategy


class MomentumStrategy(Strategy):
    """
    Momentum Strategie.
    
    Diese Strategie kauft Vermögenswerte mit positivem Momentum und verkauft
    solche mit negativem Momentum, basierend auf der Annahme, dass Preisbewegungen
    dazu neigen, in die gleiche Richtung weiterzugehen.
    """
    
    def __init__(self, lookback_period=20, threshold=0.0, name=None):
        """
        Initialisiert die Momentum Strategie.
        
        Args:
            lookback_period (int, optional): Zeitraum für die Berechnung des Momentums.
                Standardmäßig 20 Tage.
            threshold (float, optional): Momentum-Schwellenwert für Signale.
                Standardmäßig 0.0.
            name (str, optional): Name der Strategie. Wenn nicht angegeben,
                wird ein Name basierend auf den Parametern generiert.
        """
        if name is None:
            name = f"Momentum_{lookback_period}_{threshold}"
            
        super().__init__(name=name)
        
        self.lookback_period = lookback_period
        self.threshold = threshold
    
    def generate_signals(self, data):
        """
        Generiert Trading-Signale basierend auf dem Preismomentum.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            pandas.DataFrame: DataFrame mit generierten Signalen.
        """
        # Erstelle einen DataFrame für die Signale
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        
        # Berechne Momentum als prozentuale Preisänderung über den Lookback-Zeitraum
        signals['momentum'] = data['Close'].pct_change(periods=self.lookback_period)
        
        # Generiere Signale basierend auf Momentum und Schwellenwert
        signals['signal'] = 0.0
        
        # Kaufsignal, wenn Momentum über dem Schwellenwert liegt
        signals.loc[signals['momentum'] > self.threshold, 'signal'] = 1.0
        
        # Verkaufssignal, wenn Momentum unter dem Schwellenwert liegt
        signals.loc[signals['momentum'] < -self.threshold, 'signal'] = -1.0
        
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
