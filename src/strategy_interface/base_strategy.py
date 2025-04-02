"""
Basis-Strategie-Interface

Dieses Modul definiert die Basis-Klasse für Trading-Strategien und die
notwendigen Interfaces für Signalgenerierung und Positionsgrößenberechnung.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    Abstrakte Basisklasse für alle Trading-Strategien.
    
    Diese Klasse definiert die grundlegende Struktur und Schnittstellen,
    die alle Strategien implementieren müssen.
    """
    
    def __init__(self, name=None):
        """
        Initialisiert die Strategie.
        
        Args:
            name (str, optional): Name der Strategie. Wenn nicht angegeben,
                wird der Klassenname verwendet.
        """
        self.name = name if name else self.__class__.__name__
        self.positions = None
        self.signals = None
    
    @abstractmethod
    def generate_signals(self, data):
        """
        Generiert Trading-Signale basierend auf den Eingabedaten.
        
        Diese Methode muss von allen abgeleiteten Klassen implementiert werden.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            pandas.DataFrame: DataFrame mit generierten Signalen.
        """
        pass
    
    @abstractmethod
    def calculate_positions(self, data, signals):
        """
        Berechnet die Positionsgrößen basierend auf den generierten Signalen.
        
        Diese Methode muss von allen abgeleiteten Klassen implementiert werden.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            signals (pandas.DataFrame): Generierte Trading-Signale.
            
        Returns:
            pandas.DataFrame: DataFrame mit berechneten Positionsgrößen.
        """
        pass
    
    def run_strategy(self, data):
        """
        Führt die Strategie auf den gegebenen Daten aus.
        
        Diese Methode ruft generate_signals und calculate_positions auf
        und gibt die berechneten Positionen zurück.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            pandas.DataFrame: DataFrame mit berechneten Positionsgrößen.
        """
        self.signals = self.generate_signals(data)
        self.positions = self.calculate_positions(data, self.signals)
        return self.positions
    
    def get_signals(self):
        """
        Gibt die generierten Signale zurück.
        
        Returns:
            pandas.DataFrame: DataFrame mit generierten Signalen.
        """
        return self.signals
    
    def get_positions(self):
        """
        Gibt die berechneten Positionsgrößen zurück.
        
        Returns:
            pandas.DataFrame: DataFrame mit berechneten Positionsgrößen.
        """
        return self.positions


class SignalGenerator(ABC):
    """
    Abstrakte Basisklasse für Signalgeneratoren.
    
    Diese Klasse definiert die Schnittstelle für Komponenten, die
    Trading-Signale generieren.
    """
    
    @abstractmethod
    def generate(self, data):
        """
        Generiert Trading-Signale basierend auf den Eingabedaten.
        
        Diese Methode muss von allen abgeleiteten Klassen implementiert werden.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            pandas.DataFrame: DataFrame mit generierten Signalen.
        """
        pass


class PositionSizer(ABC):
    """
    Abstrakte Basisklasse für Positionsgrößenberechner.
    
    Diese Klasse definiert die Schnittstelle für Komponenten, die
    Positionsgrößen basierend auf Signalen berechnen.
    """
    
    @abstractmethod
    def calculate(self, data, signals):
        """
        Berechnet die Positionsgrößen basierend auf den generierten Signalen.
        
        Diese Methode muss von allen abgeleiteten Klassen implementiert werden.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            signals (pandas.DataFrame): Generierte Trading-Signale.
            
        Returns:
            pandas.DataFrame: DataFrame mit berechneten Positionsgrößen.
        """
        pass


class FixedPositionSizer(PositionSizer):
    """
    Positionsgrößenberechner mit fester Positionsgröße.
    
    Diese Klasse implementiert einen einfachen Positionsgrößenberechner,
    der eine feste Positionsgröße für alle Signale verwendet.
    """
    
    def __init__(self, position_size=1.0):
        """
        Initialisiert den FixedPositionSizer.
        
        Args:
            position_size (float, optional): Feste Positionsgröße.
                Standardmäßig 1.0 (100% des verfügbaren Kapitals).
        """
        self.position_size = position_size
    
    def calculate(self, data, signals):
        """
        Berechnet die Positionsgrößen basierend auf den generierten Signalen.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            signals (pandas.DataFrame): Generierte Trading-Signale.
            
        Returns:
            pandas.DataFrame: DataFrame mit berechneten Positionsgrößen.
        """
        positions = signals.copy()
        
        # Multipliziere die Signale mit der festen Positionsgröße
        for column in positions.columns:
            positions[column] = positions[column] * self.position_size
        
        return positions


class PercentagePositionSizer(PositionSizer):
    """
    Positionsgrößenberechner mit prozentualer Positionsgröße.
    
    Diese Klasse implementiert einen Positionsgrößenberechner, der
    einen bestimmten Prozentsatz des verfügbaren Kapitals für jede
    Position verwendet.
    """
    
    def __init__(self, percentage=0.1):
        """
        Initialisiert den PercentagePositionSizer.
        
        Args:
            percentage (float, optional): Prozentsatz des verfügbaren Kapitals.
                Standardmäßig 0.1 (10% des verfügbaren Kapitals).
        """
        self.percentage = percentage
    
    def calculate(self, data, signals):
        """
        Berechnet die Positionsgrößen basierend auf den generierten Signalen.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            signals (pandas.DataFrame): Generierte Trading-Signale.
            
        Returns:
            pandas.DataFrame: DataFrame mit berechneten Positionsgrößen.
        """
        positions = signals.copy()
        
        # Multipliziere die Signale mit dem Prozentsatz
        for column in positions.columns:
            positions[column] = positions[column] * self.percentage
        
        return positions


class VolatilityPositionSizer(PositionSizer):
    """
    Positionsgrößenberechner basierend auf Volatilität.
    
    Diese Klasse implementiert einen Positionsgrößenberechner, der
    die Positionsgröße basierend auf der Volatilität des Instruments
    anpasst.
    """
    
    def __init__(self, target_volatility=0.01, lookback_period=20):
        """
        Initialisiert den VolatilityPositionSizer.
        
        Args:
            target_volatility (float, optional): Zielvolatilität.
                Standardmäßig 0.01 (1% tägliche Volatilität).
            lookback_period (int, optional): Zeitraum für die Berechnung
                der Volatilität. Standardmäßig 20 Tage.
        """
        self.target_volatility = target_volatility
        self.lookback_period = lookback_period
    
    def calculate(self, data, signals):
        """
        Berechnet die Positionsgrößen basierend auf den generierten Signalen.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            signals (pandas.DataFrame): Generierte Trading-Signale.
            
        Returns:
            pandas.DataFrame: DataFrame mit berechneten Positionsgrößen.
        """
        positions = signals.copy()
        
        # Berechne die Volatilität
        returns = data['Close'].pct_change().dropna()
        volatility = returns.rolling(window=self.lookback_period).std()
        
        # Berechne den Volatilitätsfaktor
        vol_factor = self.target_volatility / volatility
        
        # Multipliziere die Signale mit dem Volatilitätsfaktor
        for column in positions.columns:
            positions[column] = positions[column] * vol_factor
        
        return positions
