"""
Machine Learning Strategie

Diese Strategie verwendet Machine Learning Algorithmen, um Preisbewegungen
vorherzusagen und entsprechende Handelssignale zu generieren.
"""

import pandas as pd
import numpy as np
from src.strategy_interface.base_strategy import Strategy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class MLStrategy(Strategy):
    """
    Machine Learning basierte Strategie.
    
    Diese Strategie verwendet einen Random Forest Classifier, um Preisbewegungen
    vorherzusagen und entsprechende Handelssignale zu generieren.
    """
    
    def __init__(self, lookback_period=10, train_size=0.7, n_estimators=100, name=None):
        """
        Initialisiert die Machine Learning Strategie.
        
        Args:
            lookback_period (int, optional): Anzahl der vergangenen Tage, die für
                die Feature-Generierung verwendet werden. Standardmäßig 10 Tage.
            train_size (float, optional): Anteil der Daten, der für das Training
                verwendet wird. Standardmäßig 0.7 (70%).
            n_estimators (int, optional): Anzahl der Bäume im Random Forest.
                Standardmäßig 100.
            name (str, optional): Name der Strategie. Wenn nicht angegeben,
                wird ein Name basierend auf den Parametern generiert.
        """
        if name is None:
            name = f"ML_RandomForest_{lookback_period}_{n_estimators}"
            
        super().__init__(name=name)
        
        self.lookback_period = lookback_period
        self.train_size = train_size
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()
    
    def create_features(self, data):
        """
        Erstellt Features für das Machine Learning Modell.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            pandas.DataFrame: DataFrame mit Features.
        """
        # Erstelle einen DataFrame für die Features
        features = pd.DataFrame(index=data.index)
        
        # Preisbasierte Features
        features['return_1d'] = data['Close'].pct_change(periods=1)
        features['return_5d'] = data['Close'].pct_change(periods=5)
        features['return_10d'] = data['Close'].pct_change(periods=10)
        
        # Volatilitätsbasierte Features
        features['volatility_5d'] = data['Close'].pct_change().rolling(window=5).std()
        features['volatility_10d'] = data['Close'].pct_change().rolling(window=10).std()
        
        # Volumenbasierte Features
        features['volume_change_1d'] = data['Volume'].pct_change(periods=1)
        features['volume_change_5d'] = data['Volume'].pct_change(periods=5)
        
        # Technische Indikatoren
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        features['rsi_14d'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        features['sma_5d'] = data['Close'].rolling(window=5, min_periods=1).mean()
        features['sma_10d'] = data['Close'].rolling(window=10, min_periods=1).mean()
        features['sma_20d'] = data['Close'].rolling(window=20, min_periods=1).mean()
        
        # Moving Average Crossovers
        features['sma_5d_10d_ratio'] = features['sma_5d'] / features['sma_10d']
        features['sma_5d_20d_ratio'] = features['sma_5d'] / features['sma_20d']
        
        # Bollinger Bands
        features['bb_20d_upper'] = features['sma_20d'] + (data['Close'].rolling(window=20).std() * 2)
        features['bb_20d_lower'] = features['sma_20d'] - (data['Close'].rolling(window=20).std() * 2)
        features['bb_position'] = (data['Close'] - features['bb_20d_lower']) / (features['bb_20d_upper'] - features['bb_20d_lower'])
        
        # Entferne NaN-Werte
        features = features.dropna()
        
        return features
    
    def create_target(self, data, horizon=1):
        """
        Erstellt die Zielvariable für das Machine Learning Modell.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            horizon (int, optional): Prognosehorizont in Tagen.
                Standardmäßig 1 Tag.
            
        Returns:
            pandas.Series: Zielvariable (1 für Aufwärtsbewegung, 0 für Abwärtsbewegung).
        """
        # Berechne zukünftige Rendite
        future_return = data['Close'].pct_change(periods=horizon).shift(-horizon)
        
        # Erstelle binäre Zielvariable (1 für positive Rendite, 0 für negative Rendite)
        target = (future_return > 0).astype(int)
        
        return target
    
    def train_model(self, features, target):
        """
        Trainiert das Machine Learning Modell.
        
        Args:
            features (pandas.DataFrame): Features für das Training.
            target (pandas.Series): Zielvariable für das Training.
        """
        # Normalisiere Features
        X = self.scaler.fit_transform(features)
        y = target.values
        
        # Initialisiere und trainiere das Modell
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
        self.model.fit(X, y)
    
    def generate_signals(self, data):
        """
        Generiert Trading-Signale basierend auf den Vorhersagen des Machine Learning Modells.
        
        Args:
            data (pandas.DataFrame): Historische Marktdaten.
            
        Returns:
            pandas.DataFrame: DataFrame mit generierten Signalen.
        """
        # Erstelle Features und Zielvariable
        features = self.create_features(data)
        target = self.create_target(data)
        
        # Stelle sicher, dass Features und Target den gleichen Index haben
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]
        
        # Bestimme Trainings- und Testdaten
        train_size = int(len(features) * self.train_size)
        
        X_train = features.iloc[:train_size]
        y_train = target.iloc[:train_size]
        
        X_test = features.iloc[train_size:]
        
        # Trainiere das Modell
        self.train_model(X_train, y_train)
        
        # Erstelle einen DataFrame für die Signale
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        
        # Normalisiere Testdaten
        X_test_scaled = self.scaler.transform(X_test)
        
        # Generiere Vorhersagen
        if len(X_test_scaled) > 0:
            predictions = self.model.predict(X_test_scaled)
            
            # Konvertiere Vorhersagen in Signale (1 für Kauf, -1 für Verkauf)
            signals.loc[X_test.index, 'signal'] = np.where(predictions == 1, 1.0, -1.0)
        
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
