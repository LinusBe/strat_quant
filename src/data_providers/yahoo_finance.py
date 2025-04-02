"""
Yahoo Finance Data Provider Module (Local Version)

Dieses Modul stellt eine Schnittstelle zur Yahoo Finance API bereit, um historische
Marktdaten für verschiedene Finanzinstrumente abzurufen. Diese Version verwendet
die yfinance-Bibliothek für die lokale Ausführung.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pathlib import Path
import yfinance as yf

class YahooFinanceProvider:
    """
    Eine Klasse zum Abrufen von Finanzdaten von Yahoo Finance.
    
    Diese Klasse bietet Methoden zum Abrufen historischer Kursdaten, 
    zum Caching von Daten und zur Vorverarbeitung für die Verwendung 
    in Trading-Strategien.
    """
    
    def __init__(self, cache_dir=None):
        """
        Initialisiert den Yahoo Finance Provider.
        
        Args:
            cache_dir (str, optional): Verzeichnis zum Speichern von gecachten Daten.
                Standardmäßig wird der 'data' Ordner im Projektverzeichnis verwendet.
        """
        # Kein ApiClient mehr - wir verwenden direkt yfinance
        
        if cache_dir is None:
            # Standardmäßig den 'data' Ordner im Projektverzeichnis verwenden
            self.cache_dir = Path(__file__).parents[2] / 'data'
        else:
            self.cache_dir = Path(cache_dir)
            
        # Stelle sicher, dass das Cache-Verzeichnis existiert
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_data(self, symbol, start_date=None, end_date=None, interval='1d', use_cache=True):
        """
        Ruft historische Kursdaten für ein bestimmtes Symbol ab.
        
        Args:
            symbol (str): Das Tickersymbol des Finanzinstruments.
            start_date (str, optional): Startdatum im Format 'YYYY-MM-DD'.
            end_date (str, optional): Enddatum im Format 'YYYY-MM-DD'.
            interval (str, optional): Zeitintervall der Daten ('1d', '1wk', '1mo', etc.).
            use_cache (bool, optional): Ob gecachte Daten verwendet werden sollen.
            
        Returns:
            pandas.DataFrame: DataFrame mit historischen Kursdaten.
        """
        # Wenn kein Enddatum angegeben ist, verwende das aktuelle Datum
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Wenn kein Startdatum angegeben ist, verwende 1 Jahr vor dem Enddatum
        if start_date is None:
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_date = (end_date_dt - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Überprüfe, ob Daten im Cache vorhanden sind
        cache_file = self._get_cache_filename(symbol, start_date, end_date, interval)
        
        if use_cache and os.path.exists(cache_file):
            print(f"Lade gecachte Daten für {symbol} von {start_date} bis {end_date}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        try:
            # Rufe Daten von der Yahoo Finance API ab
            data = self._fetch_data_from_api(symbol, start_date, end_date, interval)
            
            # Speichere die Daten im Cache
            if use_cache and not data.empty:
                data.to_csv(cache_file)
                
            return data
            
        except Exception as e:
            print(f"Fehler beim Abrufen der Daten für {symbol}: {e}")
            return None
    
    def _fetch_data_from_api(self, symbol, start_date, end_date, interval):
        """
        Ruft Daten von der Yahoo Finance API ab.
        
        Args:
            symbol (str): Das Tickersymbol des Finanzinstruments.
            start_date (str): Startdatum im Format 'YYYY-MM-DD'.
            end_date (str): Enddatum im Format 'YYYY-MM-DD'.
            interval (str): Zeitintervall der Daten.
            
        Returns:
            pandas.DataFrame: DataFrame mit historischen Kursdaten.
        """
        # Verwende yfinance, um Daten abzurufen
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False
        )
        
        # Stelle sicher, dass die Spalten standardisiert sind
        if not df.empty:
            # Stelle sicher, dass die Spalten die richtigen Namen haben
            df.columns = [col.title() for col in df.columns]
            
            # Füge 'Adj Close' hinzu, falls nicht vorhanden
            if 'Adj Close' not in df.columns and 'Close' in df.columns:
                df['Adj Close'] = df['Close']
                
        return df
    
    def _get_cache_filename(self, symbol, start_date, end_date, interval):
        """
        Generiert einen Dateinamen für die Cache-Datei.
        
        Args:
            symbol (str): Das Tickersymbol des Finanzinstruments.
            start_date (str): Startdatum im Format 'YYYY-MM-DD'.
            end_date (str): Enddatum im Format 'YYYY-MM-DD'.
            interval (str): Zeitintervall der Daten.
            
        Returns:
            str: Pfad zur Cache-Datei.
        """
        filename = f"{symbol}_{start_date}_{end_date}_{interval}.csv"
        return os.path.join(self.cache_dir, filename)
    
    def get_multiple_symbols(self, symbols, start_date=None, end_date=None, interval='1d'):
        """
        Ruft historische Kursdaten für mehrere Symbole ab.
        
        Args:
            symbols (list): Liste von Tickersymbolen.
            start_date (str, optional): Startdatum im Format 'YYYY-MM-DD'.
            end_date (str, optional): Enddatum im Format 'YYYY-MM-DD'.
            interval (str, optional): Zeitintervall der Daten.
            
        Returns:
            dict: Dictionary mit DataFrames für jedes Symbol.
        """
        data_dict = {}
        
        for symbol in symbols:
            print(f"Lade Daten für {symbol}...")
            df = self.get_data(symbol, start_date, end_date, interval)
            
            if df is not None and not df.empty:
                data_dict[symbol] = df
            
            # Kurze Pause, um die API nicht zu überlasten
            time.sleep(0.5)
        
        return data_dict
    
    def get_market_data(self, symbols, start_date=None, end_date=None, interval='1d'):
        """
        Ruft Marktdaten für mehrere Symbole ab und kombiniert sie in einem Panel.
        
        Args:
            symbols (list): Liste von Tickersymbolen.
            start_date (str, optional): Startdatum im Format 'YYYY-MM-DD'.
            end_date (str, optional): Enddatum im Format 'YYYY-MM-DD'.
            interval (str, optional): Zeitintervall der Daten.
            
        Returns:
            dict: Dictionary mit DataFrames für jede Kursdatenart (Open, High, Low, Close, Volume).
        """
        data_dict = self.get_multiple_symbols(symbols, start_date, end_date, interval)
        
        # Initialisiere Dictionaries für jede Kursdatenart
        panel = {
            'Open': pd.DataFrame(),
            'High': pd.DataFrame(),
            'Low': pd.DataFrame(),
            'Close': pd.DataFrame(),
            'Volume': pd.DataFrame(),
            'Adj Close': pd.DataFrame()
        }
        
        # Fülle die Dictionaries mit Daten
        for symbol, df in data_dict.items():
            for col in df.columns:
                if col in panel:
                    panel[col][symbol] = df[col]
        
        return panel
