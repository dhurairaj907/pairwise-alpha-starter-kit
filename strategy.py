
def get_coin_metadata():
    return {
        "targets": [
            {"symbol": "BONK", "timeframe": "1H"},
            {"symbol": "DOGE", "timeframe": "2H"},
            {"symbol": "PEPE", "timeframe": "4H"},
        ],
        "anchors": [
            {"symbol": "BTC", "timeframe": "1H"},
            {"symbol": "ETH", "timeframe": "1H"},
            {"symbol": "BNB", "timeframe": "2H"},
            {"symbol": "SOL", "timeframe": "4H"},
            {"symbol": "USDT", "timeframe": "1D"},
        ]
    }

import pandas as pd
import numpy as np

def generate_signals(candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-coin SMA crossover with volatility-adjusted position sizing.
    Anchor-based trend confirmation (e.g., BTC).
    Deterministic, no future data used.
    """
    def calculate_volatility(close_series, window=10):
        return close_series.pct_change().rolling(window).std()

    results = []

    for symbol in candles_target['symbol'].unique():
        df = candles_target[candles_target['symbol'] == symbol].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # SMA signals
        df["sma_fast"] = df["close"].rolling(window=5).mean()
        df["sma_slow"] = df["close"].rolling(window=15).mean()

        # Volatility-based sizing
        df["volatility"] = calculate_volatility(df["close"], window=10)
        df["base_position_size"] = 1 / (df["volatility"] + 1e-6)
        df["base_position_size"] = df["base_position_size"].clip(upper=1.0)

        # Entry/exit signals
        df["signal"] = 0
        df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
        df.loc[df["sma_fast"] < df["sma_slow"], "signal"] = -1

        # Anchor filter using BTC
        if "close_BTC_1H" in candles_anchor.columns:
            btc_fast = candles_anchor["close_BTC_1H"].rolling(10).mean()
            btc_slow = candles_anchor["close_BTC_1H"].rolling(20).mean()
            btc_trend = (btc_fast > btc_slow).astype(int)
            df["signal"] *= btc_trend.values[:len(df)]  # sync length

        df["position_size"] = df["signal"] * df["base_position_size"]

        results.append(df[["timestamp", "symbol", "signal", "position_size"]])

    return pd.concat(results, ignore_index=True)
