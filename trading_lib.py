#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TRADING_LIB - Dati, strategie classiche e motore di backtest (anti look-ahead)
================================================================================

Principi di rigore (per evitare risultati illusori):
  * NESSUN look-ahead: la posizione decisa alla chiusura del giorno t viene
    applicata al rendimento del giorno t+1 (esecuzione ritardata di 1 barra).
  * COSTI DI TRANSAZIONE: ogni variazione di posizione paga commissione+slippage.
  * OUT-OF-SAMPLE: la selezione dei parametri (walk-forward) usa solo dati passati.
  * BENCHMARK ONESTO: ogni strategia e' confrontata col buy & hold.

NB: profitto nel backtest != profitto reale. Overfitting, regime shift, slippage
reale, e impatto di mercato possono erodere o azzerare qualunque edge storico.
================================================================================
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ANN = 252  # giorni di trading per anno (annualizzazione)


# ==============================================================================
# 1. DATI
# ==============================================================================
def load_yahoo(symbol: str, start: str = "2010-01-01", end: Optional[str] = None,
               cache_dir: str = "data") -> pd.DataFrame:
    """
    Scarica prezzi giornalieri AGGIUSTATI (split/dividendi) da Yahoo Finance.
    Richiede che 'query1.finance.yahoo.com' sia nell'allowlist di rete.
    I dati vengono messi in cache su disco per riproducibilita'/offline.
    Ritorna DataFrame con colonne [open, high, low, close, volume], indice = data.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{symbol.replace('^','_')}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=["date"]).set_index("date")
        return df

    import yfinance as yf
    raw = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"Nessun dato per {symbol} (host in allowlist? simbolo valido?)")
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "date"
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_csv(cache)
    return df


def load_stooq(symbol: str, cache_dir: str = "data") -> pd.DataFrame:
    """
    Scarica prezzi giornalieri da Stooq (CSV, gratis, senza chiave).
    Richiede 'stooq.com' nell'allowlist (un solo host). Simboli: azioni US come
    'spy.us', 'aapl.us'; indici come '^spx'. Cache su disco.
    """
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    s = symbol.lower()
    if not s.startswith("^") and "." not in s:
        s = f"{s}.us"
    cache = os.path.join(cache_dir, f"{s.replace('^','_').replace('.','_')}_stooq.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=["date"]).set_index("date")
        return df
    url = f"https://stooq.com/q/d/l/?s={s}&i=d"
    txt = requests.get(url, timeout=20).text
    from io import StringIO
    raw = pd.read_csv(StringIO(txt))
    if "Date" not in raw.columns or raw.empty:
        raise RuntimeError(f"Stooq: risposta inattesa per {symbol} ({txt[:80]!r})")
    raw.columns = [c.lower() for c in raw.columns]
    df = raw.rename(columns={"date": "date"}).set_index(pd.to_datetime(raw["date"]))
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    df.to_csv(cache)
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Carica un CSV OHLCV dell'utente. Richiede almeno colonne 'date' e 'close'."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if c in ("date", "datetime", "time")), None)
    if date_col is None:
        raise ValueError("CSV senza colonna data riconoscibile (date/datetime/time).")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"
    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]
    for c in ("open", "high", "low", "volume"):
        if c not in df.columns:
            df[c] = df["close"] if c != "volume" else 0.0
    return df[["open", "high", "low", "close", "volume"]]


def simulate_prices(n_days: int = 3000, mu: float = 0.07, sigma: float = 0.20,
                    seed: int = 0) -> pd.DataFrame:
    """
    Prezzi sintetici (GBM) SOLO per collaudare la pipeline. NON rappresentano un
    mercato reale: il profitto qui e' un artefatto del drift, privo di significato.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / ANN
    rets = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2012-01-02", periods=n_days)
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 0.0}, index=idx).rename_axis("date")


# ==============================================================================
# 2. STRATEGIE CLASSICHE
# Ogni funzione ritorna una Serie di posizione-OBIETTIVO in [-1, 1] decisa alla
# chiusura del giorno t usando SOLO dati fino a t. Il ritardo di esecuzione di
# 1 barra e' applicato dal motore di backtest (niente look-ahead qui).
# ==============================================================================
def strat_buy_and_hold(px: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=px.index)


def strat_sma_crossover(px: pd.DataFrame, fast: int = 20, slow: int = 100,
                        allow_short: bool = False) -> pd.Series:
    c = px["close"]
    f = c.rolling(fast).mean()
    s = c.rolling(slow).mean()
    sig = np.where(f > s, 1.0, -1.0 if allow_short else 0.0)
    return pd.Series(sig, index=px.index).fillna(0.0)


def strat_tsmom(px: pd.DataFrame, lookback: int = 90, allow_short: bool = True) -> pd.Series:
    """Time-series momentum: long se il rendimento passato e' positivo."""
    mom = px["close"].pct_change(lookback)
    sig = np.where(mom > 0, 1.0, -1.0 if allow_short else 0.0)
    return pd.Series(sig, index=px.index).fillna(0.0)


def strat_mean_reversion(px: pd.DataFrame, lookback: int = 20, z_entry: float = 1.0,
                         allow_short: bool = True) -> pd.Series:
    """Mean-reversion su z-score: short se prezzo troppo alto, long se troppo basso."""
    c = px["close"]
    ma = c.rolling(lookback).mean()
    sd = c.rolling(lookback).std()
    z = (c - ma) / (sd + 1e-9)
    pos = -np.clip(z / z_entry, -1.0, 1.0)
    if not allow_short:
        pos = np.clip(pos, 0.0, 1.0)
    return pd.Series(pos, index=px.index).fillna(0.0)


# ==============================================================================
# 3. MOTORE DI BACKTEST (vettoriale, anti look-ahead)
# ==============================================================================
def run_backtest(px: pd.DataFrame, target_pos: pd.Series, cost_bps: float = 2.0) -> pd.DataFrame:
    """
    Esegue il backtest di una serie di posizioni-obiettivo.
        cost_bps: costo per unita' di turnover, in punti base sul nozionale.
    Ritorna un DataFrame con: ret (rendimento netto strategia), equity, pos, turnover.
    """
    ret = px["close"].pct_change().fillna(0.0)
    pos = target_pos.reindex(px.index).fillna(0.0).shift(1).fillna(0.0)  # esecuzione +1 barra
    turnover = (pos - pos.shift(1).fillna(0.0)).abs()
    cost = turnover * (cost_bps / 1e4)
    strat_ret = pos * ret - cost
    out = pd.DataFrame({
        "ret": strat_ret,
        "equity": (1.0 + strat_ret).cumprod(),
        "pos": pos,
        "turnover": turnover,
        "mkt_ret": ret,
    }, index=px.index)
    return out


def metrics(bt: pd.DataFrame, name: str = "") -> Dict:
    """Metriche economiche annualizzate dai rendimenti netti del backtest."""
    r = bt["ret"].values
    eq = bt["equity"].values
    n = len(r)
    years = n / ANN
    total_ret = eq[-1] - 1.0 if n else 0.0
    cagr = eq[-1] ** (1.0 / years) - 1.0 if years > 0 and eq[-1] > 0 else float("nan")
    vol = r.std() * np.sqrt(ANN)
    sharpe = (r.mean() * ANN) / (r.std() * np.sqrt(ANN) + 1e-12)
    downside = r[r < 0].std() * np.sqrt(ANN)
    sortino = (r.mean() * ANN) / (downside + 1e-12)
    roll_max = np.maximum.accumulate(eq)
    max_dd = float((eq / roll_max - 1.0).min()) if n else 0.0
    hit = float((r > 0).mean())
    ann_turn = float(bt["turnover"].sum() / years) if years > 0 else 0.0
    return {
        "name": name, "total_ret": float(total_ret), "cagr": float(cagr),
        "vol": float(vol), "sharpe": float(sharpe), "sortino": float(sortino),
        "max_dd": max_dd, "hit": hit, "ann_turnover": ann_turn,
        "calmar": float(cagr / (abs(max_dd) + 1e-12)) if max_dd != 0 else float("nan"),
    }


def print_metrics_table(rows: List[Dict], title: str) -> None:
    print(f"\n===== {title} =====")
    print(f"{'Strategia':<22}{'Tot.Ret':>10}{'CAGR':>9}{'Vol':>8}"
          f"{'Sharpe':>8}{'MaxDD':>9}{'Hit%':>7}{'Turn/y':>8}")
    for r in rows:
        print(f"{r['name']:<22}{r['total_ret']*100:>9.1f}%{r['cagr']*100:>8.1f}%"
              f"{r['vol']*100:>7.1f}%{r['sharpe']:>8.2f}{r['max_dd']*100:>8.1f}%"
              f"{r['hit']*100:>6.1f}%{r['ann_turnover']:>8.1f}")


# ==============================================================================
# 4. SELEZIONE WALK-FORWARD DEI PARAMETRI (out-of-sample onesto)
# ==============================================================================
def walk_forward(px: pd.DataFrame, strat_fn: Callable, param_grid: List[dict],
                 n_folds: int = 5, cost_bps: float = 2.0,
                 min_train: int = 252) -> Tuple[pd.Series, List[dict]]:
    """
    Per ciascun fold di TEST, sceglie i parametri che massimizzano lo Sharpe sui
    dati PRECEDENTI (training espandente), poi li applica al fold di test. Le
    posizioni OOS vengono concatenate -> equity onesta, senza ottimizzazione sul
    periodo valutato. Ritorna (posizioni_OOS, parametri_scelti_per_fold).
    """
    idx = px.index
    fold_bounds = np.linspace(min_train, len(idx), n_folds + 1, dtype=int)
    oos_pos = pd.Series(0.0, index=idx)
    chosen = []
    for i in range(n_folds):
        tr_end = fold_bounds[i]
        te_end = fold_bounds[i + 1]
        if tr_end >= te_end:
            continue
        train = px.iloc[:tr_end]
        best, best_sharpe = None, -1e9
        for params in param_grid:
            sig = strat_fn(train, **params)
            m = metrics(run_backtest(train, sig, cost_bps))
            if np.isfinite(m["sharpe"]) and m["sharpe"] > best_sharpe:
                best_sharpe, best = m["sharpe"], params
        best = best or param_grid[0]
        chosen.append({"fold": i, **best, "train_sharpe": round(best_sharpe, 2)})
        # Applica i parametri scelti a TUTTO lo storico fino a te_end, poi tieni
        # solo il segmento di test (cosi' gli indicatori hanno il warm-up corretto).
        full_sig = strat_fn(px.iloc[:te_end], **best)
        oos_pos.iloc[tr_end:te_end] = full_sig.iloc[tr_end:te_end].values
    return oos_pos, chosen


if __name__ == "__main__":
    # Collaudo della pipeline su dati SINTETICI (privo di significato economico).
    print("[collaudo] dati sintetici GBM - solo verifica meccanica della pipeline")
    px = simulate_prices(n_days=2500, seed=1)
    rows = []
    for name, fn, kw in [
        ("Buy&Hold", strat_buy_and_hold, {}),
        ("SMA 20/100", strat_sma_crossover, dict(fast=20, slow=100)),
        ("TSMOM 90", strat_tsmom, dict(lookback=90)),
        ("MeanRev 20", strat_mean_reversion, dict(lookback=20, z_entry=1.0)),
    ]:
        bt = run_backtest(px, fn(px, **kw), cost_bps=2.0)
        rows.append(metrics(bt, name))
    print_metrics_table(rows, "COLLAUDO (sintetico)")
    print("\n[OK] motore di backtest funzionante. Per risultati reali servono dati di mercato.")
