#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ESPERIMENTO DI TRADING: classico vs RL vs Buy & Hold (out-of-sample onesto)
================================================================================

Pipeline:
  1. Carica dati (Yahoo reali se l'host e' in allowlist; oppure CSV; altrimenti
     dati SINTETICI con avviso esplicito -> NON significativi).
  2. Split temporale: TRAIN (passato) / TEST (futuro mai visto).
  3. Strategie classiche: selezione dei parametri SOLO sul train (per Sharpe),
     valutazione sul test. Inoltre equity walk-forward per robustezza.
  4. Agente RL (PPO/SAC): addestrato SOLO sul train, valutato sul test.
  5. Confronto onesto col Buy & Hold sul test + grafico equity.

Uso:
    python trading_experiment.py SYMBOL            (es. SPY, AAPL, ^GSPC)
    python trading_experiment.py --csv path.csv
================================================================================
"""
from __future__ import annotations
import sys, warnings
from typing import Dict, List
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trading_lib import (
    load_yahoo, load_csv, simulate_prices,
    strat_buy_and_hold, strat_sma_crossover, strat_tsmom, strat_mean_reversion,
    run_backtest, metrics, print_metrics_table, ANN,
)
from trading_rl import make_features, train_rl, eval_rl_positions

COST_BPS = 2.0           # costo per turnover (2 bps ~ ETF liquido)
TEST_YEARS = 4           # ultimi N anni come test out-of-sample


# ------------------------------------------------------------------------------
def get_data(argv: List[str]) -> tuple[pd.DataFrame, str, bool]:
    """Ritorna (prezzi, etichetta, is_real)."""
    if "--csv" in argv:
        path = argv[argv.index("--csv") + 1]
        return load_csv(path), f"CSV:{path}", True
    symbol = next((a for a in argv[1:] if not a.startswith("--")), "SPY")
    try:
        px = load_yahoo(symbol)
        if len(px) > 300:
            return px, symbol, True
    except Exception as e:
        print(f"[avviso] dati reali non disponibili ({type(e).__name__}: {e}).")
    print("[avviso] >>> USO DATI SINTETICI: risultati NON significativi (collaudo). <<<")
    return simulate_prices(n_days=3500, seed=7), "SINTETICO", False


def metrics_on(bt_full: pd.DataFrame, idx, name: str) -> Dict:
    """Metriche su un sotto-periodo, ricostruendo l'equity da 1.0."""
    sub = bt_full.loc[idx].copy()
    sub["equity"] = (1.0 + sub["ret"]).cumprod()
    return metrics(sub, name)


# ------------------------------------------------------------------------------
def main():
    px, label, is_real = get_data(sys.argv)
    n = len(px)
    split = px.index[max(0, n - TEST_YEARS * ANN)]
    train_idx = px.index[px.index < split]
    test_idx = px.index[px.index >= split]
    print(f"\n[dati] {label}: {n} barre dal {px.index[0].date()} al {px.index[-1].date()}")
    print(f"[split] TRAIN {train_idx[0].date()}..{train_idx[-1].date()} "
          f"({len(train_idx)})  |  TEST {test_idx[0].date()}..{test_idx[-1].date()} ({len(test_idx)})")

    # ---------------- Strategie classiche: tuning su TRAIN, valutazione su TEST ----------------
    px_tr = px.loc[train_idx]
    families = {
        "SMA": (strat_sma_crossover, [dict(fast=f, slow=s) for f in (10, 20, 50) for s in (100, 150, 200) if f < s]),
        "TSMOM": (strat_tsmom, [dict(lookback=l) for l in (20, 60, 90, 120, 200)]),
        "MeanRev": (strat_mean_reversion, [dict(lookback=l, z_entry=z) for l in (10, 20, 40) for z in (1.0, 1.5, 2.0)]),
    }
    test_rows: List[Dict] = []
    classical_pos = {}
    for fam, (fn, grid) in families.items():
        best, best_sh = grid[0], -1e9
        for params in grid:
            m = metrics(run_backtest(px_tr, fn(px_tr, **params), COST_BPS))
            if np.isfinite(m["sharpe"]) and m["sharpe"] > best_sh:
                best_sh, best = m["sharpe"], params
        sig_full = fn(px, **best)                      # segnale su tutto lo storico (warm-up corretto)
        bt_full = run_backtest(px, sig_full, COST_BPS)
        classical_pos[fam] = bt_full["pos"]
        m_test = metrics_on(bt_full, test_idx, f"{fam} {best}")
        test_rows.append(m_test)
        print(f"[classico] {fam:8} best={best} (train Sharpe {best_sh:.2f})")

    # ---------------- Agente RL: train su TRAIN, valutazione su TEST ----------------
    feat_df, ret_s = make_features(px)
    feat = feat_df.values
    ret = ret_s.values
    tr_mask = (px.index < split).values
    n_tr = int(tr_mask.sum())
    for algo, steps in [("PPO", 200_000), ("SAC", 60_000)]:
        model, norm = train_rl(feat[:n_tr], ret[:n_tr], algo=algo, timesteps=steps, cost_bps=COST_BPS)
        pos_full = eval_rl_positions(model, norm, feat, ret, COST_BPS)
        pos_full.index = px.index
        bt_full = run_backtest(px, pos_full, COST_BPS)
        classical_pos[f"RL-{algo}"] = bt_full["pos"]
        test_rows.append(metrics_on(bt_full, test_idx, f"RL-{algo}"))

    # ---------------- Benchmark Buy & Hold sul test ----------------
    bh_full = run_backtest(px, strat_buy_and_hold(px), COST_BPS)
    bh_test = metrics_on(bh_full, test_idx, "Buy&Hold")
    test_rows.append(bh_test)

    # ---------------- Tabella e grafico ----------------
    print_metrics_table(test_rows, f"TEST OUT-OF-SAMPLE ({label}) - {test_idx[0].date()}..{test_idx[-1].date()}")

    plt.figure(figsize=(11, 6))
    for fam, pos in classical_pos.items():
        eq = (1.0 + run_backtest(px, pos, COST_BPS)["ret"].loc[test_idx]).cumprod()
        plt.plot(eq.index, eq.values, label=fam, linewidth=1.3)
    eq_bh = (1.0 + bh_full["ret"].loc[test_idx]).cumprod()
    plt.plot(eq_bh.index, eq_bh.values, "k--", label="Buy&Hold", linewidth=2)
    plt.title(f"Equity out-of-sample ({label}) - costi {COST_BPS} bps")
    plt.ylabel("Equity (start = 1.0)"); plt.legend(); plt.grid(alpha=0.3)
    fname = f"trading_equity_{label.replace(':','_').replace('/','_')}.png"
    plt.tight_layout(); plt.savefig(fname, dpi=110)
    print(f"\n[grafico] salvato {fname}")

    # ---------------- Verdetto onesto vs Buy & Hold ----------------
    bh_sh = bh_test["sharpe"]
    beats = [r for r in test_rows if r["name"] != "Buy&Hold" and r["sharpe"] > bh_sh]
    print("\n[verdetto] (criterio: Sharpe out-of-sample vs Buy&Hold)")
    print(f"  Buy&Hold: Sharpe {bh_sh:.2f}, Tot.Ret {bh_test['total_ret']*100:.1f}%, MaxDD {bh_test['max_dd']*100:.1f}%")
    if beats:
        for r in sorted(beats, key=lambda x: -x["sharpe"]):
            print(f"  BATTE B&H: {r['name']:<16} Sharpe {r['sharpe']:.2f} "
                  f"(Tot.Ret {r['total_ret']*100:.1f}%, MaxDD {r['max_dd']*100:.1f}%)")
    else:
        print("  Nessuna strategia batte il Buy&Hold out-of-sample (risultato comune e onesto).")
    if not is_real:
        print("\n[NB] Dati sintetici: i numeri NON hanno valore economico. Servono dati reali.")


if __name__ == "__main__":
    main()
