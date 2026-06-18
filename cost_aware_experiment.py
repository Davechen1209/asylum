#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ESPERIMENTO COST-AWARE: l'RL puo' battere il Delta-Hedge?
================================================================================

In regime di COSTI DI TRANSAZIONE ELEVATI il delta-hedging classico (che
ribilancia ad ogni passo) diventa subottimo: la strategia ottima e' una
"no-trade band" (si ribilancia solo quando il Delta residuo esce da una soglia).

Questo esperimento verifica, con protocollo rigoroso e privo di data leakage
(stessi accorgimenti di advanced_training.py), se l'agente DRL impara da solo
una politica a banda che batte:
    - Delta-Hedge pieno (ribilancio ad ogni passo);
    - Delta-Hedge con NO-TRADE BAND ottimizzata su validation (avversario forte);
    - No-Hedge.

Selezione (banda dei baseline e configurazione DRL) SOLO su validation, tramite
CVaR-5% (Expected Shortfall: economicamente sensato, cattura costi + coda).
Test (500 scenari mai visti) valutato una sola volta.
================================================================================
"""
from __future__ import annotations

import warnings
from dataclasses import replace
import numpy as np

warnings.filterwarnings("ignore")

from drl_options_hedging import EnvConfig, policy_delta_hedge, policy_no_hedge, EPS
from advanced_training import train_config, evaluate, print_table, N_VAL, N_TEST


# ------------------------------------------------------------------------------
# Baseline classica con NO-TRADE BAND: ribilancia al Delta neutro solo se il
# Delta di portafoglio supera in valore assoluto la soglia `band` (in unita').
# ------------------------------------------------------------------------------
def make_band_policy(band: float):
    def policy(obs, env):
        g = env.current_option_greeks()
        port_delta = env.option_qty * g["delta"] + env.hedge
        q_abs = abs(env.option_qty) + EPS
        if abs(port_delta) > band:
            # Fuori banda: ribilancia per azzerare il Delta.
            a = (-env.option_qty * g["delta"]) / q_abs
        else:
            # Dentro banda: mantieni la posizione (nessun trade).
            a = env.hedge / q_abs
        return np.array([np.clip(a, -1.0, 1.0)], dtype=np.float32)
    return policy


def main():
    # Regime ad ALTI COSTI: 50 bps totali (5x rispetto all'esperimento base).
    HIGH = EnvConfig(commission_rate=2e-3, slippage_rate=3e-3)
    print(f"[setup] Costi totali = {(HIGH.commission_rate + HIGH.slippage_rate)*1e4:.0f} bps "
          f"(commissione {HIGH.commission_rate*1e4:.0f} + slippage {HIGH.slippage_rate*1e4:.0f})")

    # -------------------- 1) Tuning della NO-TRADE BAND su validation --------------------
    print("\n[validation] Tuning della no-trade band (avversario classico forte)...", flush=True)
    bands = [2, 5, 10, 15, 20, 30, 45]
    band_results = []
    for b in bands:
        m = evaluate(make_band_policy(b), "val", N_VAL, HIGH, f"Band({b})")
        band_results.append((b, m))
        print(f"    band={b:>3}: mean={m['mean']:>8.2f}  std={m['std']:>7.2f}  "
              f"CVaR5={m['cvar5']:>8.2f}  costi={m['mean_cost']:>6.2f}", flush=True)
    best_band, best_band_m = max(band_results, key=lambda x: x[1]["cvar5"])  # max CVaR = min tail-risk
    print(f"    -> migliore band su validation (max CVaR-5%): {best_band}")

    # -------------------- 2) Addestramento configurazioni DRL (alti costi) --------------------
    drl_configs = [
        dict(cfg_name="PPO-mv-l0.03", algo="PPO", reward_mode="meanvar",
             risk_lambda=0.03, timesteps=250_000),
        dict(cfg_name="PPO-mv-l0.10", algo="PPO", reward_mode="meanvar",
             risk_lambda=0.10, timesteps=250_000),
        dict(cfg_name="SAC-mv-l0.05", algo="SAC", reward_mode="meanvar",
             risk_lambda=0.05, timesteps=80_000),
    ]
    trained, val_rows = {}, []
    for c in drl_configs:
        model, normalizer, act = train_config(base_market_cfg=HIGH, **c)
        trained[c["cfg_name"]] = (model, normalizer, act)
        m = evaluate(act, "val", N_VAL, HIGH, c["cfg_name"])
        val_rows.append(m)
        print(f"    [VAL] {c['cfg_name']}: mean={m['mean']:.1f}  std={m['std']:.1f}  "
              f"CVaR5={m['cvar5']:.1f}  costi={m['mean_cost']:.1f}", flush=True)

    # Baseline su validation (riferimento completo).
    val_rows += [
        best_band_m,
        evaluate(policy_delta_hedge, "val", N_VAL, HIGH, "Delta-Hedge (full)"),
        evaluate(policy_no_hedge, "val", N_VAL, HIGH, "No-Hedge"),
    ]
    print_table(val_rows, "VALIDATION (alti costi) - selezione modello")

    # -------------------- 3) Selezione del miglior DRL su validation --------------------
    drl_val = [r for r in val_rows if r["name"] in trained]
    best_name = max(drl_val, key=lambda r: r["cvar5"])["name"]  # max CVaR-5%
    print(f"\n[selezione] Miglior DRL su VALIDATION (max CVaR-5%): {best_name}")

    # -------------------- 4) TEST FINALE (500 scenari mai visti) --------------------
    best_act = trained[best_name][2]
    test_rows = [
        evaluate(best_act, "test", N_TEST, HIGH, f"{best_name} (DRL)"),
        evaluate(make_band_policy(best_band), "test", N_TEST, HIGH, f"Band({best_band}) ottima"),
        evaluate(policy_delta_hedge, "test", N_TEST, HIGH, "Delta-Hedge (full)"),
        evaluate(policy_no_hedge, "test", N_TEST, HIGH, "No-Hedge"),
    ]
    print_table(test_rows, f"TEST FINALE alti costi ({N_TEST} scenari mai visti)")

    # Esito sintetico.
    drl_t = test_rows[0]; band_t = test_rows[1]; full_t = test_rows[2]
    print("\n[esito]")
    print(f"  DRL vs Delta-Hedge pieno : P&L medio {drl_t['mean']:.1f} vs {full_t['mean']:.1f} "
          f"({'MEGLIO' if drl_t['mean'] > full_t['mean'] else 'peggio'})")
    print(f"  DRL vs Band ottima       : P&L medio {drl_t['mean']:.1f} vs {band_t['mean']:.1f} "
          f"({'MEGLIO' if drl_t['mean'] > band_t['mean'] else 'peggio'})")


if __name__ == "__main__":
    main()
