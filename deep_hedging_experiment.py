#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ESPERIMENTO DEEP HEDGING: reward su P&L TERMINALE per battere la no-trade band
================================================================================

Diagnosi dell'esperimento precedente: la reward mean-variance PER-PASSO premia
il tracking stretto del Delta (alto turnover), in conflitto col risparmio sui
costi. Qui si usa la reward corretta, stile Deep Hedging (Buehler et al. 2019):
ottimizzare la media-varianza del P&L *TERMINALE*.

Implementazione: reward DENSA via telescoping (reward_mode='terminal_mv'):
    R_t = pnl_t - lambda*(2*W_{t-1}*pnl_t + pnl_t^2)   =>   sum_t R_t = W_T - lambda*W_T^2
in attesa E[W_T] - lambda*E[W_T^2] (media-varianza terminale). Cosi' l'agente puo'
lasciar driftare il Delta quando non conviene tradare -> emerge la no-trade band.

Stesso protocollo anti data-leakage (split disgiunti, VecNormalize congelato,
selezione su validation, test toccato una sola volta).
================================================================================
"""
from __future__ import annotations
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from drl_options_hedging import EnvConfig, policy_delta_hedge, policy_no_hedge
from advanced_training import train_config, evaluate, print_table, N_VAL, N_TEST
from cost_aware_experiment import make_band_policy


def main():
    HIGH = EnvConfig(commission_rate=2e-3, slippage_rate=3e-3)  # 50 bps, come prima
    print(f"[setup] Costi totali = {(HIGH.commission_rate+HIGH.slippage_rate)*1e4:.0f} bps")

    # ---- 1) Avversario forte: no-trade band ottimizzata su validation ----
    print("\n[validation] Tuning no-trade band classica...", flush=True)
    band_results = []
    for b in [5, 10, 15, 20, 30, 45]:
        m = evaluate(make_band_policy(b), "val", N_VAL, HIGH, f"Band({b})")
        band_results.append((b, m))
        print(f"    band={b:>3}: mean={m['mean']:>8.2f} std={m['std']:>7.2f} "
              f"CVaR5={m['cvar5']:>8.2f} costi={m['mean_cost']:>6.2f}", flush=True)
    best_band, best_band_m = max(band_results, key=lambda x: x[1]["cvar5"])
    print(f"    -> migliore band: {best_band}")

    # ---- 2) DRL con reward DEEP HEDGING (media-varianza terminale) ----
    drl_configs = [
        dict(cfg_name="PPO-term-l0.002", algo="PPO", reward_mode="terminal_mv",
             risk_lambda=0.002, timesteps=250_000),
        dict(cfg_name="PPO-term-l0.005", algo="PPO", reward_mode="terminal_mv",
             risk_lambda=0.005, timesteps=250_000),
        dict(cfg_name="PPO-term-l0.01", algo="PPO", reward_mode="terminal_mv",
             risk_lambda=0.01, timesteps=250_000),
        dict(cfg_name="SAC-term-l0.005", algo="SAC", reward_mode="terminal_mv",
             risk_lambda=0.005, timesteps=80_000),
    ]
    trained, val_rows = {}, []
    for c in drl_configs:
        model, normalizer, act = train_config(base_market_cfg=HIGH, **c)
        trained[c["cfg_name"]] = (model, normalizer, act)
        m = evaluate(act, "val", N_VAL, HIGH, c["cfg_name"])
        val_rows.append(m)
        print(f"    [VAL] {c['cfg_name']}: mean={m['mean']:.1f} std={m['std']:.1f} "
              f"CVaR5={m['cvar5']:.1f} costi={m['mean_cost']:.1f}", flush=True)

    val_rows += [best_band_m,
                 evaluate(policy_delta_hedge, "val", N_VAL, HIGH, "Delta-Hedge (full)"),
                 evaluate(policy_no_hedge, "val", N_VAL, HIGH, "No-Hedge")]
    print_table(val_rows, "VALIDATION (Deep Hedging) - selezione")

    # ---- 3) Selezione su validation (max CVaR-5%) ----
    drl_val = [r for r in val_rows if r["name"] in trained]
    best_name = max(drl_val, key=lambda r: r["cvar5"])["name"]
    print(f"\n[selezione] Miglior DRL su VALIDATION (max CVaR-5%): {best_name}")

    # ---- 4) TEST FINALE (500 scenari mai visti) ----
    best_act = trained[best_name][2]
    test_rows = [
        evaluate(best_act, "test", N_TEST, HIGH, f"{best_name} (DRL)"),
        evaluate(make_band_policy(best_band), "test", N_TEST, HIGH, f"Band({best_band}) ottima"),
        evaluate(policy_delta_hedge, "test", N_TEST, HIGH, "Delta-Hedge (full)"),
        evaluate(policy_no_hedge, "test", N_TEST, HIGH, "No-Hedge"),
    ]
    print_table(test_rows, f"TEST FINALE Deep Hedging ({N_TEST} scenari mai visti)")

    drl_t, band_t, full_t = test_rows[0], test_rows[1], test_rows[2]
    print("\n[esito]")
    print(f"  DRL vs Delta-Hedge pieno : P&L {drl_t['mean']:.1f} vs {full_t['mean']:.1f}  "
          f"costi {drl_t['mean_cost']:.1f} vs {full_t['mean_cost']:.1f}  "
          f"-> {'MEGLIO' if drl_t['mean'] > full_t['mean'] else 'peggio'}")
    print(f"  DRL vs Band ottima       : P&L {drl_t['mean']:.1f} vs {band_t['mean']:.1f}  "
          f"costi {drl_t['mean_cost']:.1f} vs {band_t['mean_cost']:.1f}  "
          f"-> {'MEGLIO' if drl_t['mean'] > band_t['mean'] else 'peggio'}")


if __name__ == "__main__":
    main()
