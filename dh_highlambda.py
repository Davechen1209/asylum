#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostica: la reward terminal_mv risponde a lambda solo se molto piu' grande?
Riallena PPO con lambda alti e confronta con la band ottimizzata (band=15)."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from drl_options_hedging import EnvConfig, policy_delta_hedge, policy_no_hedge
from advanced_training import train_config, evaluate, print_table, N_VAL, N_TEST
from cost_aware_experiment import make_band_policy

def main():
    HIGH = EnvConfig(commission_rate=2e-3, slippage_rate=3e-3)
    trained, val_rows = {}, []
    for lam in [0.05, 0.2, 0.5]:
        name = f"PPO-term-l{lam}"
        m, n, act = train_config(name, "PPO", "terminal_mv", lam, timesteps=250_000, base_market_cfg=HIGH)
        trained[name] = act
        r = evaluate(act, "val", N_VAL, HIGH, name); val_rows.append(r)
        print(f"    [VAL] {name}: mean={r['mean']:.1f} std={r['std']:.1f} "
              f"CVaR5={r['cvar5']:.1f} costi={r['mean_cost']:.1f}", flush=True)
    val_rows += [evaluate(make_band_policy(15), "val", N_VAL, HIGH, "Band(15)"),
                 evaluate(policy_delta_hedge, "val", N_VAL, HIGH, "Delta-Hedge (full)")]
    print_table(val_rows, "VALIDATION lambda alti")

    best = max([r for r in val_rows if r["name"] in trained], key=lambda r: r["cvar5"])["name"]
    print(f"\n[selezione] miglior DRL: {best}")
    test_rows = [evaluate(trained[best], "test", N_TEST, HIGH, f"{best} (DRL)"),
                 evaluate(make_band_policy(15), "test", N_TEST, HIGH, "Band(15) ottima"),
                 evaluate(policy_delta_hedge, "test", N_TEST, HIGH, "Delta-Hedge (full)")]
    print_table(test_rows, f"TEST lambda alti ({N_TEST} scenari)")
    d, b = test_rows[0], test_rows[1]
    print(f"\n[esito] DRL vs Band: P&L {d['mean']:.1f} vs {b['mean']:.1f} | "
          f"costi {d['mean_cost']:.1f} vs {b['mean_cost']:.1f} -> "
          f"{'MEGLIO' if d['mean'] > b['mean'] else 'peggio'}")

if __name__ == "__main__":
    main()
