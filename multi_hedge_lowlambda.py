#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplemento all'esperimento multi-strumento: sweep a LAMBDA BASSO.

Diagnosi: con la reward mean-variance per-passo, lambda alto (>=0.1) spinge
l'agente a iper-tradare l'opzione cara (azzera il P&L per-passo a ogni costo).
Qui si esplorano lambda piccoli, dove l'agente dovrebbe tradare meno e avvicinare
la frontiera costo/rischio della gamma-band. Si testano anche PPO e SAC.

Stesso protocollo anti data-leakage; selezione su validation (max CVaR-5%).
"""
from __future__ import annotations
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from multi_hedge_env import (MultiHedgeConfig, policy_delta_gamma,
                             policy_delta_stock, policy_no_hedge, make_gamma_band_policy)
from multi_hedge_experiment import train_config, evaluate
from advanced_training import print_table, N_VAL, N_TEST


def main():
    BASE = MultiHedgeConfig(opt_spread=0.02)
    BEST_BAND = 3  # selezionato su validation nel run principale

    configs = [
        dict(cfg_name="PPO-mv-l0.005", algo="PPO", reward_mode="meanvar", risk_lambda=0.005, timesteps=250_000),
        dict(cfg_name="PPO-mv-l0.02",  algo="PPO", reward_mode="meanvar", risk_lambda=0.02,  timesteps=250_000),
        dict(cfg_name="PPO-mv-l0.05",  algo="PPO", reward_mode="meanvar", risk_lambda=0.05,  timesteps=250_000),
        dict(cfg_name="SAC-mv-l0.02",  algo="SAC", reward_mode="meanvar", risk_lambda=0.02,  timesteps=80_000),
    ]
    trained, val_rows = {}, []
    for c in configs:
        model, norm, act = train_config(base_cfg=BASE, **c)
        trained[c["cfg_name"]] = (model, norm, act)
        m = evaluate(act, "val", N_VAL, BASE, c["cfg_name"]); val_rows.append(m)
        print(f"    [VAL] {c['cfg_name']}: mean={m['mean']:.1f} std={m['std']:.1f} "
              f"CVaR5={m['cvar5']:.1f} costi={m['mean_cost']:.1f}", flush=True)

    val_rows += [evaluate(make_gamma_band_policy(BEST_BAND), "val", N_VAL, BASE, f"GBand({BEST_BAND})"),
                 evaluate(policy_delta_gamma, "val", N_VAL, BASE, "Delta-Gamma (full)")]
    print_table(val_rows, "VALIDATION lambda basso")

    best_name = max([r for r in val_rows if r["name"] in trained], key=lambda r: r["cvar5"])["name"]
    print(f"\n[selezione] Miglior DRL su VALIDATION (max CVaR-5%): {best_name}")

    best_act = trained[best_name][2]
    test_rows = [
        evaluate(best_act, "test", N_TEST, BASE, f"{best_name} (DRL)"),
        evaluate(make_gamma_band_policy(BEST_BAND), "test", N_TEST, BASE, f"GBand({BEST_BAND}) ottima"),
        evaluate(policy_delta_gamma, "test", N_TEST, BASE, "Delta-Gamma (full)"),
        evaluate(policy_delta_stock, "test", N_TEST, BASE, "Delta (stock)"),
        evaluate(policy_no_hedge, "test", N_TEST, BASE, "No-Hedge"),
    ]
    print_table(test_rows, f"TEST FINALE lambda basso ({N_TEST} scenari)")
    drl, band = test_rows[0], test_rows[1]
    print(f"\n[esito] DRL vs GBand: CVaR {drl['cvar5']:.1f} vs {band['cvar5']:.1f} | "
          f"std {drl['std']:.1f} vs {band['std']:.1f} | costi {drl['mean_cost']:.1f} vs {band['mean_cost']:.1f} "
          f"-> {'MEGLIO' if drl['cvar5'] > band['cvar5'] else 'peggio'}")


if __name__ == "__main__":
    main()
