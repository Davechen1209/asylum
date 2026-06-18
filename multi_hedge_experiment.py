#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ESPERIMENTO MULTI-STRUMENTO: l'RL puo' battere il Delta-Gamma hedge classico?
================================================================================

Scenario (vedi multi_hedge_env.py): coprire una passivita' con Delta/Gamma/Vega
usando sottostante (economico, solo Delta) + opzione di hedge (cara, con Gamma e
Vega). Il Delta-Gamma hedge classico ribilancia l'opzione ad ogni passo -> caro
e subottimo; la gamma-band (ribilancia l'opzione solo fuori soglia) e' molto piu'
efficiente. NON c'e' formula chiusa cost-aware: terreno favorevole all'RL.

Avversari classici:
    - No-Hedge, Delta (solo sottostante), Delta-Gamma (pieno), Delta-Vega;
    - GAMMA-BAND ottimizzata su validation (avversario forte).
Protocollo anti data-leakage identico agli altri esperimenti: split disgiunti,
VecNormalize congelato sul training, selezione su validation, test una volta.
================================================================================
"""
from __future__ import annotations
import time, warnings
from dataclasses import replace
from typing import Callable, Dict, List
import numpy as np
warnings.filterwarnings("ignore")

from multi_hedge_env import (
    MultiHedgeConfig, MultiHedgeEnv,
    policy_no_hedge, policy_delta_stock, policy_delta_gamma, policy_delta_vega,
    make_gamma_band_policy,
)
from advanced_training import FrozenObsNormalizer, compute_metrics, print_table, N_VAL, N_TEST

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def evaluate(act: Callable, split: str, n: int, cfg_base: MultiHedgeConfig, name: str) -> Dict:
    env = MultiHedgeEnv(replace(cfg_base, data_split=split))
    env.set_episode_index(0)
    totals, costs = [], []
    for _ in range(n):
        obs, _ = env.reset(); done = False; p, c = 0.0, 0.0
        while not done:
            obs, _, term, trunc, info = env.step(act(obs, env))
            done = term or trunc; p += info["pnl"]; c += info["transaction_cost"]
        totals.append(p); costs.append(c)
    return compute_metrics(totals, costs, name)


def train_config(cfg_name, algo, reward_mode, risk_lambda, timesteps,
                 base_cfg: MultiHedgeConfig, seed=0):
    print(f"\n>>> [{cfg_name}] algo={algo} reward={reward_mode} "
          f"lambda={risk_lambda} steps={timesteps}", flush=True)
    train_cfg = replace(base_cfg, data_split="train", reward_mode=reward_mode, risk_lambda=risk_lambda)
    venv = DummyVecEnv([lambda: Monitor(MultiHedgeEnv(train_cfg))])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    pk = dict(net_arch=[256, 256])
    if algo == "PPO":
        model = PPO("MlpPolicy", venv, verbose=0, seed=seed, gamma=0.99, n_steps=2048,
                    batch_size=256, n_epochs=10, gae_lambda=0.95, ent_coef=0.0,
                    learning_rate=3e-4, clip_range=0.2, policy_kwargs=pk)
    elif algo == "SAC":
        model = SAC("MlpPolicy", venv, verbose=0, seed=seed, gamma=0.99, learning_rate=3e-4,
                    buffer_size=200_000, batch_size=256, tau=0.005, train_freq=1,
                    gradient_steps=1, learning_starts=1000, policy_kwargs=pk)
    else:
        raise ValueError(algo)

    t0 = time.time(); model.learn(total_timesteps=timesteps, progress_bar=False)
    print(f"    training completato in {time.time()-t0:.0f}s", flush=True)
    normalizer = FrozenObsNormalizer(venv)

    def act(obs, env):
        a, _ = model.predict(normalizer(obs), deterministic=True)
        return a
    return model, normalizer, act


def main():
    BASE = MultiHedgeConfig(opt_spread=0.02)  # 2% spread sul premio dell'opzione

    # Verifica anti-leakage.
    e1 = MultiHedgeEnv(replace(BASE, data_split="train")); e1.reset()
    e2 = MultiHedgeEnv(replace(BASE, data_split="test")); e2.reset()
    assert not np.allclose(e1.prices, e2.prices), "Leakage tra split!"
    print("[anti-leakage] Split train/test distinti (bande di seed disgiunte).")

    # ---- 1) Tuning gamma-band (avversario forte) su validation ----
    print("\n[validation] Tuning gamma-band...", flush=True)
    band_res = []
    for b in [1, 2, 3, 5, 8, 12]:
        m = evaluate(make_gamma_band_policy(b), "val", N_VAL, BASE, f"GBand({b})")
        band_res.append((b, m))
        print(f"    band={b:>2}: mean={m['mean']:8.2f} std={m['std']:6.2f} "
              f"CVaR5={m['cvar5']:8.2f} costi={m['mean_cost']:6.2f}", flush=True)
    best_band, best_band_m = max(band_res, key=lambda x: x[1]["cvar5"])
    print(f"    -> migliore gamma-band: {best_band}")

    # ---- 2) DRL (reward mean-variance) ----
    configs = [
        dict(cfg_name="PPO-mv-l0.1", algo="PPO", reward_mode="meanvar", risk_lambda=0.1, timesteps=250_000),
        dict(cfg_name="PPO-mv-l0.3", algo="PPO", reward_mode="meanvar", risk_lambda=0.3, timesteps=250_000),
        dict(cfg_name="SAC-mv-l0.1", algo="SAC", reward_mode="meanvar", risk_lambda=0.1, timesteps=80_000),
    ]
    trained, val_rows = {}, []
    for c in configs:
        model, norm, act = train_config(base_cfg=BASE, **c)
        trained[c["cfg_name"]] = (model, norm, act)
        m = evaluate(act, "val", N_VAL, BASE, c["cfg_name"]); val_rows.append(m)
        print(f"    [VAL] {c['cfg_name']}: mean={m['mean']:.1f} std={m['std']:.1f} "
              f"CVaR5={m['cvar5']:.1f} costi={m['mean_cost']:.1f}", flush=True)

    val_rows += [best_band_m,
                 evaluate(policy_delta_gamma, "val", N_VAL, BASE, "Delta-Gamma (full)"),
                 evaluate(policy_delta_stock, "val", N_VAL, BASE, "Delta (stock)"),
                 evaluate(policy_no_hedge, "val", N_VAL, BASE, "No-Hedge")]
    print_table(val_rows, "VALIDATION multi-strumento - selezione")

    # ---- 3) Selezione su validation (max CVaR-5%) ----
    best_name = max([r for r in val_rows if r["name"] in trained], key=lambda r: r["cvar5"])["name"]
    print(f"\n[selezione] Miglior DRL su VALIDATION (max CVaR-5%): {best_name}")

    # ---- 4) TEST FINALE ----
    best_act = trained[best_name][2]
    test_rows = [
        evaluate(best_act, "test", N_TEST, BASE, f"{best_name} (DRL)"),
        evaluate(make_gamma_band_policy(best_band), "test", N_TEST, BASE, f"GBand({best_band}) ottima"),
        evaluate(policy_delta_gamma, "test", N_TEST, BASE, "Delta-Gamma (full)"),
        evaluate(policy_delta_stock, "test", N_TEST, BASE, "Delta (stock)"),
        evaluate(policy_no_hedge, "test", N_TEST, BASE, "No-Hedge"),
    ]
    print_table(test_rows, f"TEST FINALE multi-strumento ({N_TEST} scenari mai visti)")

    drl, band, dg = test_rows[0], test_rows[1], test_rows[2]
    print("\n[esito] (criterio: rischio - CVaR-5% e std - a parita' di copertura)")
    print(f"  DRL vs Delta-Gamma pieno : CVaR {drl['cvar5']:.1f} vs {dg['cvar5']:.1f} | "
          f"std {drl['std']:.1f} vs {dg['std']:.1f} | costi {drl['mean_cost']:.1f} vs {dg['mean_cost']:.1f} "
          f"-> {'MEGLIO' if drl['cvar5'] > dg['cvar5'] else 'peggio'}")
    print(f"  DRL vs GBand ottima      : CVaR {drl['cvar5']:.1f} vs {band['cvar5']:.1f} | "
          f"std {drl['std']:.1f} vs {band['std']:.1f} | costi {drl['mean_cost']:.1f} vs {band['mean_cost']:.1f} "
          f"-> {'MEGLIO' if drl['cvar5'] > band['cvar5'] else 'peggio'}")


if __name__ == "__main__":
    main()
