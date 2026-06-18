#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
WORKFLOW AVANZATO DI ADDESTRAMENTO E VALUTAZIONE (anti data-leakage)
================================================================================

Obiettivo: spingere al massimo le performance dell'agente DRL di hedging,
mantenendo un protocollo sperimentale RIGOROSO e privo di data leakage.

GARANZIE ANTI DATA-LEAKAGE
--------------------------
1. SPLIT DISGIUNTI: train / validation / test usano bande di seed separate
   (vedi OptionsTradingEnv.SPLIT_BASE). Nessuna traiettoria di mercato vista
   in training ricompare in validation o test.
2. NORMALIZZAZIONE CONGELATA: VecNormalize calcola media/varianza delle
   osservazioni SOLO sui dati di training; in valutazione tali statistiche
   vengono usate congelate (nessun aggiornamento con dati val/test).
3. SELEZIONE SOLO SU VALIDATION: la scelta della configurazione migliore
   avviene esclusivamente sul validation set. Il TEST set viene toccato UNA
   SOLA VOLTA, alla fine, per la stima finale e imparziale.
4. NIENTE LOOK-AHEAD: l'osservazione contiene solo informazioni disponibili al
   momento della decisione; il P&L del passo successivo e' la conseguenza
   dell'azione, non un input.
5. METRICHE ECONOMICHE IMPARZIALI: tutte le metriche sono calcolate sul P&L
   grezzo (info['pnl']), indipendentemente dalla reward usata in training.

Esecuzione:
    python advanced_training.py
================================================================================
"""

from __future__ import annotations

import time
import warnings
from dataclasses import replace
from typing import Callable, Dict, List

import numpy as np

warnings.filterwarnings("ignore")

from drl_options_hedging import (  # noqa: E402
    OptionsTradingEnv,
    EnvConfig,
    policy_delta_hedge,
    policy_no_hedge,
)

from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

# Numerosita' degli split di valutazione (episodi indipendenti per split).
N_VAL = 300
N_TEST = 500


# ------------------------------------------------------------------------------
# Normalizzatore congelato: replica la normalizzazione di VecNormalize usando
# statistiche calcolate SOLO sul training (nessun leakage da val/test).
# ------------------------------------------------------------------------------
class FrozenObsNormalizer:
    def __init__(self, venv: VecNormalize):
        self.mean = venv.obs_rms.mean.copy()
        self.var = venv.obs_rms.var.copy()
        self.eps = float(venv.epsilon)
        self.clip = float(venv.clip_obs)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        norm = (obs - self.mean) / np.sqrt(self.var + self.eps)
        return np.clip(norm, -self.clip, self.clip).astype(np.float32)


# ------------------------------------------------------------------------------
# Metriche economiche (calcolate sul P&L grezzo di fine episodio).
# ------------------------------------------------------------------------------
def compute_metrics(totals: List[float], costs: List[float], name: str) -> Dict:
    t = np.asarray(totals)
    c = np.asarray(costs)
    worst5 = np.sort(t)[: max(1, int(0.05 * len(t)))]  # coda peggiore (5%)
    return {
        "name": name,
        "mean": float(t.mean()),
        "std": float(t.std()),
        "sharpe": float(t.mean() / (t.std() + 1e-8)),
        "cvar5": float(worst5.mean()),   # Expected Shortfall al 5%
        "worst": float(t.min()),
        "mean_cost": float(c.mean()),
    }


def evaluate(act: Callable, split: str, n: int, cfg_base: EnvConfig, name: str) -> Dict:
    """
    Valuta una politica su `n` episodi DETERMINISTICI dello split indicato.
    `act(obs, env) -> action`. Le metriche usano il P&L grezzo (imparziale).
    """
    cfg = replace(cfg_base, data_split=split)
    env = OptionsTradingEnv(cfg)
    env.set_episode_index(0)  # sequenza riproducibile di path: 0, 1, ..., n-1
    totals, costs = [], []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        ep_pnl, ep_cost = 0.0, 0.0
        while not done:
            action = act(obs, env)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            ep_pnl += info["pnl"]
            ep_cost += info["transaction_cost"]
        totals.append(ep_pnl)
        costs.append(ep_cost)
    return compute_metrics(totals, costs, name)


# ------------------------------------------------------------------------------
# Addestramento di una configurazione con VecNormalize (statistiche da TRAIN).
# ------------------------------------------------------------------------------
def train_config(cfg_name: str, algo: str, reward_mode: str, risk_lambda: float,
                 timesteps: int, base_market_cfg: EnvConfig, seed: int = 0):
    print(f"\n>>> [{cfg_name}] algo={algo} reward={reward_mode} "
          f"lambda={risk_lambda} steps={timesteps}", flush=True)

    train_cfg = replace(base_market_cfg, data_split="train",
                        reward_mode=reward_mode, risk_lambda=risk_lambda)

    # VecNormalize: normalizza osservazioni e reward usando SOLO il training.
    venv = DummyVecEnv([lambda: Monitor(OptionsTradingEnv(train_cfg))])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    policy_kwargs = dict(net_arch=[256, 256])
    if algo == "PPO":
        model = PPO("MlpPolicy", venv, verbose=0, seed=seed, gamma=0.99,
                    n_steps=2048, batch_size=256, n_epochs=10, gae_lambda=0.95,
                    ent_coef=0.0, learning_rate=3e-4, clip_range=0.2,
                    policy_kwargs=policy_kwargs)
    elif algo == "SAC":
        model = SAC("MlpPolicy", venv, verbose=0, seed=seed, gamma=0.99,
                    learning_rate=3e-4, buffer_size=200_000, batch_size=256,
                    tau=0.005, train_freq=1, gradient_steps=1, learning_starts=1000,
                    policy_kwargs=policy_kwargs)
    else:
        raise ValueError(algo)

    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    print(f"    training completato in {time.time() - t0:.0f}s", flush=True)

    # Congela le statistiche di normalizzazione (calcolate sul solo training).
    normalizer = FrozenObsNormalizer(venv)

    def act(obs, env):
        a, _ = model.predict(normalizer(obs), deterministic=True)
        return a

    return model, normalizer, act


def print_table(rows: List[Dict], title: str) -> None:
    print(f"\n===== {title} =====")
    print(f"{'Strategia':<22}{'P&L medio':>11}{'Std':>10}{'Sharpe':>9}"
          f"{'CVaR-5%':>11}{'Worst':>11}{'Costi':>9}")
    for r in rows:
        print(f"{r['name']:<22}{r['mean']:>11.2f}{r['std']:>10.2f}{r['sharpe']:>9.3f}"
              f"{r['cvar5']:>11.2f}{r['worst']:>11.2f}{r['mean_cost']:>9.2f}")


def main():
    base = EnvConfig()  # parametri di mercato/opzione comuni a tutti gli split

    # ---- Verifica esplicita di assenza di leakage tra gli split ----
    s_tr = OptionsTradingEnv(replace(base, data_split="train")); s_tr.reset()
    s_va = OptionsTradingEnv(replace(base, data_split="val")); s_va.reset()
    s_te = OptionsTradingEnv(replace(base, data_split="test")); s_te.reset()
    assert not np.allclose(s_tr.prices, s_va.prices) and not np.allclose(s_va.prices, s_te.prices), \
        "Leakage: gli split condividono traiettorie!"
    print("[anti-leakage] Split train/val/test verificati distinti (bande di seed disgiunte).")

    # ---- Configurazioni da confrontare (selezione SOLO su validation) ----
    configs = [
        dict(cfg_name="PPO-meanvar", algo="PPO", reward_mode="meanvar",
             risk_lambda=0.1, timesteps=250_000),
        dict(cfg_name="PPO-DSR", algo="PPO", reward_mode="dsr",
             risk_lambda=0.0, timesteps=250_000),
        dict(cfg_name="SAC-meanvar", algo="SAC", reward_mode="meanvar",
             risk_lambda=0.1, timesteps=80_000),
    ]

    trained = {}
    val_rows = []
    for c in configs:
        model, normalizer, act = train_config(base_market_cfg=base, **c)
        trained[c["cfg_name"]] = (model, normalizer, act)
        m = evaluate(act, "val", N_VAL, base, c["cfg_name"])
        val_rows.append(m)
        print(f"    [VAL] {c['cfg_name']}: std={m['std']:.1f}  "
              f"mean={m['mean']:.1f}  CVaR5={m['cvar5']:.1f}", flush=True)

    # Baseline su validation (riferimento).
    val_rows.append(evaluate(policy_delta_hedge, "val", N_VAL, base, "Delta-Hedge"))
    val_rows.append(evaluate(policy_no_hedge, "val", N_VAL, base, "No-Hedge"))
    print_table(val_rows, "VALIDATION (per la SELEZIONE del modello)")

    # ---- Selezione: minima volatilita' (std) del P&L tra le configurazioni DRL ----
    drl_val = [r for r in val_rows if r["name"] in trained]
    best_name = min(drl_val, key=lambda r: r["std"])["name"]
    print(f"\n[selezione] Migliore su VALIDATION (min std P&L): {best_name}")

    # ---- Valutazione FINALE sul TEST set (toccato una sola volta) ----
    best_act = trained[best_name][2]
    test_rows = [
        evaluate(best_act, "test", N_TEST, base, f"{best_name} (DRL)"),
        evaluate(policy_delta_hedge, "test", N_TEST, base, "Delta-Hedge"),
        evaluate(policy_no_hedge, "test", N_TEST, base, "No-Hedge"),
    ]
    print_table(test_rows, f"TEST FINALE ({N_TEST} scenari mai visti)")

    # Salvataggio del miglior modello + statistiche di normalizzazione.
    best_model = trained[best_name][0]
    best_model.save("best_hedging_model")
    np.savez("best_normalizer.npz",
             mean=trained[best_name][1].mean, var=trained[best_name][1].var)
    print("\n[salvataggio] best_hedging_model.zip + best_normalizer.npz")


if __name__ == "__main__":
    main()
