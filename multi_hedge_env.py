#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AMBIENTE DI HEDGING MULTI-STRUMENTO (Delta / Gamma / Vega)
================================================================================

A differenza dell'ambiente base (copertura del solo Delta col sottostante), qui
l'agente copre un portafoglio con esposizione a Delta, Gamma e Vega usando DUE
strumenti:

    1. il SOTTOSTANTE        -> solo Delta (Gamma=Vega=0), costo basso;
    2. un'OPZIONE DI HEDGING -> Delta, Gamma, Vega propri, costo alto
                                (spread proporzionale al premio).

Sotto volatilita' stocastica (Heston) la passivita' ha rischio Vega reale.
Per neutralizzare Gamma o Vega bisogna usare l'opzione, ma questa e' costosa da
negoziare: il delta-gamma hedge classico la ribilancia ad OGNI passo (caro).
NON esiste una soluzione classica in forma chiusa che bilanci Greche-neutralita'
e costi -> e' lo scenario in cui l'RL puo' davvero aggiungere valore.

Si riusa il motore di drl_options_hedging.py (Black-Scholes, Heston, DSR) e lo
stesso protocollo anti data-leakage (bande di seed disgiunte per split).
================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from drl_options_hedging import (
    BlackScholes, MarketParams, simulate_heston, DifferentialSharpeRatio, EPS,
)


@dataclass
class MultiHedgeConfig:
    """Configurazione dell'ambiente di hedging multi-strumento."""
    market: MarketParams = field(default_factory=MarketParams)
    # --- Passivita' da coprire ---
    liab_type: str = "call"
    liab_qty: float = -100.0        # short 100 (posizione venduta)
    liab_moneyness: float = 1.0     # K_liab / S0
    maturity_days: int = 60
    # --- Opzione di hedging (stessa scadenza, strike diverso) ---
    hedge_type: str = "call"
    hedge_moneyness_mult: float = 1.05   # K_opt = K_liab * 1.05 (OTM)
    # --- Costi ---
    stock_commission: float = 5e-4   # 5 bps sul nozionale azionario
    stock_slippage: float = 5e-4     # 5 bps
    opt_spread: float = 0.02         # 2% del premio per contratto scambiato (bid-ask)
    opt_commission: float = 0.0
    # --- Scale delle azioni (in multipli di |liab_qty|) ---
    stock_scale: float = 1.5
    opt_scale: float = 2.5
    # --- Reward ---
    reward_mode: str = "meanvar"     # "dsr" | "meanvar" | "terminal_mv"
    risk_lambda: float = 0.1
    dsr_alpha: float = 0.05
    # --- Riproducibilita' / anti-leakage ---
    random_strike: bool = True
    data_split: Optional[str] = None
    split_offset: int = 0


class MultiHedgeEnv(gym.Env):
    """
    OBSERVATION SPACE (Box, 9 dim, normalizzato)
        [0] S / K_liab                 moneyness della passivita'
        [1] (T - t) / T0               tempo residuo in [0,1]
        [2] sigma                      volatilita' istantanea (proxy IV)
        [3] pos_stock / |liab_qty|     posizione azionaria corrente
        [4] pos_opt   / |liab_qty|     posizione opzione di hedge corrente
        [5] Delta_tot / |liab_qty|     Delta di portafoglio
        [6] Gamma_tot * S / |liab_qty| "dollar gamma" di portafoglio
        [7] Vega_tot / (|liab_qty|*VN) Vega di portafoglio normalizzato
        [8] S / K_opt                  moneyness dello strumento di hedge

    ACTION SPACE (Box, 2 dim, in [-1,1])
        a[0] -> posizione obiettivo sul sottostante  = a0 * stock_scale*|liab_qty|
        a[1] -> posizione obiettivo sull'opzione hedge = a1 * opt_scale*|liab_qty|
    """
    metadata = {"render_modes": []}
    SPLIT_BASE = {"train": 0, "val": 1_000_000_000, "test": 2_000_000_000}
    VEGA_NORM = 20.0  # scala tipica del vega per contratto ATM ~60g

    def __init__(self, config: Optional[MultiHedgeConfig] = None):
        super().__init__()
        self.cfg = config if config is not None else MultiHedgeConfig()
        self.market = self.cfg.market
        self._ep_idx = 0
        self.n_steps = int(self.cfg.maturity_days)
        self.T0 = self.cfg.maturity_days * self.market.dt

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_low = np.array([0.0, 0.0, 0.0, -3.0, -3.0, -3.0, -3.0, -3.0, 0.0], dtype=np.float32)
        obs_high = np.array([5.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.dsr = DifferentialSharpeRatio(alpha=self.cfg.dsr_alpha)
        self.reward_mode = self.cfg.reward_mode

        # Comodita' per l'inversione delle azioni nelle baseline.
        self.q_abs = abs(self.cfg.liab_qty) + EPS
        self.stock_scale_abs = self.cfg.stock_scale * abs(self.cfg.liab_qty)
        self.opt_scale_abs = self.cfg.opt_scale * abs(self.cfg.liab_qty)

        # Stato (popolato in reset()).
        self.prices = np.array([]); self.vols = np.array([])
        self.K_liab = self.market.S0; self.K_opt = self.market.S0
        self.liab_qty = self.cfg.liab_qty
        self.pos_stock = 0.0; self.pos_opt = 0.0
        self.t = 0; self.wealth = 0.0
        self._prev_liab_price = 0.0; self._prev_opt_price = 0.0

    # ------------------------------------------------------------------ greche
    def _greeks_at(self, t: int) -> Tuple[Dict, Dict]:
        S = self.prices[t]; sigma = self.vols[t]; T = self.T0 - t * self.market.dt
        g_liab = BlackScholes.price_and_greeks(S, self.K_liab, T, self.market.r,
                                               sigma, self.cfg.liab_type)
        g_opt = BlackScholes.price_and_greeks(S, self.K_opt, T, self.market.r,
                                              sigma, self.cfg.hedge_type)
        return g_liab, g_opt

    def current_greeks(self) -> Tuple[Dict, Dict]:
        """Greche (passivita', opzione hedge) nello stato corrente (per le baseline)."""
        return self._greeks_at(self.t)

    def _portfolio_greeks(self, g_liab: Dict, g_opt: Dict) -> Tuple[float, float, float]:
        delta = self.liab_qty * g_liab["delta"] + self.pos_stock + self.pos_opt * g_opt["delta"]
        gamma = self.liab_qty * g_liab["gamma"] + self.pos_opt * g_opt["gamma"]
        vega = self.liab_qty * g_liab["vega"] + self.pos_opt * g_opt["vega"]
        return delta, gamma, vega

    def _get_obs(self, g_liab: Dict, g_opt: Dict) -> np.ndarray:
        S = self.prices[self.t]; T = self.T0 - self.t * self.market.dt
        delta, gamma, vega = self._portfolio_greeks(g_liab, g_opt)
        obs = np.array([
            S / self.K_liab,
            T / (self.T0 + EPS),
            self.vols[self.t],
            self.pos_stock / self.q_abs,
            self.pos_opt / self.q_abs,
            delta / self.q_abs,
            (gamma * S) / self.q_abs,
            vega / (self.q_abs * self.VEGA_NORM),
            S / self.K_opt,
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _compute_reward(self, pnl: float, w_prev: float, terminated: bool) -> float:
        if self.reward_mode == "meanvar":
            return float(pnl - self.cfg.risk_lambda * (pnl ** 2))
        if self.reward_mode == "terminal_mv":
            lam = self.cfg.risk_lambda
            return float(pnl - lam * (2.0 * w_prev * pnl + pnl ** 2))
        return self.dsr.step(pnl)

    def set_episode_index(self, idx: int) -> None:
        self._ep_idx = idx

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.cfg.data_split is not None:
            base = self.SPLIT_BASE[self.cfg.data_split]
            rng = np.random.default_rng(base + self.cfg.split_offset + self._ep_idx)
            self._ep_idx += 1
        else:
            rng = np.random.default_rng(self.np_random.integers(0, 2 ** 31 - 1))

        self.prices, self.vols = simulate_heston(self.market, self.n_steps, rng)

        K_liab = self.market.S0 * self.cfg.liab_moneyness
        if self.cfg.random_strike:
            K_liab *= float(rng.uniform(0.95, 1.05))
        self.K_liab = K_liab
        self.K_opt = K_liab * self.cfg.hedge_moneyness_mult
        self.liab_qty = self.cfg.liab_qty

        self.pos_stock = 0.0; self.pos_opt = 0.0
        self.t = 0; self.wealth = 0.0
        self.dsr.reset()

        g_liab, g_opt = self._greeks_at(0)
        self._prev_liab_price = g_liab["price"]
        self._prev_opt_price = g_opt["price"]
        return self._get_obs(g_liab, g_opt), {"K_liab": self.K_liab, "K_opt": self.K_opt}

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        S_now = self.prices[self.t]

        # --- 1) Ribilanciamento dei due strumenti (costi) ---
        target_stock = a[0] * self.stock_scale_abs
        target_opt = a[1] * self.opt_scale_abs
        trade_stock = target_stock - self.pos_stock
        trade_opt = target_opt - self.pos_opt

        cost_stock = (self.cfg.stock_commission + self.cfg.stock_slippage) * abs(trade_stock) * S_now
        cost_opt = (self.cfg.opt_spread * abs(trade_opt) * self._prev_opt_price
                    + self.cfg.opt_commission * abs(trade_opt))
        cost = cost_stock + cost_opt

        self.pos_stock = target_stock
        self.pos_opt = target_opt

        # --- 2) Avanzamento del mercato ---
        liab_price_now = self._prev_liab_price
        opt_price_now = self._prev_opt_price
        self.t += 1
        S_next = self.prices[self.t]
        g_liab_next, g_opt_next = self._greeks_at(self.t)
        liab_price_next = g_liab_next["price"]
        opt_price_next = g_opt_next["price"]

        # --- 3) P&L di periodo (mark-to-market di tutte le posizioni) ---
        pnl = (self.liab_qty * (liab_price_next - liab_price_now)
               + self.pos_stock * (S_next - S_now)
               + self.pos_opt * (opt_price_next - opt_price_now)
               - cost)

        w_prev = self.wealth
        self.wealth += pnl
        self._prev_liab_price = liab_price_next
        self._prev_opt_price = opt_price_next

        terminated = self.t >= self.n_steps
        truncated = False
        reward = self._compute_reward(pnl, w_prev, terminated)

        delta, gamma, vega = self._portfolio_greeks(g_liab_next, g_opt_next)
        info = {
            "pnl": pnl, "wealth": self.wealth, "transaction_cost": cost,
            "cost_stock": cost_stock, "cost_opt": cost_opt,
            "trade_stock": trade_stock, "trade_opt": trade_opt,
            "pos_stock": self.pos_stock, "pos_opt": self.pos_opt,
            "portfolio_delta": delta, "portfolio_gamma": gamma, "portfolio_vega": vega,
        }
        return self._get_obs(g_liab_next, g_opt_next), float(reward), terminated, truncated, info


# ==============================================================================
# Politiche classiche di riferimento (baseline)
# ==============================================================================
def _to_action(env: MultiHedgeEnv, target_stock: float, target_opt: float) -> np.ndarray:
    """Converte posizioni-obiettivo assolute in azione normalizzata [-1,1]^2."""
    a0 = target_stock / (env.stock_scale_abs + EPS)
    a1 = target_opt / (env.opt_scale_abs + EPS)
    return np.clip(np.array([a0, a1], dtype=np.float32), -1.0, 1.0)


def policy_no_hedge(obs, env: MultiHedgeEnv) -> np.ndarray:
    return np.array([0.0, 0.0], dtype=np.float32)


def policy_delta_stock(obs, env: MultiHedgeEnv) -> np.ndarray:
    """Solo Delta col sottostante (ignora Gamma/Vega)."""
    g_liab, _ = env.current_greeks()
    target_stock = -env.liab_qty * g_liab["delta"]
    return _to_action(env, target_stock, 0.0)


def policy_delta_gamma(obs, env: MultiHedgeEnv) -> np.ndarray:
    """Delta-Gamma neutro: opzione per il Gamma, sottostante per il Delta residuo."""
    g_liab, g_opt = env.current_greeks()
    target_opt = -(env.liab_qty * g_liab["gamma"]) / (g_opt["gamma"] + EPS)
    target_stock = -(env.liab_qty * g_liab["delta"] + target_opt * g_opt["delta"])
    return _to_action(env, target_stock, target_opt)


def policy_delta_vega(obs, env: MultiHedgeEnv) -> np.ndarray:
    """Delta-Vega neutro: opzione per il Vega, sottostante per il Delta residuo."""
    g_liab, g_opt = env.current_greeks()
    target_opt = -(env.liab_qty * g_liab["vega"]) / (g_opt["vega"] + EPS)
    target_stock = -(env.liab_qty * g_liab["delta"] + target_opt * g_opt["delta"])
    return _to_action(env, target_stock, target_opt)


def make_gamma_band_policy(band: float):
    """Avversario forte: ribilancia l'opzione (cara) per il Gamma SOLO se il
    Gamma residuo supera la soglia; il sottostante (economico) azzera sempre il
    Delta dato il pos_opt corrente."""
    def policy(obs, env: MultiHedgeEnv) -> np.ndarray:
        g_liab, g_opt = env.current_greeks()
        resid_gamma = env.liab_qty * g_liab["gamma"] + env.pos_opt * g_opt["gamma"]
        if abs(resid_gamma) > band:
            target_opt = -(env.liab_qty * g_liab["gamma"]) / (g_opt["gamma"] + EPS)
        else:
            target_opt = env.pos_opt
        target_stock = -(env.liab_qty * g_liab["delta"] + target_opt * g_opt["delta"])
        return _to_action(env, target_stock, target_opt)
    return policy


# ==============================================================================
# Self-test rapido
# ==============================================================================
if __name__ == "__main__":
    cfg = MultiHedgeConfig(data_split="train")
    env = MultiHedgeEnv(cfg); env.set_episode_index(0)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), "obs fuori spazio"

    # Verifica che il delta-gamma hedge azzeri Delta e Gamma di portafoglio
    # NELL'ISTANTE della copertura (t), prima che il mercato avanzi.
    g_liab, g_opt = env.current_greeks()
    a = policy_delta_gamma(obs, env)
    env.pos_stock = a[0] * env.stock_scale_abs
    env.pos_opt = a[1] * env.opt_scale_abs
    d0, gm0, vg0 = env._portfolio_greeks(g_liab, g_opt)
    print(f"[self-test] delta-gamma hedge a t: Delta={d0:.4f} Gamma={gm0:.5f} (attesi ~0)")
    assert abs(d0) < 1e-3 and abs(gm0) < 1e-4, "delta-gamma hedge non neutralizza le greche"
    env.pos_stock = 0.0; env.pos_opt = 0.0  # ripristina prima del rollout

    # Un episodio completo con ciascuna baseline.
    for name, pol in [("no-hedge", policy_no_hedge), ("delta", policy_delta_stock),
                      ("delta-gamma", policy_delta_gamma), ("delta-vega", policy_delta_vega),
                      ("gamma-band(2)", make_gamma_band_policy(2.0))]:
        env.set_episode_index(0); obs, _ = env.reset()
        done = False; pnl = 0.0; cost = 0.0
        while not done:
            obs, r, term, trunc, info = env.step(pol(obs, env))
            done = term or trunc; pnl += info["pnl"]; cost += info["transaction_cost"]
        print(f"[self-test] {name:14}: P&L={pnl:8.2f}  costi={cost:7.2f}")
    print("=== SELF-TEST MULTI-HEDGE SUPERATO ===")
