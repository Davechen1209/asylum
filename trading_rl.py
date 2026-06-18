#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TRADING_RL - Agente di trading direzionale con Reinforcement Learning
================================================================================

Ambiente gym su feature di prezzo CAUSALI (solo passato -> niente look-ahead).
Azione = posizione obiettivo continua in [-1, 1]. Reward = pos * rendimento_t+1
- costi di turnover. Split temporale train/test (l'agente si addestra solo sul
passato; la valutazione e' sul futuro mai visto). Normalizzazione osservazioni
congelata sul training (come negli altri esperimenti, niente leakage).

Onesta': dai 4 esperimenti di hedging sappiamo che l'RL raramente batte metodi
classici ben tarati. Qui l'RL e' uno dei contendenti, non una promessa.
================================================================================
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ------------------------------------------------------------------------------
# Feature engineering CAUSALE (ogni feature al giorno t usa solo dati <= t)
# ------------------------------------------------------------------------------
def make_features(px: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    c = px["close"]
    ret = c.pct_change().fillna(0.0)
    feat = pd.DataFrame(index=px.index)
    feat["r1"] = ret
    feat["r5"] = c.pct_change(5)
    feat["mom20"] = c.pct_change(20)
    feat["mom60"] = c.pct_change(60)
    feat["mom120"] = c.pct_change(120)
    feat["vol20"] = ret.rolling(20).std() * np.sqrt(252)
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    feat["z20"] = (c - ma20) / (sd20 + 1e-9)
    # RSI(14)
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    feat["rsi"] = (100 - 100 / (1 + up / (dn + 1e-9))) / 100.0 - 0.5
    feat = feat.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feat, ret


class TradingEnv(gym.Env):
    """
    Stato:   feature causali al giorno t + posizione corrente.
    Azione:  Box([-1,1]) -> posizione obiettivo (frazione di capitale, long/short).
    Reward:  pos * ret[t+1] - cost_bps*|delta_pos| (P&L netto del periodo).
    """
    metadata = {"render_modes": []}

    def __init__(self, feat: np.ndarray, ret: np.ndarray, cost_bps: float = 2.0,
                 turnover_penalty: float = 0.0):
        super().__init__()
        self.feat = feat.astype(np.float32)
        self.ret = ret.astype(np.float32)
        self.cost = cost_bps / 1e4
        self.turnover_penalty = turnover_penalty
        self.n = len(ret)
        self.nf = feat.shape[1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        # osservazione: feature + posizione corrente
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.nf + 1,), dtype=np.float32)
        self.t = 0
        self.pos = 0.0

    def _obs(self):
        return np.concatenate([self.feat[self.t], [np.float32(self.pos)]]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = 0
        self.pos = 0.0
        return self._obs(), {}

    def step(self, action):
        new_pos = float(np.clip(action[0], -1.0, 1.0))
        r_next = self.ret[self.t + 1]
        turn = abs(new_pos - self.pos)
        pnl = new_pos * r_next - self.cost * turn
        reward = pnl - self.turnover_penalty * turn
        self.pos = new_pos
        self.t += 1
        terminated = self.t >= self.n - 1
        info = {"pnl": pnl, "pos": new_pos, "turnover": turn, "mkt_ret": r_next}
        return self._obs(), float(reward), terminated, False, info


# ------------------------------------------------------------------------------
# Addestramento e valutazione (normalizzazione congelata sul training)
# ------------------------------------------------------------------------------
def train_rl(feat_tr: np.ndarray, ret_tr: np.ndarray, algo: str = "PPO",
             timesteps: int = 200_000, cost_bps: float = 2.0,
             turnover_penalty: float = 0.0, seed: int = 0):
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    def mk():
        return Monitor(TradingEnv(feat_tr, ret_tr, cost_bps, turnover_penalty))
    venv = DummyVecEnv([mk])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    pk = dict(net_arch=[128, 128])
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

    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    print(f"    RL {algo} addestrato in {time.time()-t0:.0f}s", flush=True)

    mean = venv.obs_rms.mean.copy(); var = venv.obs_rms.var.copy()
    eps = float(venv.epsilon); clip = float(venv.clip_obs)

    def norm(o):
        return np.clip((o - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
    return model, norm


def eval_rl_positions(model, norm, feat: np.ndarray, ret: np.ndarray,
                      cost_bps: float = 2.0) -> pd.Series:
    """Genera la serie di posizioni dell'agente sul periodo (deterministico)."""
    env = TradingEnv(feat, ret, cost_bps)
    obs, _ = env.reset()
    positions = np.zeros(len(ret), dtype=np.float64)
    done = False
    while not done:
        a, _ = model.predict(norm(obs), deterministic=True)
        t_now = env.t
        obs, r, term, trunc, info = env.step(a)
        positions[t_now] = info["pos"]  # posizione decisa al giorno t_now
        done = term or trunc
    return pd.Series(positions, name="rl_pos")
