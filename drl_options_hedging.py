#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SISTEMA DI HEDGING DI OPZIONI BASATO SU DEEP REINFORCEMENT LEARNING
================================================================================

Sistema completo per la gestione e la copertura (hedging) di un portafoglio di
opzioni Europee tramite un agente di Deep Reinforcement Learning.

Componenti principali (vedi sezioni numerate nel file):
    1. FORMULE DI BLACK-SCHOLES-MERTON
       Prezzo analitico di Call/Put Europee e calcolo analitico delle Greche
       (Delta, Gamma, Vega, Theta).

    2. LOGICA DEL DIFFERENTIAL SHARPE RATIO (DSR)
       Formulazione ricorsiva di Moody & Saffell usata come funzione di reward.

    3. SIMULATORE DI MERCATO SINTETICO
       Moto Browniano Geometrico con volatilita' stocastica (modello di Heston).

    4. AMBIENTE GYMNASIUM (`OptionsTradingEnv`)
       Ambiente custom con spazio degli stati continuo (prezzo, scadenza, vol,
       posizione, Greche di portafoglio) e spazio delle azioni continuo
       (Delta-hedging sul sottostante) con costi di transazione.

    5. PIPELINE DI TRAINING E TESTING
       Addestramento con Stable-Baselines3 (PPO / SAC) e backtest comparativo
       contro le strategie baseline (Delta-hedging classico e nessuna copertura).

Esecuzione:
    python drl_options_hedging.py --algo PPO --timesteps 50000

Autore: Quant Research Team
Dipendenze: numpy, scipy, gymnasium, stable-baselines3, torch, matplotlib
================================================================================
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

# Import "soft" delle dipendenze pesanti: lo script resta utilizzabile (almeno
# per la dimostrazione dell'ambiente) anche se Stable-Baselines3 non e' presente.
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - dipendenza fondamentale
    raise ImportError(
        "E' necessario installare 'gymnasium' (pip install gymnasium)."
    ) from exc

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor

    SB3_AVAILABLE = True
except ImportError:  # pragma: no cover - gestito a runtime
    SB3_AVAILABLE = False

# Costante numerica per stabilizzare le divisioni (es. T->0, sigma->0, var->0).
EPS = 1e-8


# ==============================================================================
# 1. FORMULE DI BLACK-SCHOLES-MERTON
# ==============================================================================
class BlackScholes:
    """
    Modello analitico di Black-Scholes-Merton per opzioni Europee.

    Tutte le funzioni gestiscono i casi limite matematici:
        * tempo alla scadenza T -> 0  (l'opzione vale il valore intrinseco);
        * volatilita' sigma -> 0      (degenerazione del modello).

    Convenzioni:
        S     : prezzo spot del sottostante
        K     : strike (prezzo di esercizio)
        T     : tempo alla scadenza in anni (T - t)
        r     : tasso risk-free annuo (composto continuo)
        sigma : volatilita' annualizzata
        q     : dividend yield continuo (default 0)
    """

    @staticmethod
    def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> Tuple[float, float]:
        """Calcola i termini ausiliari d1 e d2 con protezione anti divisione-per-zero."""
        # Floor su T e sigma: evita sia la divisione per zero sia il log mal posto.
        T = max(float(T), EPS)
        sigma = max(float(sigma), EPS)
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    @staticmethod
    def _intrinsic_greeks(S: float, K: float, option_type: str) -> Dict[str, float]:
        """Valore intrinseco e Greche degeneri usate alla scadenza (T <= 0)."""
        is_call = option_type.lower() == "call"
        if is_call:
            price = max(S - K, 0.0)
            delta = 1.0 if S > K else (0.5 if S == K else 0.0)
        else:
            price = max(K - S, 0.0)
            delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
        # Alla scadenza Gamma, Vega e Theta non sono piu' definite: sono nulle.
        return {"price": price, "delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    @classmethod
    def price_and_greeks(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Dict[str, float]:
        """
        Restituisce prezzo e Greche analitiche in un unico dizionario.

        Greche restituite:
            delta : sensibilita' del prezzo rispetto a S            (dV/dS)
            gamma : sensibilita' del Delta rispetto a S             (d^2V/dS^2)
            vega  : sensibilita' rispetto alla volatilita' (per 1.0, cioe' 100%)
            theta : decadimento temporale ANNUO (per dividere per 365 -> giornaliero)
        """
        # Caso limite: opzione scaduta -> valore intrinseco e Greche degeneri.
        if T <= EPS:
            return cls._intrinsic_greeks(S, K, option_type)

        is_call = option_type.lower() == "call"
        d1, d2 = cls._d1_d2(S, K, T, r, sigma, q)
        sigma = max(float(sigma), EPS)
        T = max(float(T), EPS)
        sqrt_T = np.sqrt(T)

        pdf_d1 = norm.pdf(d1)          # densita' normale standard in d1
        disc_r = np.exp(-r * T)        # fattore di sconto risk-free
        disc_q = np.exp(-q * T)        # fattore di sconto dei dividendi

        # ----- Gamma e Vega sono identiche per Call e Put -----
        gamma = disc_q * pdf_d1 / (S * sigma * sqrt_T)
        vega = S * disc_q * pdf_d1 * sqrt_T

        if is_call:
            price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
            delta = disc_q * norm.cdf(d1)
            theta = (
                -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
                - r * K * disc_r * norm.cdf(d2)
                + q * S * disc_q * norm.cdf(d1)
            )
        else:  # put
            price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
            delta = -disc_q * norm.cdf(-d1)
            theta = (
                -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
                + r * K * disc_r * norm.cdf(-d2)
                - q * S * disc_q * norm.cdf(-d1)
            )

        return {
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
        }

    # --- Funzioni di comodita' (wrapper) ------------------------------------
    @classmethod
    def call_price(cls, S, K, T, r, sigma, q=0.0) -> float:
        return cls.price_and_greeks(S, K, T, r, sigma, "call", q)["price"]

    @classmethod
    def put_price(cls, S, K, T, r, sigma, q=0.0) -> float:
        return cls.price_and_greeks(S, K, T, r, sigma, "put", q)["price"]


# ==============================================================================
# 2. LOGICA DEL DIFFERENTIAL SHARPE RATIO (DSR)
# ==============================================================================
class DifferentialSharpeRatio:
    """
    Implementazione della formulazione ricorsiva del Differential Sharpe Ratio
    (Moody & Saffell, 1998), utilizzata come funzione di reward incrementale.

    Idea: invece di calcolare lo Sharpe Ratio su una finestra mobile, si misura
    il *contributo marginale* dell'ultimo rendimento R_t allo Sharpe Ratio
    esponenziale. Questo fornisce un segnale di reward denso, online e causale.

    Variabili di stato esponenziali (decadimento alpha):
        A_t = A_{t-1} + alpha * (R_t      - A_{t-1})   # media dei rendimenti  (== eta_t)
        B_t = B_{t-1} + alpha * (R_t^2    - B_{t-1})   # media del 2^ momento
    La varianza esponenziale e' Delta_t = B_t - A_t^2.

    Reward (DSR) al passo t, derivata dello Sharpe rispetto ad alpha (alpha->0):

                     B_{t-1} * dA_t - 0.5 * A_{t-1} * dB_t
        D_t  =  -------------------------------------------------
                          (B_{t-1} - A_{t-1}^2)^(3/2)

        con   dA_t = R_t - A_{t-1}   e   dB_t = R_t^2 - B_{t-1}.

    Il denominatore (varianza^(3/2)) e' protetto da un termine epsilon per
    gestire i casi limite (varianza nulla o numericamente negativa).
    """

    def __init__(self, alpha: float = 0.05, eps: float = 1e-6):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha deve essere in (0, 1].")
        self.alpha = alpha   # fattore di decadimento (adaptation rate)
        self.eps = eps       # stabilizzatore numerico del denominatore
        self.reset()

    def reset(self) -> None:
        """Reinizializza le statistiche esponenziali (inizio di un nuovo episodio)."""
        self.A = 0.0          # eta_t : media esponenziale dei rendimenti
        self.B = 0.0          # B_t   : media esponenziale del rendimento al quadrato
        self._warmed_up = False

    @property
    def variance(self) -> float:
        """Varianza esponenziale Delta_t = B_t - A_t^2 (sempre >= 0)."""
        return max(self.B - self.A ** 2, 0.0)

    @property
    def sharpe(self) -> float:
        """Sharpe Ratio esponenziale corrente (per monitoraggio/diagnostica)."""
        return self.A / np.sqrt(self.variance + self.eps)

    def step(self, R: float) -> float:
        """
        Aggiorna le statistiche con il nuovo rendimento R e restituisce il DSR.

        Al primissimo passo non esiste ancora uno stato (A_{t-1}, B_{t-1})
        significativo: si effettua il bootstrap e si ritorna reward nullo.
        """
        R = float(R)

        # --- Warm-up: bootstrap delle statistiche al primo rendimento ---
        if not self._warmed_up:
            self.A = self.alpha * R
            self.B = self.alpha * (R ** 2)
            self._warmed_up = True
            return 0.0

        # Stato precedente (t-1) usato nella formula differenziale.
        A_prev, B_prev = self.A, self.B
        dA = R - A_prev
        dB = R ** 2 - B_prev

        # Denominatore = (varianza)^(3/2), stabilizzato con epsilon.
        var_prev = max(B_prev - A_prev ** 2, 0.0)
        denom = (var_prev + self.eps) ** 1.5

        dsr = (B_prev * dA - 0.5 * A_prev * dB) / denom

        # Aggiornamento ricorsivo delle medie mobili esponenziali.
        self.A = A_prev + self.alpha * dA
        self.B = B_prev + self.alpha * dB

        # Clip difensivo: in regimi a varianza quasi nulla il DSR puo' esplodere.
        return float(np.clip(dsr, -10.0, 10.0))


# ==============================================================================
# 3. SIMULATORE DI MERCATO SINTETICO (GBM + VOLATILITA' STOCASTICA DI HESTON)
# ==============================================================================
@dataclass
class MarketParams:
    """Parametri del processo di mercato (modello di Heston)."""
    S0: float = 100.0       # prezzo iniziale del sottostante
    mu: float = 0.05        # drift annuo (rendimento atteso)
    r: float = 0.01         # tasso risk-free annuo
    v0: float = 0.04        # varianza istantanea iniziale (vol iniziale = 20%)
    kappa: float = 2.0      # velocita' di mean-reversion della varianza
    theta: float = 0.04     # varianza di lungo periodo
    xi: float = 0.3         # vol-of-vol (volatilita' della varianza)
    rho: float = -0.7       # correlazione tra shock di prezzo e di varianza
    dt: float = 1.0 / 252.0  # passo temporale (1 giorno di trading)


def simulate_heston(
    params: MarketParams, n_steps: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula una traiettoria di prezzo con volatilita' stocastica (Heston),
    usando lo schema di Eulero "full truncation" per garantire varianza >= 0.

    Ritorna:
        prices : array (n_steps + 1,) dei prezzi del sottostante
        vols   : array (n_steps + 1,) della volatilita' istantanea sqrt(v_t),
                 usata come proxy della volatilita' implicita per il pricing.
    """
    dt = params.dt
    prices = np.empty(n_steps + 1, dtype=np.float64)
    variances = np.empty(n_steps + 1, dtype=np.float64)
    prices[0] = params.S0
    variances[0] = params.v0

    sqrt_dt = np.sqrt(dt)
    for i in range(n_steps):
        v_pos = max(variances[i], 0.0)  # full truncation
        # Due shock gaussiani correlati (correlazione rho).
        z1 = rng.standard_normal()
        z_ind = rng.standard_normal()
        z2 = params.rho * z1 + np.sqrt(1.0 - params.rho ** 2) * z_ind

        # Evoluzione del prezzo (GBM con varianza istantanea v_pos).
        prices[i + 1] = prices[i] * np.exp(
            (params.mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z1
        )
        # Evoluzione della varianza (processo CIR di Heston, full truncation).
        variances[i + 1] = (
            variances[i] + params.kappa * (params.theta - v_pos) * dt
            + params.xi * np.sqrt(v_pos) * sqrt_dt * z2
        )

    vols = np.sqrt(np.maximum(variances, EPS))
    return prices, vols


# ==============================================================================
# 4. AMBIENTE GYMNASIUM: OptionsTradingEnv
# ==============================================================================
@dataclass
class EnvConfig:
    """Configurazione dell'ambiente di trading/hedging."""
    market: MarketParams = field(default_factory=MarketParams)
    option_type: str = "call"      # tipo di opzione nel portafoglio
    option_qty: float = -100.0     # quantita' (negativo = posizione corta venduta)
    maturity_days: int = 60        # giorni alla scadenza all'inizio dell'episodio
    moneyness: float = 1.0         # K / S0 iniziale (1.0 = At-The-Money)
    commission_rate: float = 5e-4  # commissione proporzionale (5 bps sul nozionale)
    slippage_rate: float = 5e-4    # slippage proporzionale (5 bps sul nozionale)
    dsr_alpha: float = 0.05        # fattore di decadimento del DSR
    random_strike: bool = True     # se True randomizza lievemente strike e maturita'
    # --- Reward ---
    # "dsr" | "meanvar" | "terminal_mv" (Deep Hedging, media-varianza terminale) | "entropic"
    reward_mode: str = "dsr"
    risk_lambda: float = 0.02      # peso della penalita' di varianza ("meanvar"/"terminal_mv")
    risk_aversion: float = 1.0     # coefficiente di avversione al rischio ("entropic")
    wealth_scale: float = 100.0    # scala del P&L terminale ("entropic"), stabilita' numerica
    # --- Separazione dei dati (anti data-leakage) ---
    # Bande di seed disgiunte per train/val/test: nessuna traiettoria condivisa.
    data_split: Optional[str] = None  # None -> casuale; "train"/"val"/"test" -> deterministico
    split_offset: int = 0             # offset per dare path distinti a env paralleli sullo stesso split


class OptionsTradingEnv(gym.Env):
    """
    Ambiente di Delta-hedging di un portafoglio di opzioni.

    SCENARIO
    --------
    L'agente ha in portafoglio una posizione (tipicamente CORTA) su opzioni
    Europee -- ad esempio ha venduto 100 Call -- e deve coprirne il rischio
    direzionale negoziando il sottostante. L'obiettivo e' massimizzare la
    stabilita' del P&L (misurata dal Differential Sharpe Ratio), tenendo conto
    dei costi di transazione.

    OBSERVATION SPACE (Box continuo, 6 dimensioni, valori normalizzati)
    -------------------------------------------------------------------
        [0] moneyness            : S / K
        [1] tempo alla scadenza  : (T - t) / T0          in [0, 1]
        [2] volatilita' implicita: sigma
        [3] posizione corrente   : hedge / |option_qty|
        [4] Delta di portafoglio : Delta_tot / |option_qty|
        [5] Gamma di portafoglio : (Gamma_tot * S) / |option_qty|

    ACTION SPACE (Box continuo, 1 dimensione)
    -----------------------------------------
        a in [-1, 1] -> posizione obiettivo sul sottostante:
            hedge_target = a * |option_qty|
        L'azione rappresenta quindi quante unita' di sottostante detenere per
        coprire/aggiustare il Delta del portafoglio.

    REWARD
    ------
        Differential Sharpe Ratio del P&L di periodo (vedi sezione 2).
    """

    metadata = {"render_modes": []}

    # Basi di seed DISGIUNTE per i tre split: garantiscono che le traiettorie di
    # mercato viste in training non compaiano MAI in validation o test.
    SPLIT_BASE = {"train": 0, "val": 1_000_000_000, "test": 2_000_000_000}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = config if config is not None else EnvConfig()
        self.market = self.cfg.market

        # Contatore di episodi: in modalita' split rende le traiettorie deterministiche
        # e riproducibili (path = f(split, indice episodio)).
        self._ep_idx = 0

        # Numero di passi per episodio = giorni alla scadenza.
        self.n_steps = int(self.cfg.maturity_days)
        self.T0 = self.cfg.maturity_days * self.market.dt  # scadenza iniziale (anni)

        # --- Definizione degli spazi (Gymnasium) ---
        # Azione continua: posizione obiettivo normalizzata sul sottostante.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Osservazione continua: limiti ampi (l'osservazione viene comunque clippata).
        obs_low = np.array([0.0, 0.0, 0.0, -2.0, -3.0, -3.0], dtype=np.float32)
        obs_high = np.array([5.0, 1.0, 3.0, 2.0, 3.0, 3.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Calcolatore del reward (DSR).
        self.dsr = DifferentialSharpeRatio(alpha=self.cfg.dsr_alpha)
        self.reward_mode = self.cfg.reward_mode

        # Stato interno (popolato in reset()).
        self.prices: np.ndarray = np.array([])
        self.vols: np.ndarray = np.array([])
        self.K: float = self.market.S0
        self.option_qty: float = self.cfg.option_qty
        self.hedge: float = 0.0
        self.t: int = 0
        self.wealth: float = 0.0
        self._prev_option_value: float = 0.0

    # ------------------------------------------------------------------ utils
    def _bs(self, S: float, T: float, sigma: float) -> Dict[str, float]:
        """Prezzo e Greche dell'opzione singola tramite Black-Scholes."""
        return BlackScholes.price_and_greeks(
            S=S, K=self.K, T=T, r=self.market.r, sigma=sigma, option_type=self.cfg.option_type
        )

    def current_option_greeks(self) -> Dict[str, float]:
        """Greche dell'opzione nello stato corrente (utile per le baseline)."""
        T = self.T0 - self.t * self.market.dt
        return self._bs(self.prices[self.t], T, self.vols[self.t])

    def _portfolio_greeks(self, greeks: Dict[str, float]) -> Tuple[float, float]:
        """Delta e Gamma TOTALI di portafoglio (opzioni + sottostante)."""
        # Il sottostante ha Delta = 1 e Gamma = 0 per unita'.
        port_delta = self.option_qty * greeks["delta"] + self.hedge
        port_gamma = self.option_qty * greeks["gamma"]
        return port_delta, port_gamma

    def _get_obs(self, greeks: Dict[str, float]) -> np.ndarray:
        """Costruisce il vettore di osservazione normalizzato e lo clippa nei limiti."""
        S = self.prices[self.t]
        T = self.T0 - self.t * self.market.dt
        sigma = self.vols[self.t]
        port_delta, port_gamma = self._portfolio_greeks(greeks)
        q_abs = abs(self.option_qty) + EPS

        obs = np.array(
            [
                S / self.K,                       # moneyness
                T / (self.T0 + EPS),              # tempo residuo normalizzato
                sigma,                            # volatilita' implicita
                self.hedge / q_abs,              # posizione corrente normalizzata
                port_delta / q_abs,              # Delta totale normalizzato
                (port_gamma * S) / q_abs,        # Gamma totale (in "dollar gamma")
            ],
            dtype=np.float32,
        )
        # Clip difensivo per restare dentro l'observation space.
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _transaction_cost(self, trade: float, S: float) -> float:
        """Costo di transazione = (commissione + slippage) sul nozionale scambiato."""
        notional = abs(trade) * S
        return (self.cfg.commission_rate + self.cfg.slippage_rate) * notional

    def set_episode_index(self, idx: int) -> None:
        """Imposta il contatore di episodi (per valutazioni deterministiche e ripetibili)."""
        self._ep_idx = idx

    def _compute_reward(self, pnl: float, w_prev: float = 0.0, terminated: bool = False) -> float:
        """
        Calcola il reward secondo la modalita' configurata:

            'dsr'         -> Differential Sharpe Ratio (specifica originale).

            'meanvar'     -> P&L penalizzato per la varianza PER-PASSO
                             (R = pnl - lambda*pnl^2). Spinge al tracking stretto
                             del Delta -> alto turnover (in conflitto coi costi).

            'terminal_mv' -> Media-varianza sul P&L TERMINALE, in forma DENSA via
                             telescoping. Poiche' W_T^2 = sum_t (2*W_{t-1}*pnl_t + pnl_t^2),
                             la reward densa
                                 R_t = pnl_t - lambda*(2*W_{t-1}*pnl_t + pnl_t^2)
                             ha somma  W_T - lambda*W_T^2, ovvero in attesa
                                 E[W_T] - lambda*E[W_T^2] (~ media-varianza terminale).
                             Permette al Delta di "driftare" -> fa emergere la
                             no-trade band (approccio stile Deep Hedging).

            'entropic'    -> Utilita' esponenziale sul P&L terminale (sparsa):
                             R_T = -exp(-a * W_T / scale), R_{t<T} = 0.

        Il P&L grezzo resta sempre in info['pnl'] per metriche imparziali.
        """
        mode = self.reward_mode
        if mode == "meanvar":
            return float(pnl - self.cfg.risk_lambda * (pnl ** 2))
        if mode == "terminal_mv":
            lam = self.cfg.risk_lambda
            return float(pnl - lam * (2.0 * w_prev * pnl + pnl ** 2))
        if mode == "entropic":
            if not terminated:
                return 0.0
            w = self.wealth / self.cfg.wealth_scale
            expo = float(np.clip(-self.cfg.risk_aversion * w, -10.0, 10.0))
            return float(-np.exp(expo))
        return self.dsr.step(pnl)  # default: DSR

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reinizializza l'ambiente: nuova traiettoria di mercato e stato pulito."""
        super().reset(seed=seed)  # inizializza self.np_random

        # --- Selezione della traiettoria di mercato ---
        # In modalita' split il path e' deterministico e legato a (split, indice):
        # bande di seed disgiunte => nessun data leakage tra train/val/test.
        if self.cfg.data_split is not None:
            base = self.SPLIT_BASE[self.cfg.data_split]
            heston_seed = base + self.cfg.split_offset + self._ep_idx
            self._ep_idx += 1
            rng = np.random.default_rng(heston_seed)
        else:
            rng = np.random.default_rng(self.np_random.integers(0, 2 ** 31 - 1))

        self.prices, self.vols = simulate_heston(self.market, self.n_steps, rng)

        # Strike: eventualmente randomizzato (dallo STESSO rng -> riproducibile per scenario).
        base_strike = self.market.S0 * self.cfg.moneyness
        if self.cfg.random_strike:
            base_strike *= float(rng.uniform(0.95, 1.05))
        self.K = base_strike
        self.option_qty = self.cfg.option_qty

        # Reset dello stato di portafoglio e del calcolatore di reward.
        self.hedge = 0.0
        self.t = 0
        self.wealth = 0.0
        self.dsr.reset()

        greeks = self.current_option_greeks()
        self._prev_option_value = greeks["price"]
        obs = self._get_obs(greeks)
        info = {"strike": self.K, "S0": self.prices[0]}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Esegue un passo:
            1. ribilancia la copertura alla posizione obiettivo (paga i costi);
            2. fa avanzare il mercato di un giorno;
            3. calcola il P&L di periodo e il reward (DSR);
            4. costruisce la nuova osservazione.
        """
        action = float(np.clip(action, -1.0, 1.0)[0])
        S_now = self.prices[self.t]

        # --- 1) Ribilanciamento della copertura ---
        hedge_target = action * abs(self.option_qty)
        trade = hedge_target - self.hedge
        cost = self._transaction_cost(trade, S_now)
        self.hedge = hedge_target  # nuova posizione mantenuta nell'intervallo [t, t+1]

        # --- 2) Avanzamento del mercato ---
        self.t += 1
        S_next = self.prices[self.t]
        T_next = self.T0 - self.t * self.market.dt
        sigma_next = self.vols[self.t]
        greeks_next = self._bs(S_next, T_next, sigma_next)
        option_value_next = greeks_next["price"]

        # --- 3) P&L di periodo e reward ---
        # P&L = variazione valore opzioni + P&L copertura - costi di transazione.
        pnl_options = self.option_qty * (option_value_next - self._prev_option_value)
        pnl_hedge = self.hedge * (S_next - S_now)
        pnl = pnl_options + pnl_hedge - cost

        w_prev = self.wealth                 # ricchezza cumulata PRIMA di questo passo
        self.wealth += pnl
        self._prev_option_value = option_value_next

        # --- 4) Terminazione ---
        terminated = self.t >= self.n_steps  # raggiunta la scadenza dell'opzione
        truncated = False

        # Reward (DSR / mean-variance per-passo / media-varianza TERMINALE / entropica).
        reward = self._compute_reward(pnl, w_prev=w_prev, terminated=terminated)

        obs = self._get_obs(greeks_next)

        port_delta, port_gamma = self._portfolio_greeks(greeks_next)
        info = {
            "pnl": pnl,
            "wealth": self.wealth,
            "transaction_cost": cost,
            "trade": trade,
            "hedge": self.hedge,
            "option_value": option_value_next,
            "S": S_next,
            "sigma": sigma_next,
            "T": T_next,
            "portfolio_delta": port_delta,
            "portfolio_gamma": port_gamma,
            "option_delta": greeks_next["delta"],
            "vega": greeks_next["vega"],
            "theta": greeks_next["theta"],
        }
        return obs, float(reward), terminated, truncated, info


# ==============================================================================
# 5. PIPELINE DI TRAINING E TESTING
# ==============================================================================
def make_env(config: Optional[EnvConfig] = None) -> OptionsTradingEnv:
    """Factory dell'ambiente (eventualmente avvolto da Monitor per il logging)."""
    env = OptionsTradingEnv(config)
    if SB3_AVAILABLE:
        env = Monitor(env)
    return env


def rollout(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray, OptionsTradingEnv], np.ndarray],
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Esegue un episodio completo seguendo `policy_fn` e registra le traiettorie.

    policy_fn(obs, base_env) -> action  (array shape (1,))
    Ritorna un dizionario di array numpy con la storia dell'episodio.
    """
    base_env = env.unwrapped  # accesso allo stato interno (Greche, ecc.)
    obs, _ = env.reset(seed=seed)
    done = False
    hist = {k: [] for k in ["pnl", "wealth", "hedge", "S", "portfolio_delta", "option_delta", "reward"]}
    while not done:
        action = policy_fn(obs, base_env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        hist["pnl"].append(info["pnl"])
        hist["wealth"].append(info["wealth"])
        hist["hedge"].append(info["hedge"])
        hist["S"].append(info["S"])
        hist["portfolio_delta"].append(info["portfolio_delta"])
        hist["option_delta"].append(info["option_delta"])
        hist["reward"].append(reward)
    return {k: np.asarray(v) for k, v in hist.items()}


# --- Strategie baseline (politiche deterministiche) --------------------------
def policy_delta_hedge(obs: np.ndarray, env: OptionsTradingEnv) -> np.ndarray:
    """Delta-hedging classico: annulla il Delta del portafoglio ad ogni passo."""
    greeks = env.current_option_greeks()
    # hedge desiderato per Delta neutralita': hedge = -option_qty * delta_opzione.
    desired_hedge = -env.option_qty * greeks["delta"]
    action = desired_hedge / (abs(env.option_qty) + EPS)
    return np.array([np.clip(action, -1.0, 1.0)], dtype=np.float32)


def policy_no_hedge(obs: np.ndarray, env: OptionsTradingEnv) -> np.ndarray:
    """Nessuna copertura: l'agente resta sempre flat sul sottostante."""
    return np.array([0.0], dtype=np.float32)


def evaluate(
    policy_fn: Callable, config: EnvConfig, n_episodes: int = 50, seed0: int = 10_000
) -> Dict[str, float]:
    """Valuta una politica su molti episodi e ritorna metriche aggregate di P&L."""
    env = OptionsTradingEnv(config)
    total_pnls, sharpes = [], []
    for ep in range(n_episodes):
        hist = rollout(env, policy_fn, seed=seed0 + ep)
        pnls = hist["pnl"]
        total_pnls.append(pnls.sum())
        sharpes.append(pnls.mean() / (pnls.std() + EPS))
    total_pnls = np.asarray(total_pnls)
    return {
        "mean_pnl": float(total_pnls.mean()),
        "std_pnl": float(total_pnls.std()),
        "mean_sharpe_per_step": float(np.mean(sharpes)),
        "worst_pnl": float(total_pnls.min()),
        "best_pnl": float(total_pnls.max()),
    }


def plot_results(histories: Dict[str, Dict[str, np.ndarray]], filename: str = "hedging_results.png") -> None:
    """Salva un grafico comparativo delle traiettorie (richiede matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # backend non interattivo (ambiente headless)
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib non disponibile: grafico saltato.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ref = next(iter(histories.values()))
    steps = np.arange(len(ref["S"]))

    # Prezzo del sottostante.
    axes[0, 0].plot(steps, ref["S"], color="black")
    axes[0, 0].set_title("Prezzo del sottostante")
    axes[0, 0].set_xlabel("Giorno"); axes[0, 0].set_ylabel("S")

    # P&L cumulato (wealth) per ciascuna strategia.
    for name, h in histories.items():
        axes[0, 1].plot(steps, h["wealth"], label=name)
    axes[0, 1].set_title("P&L cumulato"); axes[0, 1].legend()
    axes[0, 1].set_xlabel("Giorno"); axes[0, 1].set_ylabel("Wealth")

    # Posizione di copertura.
    for name, h in histories.items():
        axes[1, 0].plot(steps, h["hedge"], label=name)
    axes[1, 0].set_title("Posizione di hedge sul sottostante"); axes[1, 0].legend()
    axes[1, 0].set_xlabel("Giorno"); axes[1, 0].set_ylabel("Unita'")

    # Delta di portafoglio (obiettivo: vicino a zero).
    for name, h in histories.items():
        axes[1, 1].plot(steps, h["portfolio_delta"], label=name)
    axes[1, 1].axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    axes[1, 1].set_title("Delta di portafoglio"); axes[1, 1].legend()
    axes[1, 1].set_xlabel("Giorno"); axes[1, 1].set_ylabel("Delta")

    fig.tight_layout()
    fig.savefig(filename, dpi=110)
    print(f"[plot] Grafico salvato in: {filename}")


def train_agent(algo: str, config: EnvConfig, total_timesteps: int, seed: int = 0):
    """Addestra l'agente DRL (PPO o SAC) sull'ambiente di hedging."""
    if not SB3_AVAILABLE:
        raise RuntimeError(
            "Stable-Baselines3 non e' installato. "
            "Esegui: pip install stable-baselines3 torch"
        )

    env = make_env(config)
    # Validazione di conformita' dell'ambiente alle API Gymnasium.
    check_env(env, warn=True)

    algo = algo.upper()
    common = dict(policy="MlpPolicy", env=env, verbose=1, seed=seed)
    if algo == "PPO":
        model = PPO(n_steps=2048, batch_size=64, gae_lambda=0.95, gamma=0.99,
                    ent_coef=0.0, learning_rate=3e-4, **common)
    elif algo == "SAC":
        model = SAC(learning_rate=3e-4, buffer_size=100_000, batch_size=256,
                    tau=0.005, gamma=0.99, train_freq=1, **common)
    else:
        raise ValueError(f"Algoritmo non supportato: {algo} (usa PPO o SAC).")

    print(f"\n=== Avvio addestramento {algo} per {total_timesteps} timesteps ===")
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model


def run_pipeline(algo: str = "PPO", total_timesteps: int = 50_000, eval_episodes: int = 50) -> None:
    """Pipeline completa: training -> valutazione -> backtest comparativo -> grafico."""
    config = EnvConfig()

    # 1) Addestramento dell'agente DRL.
    model = train_agent(algo, config, total_timesteps)

    # Politica dell'agente come funzione compatibile con rollout/evaluate.
    def policy_agent(obs: np.ndarray, env: OptionsTradingEnv) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action

    # 2) Valutazione aggregata (agente vs baseline) su scenari out-of-sample.
    print("\n=== Valutazione su episodi out-of-sample ===")
    results = {
        f"DRL-{algo}": evaluate(policy_agent, config, eval_episodes),
        "Delta-Hedge": evaluate(policy_delta_hedge, config, eval_episodes),
        "No-Hedge": evaluate(policy_no_hedge, config, eval_episodes),
    }
    print(f"\n{'Strategia':<14}{'P&L medio':>12}{'Std P&L':>12}{'P&L peggiore':>14}")
    for name, m in results.items():
        print(f"{name:<14}{m['mean_pnl']:>12.2f}{m['std_pnl']:>12.2f}{m['worst_pnl']:>14.2f}")

    # 3) Backtest su un singolo episodio comune per il confronto grafico.
    eval_seed = 999
    histories = {
        f"DRL-{algo}": rollout(OptionsTradingEnv(config), policy_agent, seed=eval_seed),
        "Delta-Hedge": rollout(OptionsTradingEnv(config), policy_delta_hedge, seed=eval_seed),
        "No-Hedge": rollout(OptionsTradingEnv(config), policy_no_hedge, seed=eval_seed),
    }
    plot_results(histories)

    # 4) Salvataggio del modello addestrato.
    model_path = f"drl_hedging_{algo.lower()}"
    model.save(model_path)
    print(f"\n[modello] Salvato in: {model_path}.zip")


def demo_without_sb3() -> None:
    """Dimostrazione del solo ambiente (agente casuale) se SB3 non e' installato."""
    print("\n[demo] Stable-Baselines3 non disponibile: eseguo un rollout casuale.")
    env = OptionsTradingEnv()
    obs, info = env.reset(seed=0)
    print(f"[demo] Osservazione iniziale: {np.round(obs, 4)}")
    total_reward = 0.0
    for _ in range(env.n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"[demo] Reward cumulato (DSR): {total_reward:.4f} | Wealth finale: {info['wealth']:.2f}")


def self_test() -> None:
    """Test rapidi di correttezza (Black-Scholes, parita' Put-Call, DSR, env)."""
    print("=== SELF-TEST ===")
    # Put-Call parity: C - P = S e^{-qT} - K e^{-rT}.
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.02, 0.2
    c = BlackScholes.call_price(S, K, T, r, sigma)
    p = BlackScholes.put_price(S, K, T, r, sigma)
    parity_lhs = c - p
    parity_rhs = S - K * np.exp(-r * T)
    assert abs(parity_lhs - parity_rhs) < 1e-6, "Parita' Put-Call violata!"
    print(f"[OK] Parita' Put-Call: C-P={parity_lhs:.4f} ~ S-Ke^(-rT)={parity_rhs:.4f}")

    # Greche finite e coerenti (gamma > 0, vega > 0 per opzione ATM).
    g = BlackScholes.price_and_greeks(S, K, T, r, sigma, "call")
    assert g["gamma"] > 0 and g["vega"] > 0, "Greche non valide!"
    print(f"[OK] Greche Call ATM: {dict((k, round(v, 4)) for k, v in g.items())}")

    # Casi limite: scadenza e volatilita' nulla non generano NaN/inf.
    g_exp = BlackScholes.price_and_greeks(110.0, 100.0, 0.0, r, sigma, "call")
    assert g_exp["price"] == 10.0, "Valore intrinseco alla scadenza errato!"
    g_zero_vol = BlackScholes.price_and_greeks(S, K, T, r, 0.0, "call")
    assert np.isfinite(g_zero_vol["price"]), "NaN con volatilita' nulla!"
    print("[OK] Casi limite (T=0, sigma=0) gestiti senza NaN/inf.")

    # DSR: esecuzione su rendimenti casuali senza esplosioni numeriche.
    dsr = DifferentialSharpeRatio(alpha=0.05)
    rng = np.random.default_rng(0)
    rewards = [dsr.step(float(x)) for x in rng.normal(0.01, 0.1, size=200)]
    assert all(np.isfinite(rewards)), "DSR ha prodotto valori non finiti!"
    print(f"[OK] DSR stabile su 200 passi (Sharpe esponenziale finale={dsr.sharpe:.3f}).")

    # Ambiente: conformita' di base alle API Gymnasium.
    env = OptionsTradingEnv()
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs), "Osservazione fuori dallo spazio!"
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    assert np.isfinite(rew), "Reward non finito!"
    print("[OK] Ambiente: reset()/step() conformi e numericamente stabili.")
    print("=== SELF-TEST SUPERATO ===\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRL Options Hedging con Black-Scholes e DSR.")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"],
                        help="Algoritmo DRL per spazi d'azione continui.")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Numero totale di timesteps di addestramento.")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Numero di episodi per la valutazione out-of-sample.")
    parser.add_argument("--self-test", action="store_true",
                        help="Esegue solo i test di correttezza ed esce.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # I test di correttezza vengono sempre eseguiti per primi (rapidi).
    self_test()

    if args.self_test:
        pass  # solo i test
    elif SB3_AVAILABLE:
        run_pipeline(algo=args.algo, total_timesteps=args.timesteps,
                     eval_episodes=args.eval_episodes)
    else:
        demo_without_sb3()
