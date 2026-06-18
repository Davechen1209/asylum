# DRL Options Hedging — Black-Scholes & Differential Sharpe Ratio

Sistema di **Deep Reinforcement Learning** per la gestione e la copertura
(*hedging*) di un portafoglio di opzioni Europee. Il modello di
**Black-Scholes-Merton** definisce nativamente lo stato (prezzi e Greche) e il
**Differential Sharpe Ratio (DSR)** di Moody & Saffell funge da funzione di
reward online e incrementale.

Tutto il codice è contenuto, in forma modulare e commentata in italiano, nel
file [`drl_options_hedging.py`](drl_options_hedging.py).

---

## 1. Architettura

Il file è organizzato in 5 sezioni chiare:

| Sezione | Componente | Descrizione |
|--------:|------------|-------------|
| 1 | `BlackScholes` | Prezzo analitico di Call/Put Europee e Greche analitiche (**Delta, Gamma, Vega, Theta**), con gestione dei casi limite `T→0` e `σ→0`. |
| 2 | `DifferentialSharpeRatio` | Formulazione **ricorsiva** del DSR con medie mobili esponenziali `A_t` (η, rendimenti) e `B_t` (secondo momento), fattore di decadimento `α=0.05`, denominatore stabilizzato con epsilon. |
| 3 | `simulate_heston` | Dati sintetici: **GBM con volatilità stocastica** (modello di Heston, schema *full-truncation* di Eulero). |
| 4 | `OptionsTradingEnv` | Ambiente **Gymnasium** custom: spazio degli stati e delle azioni continui, costi di transazione (commissioni + slippage). |
| 5 | Pipeline | Training con **Stable-Baselines3 (PPO/SAC)**, valutazione out-of-sample e **backtest comparativo** vs baseline. |

### Spazio degli stati (Observation Space) — `Box` continuo (6 dim.)

| idx | Variabile | Descrizione |
|----:|-----------|-------------|
| 0 | `S / K` | moneyness del sottostante |
| 1 | `(T-t) / T0` | tempo residuo alla scadenza (normalizzato) |
| 2 | `σ` | volatilità implicita |
| 3 | `hedge / |q|` | posizione corrente sul sottostante |
| 4 | `Δ_tot / |q|` | **Delta totale** di portafoglio (Black-Scholes) |
| 5 | `(Γ_tot · S) / |q|` | **Gamma totale** di portafoglio (Black-Scholes) |

### Spazio delle azioni (Action Space) — `Box` continuo (1 dim.)

`a ∈ [-1, 1]` → posizione obiettivo sul sottostante: `hedge_target = a · |q|`.
L'agente decide quante unità di sottostante detenere per neutralizzare/aggiustare
il Delta (Delta-hedging).

### Reward — Differential Sharpe Ratio

```
            B_{t-1}·ΔA_t − ½·A_{t-1}·ΔB_t
  D_t  =  ─────────────────────────────────────
                 (B_{t-1} − A_{t-1}²)^{3/2}
```

con `ΔA_t = R_t − A_{t-1}`, `ΔB_t = R_t² − B_{t-1}` e `R_t` = P&L di periodo del
portafoglio. È un segnale **denso, causale e scale-invariante** che premia la
stabilità del P&L corretta per il rischio.

---

## 2. Installazione

```bash
pip install -r requirements.txt
```

> Lo script funziona anche senza `stable-baselines3`: in tal caso esegue una
> dimostrazione dell'ambiente con un agente casuale.

---

## 3. Esecuzione

```bash
# Test di correttezza rapidi (Black-Scholes, parità Put-Call, DSR, env)
python drl_options_hedging.py --self-test

# Pipeline completa: training + valutazione + backtest + grafico
python drl_options_hedging.py --algo PPO --timesteps 50000

# In alternativa, Soft Actor-Critic (più sample-efficient)
python drl_options_hedging.py --algo SAC --timesteps 50000
```

Parametri principali della CLI:

| Flag | Default | Descrizione |
|------|---------|-------------|
| `--algo` | `PPO` | algoritmo DRL (`PPO` o `SAC`) |
| `--timesteps` | `50000` | timesteps totali di addestramento |
| `--eval-episodes` | `50` | episodi per la valutazione out-of-sample |
| `--self-test` | – | esegue solo i test di correttezza |

Output prodotti: `hedging_results.png` (grafico comparativo) e
`drl_hedging_<algo>.zip` (modello addestrato).

---

## 4. Interpretazione dei risultati

La pipeline confronta l'agente DRL con due baseline:

- **Delta-Hedge** — copertura classica che azzera il Delta ad ogni passo;
- **No-Hedge** — nessuna copertura (esposizione direzionale piena).

Il Delta-hedging riduce drasticamente la **volatilità del P&L** rispetto al
No-Hedge: è il segno che le Greche di Black-Scholes e la contabilità del P&L
sono corrette. Con sufficienti timesteps, l'agente DRL impara a coprire
gestendo al contempo i **costi di transazione**, ottimizzando il DSR invece di
neutralizzare ciecamente il Delta.

---

## 5. Note tecniche e casi limite

- **Divisioni per zero** gestite con un termine `EPS` (volatilità, scadenza,
  varianza del DSR).
- **Volatilità del modello di Heston** mantenuta non-negativa (*full
  truncation*); la vol istantanea è usata come *proxy* della volatilità
  implicita per il pricing.
- L'ambiente è validato con `stable_baselines3.common.env_checker.check_env`.
- Il DSR è **scale-invariante**: il P&L grezzo può essere passato direttamente
  come rendimento senza ulteriore normalizzazione.

> ⚠️ Codice a scopo di ricerca/didattico. Non costituisce consulenza
> finanziaria né è pronto per il trading in produzione.
