# Risultati sperimentali — DRL Options Hedging

Esperimento per spingere le performance dell'agente DRL mantenendo un
protocollo rigoroso e privo di **data leakage** (script: `advanced_training.py`).

## Protocollo (anti data-leakage)

- **Split disgiunti** train / validation / test su bande di seed separate
  (`OptionsTradingEnv.SPLIT_BASE`): nessuna traiettoria di mercato condivisa
  (verificato con assert all'avvio).
- **`VecNormalize` congelato**: media/varianza delle osservazioni calcolate
  **solo sul training** e usate congelate in valutazione (`FrozenObsNormalizer`).
- **Selezione solo su validation** (min std del P&L); **test toccato una volta**
  (500 scenari mai visti) per la stima finale imparziale.
- **Metriche economiche** calcolate sul **P&L grezzo**, indipendenti dalla
  reward usata in training.

## Configurazioni confrontate

| Config | Algoritmo | Reward | λ | Timesteps |
|--------|-----------|--------|---|-----------|
| PPO-meanvar | PPO | mean-variance | 0.1 | 250k |
| PPO-DSR | PPO | DSR (specifica) | – | 250k |
| SAC-meanvar | SAC | mean-variance | 0.1 | 80k |

## Validation (selezione del modello)

| Strategia | P&L medio | Std | Sharpe | CVaR-5% | Worst | Costi |
|-----------|----------:|----:|-------:|--------:|------:|------:|
| PPO-meanvar | -17.49 | 110.63 | -0.158 | -291.84 | -409.22 | 32.07 |
| PPO-DSR | -175.11 | 1421.54 | -0.123 | -3819.48 | -5160.01 | 34.85 |
| **SAC-meanvar** ✅ | -31.02 | **85.32** | -0.364 | -225.55 | -284.13 | 31.12 |
| Delta-Hedge | -14.28 | 77.84 | -0.183 | -186.20 | -259.04 | 28.15 |
| No-Hedge | -114.17 | 639.28 | -0.179 | -1772.49 | -2499.60 | 0.00 |

Modello selezionato (min std su validation): **SAC-meanvar**.

## Test finale (500 scenari mai visti)

| Strategia | P&L medio | Std | Sharpe | CVaR-5% | Worst | Costi |
|-----------|----------:|----:|-------:|--------:|------:|------:|
| SAC-meanvar (DRL) | -14.56 | 89.72 | -0.162 | -221.33 | -389.04 | 31.28 |
| **Delta-Hedge** | **-7.94** | **75.95** | **-0.105** | **-182.30** | **-302.00** | **28.53** |
| No-Hedge | -17.78 | 564.82 | -0.031 | -1511.56 | -2649.09 | 0.00 |

## Conclusioni

1. **L'agente DRL impara a coprirsi**: riduzione del rischio (std) da ~565
   (No-Hedge) a ~90, cioè ~6.3×. Enorme miglioramento rispetto al setup base
   non normalizzato (std ~629).
2. **Non supera il Delta-Hedge classico** su nessuna metrica: per la copertura
   lineare di una singola opzione, il Delta di Black-Scholes e' gia' quasi
   ottimo. L'agente ribilancia leggermente troppo (costi 31.3 vs 28.5).
3. **Il reward conta piu' dell'algoritmo**: il DSR puro (specifica originale)
   non induce hedging (std 1421); serve una penalita' di varianza esplicita.

## Limiti e direzioni per spingere oltre

- **Ricerca iperparametri** (es. Optuna) su validation: λ, dimensione rete,
  γ, learning rate, durata training.
- **Reward cost-aware / no-trade band**: penalizzare esplicitamente il turnover
  per far emergere il risparmio sui costi (vero potenziale vantaggio sull'RL).
- **Budget di training maggiore** (qui limitato dalla CPU).
- **Setting dove la baseline NON e' ottima**: costi/market-impact elevati,
  ribilanciamento discreto, **hedging multi-strumento** (Gamma/Vega, non solo
  Delta), vincoli di posizione — e' li' che l'RL puo' davvero battere il
  delta-hedging.

> Nota metodologica: i risultati sono riportati senza selezione a posteriori
> sul test. Tutte le decisioni di modello sono state prese sul validation set.

---

# Esperimento 2 — Cost-aware: l'RL puo' battere il Delta-Hedge?

Script: `cost_aware_experiment.py`. Regime ad **alti costi (50 bps)**, dove il
delta-hedging continuo e' subottimo e la strategia ottima e' una **no-trade
band**. Avversario forte: delta-hedge a band ottimizzata su validation (CVaR-5%).

## Test finale (alti costi, 500 scenari mai visti)

| Strategia | P&L medio | Std | Sharpe | CVaR-5% | Worst | Costi |
|-----------|----------:|----:|-------:|--------:|------:|------:|
| **Band(15) ottima** 🏆 | **-58.43** | 107.30 | -0.545 | **-297.01** | -417.85 | **75.73** |
| SAC-meanvar (DRL) | -113.41 | 101.34 | -1.119 | -349.71 | -484.98 | 132.62 |
| Delta-Hedge (full) | -122.05 | 90.52 | -1.348 | -325.69 | -403.70 | 142.64 |
| No-Hedge | -17.78 | 564.82 | -0.031 | -1511.56 | -2649.09 | 0.00 |

## Conclusioni

1. **L'RL batte il delta-hedge ingenuo** (P&L -113 vs -122, costi 133 vs 143):
   marginale ma reale.
2. **L'RL perde nettamente contro la no-trade band classica** (-113 vs -58):
   la band taglia i costi (76 vs 133) ribilanciando meno, a parita' di rischio.
   L'agente NON ha scoperto la struttura a banda (turnover quasi da hedge pieno).
3. **Causa diagnosticata:** la reward mean-variance (`pnl - λ·pnl²`) penalizza
   la varianza del P&L *per-passo*, spingendo verso un tracking stretto del Delta
   (alto turnover). Questa reward **e' in conflitto** con l'obiettivo di
   risparmiare costi. La band vince perche' ottimizza direttamente il trade-off
   costo/rischio con un singolo parametro interpretabile.

## Lezione

Anche in uno scenario costruito a favore dell'RL, un metodo classico semplice e
ben tarato vince. Per battere davvero la band serve **ridisegnare la reward**
sull'obiettivo economico corretto (media-varianza del P&L **terminale**, stile
*Deep Hedging* di Buehler et al.) anziche' sulla varianza per-passo, e/o un
budget di training/tuning molto maggiore.

---

# Esperimento 3 — Deep Hedging (reward su P&L terminale)

Script: `deep_hedging_experiment.py` + `dh_highlambda.py`. Seguendo la lezione
dell'esp. 2, si e' implementata la reward Deep Hedging sull'obiettivo TERMINALE
(`reward_mode='terminal_mv'`, forma densa via telescoping; vedi
`drl_options_hedging._compute_reward`). Obiettivo: far emergere la no-trade band.

## Test finale (alti costi, 500 scenari mai visti)

| Strategia | P&L medio | Std | Sharpe | CVaR-5% | Costi |
|-----------|----------:|----:|-------:|--------:|------:|
| **Band(15) ottima** 🏆 | -58.43 | **107.30** | **-0.545** | **-297.01** | 75.73 |
| Delta-Hedge (full) | -122.05 | 90.52 | -1.348 | -325.69 | 142.64 |
| PPO-terminal_mv (DRL) | +14.97 | 570.84 | 0.026 | -1687.93 | 50.48 |
| No-Hedge | -17.78 | 564.82 | -0.031 | -1511.56 | 0.00 |

## Esito: la reward terminal_mv e' degenere

Sweep di lambda su **due ordini di grandezza** (0.002 -> 0.5): output PRATICAMENTE
IDENTICO (mean ~35, std ~565, costi ~50 su validation). L'agente **ignora del
tutto il termine di varianza** e collassa sempre su una politica **quasi-no-hedge**.

| lambda | P&L medio (val) | Std | Costi |
|-------:|---------------:|----:|------:|
| 0.002 | 35.9 | 565.7 | 50.0 |
| 0.01  | 34.1 | 565.1 | 51.3 |
| 0.05  | 35.8 | 565.5 | 50.1 |
| 0.20  | 34.7 | 565.0 | 50.8 |
| 0.50  | 35.0 | 564.9 | 50.8 |

**ATTENZIONE al "MEGLIO" ingannevole:** il DRL ha P&L medio (+15) superiore alla
band (-58), ma SOLO perche' ha smesso di coprirsi (in questo mercato con drift il
No-Hedge ha la media migliore). Il suo profilo di rischio e' quello del No-Hedge:
Std e CVaR ~5x PEGGIORI della band. Non e' hedging, e' speculazione. Su base
risk-adjusted (l'obiettivo vero) **fallisce completamente**.

## Causa diagnosticata

Il termine telescoping di varianza `2*W_{t-1}*pnl_t` e' path-dependent e ad alta
varianza: vincola solo la SOMMA terminale e non offre quasi nessun segnale di
gradiente per-passo. PPO non riesce a ottimizzarlo e ripiega sul solo termine di
media -> politica no-hedge. (La reward `meanvar` per-passo, pur subottimale,
almeno fornisce un gradiente denso e fa coprire l'agente.)

---

# Conclusione complessiva

Su **tre** esperimenti e molteplici reward (DSR, mean-variance per-passo,
Deep Hedging terminale) con PPO e SAC, **nessuna variante RL ha battuto i metodi
classici ben tarati su base risk-adjusted**:

- Caso base: il Delta di Black-Scholes resta imbattuto.
- Alti costi: la **no-trade band** ottimizzata resta il vincitore netto
  (miglior Sharpe e CVaR, minor tail-risk).

L'RL **impara a coprirsi** e batte le baseline ingenue, ma non i metodi classici
appropriati. Per un eventuale superamento servirebbero: una reward terminale con
riduzione della varianza del gradiente (es. utilita' entropica ben scalata o RL
distribuzionale per CVaR), e un budget di training/tuning molto maggiore — senza
garanzia di successo su questo problema (hedging Delta lineare), dove la
soluzione classica e' gia' quasi ottima.
