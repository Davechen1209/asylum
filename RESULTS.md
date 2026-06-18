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
