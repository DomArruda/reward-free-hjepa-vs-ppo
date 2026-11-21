# Reward-Free H-JEPA vs. PPO: Action-Blind Collapse, Compute Efficiency, and Regime-Shift Adaptation

This repository presents a two-domain empirical study evaluating a 2-Level Hierarchical Joint-Embedding Predictive Architecture (**H-JEPA**) with **SIGReg** (isotropic Gaussian anti-collapse regularization) and auxiliary **Inverse Dynamics (IDM)** regularization against a **Flat 1-Level JEPA** ablation and a dense-reward **PPO Actor-Critic** baseline.

Across $N = 8$ random seeds in both **deterministic continuous control** (GPU-vectorized Four-Rooms) and **out-of-distribution financial time series** (leak-free 2022 macroeconomic rate shock via `yfinance`), we evaluate task reliability, causal action controllability, wall-clock compute efficiency, and sample efficiency.

---

## Core Empirical Findings

1. **SIGReg Prevents Dimensional Collapse, Not Action-Blind Collapse (Replicated Across Both Domains):**
   Training an unconstrained MLP encoder with `Pred MSE + SIGReg` alone (`No-InvDyn`) collapses to **literal `0.0000` action sensitivity** across all 8 seeds in *both* noise-free 4D physics (Four-Rooms: `100.0%` failure, Welch's $p_{\text{adj}} = 5.96 \times 10^{-5}$) and noisy 17D financial markets (2022 rate shock: `-24.83%` max drawdown). Adding an auxiliary Inverse Dynamics head immediately restores action sensitivity (`0.0596` in Four-Rooms; `0.0361` in Finance) and unlocks closed-loop control.
2. **Reward-Free JEPA Achieves 2.7×–2.9× Wall-Clock Efficiency and 8.1× Sample Efficiency Over PPO:**
   Once multi-episode CEM planning is fully vectorized on CUDA, reward-free `Full H-JEPA` (`9.67 %/s`, `14.57 %/100k steps`) and `Flat 1-Level JEPA` (`10.44 %/s`, `13.51 %/100k steps`) significantly outperform dense-reward `PPO` (`3.61 %/s`, `1.80 %/100k steps`) on compute and sample efficiency (Kruskal-Wallis $p = 6.82 \times 10^{-6}$ and $p = 6.61 \times 10^{-6}$), while maintaining statistical parity with PPO on cross-room failure rate (Welch's $p_{\text{adj}} = 0.200$, ns).
3. **2-Level Temporal Hierarchy Introduces Compounding-Error Overhead in Low-D and Exogenous-Noise Regimes:**
   In Four-Rooms, `Full H-JEPA` (`42.7%`) showed no statistically significant reliability advantage over `Flat 1-Level JEPA` (`46.9%`, $p_{\text{adj}} = 0.597$, ns) while doubling test-time planning latency (`1.69s` vs. `0.85s`). Under the 2022 rate shock, unrolling 20-day Level-2 macro-horizons over noisy exogenous market features hurt `Full H-JEPA` (`-21.24%` MaxDD, `-16.96%` return) relative to `Flat 1-Level JEPA` (`-17.75%` MaxDD, `-11.11%` return)—which matched `PPO` on max drawdown (`-17.95%`) zero-shot without rewards, though PPO recovered faster in 2023 (`-2.79%` return).

---

## Experiment 1: Continuous Four-Rooms (Endogenous 4D Physics)

### Setup
* **Environment:** Continuous Four-Rooms ($[-1, 1]^2$, 4D state $[x, y, v_x, v_y]$, 2D bounded force $a \in [-1, 1]^2$, wall collisions with 0.3× elastic bounce).
* **Architecture:** Shared encoder $E_\theta: \mathbb{R}^4 \to \mathbb{R}^{32}$ regularized via **SIGReg** (Epps-Pulley 1D characteristic function test against $\mathcal{N}(0, I)$), Level-1 predictor $F^{(1)}$ (1-step), Level-2 macro-predictor $F^{(2)}$ ($k = 6$ steps) anchored via cross-level composition consistency ($\mathcal{L}_{\text{cons}}$), and auxiliary Inverse Dynamics heads $I^{(1)}, I^{(2)}$.
* **Compute Budget ($N = 8$ seeds, 24 cross-room episodes/seed, fully vectorized on CUDA):**
  * **JEPA Variants:** `393,216` reward-free transitions (`4,096` trajectories × `96` steps), evaluated via vectorized multi-episode GPU CEM.
  * **PPO Baseline:** `3,840,000` reward-labeled transitions (`512` envs × `150` steps × `50` updates) with dense distance + goal rewards.

### Causal Mechanism Check: Action Sensitivity $\|\hat{z}(s, +1) - \hat{z}(s, -1)\|_2^2$

| Model Variant | Level-1 Sensitivity | Level-2 Sensitivity |
| :--- | :---: | :---: |
| **With Inverse Dynamics Head** | `0.0024 ± 0.0003` | `0.0596 ± 0.0072` |
| **Without Inverse Dynamics** | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` |

### Failure Rate, Compute Breakdown & Efficiency ($N = 8$ Seeds)

| Condition | Fail % (Mean ± SD) | Train (s) | Eval (s) | Wall-Clock Eff (%/s) | Sample Eff (%/100k Steps) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **H-JEPA + SIGReg + InvDyn (Full)** | 42.7 ± 16.0% | `4.23s` | `1.69s` | `9.67` | **`14.57`** |
| **H-JEPA + SIGReg (No InvDyn)** | 100.0 ± 0.0% | `3.85s` | `1.68s` | `0.00` | `0.00` |
| **Flat 1-Level JEPA + InvDyn (Ablation)** | 46.9 ± 14.7% | `4.23s` | `0.85s` | **`10.44`** | `13.51` |
| **PPO Actor-Critic (RL Baseline)** | **30.7 ± 10.2%** | `18.98s` | **`0.22s`** | `3.61` | `1.80` |

### Statistical Inference (Four-Rooms)
Because `No-InvDyn` saturates at `100.0%` failure (zero variance), standard ANOVA's homoscedasticity assumption is violated. We report **Kruskal-Wallis** as our primary non-parametric omnibus test and **Welch's two-tailed $t$-tests** (Holm-Bonferroni corrected) for pairwise contrasts.

* **Omnibus Kruskal-Wallis:**
  * Failure Rate (%): $H = 20.41$, $p = 1.395 \times 10^{-4}$ (**Reject $H_0$**)
  * Wall-Clock Efficiency (%/s): $H = 26.70$, $p = 6.821 \times 10^{-6}$ (**Reject $H_0$**)
  * Sample Efficiency (%/100k Steps): $H = 26.76$, $p = 6.611 \times 10^{-6}$ (**Reject $H_0$**)

**Two-Tailed Welch Post-Hoc Contrasts vs. Full H-JEPA (Failure Rate %, Holm-Bonferroni):**

| Comparison | Δ Fail % | $t$-stat | $p_{\text{adj}}$ | Result |
| :--- | :---: | :---: | :---: | :---: |
| vs. `H-JEPA + SIGReg (No InvDyn)` | `-57.29%` | `-10.11` | $5.96 \times 10^{-5}$ | **Significant** |
| vs. `Flat 1-Level JEPA + InvDyn` | `-4.17%` | `-0.54` | $5.97 \times 10^{-1}$ | ns |
| vs. `PPO Actor-Critic (RL Baseline)` | `+11.98%` | `+1.78` | $2.00 \times 10^{-1}$ | ns |

---

## Experiment 2: 2022 Macroeconomic Rate Shock (Exogenous + Endogenous 17D Finance)

### Setup
* **Data & Strict Leakage Prevention:** Daily returns for 5 ETFs (`SPY`, `TLT`, `GLD`, `XLE`, `SHY`) via `yfinance`. Market features (20-day rolling mean and volatility, 10D) are **shifted by 1 trading day (`.shift(1)`)** and standardized using *only* training statistics so state $s_t$ strictly observes returns through $t - 1$ before selecting simplex portfolio weights $w_t \in \Delta^4$ for day $t$ (`10` bps transaction cost).
* **Regime Split:**
  * **Train (2015–2021, 1,742 trading days):** Low-rate bull market where equities (`SPY`) and long treasuries (`TLT`) rallied with negative correlation.
  * **OOD Test (2022–2023, 375 trading days):** Federal Reserve rate-hiking shock where both `SPY` and `TLT` drew down simultaneously (60/40 benchmark dropped below `-25%` MaxDD).
* **Zero-Shot Objective:** JEPA models train reward-free on random 60-day rollouts from 2015–2021 (macro-stride $k = 5$ trading days = 1 week) and plan zero-shot in 2022–2023 toward a defensive, zero-drawdown target state. PPO trains directly on 2015–2021 risk-adjusted portfolio rewards and is evaluated zero-shot in 2022–2023.

### Causal Mechanism Check: Action Sensitivity (`100% SPY` vs. `100% SHY`)

| Model Variant | Level-2 Sensitivity |
| :--- | :---: |
| **With Inverse Dynamics Head** | `0.0361` |
| **Without Inverse Dynamics** | `0.0000` |

### Zero-Shot 2022–2023 Rate-Shock Performance ($N = 8$ Seeds)

| Condition | Max Drawdown % (Mean ± SD) | Cum Return % (Mean ± SD) | Sharpe |
| :--- | :---: | :---: | :---: |
| **H-JEPA + SIGReg + InvDyn (Full)** | -21.24 ± 2.09% | -16.96 ± 2.83% | `-0.99` |
| **H-JEPA + SIGReg (No InvDyn)** | -24.83 ± 4.93% | -20.11 ± 6.86% | `-1.08` |
| **Flat 1-Level JEPA + InvDyn** | **-17.75 ± 2.96%** | -11.11 ± 4.07% | `-0.60` |
| **PPO Actor-Critic (2015–21 RL)** | -17.95 ± 1.91% | **-2.79 ± 3.27%** | **`-0.10`** |

* **Omnibus Kruskal-Wallis (Max Drawdown):** $H = 15.599$, $p = 1.370 \times 10^{-3}$ (**Reject $H_0$**)

### Key Takeaway on Exogenous vs. Endogenous Dynamics
While Four-Rooms is `100%` endogenous (agent actions drive every state variable), `10` of the `17` state dimensions in the finance benchmark are **exogenous market features** unaffected by portfolio weights. Predicting all `17` dimensions through a single unfactored latent MSE loss allows exogenous market noise to compound over Level-2's 20-day macro-horizon, causing `Full H-JEPA` to trail both `Flat 1-Level JEPA` and `PPO`. Notably, reward-free `Flat 1-Level JEPA` still matched PPO on peak 2022 drawdown (`-17.75%` vs. `-17.95%`) and clearly outperformed a passive 60/40 (`SPY`/`TLT`) portfolio.

---

## Limitations & Future Directions

* **Factoring Exogenous vs. Endogenous Latent States:** Financial and macroeconomic world models likely require explicit state factorization—separating uncontrollable market dynamics from controllable portfolio/balance-sheet transitions—before multi-step hierarchical rollouts can outperform myopic planning.
* **Low-Dimensional State Spaces vs. Pixel Observations:** Both benchmarks use compact vector observations (`4D` and `17D`) rather than high-dimensional video, where multi-level H-JEPA architectures with separate per-level encoders and top-down lifting are specifically designed to discard high-frequency visual clutter.
* **Methodological Note on Leakage & Action Parameterization:** Earlier prototyping revealed that unshifted rolling features allow PPO to exploit same-day look-ahead leakage (`+46%` return in 2022), and raw softmax logits break inverse-dynamics identifiability due to shift invariance. All reported results above use strictly lagged features (`.shift(1)`) and simplex-projected weight actions.

---

## Repository Structure & Usage

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── losses.py             # SIGRegLoss (Epps-Pulley) & Inverse Dynamics helper
│   ├── models.py             # HierarchicalJEPA (Four-Rooms & Finance), train_hjepa, GoalPPO, PortfolioPPO
│   ├── planners.py           # Vectorized GPU CEM (Euclidean & Simplex)
│   └── stats_utils.py        # Kruskal-Wallis, Alexander-Govern, Welch + Holm-Bonferroni
└── experiments/
    ├── run_four_rooms.py     # Experiment 1: Vectorized Four-Rooms (N = 8 seeds)
    └── run_rate_shock.py     # Experiment 2: Leak-free 2022 Rate Shock (N = 8 seeds)
```

```bash
pip install -r requirements.txt
python experiments/run_four_rooms.py
python experiments/run_rate_shock.py
```
