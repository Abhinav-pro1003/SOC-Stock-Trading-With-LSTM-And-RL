# Week 6 — Reinforcement Learning for Algorithmic Trading (AAPL)

## 📌 Overview

Up to this point, the project has used **supervised** models (LSTM/GRU price
and direction predictors, routed by market regime) to drive a set of
**rule-based** trading strategies. This week, we replace the fixed
if/then trading rules with agents that *learn* their own trading policy
through trial and error: two **reinforcement learning (RL)** agents — a
**Double Dueling DQN** and a **recurrent PPO** — that consume the same
LSTM/GRU signals (plus a richer set of engineered features) and decide,
bar by bar, whether to buy, sell, or hold AAPL.

Across the two notebooks this week, we:

1. Built a custom Gym-style `StockTradingEnv` with a **risk-adjusted reward
   function** — a Sortino-style downside penalty, an exponential drawdown
   penalty, an overtrading penalty, and holding-time shaping — instead of
   just raw portfolio return.
2. Engineered a richer, **stationary** state representation for the agents:
   Garman-Klass volatility, ADX trend strength, multi-horizon log-returns,
   EMA-distance features, and short-horizon **frame-stacking** for temporal
   context.
3. Upgraded the DQN agent to a full **Double Dueling DQN** (separate
   value/advantage streams + Double-DQN target) with **action masking**, so
   the agent never wastes exploration on actions it can't legally take
   (e.g. buying with $0 cash).
4. Replaced PPO's MLP backbone with a **recurrent (GRU-cell) actor-critic**,
   trained with **chunked truncated BPTT**, GAE advantage estimation, and
   the same action masking.
5. Automated hyperparameter search with **Optuna**, optimizing a custom
   objective that rewards return while heavily penalizing drawdown and
   overtrading.
6. Evaluated both RL agents against the four rule-based strategies (S1–S4)
   and Buy & Hold from previous weeks, on a held-out test period.

## 📁 Files

| File | What it is |
|---|---|
| `05_rl_trading_agent.ipynb` | **Solved** notebook: environment, DQN, PPO, training, Optuna search |
| `05_rl_trading_agent_ASSIGNMENT.ipynb` | Same notebook with **7 TODOs** for you to complete |
| `06_rl_evaluation_and_comparison.ipynb` | **Solved** notebook: trade analysis, plots, full strategy comparison |
| `06_rl_evaluation_and_comparison_ASSIGNMENT.ipynb` | Same notebook with **6 TODOs** for you to complete |
| `data/`, `models/`, `reports/` | Feature CSVs, trained LSTM/GRU weights, and saved results from earlier weeks — used as-is, no changes needed |

**Run order:** `05` (or `05_ASSIGNMENT`) must be run to completion first — it
saves `reports/rl_results.pkl`, which `06` (or `06_ASSIGNMENT`) loads.

---

## 📝 Assignment

Work through **`05_rl_trading_agent_ASSIGNMENT.ipynb`**, then
**`06_rl_evaluation_and_comparison_ASSIGNMENT.ipynb`**, in that order.
Everything except the pieces below is already implemented — data loading,
feature engineering, model loading, training loops, and plotting scaffolding
are all done for you. **13 TODOs total**, weighted toward quick, focused
wins since this is a shorter assignment:

| # | Difficulty | Notebook | What you implement |
|---|---|---|---|
| 1 | 🟢 Easy | 05 | `get_action_mask()` — which actions are currently legal |
| 2 | 🟢 Easy | 05 | Exponential drawdown penalty |
| 3 | 🟢 Easy | 05 | Overtrading penalty |
| 4 | 🟡 Medium | 05 | Dueling DQN's `V + A - mean(A)` combination |
| 5 | 🟢 Easy | 05 | Epsilon-greedy decay |
| 6 | 🟡 Medium | 05 | Double DQN target computation |
| 7 | 🔴 Hard | 05 | GAE backward recursion |
| 8 | 🟢 Easy | 06 | Build the strategy comparison table |
| 9 | 🟢 Easy | 06 | Per-trade PnL and % return |
| 10 | 🟡 Medium | 06 | Normalize + plot equity curves |
| 11 | 🟢 Easy | 06 | Drawdown series (running peak + %) |
| 12 | 🟢 Easy | 06 | Annualized volatility |
| 13 | 🟡 Medium | 06 | Rolling Sharpe ratio |

**Total: 8 Easy · 4 Medium · 1 Hard.**

Every TODO has a markdown cell directly above it explaining exactly what to
compute and why — you shouldn't need to derive any formula from scratch,
just translate the explanation into a few lines of code.

**How to know you're done:** every incomplete TODO raises
`NotImplementedError` when its cell runs. Once all 13 are filled in, both
notebooks should run top-to-bottom without errors and produce a full
comparison table + plots of DQN vs. PPO vs. the Week 4/5 rule-based
strategies vs. Buy & Hold.

---

## 📚 Resources

### Reinforcement Learning fundamentals
- Mnih et al., [*Playing Atari with Deep Reinforcement Learning*](https://arxiv.org/abs/1312.5602) (2013) — the original DQN paper
- van Hasselt et al., [*Deep Reinforcement Learning with Double Q-learning*](https://arxiv.org/abs/1509.06461) (2015) — Double DQN
- Wang et al., [*Dueling Network Architectures for Deep Reinforcement Learning*](https://arxiv.org/abs/1511.06581) (2015) — Dueling DQN
- Schaul et al., [*Prioritized Experience Replay*](https://arxiv.org/abs/1511.05952) (2015)

### PPO / policy gradients
- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017) — the PPO paper, incl. the clipped surrogate objective
- Schulman et al., [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438) (2015) — the GAE paper (TODO 7)
- OpenAI, [Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — a much gentler, code-annotated walkthrough of PPO than the paper

### Recurrent policies
- Stable-Baselines3 Contrib, [RecurrentPPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html) — a production implementation of the same GRU/LSTM-backbone idea used in notebook 05, useful for comparing design choices

### Hyperparameter optimization
- [Optuna documentation](https://optuna.readthedocs.io/en/stable/) and [official tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- Akiba et al., [*Optuna: A Next-Generation Hyperparameter Optimization Framework*](https://arxiv.org/abs/1907.10902) (2019)

### Risk & performance metrics used in the reward function and evaluation
- Investopedia, [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp)
- Investopedia, [Sortino Ratio](https://www.investopedia.com/terms/s/sortinoratio.asp) — the asymmetric downside-only risk measure the environment's reward function is modeled on
- Investopedia, [Maximum Drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp)
- Investopedia, [Average Directional Index (ADX)](https://www.investopedia.com/terms/a/adx.asp) — used as the trend-strength feature in the RL state
