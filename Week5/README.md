# Week 5 — From Predictions to Profits: Rule-Based Trading Strategies

> [!IMPORTANT]
> **Submission Link:** Please submit your completed notebook and write-up via the following form:
> 🔗 [Google Form Submission Link](https://docs.google.com/forms/d/e/1FAIpQLSeF9JAXeZsif4aFihhKukyFvJzWbF-az8s9aGG8AYhsm9L1ng/viewform?usp=dialog)

This assignment bridges the gap between the ML-based return predictions built in Week 4 and
systematic trade execution. You will load the **five pre-trained models** (baseline, uptrend,
downtrend, sideways, high_volatility) and combine them with classic technical rules to build four
production-style trading strategies, then backtest and evaluate them against a Buy-and-Hold
benchmark. There is a single notebook this week, and it must be run top to bottom.

---

## Notebook & Pipeline Structure

### `Trading_Strategies.ipynb`

The data loading, model architecture, and backtesting/evaluation engine are already built for
you. Your job is the trading logic — look for `# TODO:` markers throughout.

| # | Strategy | Models Used | Core Idea |
|---|---|---|---|
| 1 | **Dynamic Regime-Routed** | uptrend_lstm, downtrend_lstm, sideways_bilstm | Gating mechanism routes each day to the regime-specialist model |
| 2 | **Momentum + LSTM Filter** | baseline_lstm | EMA trend + LSTM validation avoids bull traps |
| 3 | **Mean Reversion + BiLSTM** | sideways_bilstm | Bollinger / RSI oversold + BiLSTM bounce confirmation |
| 4 | **Volatility Breakout + ATR** | high_volatility_gru | GRU-predicted breakout sized by ATR, with a dynamic trailing stop |

A Buy-and-Hold benchmark is included for comparison.

- **Data Loading Pipeline:** Loads the same pre-processed feature CSV used in Week 4
  (`data/processed/AAPL_features.csv`), sorts and cleans it, and reconstructs the identical
  80/20 train/test split.
- **Model Architecture & Weight Loading:** Re-declares the `TemporalAttention` and
  `ReturnSequenceModel` classes from Week 4 so the saved `.pt` weights for all five models can be
  loaded without importing the earlier notebooks. Regime classification (`classify_regimes()`) is
  reused as-is.
- **Signal Generation Engine:** Sliding-window sequence construction and batched inference are
  provided; each of the four strategy functions must turn model predictions and/or technical
  indicators into a signal array of `{-1, 0, +1}` (short / flat / long) over the test set.
- **Master Trend Filter (Section 5b):** A shared veto layer applied to Strategies 1 and 4 that
  blocks trades fighting an *established* trend.
- **Risk-Managed Backtester:** A vectorized engine simulating each strategy on the test set with
  a lagged signal, a 0.05% one-way transaction cost drag, a hard stop-loss, and
  volatility-targeted position sizing. Two formula-driven gaps (vol-targeted sizing and the
  transaction-cost drag) are left for you to complete.
- **Performance Evaluation:** Annualised Sharpe Ratio, Max Drawdown, Total Return, Annualised
  Return, and Win Rate — formulas are fully specified in each function's docstring; only the
  implementation is missing.
- **Outputs:** Saves five diagnostic figures (`fig_cumulative_returns.png`,
  `fig_drawdown_profiles.png`, `fig_signal_maps.png`, `fig_sharpe_mdd.png`,
  `fig_monthly_heatmap.png`) and the final comparison table
  (`week5_strategy_performance.csv`) into `reports/`.

> [!WARNING]
> The hyperparameters at the top of Section 0 are **deliberately bad starting points**. Don't
> assume a negative Sharpe ratio or a wall of zero-signals means your logic is wrong — it may just
> mean the knobs need turning. Implement first, tune second.

---

## What You Need to Do

1. Complete the `# TODO:` sections in Strategies 1–4 (Section 5).
2. Complete the Master Trend Filter (Section 5b).
3. Fill in the two backtester gaps (Section 6): volatility-targeted position sizing and the
   transaction-cost drag.
4. Implement the performance metric formulas (Section 7): `sharpe_ratio`, `max_drawdown`,
   `total_return`, `annualised_return`.
5. Fill in the small plotting `# TODO:`s in Figures 1, 2, 4, and the best-strategy lookup in
   Section 9 — the formula/method is always given, only the line of code is missing.
6. Tune the Section 0 hyperparameters until results are sensible, and complete the write-up in
   Section 10.

---

## Write-Up (Section 10)

For each strategy, address:
1. What did your **initial** run look like (Sharpe, MDD, trade count) before touching any
   hyperparameters? What did that tell you about what was wrong?
2. Which hyperparameter(s) mattered most once your logic was correct? Trial-and-error, or
   reasoning about what the parameter controls?
3. Strategy 3: how did you decide when a "normal-conviction" reading should be allowed to trade
   versus when the extreme/regime-gated path was needed? What happens if the regime gate is
   removed entirely?
4. Strategy 4: what did you observe about the interaction between the trailing stop and the entry
   logic when they were structured incorrectly? What tipped you off?
5. Compare your best strategy's Sharpe Ratio and Max Drawdown to Buy & Hold. Is it actually better
   on a risk-adjusted basis, or just on total return? Which metric would you show a risk manager,
   and why?

### Grading Rubric
- **Correctness:** does each strategy implement the described behavior without altering function
  signatures or the backtesting/evaluation code?
- **Tuning:** is there evidence of deliberate, reasoned hyperparameter search (not just
  copy-pasted values)?
- **Write-up:** do your answers demonstrate understanding of *why* a change helped, not just
  *that* it helped?

---

## 🌟 Bonus Question of the Week

Every strategy this week is still a **fixed rule** — you chose the thresholds, the confirmation
windows, the regime-routing logic, and no part of it adapts based on how well it's actually
performing. Reinforcement Learning replaces that fixed rulebook with an agent that *learns* a
policy from trial and error.

**Bonus task:** Reframe Strategy 2 (Momentum + LSTM Filter) as an RL problem, without necessarily
training anything.

1. Define a minimal **state** for a daily-rebalanced trading agent using only signals already
   available in this notebook (e.g. `pred_baseline`, `trend_score`, current position). Keep it to
   3–5 features — what's the smallest state that still lets the agent tell "should I flip
   position today?"
2. Define the **action space**. Is `{long, flat, short}` enough, or does allowing partial position
   sizes (as in the vol-targeted backtester) change what the agent can learn?
3. Design a **reward function** using only quantities you already compute in Section 6
   (`market_ret`, `tc`, the lagged signal). Should the reward be raw daily P&L, P&L minus
   transaction costs, or something risk-adjusted (e.g. a rolling Sharpe-like term)? Justify your
   choice — what bad behavior would a naive raw-P&L reward encourage an agent to learn?
4. One well-known failure mode in trading RL is an agent that learns to **never trade** (reward ≈
   0 is often "safe" under a naive reward). Given your reward design from (3), would this agent be
   vulnerable to that failure mode? What would you change if it were?
5. *(Optional, if you want to actually implement it)* Use `pred_baseline` and `trend_score` as a
   2-D state, discretize each into a small number of bins, and train a tabular Q-learning agent
   over the training split to choose `{-1, 0, +1}` each day. Compare its test-set Sharpe and trade
   count against your tuned Strategy 2.

This won't be graded on getting a "correct" design — it's meant to get you thinking in state /
action / reward terms ahead of next week.

---

## Resources

### Background & Refreshers

**Mastering Vectorized Backtesting in Python for Algorithmic Trading** — Covers vectorized return
computation, transaction cost drag, and drawdown calculation, mirroring the engine in Section 6.
🔗 https://onepagecode.substack.com/p/mastering-vectorized-backtesting-3d6

**How to Implement a Backtester in Python** — Compares vectorized vs. event-driven backtesting
and shows how per-trade transaction costs reduce cumulative return, relevant to Section 6.
🔗 https://medium.com/@diogomatoschaves/how-to-implement-a-backtester-in-python-030b968f6e8d

**Volatility Targeting Explained** — Background on scaling position size inversely to realized
volatility to keep portfolio risk constant, directly relevant to the `vol_scalar` TODO in the
risk-managed backtester.
🔗 https://stoffelwealth.com/volatility-targeting-a-guide-to-stabilizing-portfolio-risk/

**Dynamic Trailing Stops using ATR** — Discusses how an ATR-based trailing stop adapts to
volatility versus a fixed stop-loss, relevant to Strategy 4's trailing-stop TODO.
🔗 https://pyquantlab.medium.com/dynamic-trailing-stops-using-atr-2d3c4e95ddc0

### Research Papers

**Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach** —
Empirically compares regime-guided strategies against buy-and-hold across multiple markets,
including transaction costs, and evaluates Sharpe ratio and max drawdown improvements — directly
relevant to comparing Strategy 1 against Buy & Hold.
🔗 https://arxiv.org/html/2402.05272v2

**A Hybrid Learning Approach to Detecting Regime Switches in Financial Markets** — Builds and
evaluates trading strategies conditioned on detected market regimes, useful background for why
regime-specialist routing (Strategy 1) can outperform a single global model.
🔗 https://arxiv.org/pdf/2108.05801

### Videos

**The Sharpe Ratio Explained (by a quant trader)** — Covers the exact risk-adjusted performance
metric implemented in Section 7.
🔗 https://www.youtube.com/watch?v=9HD6xo2iO1g

**Exit Strategies With An ATR Trailing Stop** — Walks through how an ATR trailing stop is
constructed and why it only ever tightens in the favorable direction, directly relevant to the
Strategy 4 TODO.
🔗 https://www.youtube.com/watch?v=_Nt7J8xH6c4

**How To Use the ATR Trailing Stop Indicator for Entries & Exits** — Practical demonstration of
combining ATR-based stops with entry signals, useful companion for Strategy 4 and the master
trend filter.
🔗 https://www.youtube.com/watch?v=6ExCRhKD2Lo