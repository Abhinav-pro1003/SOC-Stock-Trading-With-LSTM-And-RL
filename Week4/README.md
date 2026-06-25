# Week 4 — LSTM-Based Stock Return Prediction

> [!IMPORTANT]
> **Submission Link:** Please submit your completed notebooks and reports via the following form:
> 🔗 [Google Form Submission Link](https://docs.google.com/forms/d/e/1FAIpQLSf7VzlNfNLZRnMtSotaUnppHl760Z8PYj4lz7W4im-lar8gGg/viewform?usp=header)

This assignment walks you through building a full deep-learning pipeline for stock market direction forecasting. You will engineer features from raw OHLCV data, design and train an LSTM model with temporal attention, extend it with market-regime awareness, and critically evaluate the results through visualisation. The three notebooks must be run in order.

---

## Notebooks & Pipeline Structure

### 1. `Baseline_lstm_01.ipynb` (Feature Engineering & Baseline LSTM)
The entry point of the pipeline. It handles feature preprocessing and establishes a baseline return forecasting model.
- **Data Preprocessing & Indicator Calculation:** Checks the input dataset `data/processed/NVDA_features.csv` (which contains date, open, high, low, close, and volume). If any of the required indicator columns are missing, they are automatically computed and written back to the same CSV. Indicator calculations include:
  - **RSI (14-period):** Relative Strength Index scaled to `[0, 1]`.
  - **MACD Family:** MACD line (EMA12 − EMA26), signal line (9-period EMA of MACD), and their difference (histogram).
  - **Bollinger Bands:** Upper, lower, band width, and `%B` based on a 20-day SMA and 2-sigma deviation.
  - **Stochastic Oscillator (14, 3):** `%K` and `%D` smoothed, scaled to `[0, 1]`.
  - **On-Balance Volume (OBV):** Cumulative OBV and OBV percentage change.
  - **VWAP:** Cumulative volume-weighted average price and its ratio to close.
- **Sequence Construction & Winsorisation:** Implements custom data preprocessing functions:
  - `make_sequences()`: Converts 2D scaled feature arrays and target labels into overlapping sequence windows of size `SEQUENCE_LENGTH = 5` (e.g., shape `(N, seq_len, n_features)`).
  - `winsorise()`: Symmetrically clips extreme return targets to `±3.0` standard deviations on the training set to prevent outlier gradients.
- **Model Architecture:** Implements a custom neural network using PyTorch:
  - `TemporalAttention`: A time-step attention mechanism that computes attention weights using a linear layer and applies softmax along the time dimension to obtain a weighted context vector from LSTM hidden states.
  - `ReturnSequenceModel`: Integrates the recurrent layer (supports LSTM, GRU, and BiLSTM architectures), temporal attention, layer normalization, GELU activation, and a small regression head.
- **Loss Function:** Implements a custom `DirectionWeightedLoss` combining:
  - A weighted Huber loss (using balanced inverse-frequency weights for Up/Down classes to penalize prediction magnitude errors).
  - A weighted Binary Cross Entropy (BCE) loss on predicted returns scaled by a temperature parameter to penalize incorrect direction classification.
- **Outputs:** Saves the trained baseline weights to `models/baseline_lstm.pt`, the fitted scaler to `models/baseline_scaler.pkl`, and evaluation metrics/history to `reports/baseline_results.pkl`.

### 2. `Regime_detection_02.ipynb` (Regime Detection & Training)
Extends the baseline model by dividing market conditions into distinct regimes and training specialist models tailored to each regime's characteristics.
- **Market Regime Classification:** Implements the `classify_regimes()` function to partition data into:
  - `uptrend` and `downtrend`: Momentum-driven regimes identified via the relationship between close price and 50-day SMA ratio, confirmed by MACD signal directions.
  - `high_volatility`: Flagged when 10-day realized volatility (rolling standard deviation of returns) exceeds a specified percentile (e.g., 75th percentile).
  - `sideways`: The fallback regime for range-bound markets.
  - *Smoothing:* Applies a majority-vote rolling filter (e.g., 5-day window) to ensure regime assignments are contiguous.
- **Custom Architectures per Regime:** Trains specialized models mapping to specific dynamics:
  - **LSTM:** Utilized for `uptrend` and `downtrend` to carry long-term directional momentum.
  - **GRU:** Used for `high_volatility` to ensure faster adaptation with fewer parameters.
  - **BiLSTM:** Applied to `sideways` to capture both forward and backward temporal dependencies within the sequence window, modeling mean-reverting behavior.
- **Handling Small-Data Regimes:**
  - `transfer_from_base()`: Copies compatible recurrent weights from the baseline model and optionally freezes recurrent parameters to perform head-only fine-tuning when training data is limited (e.g., under 150 samples).
  - `kfold_train()`: Conducts non-shuffled 5-fold cross-validation on time-series slices for small regimes (e.g., under 400 samples) to monitor performance stability.
- **Outputs:** Saves regime-specific model weights to `models/` and consolidated histories, metrics, and cross-validation results to `reports/regime_results.pkl`.

### 3. `Visual_plots_03.ipynb` (Visualization & Evaluation)
A read-only evaluation notebook that compares all models to determine whether the extra complexity of the regime-specific approach is justified. It loads the results from `reports/` and generates:
- **Loss Curves:** Compares training and validation loss per model over training epochs.
- **Validation Accuracy Curves:** Tracks validation directional accuracy across training epochs, overlaying 5-fold CV error bands for small-data models.
- **Confusion Matrices:** Displays 2x2 matrices of Up vs. Down predictions for every model.
- **Model Comparison Charts:** Grouped horizontal bar charts comparing Direction Accuracy, F1 Score, and Test RMSE against the full-dataset baseline model.
- **Return Scatter Plots & Rolling Accuracy:** Plots predicted vs. actual returns alongside a rolling accuracy timeline to assess stability.
- **Directional Bias Assessment:** Grouped bar chart comparing the total count of actual "Up" days against predicted "Up" days for each model.
- **Outputs:** Saves diagnostic figures (`fig_loss_curves.png`, `fig_accuracy_curves.png`, `fig_confusion_matrices.png`, `fig_baseline_vs_regime.png`, `fig_prediction_quality.png`, and `fig_actual_vs_predicted_up.png`) into `reports/`.

---

## Resources

### Tutorials & Notebooks

**Stock Market Analysis & Prediction using LSTM** — Kaggle notebook covering end-to-end EDA, feature preparation, and an LSTM price-prediction model in Python. Good reference for the data-loading and plotting patterns used in notebook 1.
🔗 https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm

**LSTM for Stock Market Prediction** — DataCamp step-by-step tutorial on building and training an LSTM network for sequential financial data, including data normalisation and sequence windowing.
🔗 https://www.datacamp.com/tutorial/lstm-python-stock-market

---

### Research Papers

**LSTM in Financial Forecasting** *(Borsa Istanbul Review, ScienceDirect)* — Peer-reviewed study evaluating LSTM architectures on financial time-series forecasting tasks, discussing sequence length, feature selection, and evaluation metrics relevant to notebooks 1 and 2.
🔗 https://www.sciencedirect.com/science/article/abs/pii/S2452306223000916

**Robust Anomaly Detection in Financial Markets Using LSTM Autoencoders and GANs** *(Preprints, 2025)* — Proposes a hybrid LSTM Autoencoder + GAN framework for detecting sudden regime shifts and structural breaks in daily return data — directly relevant to the regime-detection ideas in notebook 2.
🔗 https://www.preprints.org/manuscript/202506.1907

---

### Videos

**Market Regime Detection using Statistical and ML-Based Approaches** — Conference talk (AI in Finance, Texas State University) by Jason Ramchandani (LSEG London) covering the theory and practical implementation of regime classification, directly mapping to the `classify_regimes()` task in notebook 2.
🔗 https://www.youtube.com/watch?v=-53N3EFl4Ic

**Regime Switching Models with Machine Learning** — UCL PhD talk by Piotr Pomorski on combining Hidden Markov Models and ML classifiers for market regime switching — useful background for understanding why a single model underperforms across all regimes.
🔗 https://www.youtube.com/watch?v=t4iRLM3nDUk

**Stock Trend Prediction Using Python, LSTM & Flask** — Full project walkthrough (Project 44) building an LSTM-based stock trend predictor with a Flask front-end. Good practical companion to notebook 1, covering model training, saving, and serving.
🔗 https://www.youtube.com/watch?v=7dnBxHbVMV8

**Stock Price Prediction in Python with PyTorch — Full Tutorial** — Hands-on tutorial implementing an LSTM stock predictor in PyTorch, covering sequence construction, training loops, and evaluation — closely mirrors the architecture you will implement in notebooks 1 and 2.
🔗 https://www.youtube.com/watch?v=IJ50ew8wi-0