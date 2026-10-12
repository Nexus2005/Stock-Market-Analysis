# 📈 Indian Stock Market (2000–2020): Ups & Downs Analysis and Future Return Prediction

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook_Running-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/omkarahirrao/indian-market-stock-returns-analysis-2000-2020)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()

> **Applied Financial Data Science, Regime Modeling & Quantitative Machine Learning Mini-Project**  
> Analyzing two decades of Indian market turbulence, structural growth, volatility clustering, and predicting future stock returns using machine learning.

---

## 📌 Quick Access & Links

* 🌐 **Live Kaggle Notebook:** [Kaggle - Indian Market Stock Returns Analysis (2000–2020)](https://www.kaggle.com/code/omkarahirrao/indian-market-stock-returns-analysis-2000-2020)
* 📄 **Complete PDF Report:** [`Indian_Market_Stock_Returns_Analysis_Report.pdf`](./Indian_Market_Stock_Returns_Analysis_Report.pdf)
* 📓 **Jupyter Notebook File:** [`indian_market_stock_returns_analysis.ipynb`](./indian_market_stock_returns_analysis.ipynb)
* 📂 **Dataset Source:** [Kaggle - sagara9595/stock-data](https://www.kaggle.com/datasets/sagara9595/stock-data)

---

## 📖 Project Overview

Between **2000 and 2020**, the Indian equity market experienced extraordinary wealth creation alongside violent macroeconomic shocks:
1. **The Dot-Com Aftermath & Structural Reforms (2000–2003)**
2. **The Golden Bull Run (2003–2007)** (High GDP growth and industrial surge)
3. **The Global Financial Crisis Subprime Crash (2008)** (-58% market plunge over 14 months)
4. **Post-Crisis Recovery & High-Inflation Consolidation (2009–2013)**
5. **The Structural Expansion Era (2014–2019)** (Modi administration, Demonetization, GST reforms)
6. **The COVID-19 Flash Crash & V-Shaped Rebound (Q1 2020)** (-38% crash in 35 trading days)

This project conducts an exhaustive quantitative analysis of the **"ups and downs"** (cyclical regimes, drawdowns, volatility clustering, fat tails) across India's largest sectoral bellwethers and trains **machine learning models** (Regression & Classification) to forecast future returns, validated with an out-of-sample **algorithmic trading backtest**.

---

## 🏛️ Assets Analyzed (Over 5,000 Trading Days Each)

* **Reliance Industries (`RELIANCE.NS`)** - Energy, Telecom & Retail Conglomerate
* **Infosys (`INFY.NS`)** & **Tata Consultancy Services (`TCS.NS`)** - Global IT Services
* **HDFC Bank (`HDFCBANK.NS`)** - Private Banking & Financial Services
* **State Bank of India (`SBIN.NS`)** - Public Sector Banking
* **Tata Motors (`TATAMOTORS.NS`)** - Automotive & Global Manufacturing
* **ITC Ltd (`ITC.NS`)** - FMCG & Consumer Staples
* **Larsen & Toubro (`LT.NS`)** - Infrastructure & Capital Goods
* **`NIFTY_COMPOSITE`** - Equal-Weighted Indian Market Proxy (Jan 3, 2000 = 100)

---

## 📊 1. 20-Year Historical Risk & Return Profile (2000–2020)

| Asset / Index | Sector | Start Price (₹) | End Price (₹) | Total Return (%) | CAGR (%) | Annualized Vol (%) | Sharpe Ratio ($R_f=6\%$) | Max Drawdown (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | Energy / Telecom | 53.40 | 1,579.95 | **+2,858.7%** | 18.06% | 31.84% | **0.38** | -59.85% |
| **INFY** | IT Services | 127.02 | 708.85 | **+458.1%** | 8.80% | 35.61% | **0.08** | -60.84% |
| **HDFCBANK** | Private Banking | 17.00 | 1,029.50 | **+5,955.9%** | 22.37% | 29.56% | **0.55** | -42.87% |
| **SBIN** | PSU Banking | 24.34 | 187.80 | **+671.6%** | 10.51% | 39.46% | **0.11** | -66.12% |
| **TATAMOTORS** | Automotive | 33.20 | 112.50 | **+238.9%** | 6.16% | 46.47% | **0.00** | -86.53% |
| **ITC** | FMCG / Staples | 16.33 | 195.95 | **+1,100.0%** | 12.94% | 29.74% | **0.23** | -43.15% |
| **NIFTY_COMPOSITE** | Equal-Weighted Proxy | 100.00 | 1,385.20 | **+1,285.2%** | **13.73%** | **23.36%** | **0.33** | **-58.21%** |

---

## 📉 2. Market Ups & Downs: Crisis Anatomy & Tail Risk

### 2008 Lehman GFC vs. 2020 COVID Flash Crash
* **2008 Global Financial Crisis:** Slow-motion, protracted 14-month credit contraction. Peak-to-trough collapse of **-58.2%**, taking ~3 years to recover.
* **2020 COVID Flash Crash:** Violent panic. The market dropped **-38.1% in just 35 trading days**, followed by an unprecedented liquidity-driven V-shaped rebound.
* **Mandelbrot Volatility Clustering:** 30-day annualized rolling volatility spiked above **65%** during both crisis epochs.

### Return Distribution & Non-Gaussian Fat Tails
* **Daily Return Skewness:** `-0.31` (Strong negative downside asymmetry).
* **Excess Kurtosis:** `5.42` (High leptokurtosis; extreme multi-sigma events occur far more frequently than normal Gaussian models predict).
* **1-Day 95% Historical Value-at-Risk (VaR):** `2.15%`
* **1-Day 99% Historical Value-at-Risk (VaR):** `4.32%`
* **99% Conditional VaR (Expected Shortfall / CVaR):** `5.84%`

---

## 🛠️ 3. Quantitative Feature Engineering

To prevent lookahead bias and domain shift over 20 years of price appreciation, all features are engineered to be **stationary and scale-invariant**:
1. **Trend Deviations:** Distance from 20-day, 50-day, and 200-day SMAs:
   $$\text{Dist}_{\text{SMA}} = \frac{\text{Close} - \text{SMA}}{\text{SMA}}$$
2. **Golden / Death Cross Signal:**
   $$\text{Spread} = \frac{\text{SMA}_{50} - \text{SMA}_{200}}{\text{SMA}_{200}}$$
3. **Momentum Indicators:**
   * 14-Day Relative Strength Index (`RSI_14`)
   * Normalized MACD:
     $$\text{MACD}_{\text{norm}} = \frac{\text{EMA}_{12} - \text{EMA}_{26}}{\text{Close}}$$
4. **Volatility Indicators:**
   * Bollinger Bandwidth & Bollinger %B
   * 20-Day Annualized Historical Volatility (`Hist_Vol_20d`)
5. **Autoregressive Lags:** Historical returns over 1, 2, 3, 5, 10, and 21 trading days.
6. **Target Variables:**
   * **Task A (Regression):** 5-day continuous forward return (`Target_Return_5d`)
   * **Task B (Classification):** 5-day market regime direction (`Target_Direction_5d`: binary 1 for Up, 0 for Down)

---

## 🤖 4. Machine Learning Model Evaluation

### Validation Setup (Zero Lookahead Leakage)
* **Training Set (2000–2015):** 22,553 samples (15 years)
* **Validation Set (2016–2017):** 2,964 samples (2 years)
* **Out-of-Sample Test Set (2018–2020):** 3,528 samples (includes the COVID crash)

### Task A: Future Return Prediction (Regression)
| Model | Test RMSE | Test MAE | Test $R^2$ | Directional Sign Match (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (L2)** | 0.0558 | 0.0381 | -0.012 | 51.4% |
| **Random Forest Regressor** | **0.0549** | **0.0375** | **+0.018** | **52.8%** |
| **HistGradientBoosting Regressor** | 0.0562 | 0.0389 | -0.026 | 50.9% |

### Task B: Market Direction Prediction (Classification: Up vs Down)
| Model | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | 51.2% | 52.8% | 61.4% | 56.8% | 51.8% |
| **Random Forest Classifier** | **52.4%** | **54.1%** | **63.8%** | **58.5%** | **53.2%** |
| **HistGradientBoosting Classifier** | 50.8% | 52.3% | 58.7% | 55.3% | 51.1% |

### Top Predictive Features (Feature Importance)
1. `Dist_SMA200` (Distance from 200-Day SMA - Macro Trend Regime)
2. `RSI_14` (Momentum Mean-Reversion Exhaustion)
3. `Hist_Vol_20d` (Volatility Regime Indicator)
4. `Golden_Cross_Spread` (Medium vs Long-Term Moving Average Spread)

---

## 💰 5. Algorithmic Trading Strategy Backtest (2018–2020)

We simulated an algorithmic execution strategy: **Long when the model predicts Up**, and **Cash (0% risk) when the model predicts Down**, incorporating realistic 10 bps (0.10%) transaction friction:

| Metric | Passive Buy & Hold Benchmark | ML Risk-Managed Strategy | Strategic Impact |
| :--- | :---: | :---: | :---: |
| **Total Cumulative Return** | +71.7% | +49.7% | Healthy positive returns |
| **Annualized Volatility** | 33.2% | **18.6%** | **-44% Volatility Reduction** |
| **Sharpe Ratio ($R_f=6\%$)** | 0.58 | **0.78** | **+34.5% Higher Risk-Adjusted Alpha** |
| **Maximum Drawdown (COVID Crash)** | **-45.1%** | **-15.3%** | **Protected Capital (66% Drawdown Reduction)** |

---

## 📂 Repository Structure

```
├── .gitignore                                         # Ignores large raw dataset CSVs & caches
├── README.md                                          # Complete project documentation
├── Indian_Market_Stock_Returns_Analysis_Report.pdf    # Full rendered project report PDF
├── vertopal.com_indian_market_stock_returns_analysis.pdf # Original PDF export
├── indian_market_stock_returns_analysis.ipynb         # Fully executed Jupyter Notebook
├── build_full_notebook.py                             # Automated notebook reproduction pipeline
└── kaggle_kernel/                                     # Kaggle kernel configuration
    └── kernel-metadata.json                           # Kaggle deployment metadata
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Nexus2005/Stock-Market-Analysis.git
cd Stock-Market-Analysis
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter kaggle
```

### 3. Download the Dataset
```bash
kaggle datasets download -d sagara9595/stock-data --unzip -p data
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook indian_market_stock_returns_analysis.ipynb
```

---

## 👤 Author
**Omkar Ahirrao**  
* GitHub: [@Nexus2005](https://github.com/Nexus2005)  
* Kaggle: [@omkarahirrao](https://www.kaggle.com/omkarahirrao)
