# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative investment strategy repository focused on S&P 500 industry rotation using momentum signals with Mean-Variance Optimization. The strategy allocates across 69 GICS Industries based on Mu_lag-day returns and Sigma_lag-day volatility, optimizing for maximum Sharpe Ratio.

## Key Components

### Main Strategy File
- `S&P500_Industry_momentum.py`: Core implementation of the industry momentum strategy
  - Uses grid search optimization across Mu_lag, Sigma_lag, and Investment Horizon
  - Implements Mean-Variance Optimization (MVO) with Sharpe Ratio maximization
  - Includes backtesting with transaction costs
  - Generates performance metrics

### Data Files
- `data/FG_PRICE.csv`: FactSet price data for GICS Industries
- `data/FG_MKT_VALUE.csv`: FactSet market value data
- `data/Eco.csv`: Economic data including US Treasury rates (for risk-free rate)

### Data Variables
- `fg_price`: Price data loaded from FG_PRICE.csv
- `cash_return_daily_BenchmarkFrequency`: Daily risk-free rate returns

### Custom Package: topquant_ksk
Located at `../../topquant-ksk/src/topquant_ksk/`, this package provides:
- `load_data.py`: Functions for loading FactSet and DataGuide data
  - `load_FactSet_TimeSeriesData()`: Loads time series data from CSV/Excel
  - `load_DataGuide_EconomicData()`: Loads economic data
- `metrics.py`: Risk and return metrics calculation
  - `get_RiskReturnProfile()`: Calculates CAGR, Sharpe, Information Ratio, MDD, etc.

## Commands

### Running the Main Strategy
```bash
# Navigate to the strategy directory
cd C:\Users\CHECK\Desktop\TopQuantKSK\topquant_ksk_strategy\Industry_momentum

# Run the main strategy file
python S&P500_Industry_momentum.py
```

### Installing Dependencies
The project requires the following Python packages:
- pandas >= 1.5.0
- numpy >= 1.20.0
- scipy (for optimization)
- tqdm
- joblib (for parallel processing)
- matplotlib (for visualization)
- topquant_ksk (local package)

To install the local topquant_ksk package:
```bash
cd C:\Users\CHECK\Desktop\TopQuantKSK\topquant-ksk
pip install -e .
```

## Architecture

### Strategy Workflow
1. **Data Loading**: Load FactSet time series data for 69 GICS Industries
2. **Return Calculation**: Calculate daily returns and excess returns (minus risk-free rate)
3. **Cumulative Return Calculation**:
   - Cumulative Excess Return: `CumExcessRetValue = (excess_ret + 1).cumprod()`
   - Cumulative Log Return: `CumLogRetValue = np.log(RET + 1).cumsum()`
4. **Covariance Calculation**: Weekly (5-day) log returns, annualized (*52)
5. **MVO Optimization**: Maximize Sharpe Ratio using scipy.optimize.minimize (SLSQP)
6. **Grid Search**: Parallel search across Mu_lag x Sigma_lag x Investment Horizon
7. **Backtesting**: Rebalancing at investment horizon intervals with transaction costs
8. **Performance Analysis**: Calculate risk-return metrics vs S&P 500 benchmark

### Key Parameters
| Parameter | Description | Range |
|-----------|-------------|-------|
| Mu_lag | Return lookback period (days) | 60-760 (20-day step) |
| Sigma_lag | Volatility lookback period (days) | 60-760 (20-day step) |
| Investment Horizon | Rebalancing frequency (days) | [20, 40, 60] |
| Transaction Cost | Cost per turnover | 0.15% (15bp) |
| Train/Test Split | Data split for optimization | 70/30 |

### Mean-Variance Optimization Details

#### Sharpe Ratio Calculation
```
Sharpe = ExcessLogReturn_annual / Vol_annual
```

#### Annualized Excess Log Return
```python
ret_series = ExcessRet_InvestmentHorizon @ Weight
ExcessRet_plusOne = (ret_series + 1).prod() ** ((252/investment_horizon) / len(ret_series))
ExcessLogRet_annual = np.log(ExcessRet_plusOne)
```

#### Annualized Volatility
```python
Vol_annual = np.sqrt(Weight.T @ VarCov_annual @ Weight)
# VarCov_annual = np.cov(LogRet_weekly.T) * 52
```

#### Optimization Setup
- **Method**: scipy.optimize.minimize with SLSQP
- **Objective**: Minimize -Sharpe(Weight) (equivalent to maximizing Sharpe)
- **Constraints**: sum(Weight) = 1
- **Bounds**: weight_i >= 0 for all i (Long Only)

### Key Functions

#### MVO Optimization Functions
```python
def port_ExcessLogReturn_annual_from_ExcessReturn_N_Weight(Weight, ExcessRet, horizon):
    """Calculate annualized excess log return for portfolio"""
    ret_series = ExcessRet @ Weight
    ExcessRet_plusOne = (ret_series + 1).prod() ** ((252/horizon) / len(ret_series))
    return np.log(ExcessRet_plusOne)

def port_vol_annual(Weight, VarCov_annual):
    """Calculate annualized portfolio volatility"""
    return np.sqrt(Weight.T @ VarCov_annual @ Weight)

def Sharpe(Weight, ExcessRet, horizon, VarCov_annual):
    """Calculate Sharpe Ratio"""
    return port_ExcessLogReturn_annual(...) / port_vol_annual(...)

def Get_Sharpe_Maximizing_Weight(bounds, Initial_Weight, ExcessRet, horizon, VarCov):
    """Optimize portfolio weights to maximize Sharpe Ratio using SLSQP"""
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    result = minimize(lambda w: -Sharpe(w, ...), Initial_Weight,
                      bounds=bounds, constraints=cons, method='SLSQP')
    return result.x
```

#### Data Processing Functions
- Calculate daily returns: `RET_DAILY = price.pct_change()`
- Calculate excess returns: `excess_ret = (RET_DAILY.T - Rf_DailyRet).T`
- Calculate weekly log returns for covariance: `LogRet_weekly = CumLogRetValue[::5].diff()`

### Multiprocessing Implementation
The strategy uses joblib's Parallel processing to optimize across (Mu_lag, Sigma_lag, Investment Horizon) parameter combinations efficiently, utilizing all available CPU cores minus one.

## Important Notes

- The strategy uses S&P 500 GICS Industries (69 industries) as the investment universe
- No screener applied - all 69 industries are eligible for investment
- Benchmark is S&P 500 Index
- Transaction costs are set at 0.15% (15bp) per turnover
- Train/test split is 70/30 for parameter optimization
- Rebalancing occurs at investment horizon intervals (20, 40, or 60 days)
- All weights are constrained to be non-negative (Long Only)
- No individual weight cap applied
- Risk-free rate is sourced from US Treasury rates in economic data
- Covariance matrix is calculated from weekly (5-day) log returns and annualized (*52)

## Investment Universe

S&P 500 GICS Industries classification:
- 11 Sectors -> 24 Industry Groups -> 69 Industries
- Examples: Aerospace & Defense, Airlines, Banks, Biotechnology, etc.
