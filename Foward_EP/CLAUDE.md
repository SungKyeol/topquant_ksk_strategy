# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative investment strategy repository focused on S&P 500 sector rotation using forward E/P (Earnings/Price) spread signals. The main strategy implementation is in the `Foward_EP` directory, which performs sector rotation based on Bollinger Band-style signals derived from E/P spreads relative to the S&P 500 benchmark.

## Key Components

### Main Strategy File
- `S&P500_forward_EP_V4.py`: Core implementation of the forward E/P sector rotation strategy
  - Uses multiprocessing for grid search optimization
  - Implements Bollinger Band signals on E/P spreads
  - Includes backtesting with transaction costs
  - Generates performance metrics and visualization

### Data Files
- `data/data.csv`: FactSet time series data containing price, total return, market value, and forward EPS data
- `data/Eco.xlsx`: Economic data including US Treasury rates and USD/KRW exchange rates
- `data/data2.xlsx`, `data/data3.xlsx`: Additional data files for analysis

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
cd C:\Users\CHECK\Desktop\TopQuantKSK\topquant_ksk_strategy\Foward_EP

# Run the main strategy file
python S&P500_forward_EP_V4.py
```

### Installing Dependencies
The project requires the following Python packages:
- pandas >= 1.5.0
- numpy >= 1.20.0
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
1. **Data Loading**: Load FactSet time series and economic data
2. **Signal Generation**: Calculate E/P spreads and Bollinger Band signals
3. **Grid Search Optimization**: Parallel search across parameter combinations:
   - N: Moving average lookback period (20 to 780 days)
   - S: Standard deviation lookback period (20 to 780 days)
   - K: Bollinger Band width multiplier (1.0 to 3.0)
   - active_bet_multiplier: Position sizing multiplier
   - rebal_freq: Rebalancing frequency in weeks
4. **Backtesting**: Simulate portfolio performance with transaction costs
5. **Performance Analysis**: Calculate risk-return metrics and generate reports

### Key Functions in Main Strategy
- `compute_daily_weights_from_rebal_targets()`: Calculates daily portfolio weights accounting for intra-period drift
- `process_ns_combination()`: Processes one (N,S) parameter combination
- `calculate_latest_weights_and_signals()`: Computes current signals and weights
- `process_tunable_parameter_combination()`: Detailed processing for single parameter set
- `compute_param_partial_risk_return_profiles()`: Sensitivity analysis across parameters

### Multiprocessing Implementation
The strategy uses joblib's Parallel processing to optimize across parameter combinations efficiently, utilizing all available CPU cores minus one.

## Important Notes

- The strategy uses S&P 500 sector indices, excluding the overall S&P 500 from the investment universe
- Transaction costs are set at 0.10% (10 basis points) per turnover
- Train/test split is 70/30 for parameter optimization
- Rebalancing typically occurs weekly (4-week default)
- All weights are constrained to be non-negative (no shorting)
- The strategy implements market-cap weighted benchmark with active tilts