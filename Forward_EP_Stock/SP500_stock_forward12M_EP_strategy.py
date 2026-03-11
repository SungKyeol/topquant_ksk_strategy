#pip install --upgrade "topquant-ksk[all]"
import pandas as pd
import numpy as np
import topquant_ksk
from topquant_ksk.db import DBConnection
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed
import os

load_dotenv(find_dotenv())
ID = os.getenv("DB_ID")
PW = os.getenv("DB_PW")

conn = DBConnection(db_user=ID, db_password=PW, local_host=False)
#conn.tools.check_existing_tables()

# ============================================================
# 1. 데이터 로드
# ============================================================
stock_data = conn.download.fetch_timeseries_table(
    table_name="public.daily_adjusted_time_series_data_stock",
    item_names=['close_pr', 'close_tr', 'close_post', 'intra_vwap_price',
                'forward_next_twelve_months_annual_eps_adjusted', 'marketcap_security','marketcap_company'],
    etf_ticker=["SPY-US"],
)

index_data = conn.download.fetch_timeseries_table(
    table_name="public.daily_adjusted_time_series_data_index",
    item_names=['close_pr', 'close_tr'],
    save_and_reload_pickle_cache=True,    
)

macro_data = conn.download.fetch_timeseries_table(
    table_name="public.macro_time_series",
    item_names=['ytm'],
    save_and_reload_pickle_cache=True,        
)

# ============================================================
# 2. 데이터 전처리
# ============================================================
# 개별종목 항목 추출
close_pr = stock_data['close_pr']
close_tr = stock_data['close_tr']
close_post = stock_data['close_post']
intra_vwap = stock_data['intra_vwap_price']
forward_eps = stock_data['forward_next_twelve_months_annual_eps_adjusted']
marketcap_security = stock_data['marketcap_security']
marketcap_company = stock_data['marketcap_company']

# SPX BM 수익률

spx_close_tr = index_data['close_tr']['SPX Index']
BM = spx_close_tr.pct_change()

# 무위험 수익률 (US 3M Bill)
rf_ytm = macro_data['ytm'].filter(like='3 Month')
rf_ytm
if rf_ytm.shape[1] == 1:
    rf_ytm = rf_ytm.iloc[:, 0]
cash_return_daily = rf_ytm / 100 / 365
cash_return_daily = cash_return_daily.reindex(close_pr.index, method='ffill')

# 수익률 계산
price_return_daily = close_pr.pct_change()
total_return_daily = close_tr.pct_change()
# total return 결측은 price return으로 대체
total_return_daily[total_return_daily.isna()] = price_return_daily

# Forward E/P (1일 shift로 look-ahead bias 방지)
EP_forward = forward_eps.shift(1) / close_post

# BM 비중 (시가총액 기반)
bm_weights = marketcap_security.div(marketcap_security.sum(axis=1), axis=0)

# ============================================================
# 3. 그리드 서치 함수 정의
# ============================================================
transaction_cost_rate = 0.0010

def process_ns_combination(N, S, param_grid, EP_forward, bm_weights,
                           price_return_daily, total_return_daily,
                           transaction_cost_rate):
    local_results_dict = {}

    middle_band = EP_forward.rolling(window=N).mean()
    std_dev = EP_forward.rolling(window=S).std()

    for K in param_grid['K']:
        upper_band = middle_band + (K * std_dev)
        lower_band = middle_band - (K * std_dev)
        percent_b = (EP_forward - lower_band) / (upper_band - lower_band)
        investment_attractiveness = percent_b - 0.5
        weighted_signal = investment_attractiveness * bm_weights

        for active_bet_multiplier in param_grid['active_bet_multiplier']:
            mean_weighted_signal = weighted_signal.mean(axis=1)
            neutralized_signal = weighted_signal.sub(mean_weighted_signal, axis=0)
            active_weight = neutralized_signal * active_bet_multiplier
            final_weight = bm_weights + active_weight
            final_weight[final_weight < 0] = 0
            final_weight = final_weight.div(final_weight.sum(axis=1), axis=0)

            for rebal_freq in param_grid['rebal_freq']:
                try:
                    start_date = '2007-09-01'
                    weight_df = final_weight.loc[start_date:].dropna(how='all')
                    if weight_df.empty:
                        continue

                    rebal_freq_str = f"{rebal_freq}W"
                    target_weights = weight_df.resample(rebal_freq_str).last()

                    pfl_ret, daily_weights, turnover = \
                        topquant_ksk.compute_daily_weights_rets_from_rebal_targets(
                            target_weights_at_rebal_time=target_weights,
                            price_return_daily=price_return_daily,
                            total_return_daily=total_return_daily,
                            transaction_cost_rate=transaction_cost_rate,
                        )

                    params_key = (N, S, K, active_bet_multiplier, rebal_freq)
                    local_results_dict[params_key] = pfl_ret
                except Exception:
                    continue
    return local_results_dict


# ============================================================
# 4. 메인: 그리드 서치 실행
# ============================================================
if __name__ == '__main__':
    param_grid = {
        'N': list(range(20, 240 * 3 + 60, 20)),
        'S': list(range(20, 240 * 3 + 60, 20)),
        'K': [1, 1.5, 2, 2.5, 3],
        'active_bet_multiplier': [0.5],
        'rebal_freq': [4],
    }

    ns_combinations = list(product(param_grid['N'], param_grid['S']))
    n_cores = max(os.cpu_count() - 1, 1)

    print(f"Starting grid search with {len(ns_combinations)} (N,S) combinations on {n_cores} cores...")

    results_list = Parallel(n_jobs=n_cores)(
        delayed(process_ns_combination)(
            N, S, param_grid, EP_forward, bm_weights,
            price_return_daily, total_return_daily, transaction_cost_rate
        ) for N, S in tqdm(ns_combinations, desc="Processing (N,S)")
    )

    print("\nMerging results...")
    final_results_dict = {}
    for d in results_list:
        final_results_dict.update(d)

    all_returns_df = pd.DataFrame(final_results_dict)
    all_returns_df.columns.names = ['N', 'S', 'K', 'active_bet_multiplier', 'rebal_freq']

    # ============================================================
    # 5. Train/Test Split + 성과 분석
    # ============================================================
    split_point = int(len(all_returns_df) * 0.7)
    all_returns_df_train = all_returns_df.iloc[:split_point]
    all_returns_df_test = all_returns_df.iloc[split_point:]

    risk_return_profile_train = topquant_ksk.get_RiskReturnProfile(
        all_returns_df_train,
        cash_return_daily.loc[all_returns_df_train.index],
        BM.loc[all_returns_df_train.index],
    )
    risk_return_profile_test = topquant_ksk.get_RiskReturnProfile(
        all_returns_df_test,
        cash_return_daily.loc[all_returns_df_test.index],
        BM.loc[all_returns_df_test.index],
    )

    # 최적 파라미터 (BM 행 제외)
    profile_no_bm = risk_return_profile_train.iloc[:-1]
    best_param_IR = profile_no_bm['Information_Ratio'].astype(float).idxmax()
    best_param_SR = profile_no_bm['Sharpe_Ratio'].astype(float).idxmax()

    print(f"\n최적 파라미터 Information Ratio: {best_param_IR}")
    print(f"최적 파라미터 Sharpe Ratio: {best_param_SR}")

    best_idx_IR = risk_return_profile_train.index.get_loc(best_param_IR)
    best_idx_SR = risk_return_profile_train.index.get_loc(best_param_SR)

    result_train = risk_return_profile_train.iloc[[best_idx_IR, best_idx_SR, -1]].T
    result_train.columns = ['Best IR', 'Best SR', 'Benchmark']

    result_test = risk_return_profile_test.iloc[[best_idx_IR, best_idx_SR, -1]].T
    result_test.columns = ['Best IR', 'Best SR', 'Benchmark']

    print("\n" + "=" * 80)
    print("              S&P500 개별종목 Forward E/P 전략 성과 보고서")
    print("=" * 80)

    print(f"\nTrain: {all_returns_df_train.index[0]:%Y-%m-%d} ~ {all_returns_df_train.index[-1]:%Y-%m-%d}")
    print(result_train)

    print(f"\nTest: {all_returns_df_test.index[0]:%Y-%m-%d} ~ {all_returns_df_test.index[-1]:%Y-%m-%d}")
    print(result_test)

    # ============================================================
    # 6. 시각화
    # ============================================================
    import matplotlib.pyplot as plt
    try:
        plt.rc('font', family='Malgun Gothic')
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False

    strategy_cum = (1 + all_returns_df[best_param_SR]).cumprod()
    bm_cum = (1 + BM.loc[strategy_cum.index]).cumprod()

    plt.figure(figsize=(14, 7))
    plt.plot(strategy_cum.index, strategy_cum, label='Strategy (Best SR)')
    plt.plot(bm_cum.index, bm_cum, label='S&P 500', linestyle='--')
    plt.title('Strategy vs Benchmark 누적 수익률', fontsize=16)
    plt.ylabel('누적 수익률')
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('cumulative_return.png')
    plt.show()

    # ============================================================
    # 7. 최신 비중 산출
    # ============================================================
    def calculate_latest_weights_and_signals(params, EP_forward, bm_weights):
        N, S, K, active_bet_multiplier, rebal_freq = params
        latest_date = EP_forward.index[-1]

        latest_ep = EP_forward.loc[latest_date]
        latest_bm = bm_weights.loc[latest_date]
        middle_band = EP_forward.rolling(window=N).mean().loc[latest_date]
        std_dev = EP_forward.rolling(window=S).std().loc[latest_date]

        upper_band = middle_band + (K * std_dev)
        lower_band = middle_band - (K * std_dev)
        percent_b = (latest_ep - lower_band) / (upper_band - lower_band)
        investment_attractiveness = percent_b - 0.5
        weighted_signal = investment_attractiveness * latest_bm

        mean_ws = weighted_signal.mean()
        neutralized_signal = weighted_signal - mean_ws
        active_weight = neutralized_signal * active_bet_multiplier
        final_weight = latest_bm + active_weight
        final_weight[final_weight < 0] = 0
        final_weight = final_weight / final_weight.sum()

        report = pd.DataFrame({
            'BM Weight(%)': (latest_bm * 100).round(2),
            'E/P(%)': (latest_ep * 100).round(3),
            'SMA E/P(%)': (middle_band * 100).round(3),
            '%B': percent_b.round(3),
            'Attractiveness': investment_attractiveness.round(3),
            'Active Weight(%)': (active_weight * 100).round(2),
            'Final Weight(%)': (final_weight * 100).round(2),
        })
        return report.sort_values(by='Active Weight(%)', ascending=False)

    latest_report = calculate_latest_weights_and_signals(best_param_SR, EP_forward, bm_weights)
    print(f"\n최신 기준일({EP_forward.index[-1]:%Y-%m-%d}) 비중 (상위 20종목):")
    print(latest_report.head(20))
