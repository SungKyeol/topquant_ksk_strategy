import pandas as pd
import numpy as np
import topquant_ksk
from scipy.optimize import minimize
import sys
import importlib
import os

# tools 모듈을 동적으로 import하고 리로드하는 함수
def reload_tools():
    try:
        if 'tools' in sys.modules:
            importlib.reload(sys.modules['tools'])
        else:
            import topquant_ksk
        return True
    except Exception as e:
        print(f"Error reloading tools module: {e}")
        return False

reload_tools()
from topquant_ksk import *

# MVO 최적화 헬퍼 함수들 (원본 파일에서 복사)
def port_ExcessLogReturn_annual_from_ExcessReturn_N_Weight(initial_weights, ExcessRet_InvestmentHorizon, Mu_lag):
    ret_series = ExcessRet_InvestmentHorizon @ initial_weights
    cum_ret = (ret_series + 1).prod()
    annual_ret = cum_ret ** (252 / Mu_lag)
    ExcessLogRet = np.log(annual_ret)
    return ExcessLogRet

def port_vol_annual(Weight, VarCov_annual):
    return np.sqrt(Weight.T @ VarCov_annual @ Weight)

def Sharpe(Weight, ExcessRet_InvestmentHorizon, Mu_lag, VarCov_annual):
    return port_ExcessLogReturn_annual_from_ExcessReturn_N_Weight(
        Weight, ExcessRet_InvestmentHorizon, Mu_lag
    ) / port_vol_annual(Weight, VarCov_annual)

def Get_Sharpe_Maximizing_Weight(bound_list_of_tuple, Initial_Weight, ExcessRet_InvestmentHorizon, Mu_lag, VarCov_annual):
    cons = [
        {'type': 'eq', 'fun': lambda Weight: Weight.sum() - 1},
    ]

    Sharpe_Maximizing_result = minimize(
        lambda Weight: -Sharpe(Weight,
                               ExcessRet_InvestmentHorizon=ExcessRet_InvestmentHorizon,
                               Mu_lag=Mu_lag,
                               VarCov_annual=VarCov_annual),
        Initial_Weight,
        bounds=bound_list_of_tuple,
        constraints=cons,
        method='SLSQP',
        options={'disp': False}
    )

    Sharpe_Maximizing_Weight = Sharpe_Maximizing_result.x
    return Sharpe_Maximizing_Weight

def run_single_backtest(params, PRICE_DAILY, CumExcessRetValue_DAILY,
                        CumLogRetValue_DAILY, all_assets,
                        number_of_all_assets, sample_initial_drop, transaction_cost):
    """
    단일 파라미터 조합에 대한 백테스트를 실행합니다.
    """
    investment_horizon, Sigma_lag, Mu_lag = params
    param_key = (investment_horizon, Sigma_lag, Mu_lag)

    # 리밸런싱 날짜 설정
    rebalancing_dates = PRICE_DAILY.index[sample_initial_drop::investment_horizon]

    price_at_rebalancing_freq = PRICE_DAILY.loc[rebalancing_dates]
    ret_xM_realized_in_rebalancing_freq = price_at_rebalancing_freq.shift(-1) / price_at_rebalancing_freq - 1

    # 비중 초기화 (전체 자산 기준)
    weight_at_rebalancing_freq = np.zeros((len(rebalancing_dates), number_of_all_assets))

    for th, date in enumerate(rebalancing_dates):
        date_location_in_TradingDate = PRICE_DAILY.index.get_loc(date)

        # Mu, Sigma 기간 시작점 계산
        Mu_location_start = date_location_in_TradingDate - Mu_lag
        Sigma_location_start = date_location_in_TradingDate - Sigma_lag

        # 가격 데이터 스크리닝: Mu_lag + Sigma_lag 기간 동안 가격이 모두 있는 Industry만 선별
        lookback_start = min(Mu_location_start, Sigma_location_start)
        lookback_start_date = PRICE_DAILY.index[lookback_start]
        price_slice = PRICE_DAILY.loc[lookback_start_date:date]
        valid_assets_mask = price_slice.notna().all(axis=0)
        investing_assets = valid_assets_mask[valid_assets_mask].index.tolist()
        number_of_investing_assets = len(investing_assets)

        # 초기 비중 설정 (투자 가능 자산에 동일 비중)
        individual_weight = 1 / number_of_investing_assets
        Initial_Weight = np.full(shape=number_of_investing_assets, fill_value=individual_weight)

        # Bounds: Long Only, 비중 상한 없음 (0 ~ 1)
        bound_list_of_tuple = list(zip(
            np.zeros(number_of_investing_assets),
            np.ones(number_of_investing_assets)
        ))

        # Mu 기간 초과수익률 계산
        Mu_start_date = PRICE_DAILY.index[Mu_location_start]

        # investment_horizon 간격으로 슬라이싱 후, 마지막 날짜(date)가 포함 안 됐으면 추가
        CumExcessRetValue_sliced = CumExcessRetValue_DAILY.loc[Mu_start_date:date:investment_horizon, investing_assets]
        if CumExcessRetValue_sliced.index[-1] != date:
            last_row = CumExcessRetValue_DAILY.loc[[date], investing_assets]
            CumExcessRetValue_sliced = pd.concat([CumExcessRetValue_sliced, last_row])
        CumExcessRetValue_sliced = CumExcessRetValue_sliced.values

        # 슬라이스가 1개뿐이면 시작점과 끝점으로 1개 수익률 계산
        if len(CumExcessRetValue_sliced) < 2:
            start_value = CumExcessRetValue_DAILY.loc[Mu_start_date, investing_assets].values
            end_value = CumExcessRetValue_DAILY.loc[date, investing_assets].values
            ExcessRet_InvestmentHorizon = ((end_value / start_value) - 1).reshape(1, -1)
        else:
            ExcessRet_InvestmentHorizon = CumExcessRetValue_sliced[1:] / CumExcessRetValue_sliced[:-1] - 1

        # Sigma 기간 공분산 계산 (주간 로그수익률 기반)
        Sigma_start_date = PRICE_DAILY.index[Sigma_location_start]
        CumLogRetValue_sliced_Weekly = CumLogRetValue_DAILY.loc[Sigma_start_date:date:5, investing_assets].values

        if len(CumLogRetValue_sliced_Weekly) < 2:
            weight_at_rebalancing_freq[th, :] = 1 / number_of_all_assets
            continue

        LogRet_weekly_sliced = CumLogRetValue_sliced_Weekly[1:] - CumLogRetValue_sliced_Weekly[:-1]
        VarCov_annual = np.cov(LogRet_weekly_sliced.T) * 52

        # MVO 최적화
        try:
            optimal_weight = Get_Sharpe_Maximizing_Weight(
                bound_list_of_tuple, Initial_Weight,
                ExcessRet_InvestmentHorizon, Mu_lag, VarCov_annual
            )
            # 최적화된 비중을 전체 자산 배열에 매핑
            for idx, asset in enumerate(investing_assets):
                asset_idx = all_assets.index(asset)
                weight_at_rebalancing_freq[th, asset_idx] = optimal_weight[idx]

        except Exception as e:
            # 최적화 실패 시 동일 비중
            weight_at_rebalancing_freq[th, :] = 1 / number_of_all_assets

    # 비중 정규화
    weight_input_array = weight_at_rebalancing_freq.round(3).copy()
    row_sums = weight_input_array.sum(axis=1).reshape(-1, 1)
    row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
    weight_input_array = weight_input_array / row_sums

    # 수익률 계산
    return_before_TransactionCost = weight_input_array * ret_xM_realized_in_rebalancing_freq.values
    pfl_return_before_TransactionCost = np.nansum(return_before_TransactionCost, axis=1)[:-1]

    # 거래비용 계산 (가격 변동으로 drift된 비중 기준)
    ret_values = ret_xM_realized_in_rebalancing_freq.values
    drifted_weight = weight_input_array[:-1] * (1 + ret_values[:-1])
    drifted_weight_sum = np.nansum(drifted_weight, axis=1, keepdims=True)
    drifted_weight_sum[drifted_weight_sum == 0] = 1  # 0으로 나누기 방지
    drifted_weight = drifted_weight / drifted_weight_sum

    # turnover = |새 목표 비중 - drift된 비중|
    turnover_at_rebalancing = np.abs(weight_input_array[1:] - drifted_weight).sum(axis=1)
    turnover_at_rebalancing = np.concatenate([[1], turnover_at_rebalancing])  # 첫 번째는 100% 투자

    rebalancing_cost = turnover_at_rebalancing * transaction_cost
    rebalancing_cost_multiplier = 1 - rebalancing_cost

    # 거래비용 차감 후 수익률
    pfl_return_after_TransactionCost = pfl_return_before_TransactionCost * rebalancing_cost_multiplier[:-1]

    return (param_key, pfl_return_after_TransactionCost, turnover_at_rebalancing[:-1])

#################################################################################
# 메인 테스트 코드
#################################################################################

print("="*60)
print("run_single_backtest 함수 테스트")
print("파라미터: (investment_horizon=20, Sigma_lag=240, Mu_lag=240)")
print("="*60)

# 데이터 로드
print("\n[1/5] 데이터 로드 중...")
fg_price = topquant_ksk.load_FactSet_TimeSeriesData(
    filename='FG_PRICE.csv',
    column_spec=['Symbol Name','Symbol','Symbol Type','Asset Type','Item Name'],
    sheet_name=None
)
fg_price = fg_price.ffill()

economic_data = topquant_ksk.load_FactSet_TimeSeriesData(
    filename='Eco.csv',
    column_spec=['Symbol Name','Symbol','Symbol Type','Asset Type','Item Name'],
    sheet_name=None
)

# rf
rawdata_rf = economic_data[('US Benchmark Bond - 1 Year', 'TRYUS1Y-FDS', 'Yield', 'Bond', 'FG_YIELD')]/100/365
rawdata_rf_pfl_value = (rawdata_rf+1).cumprod()
rawdata_rf_pfl_value = rawdata_rf_pfl_value.reindex(fg_price.index, method='ffill')
cash_return_daily_BenchmarkFrequency = rawdata_rf_pfl_value.pct_change(fill_method=None).dropna()

print("데이터 로드 완료!")

# 데이터 전처리
print("\n[2/5] 데이터 전처리 중...")
PRICE_DAILY = fg_price.copy().drop(['S&P 500'], axis=1, level=0)
PRICE_DAILY.columns = PRICE_DAILY.columns.get_level_values(0)
RET_DAILY = PRICE_DAILY.pct_change()
Rf_DailyRet = cash_return_daily_BenchmarkFrequency

# 초과수익률 계산
excess_ret = (RET_DAILY.T - Rf_DailyRet).T
CumExcessRetValue_DAILY = (excess_ret + 1).cumprod()
CumExcessRetValue_DAILY[np.isnan(CumExcessRetValue_DAILY) & ~np.isnan(CumExcessRetValue_DAILY.shift(-1))] = 1

# 누적 로그수익률 계산
CumLogRetValue_DAILY = np.log(RET_DAILY + 1).cumsum()
CumLogRetValue_DAILY[np.isnan(CumLogRetValue_DAILY) & ~np.isnan(CumLogRetValue_DAILY.shift(-1))] = 0

all_assets = PRICE_DAILY.columns.tolist()
number_of_all_assets = len(all_assets)

print(f"전처리 완료! 자산 개수: {number_of_all_assets}")

# 파라미터 설정
print("\n[3/5] 파라미터 설정...")
transaction_cost = 0.0015
Mu_lag_max = 760
Sigma_lag_max = 760
sample_initial_drop = max(Mu_lag_max, Sigma_lag_max)

# 테스트 파라미터
test_params = (20, 240, 240)  # (investment_horizon, Sigma_lag, Mu_lag)
print(f"테스트 파라미터: investment_horizon={test_params[0]}, Sigma_lag={test_params[1]}, Mu_lag={test_params[2]}")

# 백테스트 실행
print("\n[4/5] 백테스트 실행 중...")
import time
start_time = time.time()

param_key, pfl_return, turnover = run_single_backtest(
    test_params, PRICE_DAILY, CumExcessRetValue_DAILY,
    CumLogRetValue_DAILY, all_assets,
    number_of_all_assets, sample_initial_drop, transaction_cost
)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"백테스트 완료! (실행 시간: {elapsed_time:.2f}초)")

# 결과 출력
print("\n[5/5] 결과 분석...")
print("="*60)
print(f"파라미터 키: {param_key}")
print(f"수익률 배열 길이: {len(pfl_return)}")
print(f"Turnover 배열 길이: {len(turnover)}")
print()
print(f"수익률 통계:")
print(f"  - 평균: {np.mean(pfl_return):.6f}")
print(f"  - 표준편차: {np.std(pfl_return):.6f}")
print(f"  - 최소값: {np.min(pfl_return):.6f}")
print(f"  - 최대값: {np.max(pfl_return):.6f}")
print(f"  - 누적 수익률: {((1 + pfl_return).prod() - 1):.4%}")
print()
print(f"Turnover 통계:")
print(f"  - 평균: {np.mean(turnover):.6f}")
print(f"  - 최소값: {np.min(turnover):.6f}")
print(f"  - 최대값: {np.max(turnover):.6f}")
print("="*60)
print("\n테스트 성공!")
