import pandas as pd
import numpy as np
from tqdm import tqdm # 루프 진행상황 표시
from itertools import product # 조합 생성
from joblib import Parallel, delayed # 병렬 처리
import os             # 운영체제 제어
from scipy.optimize import minimize  # MVO 최적화
import topquant_ksk


fg_price=topquant_ksk.load_FactSet_TimeSeriesData(
    filename='FG_PRICE.csv',
    column_spec=['Symbol Name','Symbol','Symbol Type','Asset Type','Item Name'],
    sheet_name=None  # CSV 파일이므로 sheet_name을 None으로 설정
) # 데이터 로드
fg_price=fg_price.ffill()


# 지수, rf 데이터 로드
economic_data=topquant_ksk.load_FactSet_TimeSeriesData(
    filename='Eco.csv',
    column_spec=['Symbol Name','Symbol','Symbol Type','Asset Type','Item Name'],
    sheet_name=None  # CSV 파일이므로 sheet_name을 None으로 설정
) # 데이터
#rf
rawdata_rf=economic_data[('US Benchmark Bond - 1 Year', 'TRYUS1Y-FDS', 'Yield', 'Bond', 'FG_YIELD')]/100/365
rawdata_rf_pfl_value=(rawdata_rf+1).cumprod()                 #누적 수익률 계산
rawdata_rf_pfl_value=rawdata_rf_pfl_value.reindex(fg_price.index,method='ffill') #인덱스 맞추기 'ffill은 결측값을 앞의 값으로 채워줌'
cash_return_daily_BenchmarkFrequency= rawdata_rf_pfl_value.pct_change(fill_method=None).dropna() #일별 수익률 계산


#################################################################################
# MVO 최적화 헬퍼 함수들
#################################################################################

def port_ExcessLogReturn_annual_from_ExcessReturn_N_Weight(initial_weights, ExcessRet_InvestmentHorizon, Mu_lag):
    """
    포트폴리오의 연율화 초과 로그수익률을 계산합니다.

    Parameters
    ----------
    initial_weights : np.ndarray
        각 자산의 포트폴리오 비중 (1D array, shape: n_assets)
    ExcessRet_InvestmentHorizon : np.ndarray
        Investment Horizon 주기별 초과수익률 (2D array, shape: n_periods x n_assets)
    Mu_lag : int
        수익률 lookback 기간 (영업일 단위, 예: 60, 120, 240)

    Returns
    -------
    float
        연율화된 초과 로그수익률

    Notes
    -----
    계산 과정:
    1. 포트폴리오 수익률 시리즈 = ExcessRet_InvestmentHorizon @ weights
    2. 누적 수익률 = (1 + ret_series).prod()
    3. 연율화 = 누적수익률 ^ (252 / Mu_lag)
    4. 로그 변환 = ln(연율화 수익률)
    """
    ret_series = ExcessRet_InvestmentHorizon @ initial_weights
    cum_ret = (ret_series + 1).prod()
    annual_ret = cum_ret ** (252 / Mu_lag)
    ExcessLogRet = np.log(annual_ret)
    return ExcessLogRet

def port_vol_annual(Weight, VarCov_annual):
    """
    포트폴리오의 연율화 변동성을 계산합니다.

    Parameters
    ----------
    Weight : np.ndarray
        각 자산의 포트폴리오 비중 (1D array, shape: n_assets)
    VarCov_annual : np.ndarray
        연율화된 공분산 행렬 (2D array, shape: n_assets x n_assets)

    Returns
    -------
    float
        연율화된 포트폴리오 변동성 (표준편차)

    Notes
    -----
    계산 공식: sqrt(W' @ VarCov @ W)
    - W: 비중 벡터
    - VarCov: 연율화 공분산 행렬 (주간 수익률 공분산 * 52)
    """
    return np.sqrt(Weight.T @ VarCov_annual @ Weight)

def Sharpe(Weight, ExcessRet_InvestmentHorizon, Mu_lag, VarCov_annual):
    """
    포트폴리오의 Sharpe Ratio를 계산합니다.

    Parameters
    ----------
    Weight : np.ndarray
        각 자산의 포트폴리오 비중 (1D array, shape: n_assets)
    ExcessRet_InvestmentHorizon : np.ndarray
        Investment Horizon 주기별 초과수익률 (2D array, shape: n_periods x n_assets)
    Mu_lag : int
        수익률 lookback 기간 (영업일 단위, 예: 60, 120, 240)
    VarCov_annual : np.ndarray
        연율화된 공분산 행렬 (2D array, shape: n_assets x n_assets)

    Returns
    -------
    float
        Sharpe Ratio (연율화 초과수익률 / 연율화 변동성)

    Notes
    -----
    Sharpe Ratio = 연율화 초과 로그수익률 / 연율화 변동성
    - 초과수익률: 무위험이자율 대비 초과 수익 (Mu_lag 기준 연율화)
    - 변동성: 포트폴리오 표준편차
    """
    return port_ExcessLogReturn_annual_from_ExcessReturn_N_Weight(
        Weight, ExcessRet_InvestmentHorizon, Mu_lag
    ) / port_vol_annual(Weight, VarCov_annual)

def Get_Sharpe_Maximizing_Weight(bound_list_of_tuple, Initial_Weight, ExcessRet_InvestmentHorizon, Mu_lag, VarCov_annual):
    """
    Sharpe Ratio를 최대화하는 최적 포트폴리오 비중을 계산합니다.

    Parameters
    ----------
    bound_list_of_tuple : list of tuple
        각 자산의 비중 제약조건 [(min1, max1), (min2, max2), ...]
        Long Only의 경우: [(0, 1), (0, 1), ...]
    Initial_Weight : np.ndarray
        최적화 시작점 (초기 비중, 1D array, shape: n_assets)
    ExcessRet_InvestmentHorizon : np.ndarray
        Investment Horizon 주기별 초과수익률 (2D array, shape: n_periods x n_assets)
    Mu_lag : int
        수익률 lookback 기간 (영업일 단위, 예: 60, 120, 240)
    VarCov_annual : np.ndarray
        연율화된 공분산 행렬 (2D array, shape: n_assets x n_assets)

    Returns
    -------
    np.ndarray
        Sharpe Ratio를 최대화하는 최적 비중 (1D array, shape: n_assets)

    Notes
    -----
    최적화 설정:
    - 목적함수: min(-Sharpe) = max(Sharpe)
    - 제약조건: sum(Weight) = 1 (완전투자)
    - 비중제약: bounds로 지정 (Long Only: 0 <= w_i <= 1)
    - 방법: SLSQP (Sequential Least Squares Programming)
    """
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

#################################################################################
# 파라미터 설정
#################################################################################

transaction_cost = 0.0010  # 10bp
lag_max = 760
lag_range = range(20, lag_max + 1, 20)  # Mu_lag = Sigma_lag 동일
investment_horizon_range = [20]

#################################################################################
# 데이터 전처리
################################################################################
fg_price_IND=fg_price.loc[:,(slice(None),slice(None),'GICS_IND',slice(None),slice(None))]
fg_price_IND_GROUP=fg_price.loc[:,(slice(None),slice(None),'GICS_IND_GROUP',slice(None),slice(None))]
fg_price_SEC=fg_price.loc[:,(slice(None),slice(None),'GICS_SEC',slice(None),slice(None))]

PRICE_DAILY = fg_price_IND_GROUP
PRICE_DAILY.columns=PRICE_DAILY.columns.get_level_values(0)
RET_DAILY = PRICE_DAILY.pct_change()
Rf_DailyRet = cash_return_daily_BenchmarkFrequency

# 초과수익률 계산
excess_ret = (RET_DAILY.T - Rf_DailyRet).T
CumExcessRetValue_DAILY = (excess_ret + 1).cumprod()
CumExcessRetValue_DAILY[np.isnan(CumExcessRetValue_DAILY) & ~np.isnan(CumExcessRetValue_DAILY.shift(-1))] = 1  # inception Value 1


# 누적 로그수익률 계산
CumLogRetValue_DAILY = np.log(RET_DAILY + 1).cumsum()
CumLogRetValue_DAILY[np.isnan(CumLogRetValue_DAILY) & ~np.isnan(CumLogRetValue_DAILY.shift(-1))] = 0  # inception Value 0

sample_initial_drop = lag_max

# 전체 자산 목록 (컬럼 인덱스 매핑용)
all_assets = PRICE_DAILY.columns.tolist()
number_of_all_assets = len(all_assets)

#################################################################################
# Grid Search 및 백테스트
#################################################################################

def run_single_backtest(params, PRICE_DAILY, CumExcessRetValue_DAILY,
                        CumLogRetValue_DAILY, all_assets,
                        number_of_all_assets, sample_initial_drop, transaction_cost):
    """
    단일 파라미터 조합에 대한 백테스트를 실행합니다.

    Parameters
    ----------
    params : tuple
        (investment_horizon, Sigma_lag, Mu_lag) 파라미터 조합
    PRICE_DAILY : pd.DataFrame
        일별 가격 데이터
    CumExcessRetValue_DAILY : pd.DataFrame
        누적 초과수익률 데이터
    CumLogRetValue_DAILY : pd.DataFrame
        누적 로그수익률 데이터
    all_assets : list
        전체 자산 목록
    number_of_all_assets : int
        전체 자산 개수
    sample_initial_drop : int
        초기 샘플 drop 개수
    transaction_cost : float
        거래비용 (예: 0.0015 = 15bp)

    Returns
    -------
    tuple
        (param_key, pfl_return_after_TransactionCost, turnover_at_rebalancing)
    """
    investment_horizon, Sigma_lag, Mu_lag = params
    param_key = (investment_horizon, Sigma_lag, Mu_lag)

    # 리밸런싱 날짜 설정
    rebalancing_dates = PRICE_DAILY.index[sample_initial_drop::investment_horizon]


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
    weight_input_df = pd.DataFrame(weight_input_array, index=rebalancing_dates, columns=all_assets)

    pfl_return_series_after_cost_daily, daily_eod_weights, portfolio_turnover_series = topquant_ksk.compute_daily_weights_rets_from_rebal_targets(
                                                                                                        target_weights_at_rebal_time=weight_input_df,
                                                                                                        price_return_daily=PRICE_DAILY.pct_change(),
                                                                                                        total_return_daily=PRICE_DAILY.pct_change(),
                                                                                                        transaction_cost_rate=transaction_cost
                                                                                                        )

    return (param_key, pfl_return_series_after_cost_daily, portfolio_turnover_series, weight_input_df)


# dict 방식으로 결과 저장 (수익률, 비중, turnover)
pfl_return_dict = {}  # key: (Horizon, Sigma_lag, Mu_lag), value: 수익률 시리즈
weight_dict = {}      # key: (Horizon, Sigma_lag, Mu_lag), value: 비중 DataFrame
turnover_dict = {}    # key: (Horizon, Sigma_lag, Mu_lag), value: turnover 시리즈

# 파라미터 조합 생성 (Mu_lag = Sigma_lag 동일)
param_combinations = [(h, lag, lag) for h in investment_horizon_range for lag in lag_range]
print(f"\n{'='*60}")
print(f"Grid Search 시작")
print(f"총 파라미터 조합 수: {len(param_combinations)}")
print(f"(Horizon: {len(investment_horizon_range)}, lag: {len(list(lag_range))})")
print(f"병렬 처리: 12 cores")
print(f"{'='*60}\n")

# 병렬 실행
results = Parallel(n_jobs=12, verbose=10)(
    delayed(run_single_backtest)(
        params, PRICE_DAILY, CumExcessRetValue_DAILY,
        CumLogRetValue_DAILY, all_assets,
        number_of_all_assets, sample_initial_drop, transaction_cost
    ) for params in param_combinations
)

# 결과 수집
for param_key, pfl_return, turnover, weight in results:
    pfl_return_dict[param_key] = pfl_return
    turnover_dict[param_key] = turnover
    weight_dict[param_key] = weight

print(f"\n{'='*60}")
print(f"Grid Search 완료!")
print(f"총 파라미터 조합 수: {len(pfl_return_dict)}")
print(f"(Horizon: {len(investment_horizon_range)}, lag: {len(list(lag_range))})")
print(f"{'='*60}")

# dict를 MultiIndex DataFrame으로 변환
# 수익률 DataFrame 생성 (Series들이 자동으로 index 정렬)
pfl_return_df = pd.DataFrame(pfl_return_dict)
pfl_return_df.columns = pd.MultiIndex.from_tuples(
    pfl_return_df.columns,
    names=['Horizon', 'Sigma_lag', 'Mu_lag']
)

# Turnover DataFrame 생성
turnover_df = pd.DataFrame(turnover_dict)
turnover_df.columns = pd.MultiIndex.from_tuples(
    turnover_df.columns,
    names=['Horizon', 'Sigma_lag', 'Mu_lag']
)

# Weight DataFrame 생성 (MultiIndex 컬럼)
weight_df = pd.concat(weight_dict, axis=1)
weight_df.columns = pd.MultiIndex.from_tuples(
    [(h, s, m, asset) for (h, s, m), df in weight_dict.items() for asset in df.columns],
    names=['Horizon', 'Sigma_lag', 'Mu_lag', 'Asset']
)

# 결과를 pkl 파일로 저장
result_dir = 'result'
os.makedirs(result_dir, exist_ok=True)  # 폴더가 없으면 생성

pfl_return_df.to_pickle(os.path.join(result_dir, 'pfl_return_df_ind_group.pkl'))
turnover_df.to_pickle(os.path.join(result_dir, 'turnover_df_ind_group.pkl'))
weight_df.to_pickle(os.path.join(result_dir, 'weight_df_ind_group.pkl'))

weight_df