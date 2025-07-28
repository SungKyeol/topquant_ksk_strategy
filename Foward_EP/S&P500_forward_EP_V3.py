import pandas as pd
import numpy as np
import topquant_ksk 
from tqdm import tqdm
import sys
import importlib
from itertools import product
from joblib import Parallel, delayed
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

rawdata=topquant_ksk.load_FactSet_TimeSeriesData('data.csv')
rawdata_index=list(rawdata.index)
rawdata_index[-1]=rawdata_index[-2]+pd.Timedelta(days=1)
rawdata.index=rawdata_index


economic_data=topquant_ksk.load_DataGuide_EconomicData('Eco.xlsx')
rawdata_rf=economic_data['국채금리_미국국채(1년)(%)']/100/365
rawdata_rf_pfl_value=(rawdata_rf+1).cumprod()
rawdata_rf_pfl_value=rawdata_rf_pfl_value.reindex(rawdata.index,method='ffill')
cash_return_daily_BenchmarkFrequency= rawdata_rf_pfl_value.pct_change().shift(1).dropna()
usd_krw=economic_data['시장평균_미국(달러)(통화대원)'].reindex(rawdata.index,method='ffill')

rawdata_FMA_EPS=rawdata['FMA_EPS'].shift(1)
rawdata_FG_PRICE=rawdata['FG_PRICE']
rawdata_FG_TOTAL_RET_IDX=rawdata['FG_TOTAL_RET_IDX']
rawdata_FG_MKT_VALUE=rawdata['FG_MKT_VALUE']

rawdata_FG_PRICE_daily_ret=rawdata_FG_PRICE.pct_change()
rawdata_FG_TOTAL_RET_IDX_daily_ret=rawdata_FG_TOTAL_RET_IDX.pct_change()
rawdata_FG_TOTAL_RET_IDX_daily_ret[pd.isna(rawdata_FG_TOTAL_RET_IDX_daily_ret)]=rawdata_FG_PRICE_daily_ret
Total_return_price=(rawdata_FG_TOTAL_RET_IDX_daily_ret+1).cumprod()

rawdata_FG_TOTAL_RET_IDX_daily_drop_SP50=rawdata_FG_TOTAL_RET_IDX_daily_ret.drop(columns=[('S&P 500', 'SP50')],axis=1)

## 1단계: P/E(E/P) 스프레드 계산 (제공된 코드)
EP_Forward_1Y=rawdata_FMA_EPS/rawdata_FG_PRICE
EP_spread=(EP_Forward_1Y.T-EP_Forward_1Y[('S&P 500', 'SP50')]).T
EP_spread = EP_spread.drop(columns=[('S&P 500', 'SP50')])

# 시가총액 비중(BM 비중) 계산
# S&P500을 제외한 섹터들로만 BM 구성
sector_market_value = rawdata_FG_MKT_VALUE.drop(columns=[('S&P 500', 'SP50')])
total_market_value = sector_market_value.sum(axis=1)
bm_weights = sector_market_value.div(total_market_value, axis=0)

todays_date=pd.Timestamp.today()
transaction_cost_rate = 0.0010
################################################################################################################    
# 멀티프로세싱을 위한 함수 정의
def process_ns_combination(N, S, param_grid, EP_spread, bm_weights, return_daily, transaction_cost_rate):
    """(N, S) 조합 하나에 대한 모든 하위 루프를 실행하고 결과 딕셔너리를 반환"""
    
    # 이 함수 내에서 사용할 변수
    local_results_dict = {}
    

    # N, S에 종속된 계산
    middle_band = EP_spread.rolling(window=N).mean()
    std_dev = EP_spread.rolling(window=S).std()
    
    # 하위 루프 실행
    for K in param_grid['K']:
        upper_band = middle_band + (K * std_dev)
        lower_band = middle_band - (K * std_dev)
        percent_b = (EP_spread - lower_band) / (upper_band - lower_band)
        investment_attractiveness = percent_b - 0.5
        weighted_signal = investment_attractiveness * np.sqrt(bm_weights)
        
        for active_bet_multiplier in param_grid['active_bet_multiplier']:
            mean_weighted_signal = weighted_signal.mean(axis=1)
            neutralized_signal = weighted_signal.sub(mean_weighted_signal, axis=0)
            active_weight = neutralized_signal * active_bet_multiplier
            final_weight = bm_weights + active_weight
            final_weight[final_weight < 0] = 0
            final_weight = final_weight.div(final_weight.sum(axis=1), axis=0)
            
            for rebal_freq in param_grid['rebal_freq']:
                try:
                    # 백테스트 시뮬레이션
                    start_date = '2007-09-01'
                    weight_df = final_weight.loc[start_date:].dropna()
                    if weight_df.empty: continue

                    rebal_freq_str = str(rebal_freq) + 'W'
                    target_weights_at_rebal_time = weight_df.resample(rebal_freq_str).last()[:todays_date]
                    
                    # (기존 백테스트 로직과 동일)
                    turnover_at_rebal_time = target_weights_at_rebal_time.diff().abs().sum(axis=1).iloc[1:]
                    pfl_return_series = (weight_df.shift(1) * return_daily.loc[weight_df.index[0]:]).sum(axis=1).dropna()
                    transaction_cost_at_rebal_time = turnover_at_rebal_time * -transaction_cost_rate
                    indexer = pfl_return_series.index.get_indexer(transaction_cost_at_rebal_time.index, method='ffill')
                    transaction_cost_at_rebal_time.index = pfl_return_series.index[indexer]
                    pfl_return_series_after_cost = pfl_return_series + transaction_cost_at_rebal_time.reindex(pfl_return_series.index).fillna(0)
                    
                    params_key = (N, S, K, active_bet_multiplier, rebal_freq)
                    local_results_dict[params_key] = pfl_return_series_after_cost
                except Exception as e:
                    continue
    return local_results_dict

# --- 메인 코드 ---
if __name__ == '__main__':
    # 파라미터 격자 설정
    param_grid = {
        'N': [i for i in range(60, 240*3+20, 20)],
        'S': [i for i in range(60, 240*3+20, 20)],
        'K': [1.0, 1.5, 2.0, 2.5, 3.0],
        'active_bet_multiplier': [i / 10 for i in range(1, 11)],
        'rebal_freq': [1, 2, 4, 8, 12]
    }
    
    # 병렬 처리할 (N, S) 조합 생성
    ns_combinations = list(product(param_grid['N'], param_grid['S']))
    
    # CPU 코어 수 설정 (os.cpu_count()는 가용한 모든 코어 사용, -1도 동일)
    n_cores = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    
    print(f"Starting grid search with {len(ns_combinations)} (N, S) combinations on {n_cores} cores...")
    
    # joblib을 사용하여 병렬 처리 실행
    # 각 (N,S) 조합에 대해 process_ns_combination 함수를 병렬로 실행
    results_list_of_dicts = Parallel(n_jobs=n_cores)(
        delayed(process_ns_combination)(
            N, S, param_grid, EP_spread, bm_weights, 
            rawdata_FG_TOTAL_RET_IDX_daily_drop_SP50,  # 백테스트에 필요한 데이터 전달
             transaction_cost_rate
        ) for N, S in tqdm(ns_combinations, desc="Processing (N,S) Combinations")
    )
    
    # 병렬 처리된 결과들을 하나의 딕셔너리로 통합
    print("\nMerging results from all processes...")
    final_results_dict = {}
    for res_dict in results_list_of_dicts:
        final_results_dict.update(res_dict)
        
    # 최종 데이터프레임 생성
    all_returns_df = pd.DataFrame(final_results_dict)
    all_returns_df.columns.names=['N','S','K','active_bet_multiplier','rebal_freq']


BM=rawdata_FG_TOTAL_RET_IDX_daily_ret[('S&P 500', 'SP50')]

risk_return_profile=topquant_ksk.get_RiskReturnProfile(all_returns_df,cash_return_daily_BenchmarkFrequency,BM)

# 최적 파라미터 찾기 (Benchmark 제외)
risk_return_profile_no_benchmark = risk_return_profile[:-1]  # Benchmark 행 제외
best_param = risk_return_profile_no_benchmark['Information_Ratio'].astype(float).idxmax()

print("최적 파라미터:", best_param)
print("\n최적 파라미터의 성과:")

# 인덱스에서 best_param의 위치를 찾아서 iloc로 접근
best_param_idx = risk_return_profile.index.get_loc(best_param)
print(risk_return_profile.iloc[[best_param_idx,-1]].T)