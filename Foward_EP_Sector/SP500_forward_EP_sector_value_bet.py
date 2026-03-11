import pandas as pd
import numpy as np
import topquant_ksk 
from tqdm import tqdm # 루프 진행상황 표시
import sys          # 시스템 제어
from itertools import product # 조합 생성
from joblib import Parallel, delayed # 병렬 처리
import os             # 운영체제 제어

import importlib    # 모듈 동적 로드
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

rawdata=topquant_ksk.load_FactSet_TimeSeriesData(
    filename='data3.csv',
    column_spec=['Symbol Name','Symbol','Symbol Type','Asset Type','Item Name'],
    sheet_name=None  # CSV 파일이므로 sheet_name을 None으로 설정
) # 데이터 로드

rawdata

#마지막 날짜가 이상하게 찍혀서 조정해줌.
rawdata_index=list(rawdata.index) #인덱스를 리스트로 바꾸고
rawdata_index[-1]=rawdata_index[-2]+pd.Timedelta(days=1) #맨 마지막 element를 바꿔줌.
rawdata.index=rawdata_index # 기존 데이터 프레임에 다시 씌우기

economic_data=topquant_ksk.load_DataGuide_EconomicData('Eco.xlsx') #데이터 로드
rawdata_rf=economic_data['국채금리_미국국채(1년)(%)']/100/365    #하루기준 rf로 바주기
rawdata_rf_pfl_value=(rawdata_rf+1).cumprod()                 #누적 수익률 계산
rawdata_rf_pfl_value=rawdata_rf_pfl_value.reindex(rawdata.index,method='ffill') #인덱스 맞추기 'ffill은 결측값을 앞의 값으로 채워줌'
cash_return_daily_BenchmarkFrequency= rawdata_rf_pfl_value.pct_change().shift(1).dropna() #일별 수익률 계산
usd_krw=economic_data['시장평균_미국(달러)(통화대원)'].reindex(rawdata.index,method='ffill') #통화 환율 데이터 로드

rawdata_FMA_EPS=rawdata['FMA_EPS'].shift(1)
rawdata_FG_PRICE=rawdata['FG_PRICE']
rawdata_FG_TOTAL_RET_IDX=rawdata['FG_TOTAL_RET_IDX']
rawdata_FG_MKT_VALUE=rawdata['FG_MKT_VALUE']

rawdata_FG_PRICE_daily_ret=rawdata_FG_PRICE.pct_change()
rawdata_FG_TOTAL_RET_IDX_daily_ret=rawdata_FG_TOTAL_RET_IDX.pct_change()

rawdata_FG_TOTAL_RET_IDX_daily_ret[pd.isna(rawdata_FG_TOTAL_RET_IDX_daily_ret)]=rawdata_FG_PRICE_daily_ret # 결측값 처리, total return 결측값은 price return으로 채워줌.
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

# return_daily=rawdata_FG_TOTAL_RET_IDX_daily_drop_SP50
# return_daily

def compute_daily_weights_from_rebal_targets(target_weights_at_rebal_time: pd.DataFrame, return_daily: pd.DataFrame) -> pd.DataFrame:
    """리밸런싱 타깃 비중(End-of-Day, EOD)과 intra-period 누적 수익률로 일별 실제 End-of-Day 비중을 계산한다.
    - 리밸런싱일의 타깃 비중은 해당 일자의 EOD 비중으로 유지
    - 리밸런싱 구간 내에서는 다음 날부터 현재일까지의 누적 수익률로 비중이 드리프트되어 EOD 비중이 됨
    - 일별 포트 수익률 계산 시에는 EOD 비중을 하루 쉬프트하여 SOD 비중으로 사용
    """
    all_days = return_daily.index
    rebal_dates = target_weights_at_rebal_time.index
    # 각 날짜별 속한 리밸런싱 시작일(이전 리밸런싱일) 라벨 생성
    anchor_series = pd.Series(rebal_dates, index=rebal_dates).reindex(all_days, method='ffill')

    # 그룹별 누적 수익률(리밸런싱일 포함)과 그룹 첫날(리밸런싱일)의 (1+r) 값 추출
    ret_plus_one = return_daily + 1.0
    cumprod_to_date = ret_plus_one.groupby(anchor_series).cumprod()
    first_day_factor = cumprod_to_date.groupby(anchor_series).transform('first')

    # 리밸런싱일의 EOD 비중이 타깃이 되도록, (리밸런싱일+1)부터의 누적수익률만 반영
    # cumprod_excluding_rebal_day: d0(리밸런싱일)에서는 1, d0+1에서는 (1+r_{d0+1}), ...
    cumprod_excluding_rebal_day = (cumprod_to_date / first_day_factor).fillna(1.0)

    # 타깃 비중을 일별로 확장해 매핑
    tw_daily = target_weights_at_rebal_time.reindex(anchor_series.values).set_index(all_days)

    # 비정규화 일별 가치 및 정규화된 실제 EOD 비중
    unnormalized_values = tw_daily * cumprod_excluding_rebal_day
    daily_eod_weights = unnormalized_values.div(unnormalized_values.sum(axis=1), axis=0).fillna(0)
    daily_eod_weights=daily_eod_weights[rebal_dates[0]:]
    return daily_eod_weights

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
                    
                    # 가격 변동을 반영한 리밸런싱 직전 비중 계산 (루프 없이)
                    ret_plus_one = (return_daily.loc[weight_df.index[0]:] + 1.0)
                    period_return_prod = ret_plus_one.resample(rebal_freq_str).prod().reindex(target_weights_at_rebal_time.index)
                    prev_target_weights = target_weights_at_rebal_time.shift(1)
                    pre_rebal_unnorm = prev_target_weights * period_return_prod
                    pre_rebal_weights = pre_rebal_unnorm.div(pre_rebal_unnorm.sum(axis=1), axis=0)
                    turnover_at_rebal_time = (pre_rebal_weights - target_weights_at_rebal_time).abs().sum(axis=1).dropna()
                    turnover_at_rebal_time.iloc[0]=1 # 첫 번째 날짜는 1로 설정
                    
                    # 일별 실제 비중(EOD, 리밸런싱 후 intra-period 드리프트) 기반 수익률
                    daily_returns = return_daily.loc[target_weights_at_rebal_time.index[0]:]
                    daily_eod_weights = compute_daily_weights_from_rebal_targets(target_weights_at_rebal_time, daily_returns)
                    pfl_return_series = (daily_eod_weights.shift(1) * daily_returns).sum(axis=1).dropna()
                    
                    # 거래비용 반영
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
        'N': [i for i in range(20, 240*3+60, 20)],
        'S': [i for i in range(20, 240*3+60, 20)],
        'K': [1,1.5,2,2.5,3],
        'active_bet_multiplier': [0.5],
        'rebal_freq': [4]
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

#train test split
split_point=int(len(all_returns_df)*0.7)
all_returns_df_train=all_returns_df.iloc[:split_point]
all_returns_df_test=all_returns_df.iloc[split_point:]


all_returns_df_train

risk_return_profile_train=topquant_ksk.get_RiskReturnProfile(all_returns_df_train,cash_return_daily_BenchmarkFrequency.loc[all_returns_df_train.index],BM.loc[all_returns_df_train.index])
risk_return_profile_test=topquant_ksk.get_RiskReturnProfile(all_returns_df_test,cash_return_daily_BenchmarkFrequency.loc[all_returns_df_test.index],BM.loc[all_returns_df_test.index])


# 최적 파라미터 찾기 (Benchmark 제외)
risk_return_profile_no_benchmark = risk_return_profile_train[:-1]  # Benchmark 행 제외
best_param_IR = risk_return_profile_no_benchmark['Information_Ratio'].astype(float).idxmax()
best_param_SR = risk_return_profile_no_benchmark['Sharpe_Ratio'].astype(float).idxmax()

print("최적 파라미터 Information Ratio:", best_param_IR)
print("최적 파라미터 Sharpe Ratio:", best_param_SR)
print("\n최적 파라미터의 성과:")

# 인덱스에서 best_param의 위치를 찾아서 iloc로 접근
best_param_idx_IR = risk_return_profile_train.index.get_loc(best_param_IR)
best_param_idx_SR = risk_return_profile_train.index.get_loc(best_param_SR)
result_train=risk_return_profile_train.iloc[[best_param_idx_IR,best_param_idx_SR,-1]].T
result_train.columns=['Best Information_Ratio','Best Sharpe_Ratio','Benchmark']
result_train

result_test=risk_return_profile_test.iloc[[best_param_idx_IR,best_param_idx_SR,-1]].T
result_test.columns=['Best Information_Ratio','Best Sharpe_Ratio','Benchmark']
result_test


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic)
# 사용하시는 OS에 맞는 폰트 이름을 입력해주세요.
try:
    plt.rc('font', family='Malgun Gothic')
except:
    try:
        plt.rc('font', family='AppleGothic')
    except:
        print("한글 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

print("\n" + "="*80)
print("              퀀트 전략 최종 성과 보고서 (S&P500 섹터 로테이션)            ")
print("="*80)

# --- 1. 최적 전략 성과 요약 ---
print("\n[1. 최적 전략 성과 요약 (Information Ratio 기준)]")
print(f"▶ 최적 파라미터 Information Ratio: N={best_param_IR[0]}, S={best_param_IR[1]}, K={best_param_IR[2]}, Multiplier={best_param_IR[3]}, Rebal Freq={best_param_IR[4]}주")
print(f"▶ 최적 파라미터 Sharpe Ratio: N={best_param_SR[0]}, S={best_param_SR[1]}, K={best_param_SR[2]}, Multiplier={best_param_SR[3]}, Rebal Freq={best_param_SR[4]}주")

# Train/Test 기간의 성과 테이블 출력 (기존 코드 활용)
print("\n▶ Train 기간 성과:")
print("train set from {} to {}".format(all_returns_df_train.index[0].strftime('%Y-%m-%d'),all_returns_df_train.index[-1].strftime('%Y-%m-%d')))
print(result_train)

result_train.to_clipboard()

print("\n▶ Test 기간 성과:")
print("test set from {} to {}".format(all_returns_df_test.index[0].strftime('%Y-%m-%d'),all_returns_df_test.index[-1].strftime('%Y-%m-%d')))
print(result_test)

result_test.to_clipboard()

def calculate_latest_weights_and_signals(params, EP_spread, bm_weights, EP_Forward_1Y):
    """주어진 파라미터로 최신 시점의 시그널과 비중을 계산하는 함수 (SMA EP Spread 추가)"""
    N, S, K, active_bet_multiplier, rebal_freq = params
    
    # 최신 시점의 데이터 추출
    latest_date = EP_spread.index[-1]
    latest_spread = EP_spread.loc[latest_date]
    latest_bm_weights = bm_weights.loc[latest_date]
    latest_ep_forward = EP_Forward_1Y.loc[latest_date]
    
    # 최신 시점의 이동평균(SMA) 및 표준편차 계산
    middle_band = EP_spread.rolling(window=N).mean().loc[latest_date]
    std_dev = EP_spread.rolling(window=S).std().loc[latest_date]
    
    # 최신 시점의 시그널 계산
    upper_band = middle_band + (K * std_dev)
    lower_band = middle_band - (K * std_dev)
    percent_b = (latest_spread - lower_band) / (upper_band - lower_band)
    investment_attractiveness = percent_b - 0.5
    weighted_signal = investment_attractiveness * np.sqrt(latest_bm_weights)
    
    # 최신 시점의 비중 계산
    mean_weighted_signal = weighted_signal.mean()
    neutralized_signal = weighted_signal - mean_weighted_signal
    active_weight = neutralized_signal * active_bet_multiplier
    final_weight = latest_bm_weights + active_weight
    final_weight[final_weight < 0] = 0
    final_weight = final_weight / final_weight.sum()
    
    # 보고서용 데이터프레임 생성
    report_df = pd.DataFrame({
        'BM 비중(%)': (latest_bm_weights * 100).round(2),
        'Sector E/P(%)': latest_ep_forward.drop(('S&P 500', 'SP50')).round(3)*100,
        'BM E/P(%)': latest_ep_forward[('S&P 500', 'SP50')].round(3)*100,
        'E/P Spread(%p)': latest_spread.round(3)*100,
        'Historical EP Spread(%p)': middle_band.round(3)*100, # ★★★ 이동평균(중심선) 값 추가 ★★★
        '%B Signal': percent_b.round(2),
        '투자 매력도 점수': investment_attractiveness.round(2),
        '액티브 비중(%)': (active_weight * 100).round(2),
        '최종 비중(%)': (final_weight * 100).round(2)
    })
    report_df.index = [col[0] for col in report_df.index] # 인덱스 이름 정리
    return report_df.sort_values(by='액티브 비중(%)', ascending=False)

# --- 함수 호출 예시 ---
# latest_weights_report = calculate_latest_weights_and_signals(best_param_SR, EP_spread, bm_weights, EP_Forward_1Y)
# print(f"▶ 최신 기준일({EP_spread.index[-1].strftime('%Y-%m-%d')}) 비중 산출 근거:")
# print(latest_weights_report)
# --- 함수 호출 예시 ---
# 최적 파라미터로 최신 비중 상세 분석
latest_weights_report = calculate_latest_weights_and_signals(best_param_SR, EP_spread, bm_weights, EP_Forward_1Y)
print(f"▶ 최신 기준일({EP_spread.index[-1].strftime('%Y-%m-%d')}) 비중 산출 근거:")
print(latest_weights_report)

latest_weights_report.to_clipboard()
# --- 4. 시각화 자료 ---
print("\n[4. 시각화 자료]")

# 최적 전략과 BM의 누적 수익률 계산
strategy_cum_returns = (1 + all_returns_df[best_param_SR]).cumprod()
bm_cum_returns = (1 + BM.loc[strategy_cum_returns.index]).cumprod()


# 날짜 정렬
aligned_strategy, aligned_bm = strategy_cum_returns.align(bm_cum_returns, join='inner')

# 그래프 생성
plt.figure(figsize=(14, 7))
plt.plot(aligned_strategy.index, aligned_strategy, label=f'최적 전략 (IR 기준)')
plt.plot(aligned_bm.index, aligned_bm, label='Benchmark (S&P 500)', linestyle='--')
plt.title('최적 전략 vs. 벤치마크 누적 수익률', fontsize=16)
plt.ylabel('누적 수익률', fontsize=12)
plt.yscale('log')
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('cumulative_return.png')

N, S, K, active_bet_multiplier, rebal_freq = best_param_SR

middle_band = EP_spread.rolling(window=N).mean()
std_dev = EP_spread.rolling(window=S).std()
    
upper_band = middle_band + (K * std_dev)
lower_band = middle_band - (K * std_dev)
percent_b = (EP_spread - lower_band) / (upper_band - lower_band)
investment_attractiveness = percent_b - 0.5
weighted_signal = investment_attractiveness * np.sqrt(bm_weights)
        
mean_weighted_signal = weighted_signal.mean(axis=1)
neutralized_signal = weighted_signal.sub(mean_weighted_signal, axis=0)
active_weight = neutralized_signal * active_bet_multiplier
final_weight = bm_weights + active_weight
final_weight[final_weight < 0] = 0
final_weight = final_weight.div(final_weight.sum(axis=1), axis=0)
            
# 백테스트 시뮬레이션
start_date = '2007-09-01'
weight_df = final_weight.loc[start_date:].dropna()

rebal_freq_str = str(rebal_freq) + 'W'
target_weights_at_rebal_time = weight_df.resample(rebal_freq_str).last()[:todays_date]

# 가격 변동을 반영한 리밸런싱 직전 비중 기반 Turnover 계산 (루프 없이)
ret_plus_one = (rawdata_FG_TOTAL_RET_IDX_daily_drop_SP50.loc[weight_df.index[0]:] + 1.0)
period_return_prod = ret_plus_one.resample(rebal_freq_str).prod().reindex(target_weights_at_rebal_time.index)
prev_target_weights = target_weights_at_rebal_time.shift(1)
pre_rebal_unnorm = prev_target_weights * period_return_prod
pre_rebal_weights = pre_rebal_unnorm.div(pre_rebal_unnorm.sum(axis=1), axis=0)
turnover_at_rebal_time = (pre_rebal_weights - target_weights_at_rebal_time).abs().sum(axis=1).dropna()

# 일별 실제 비중(리밸런싱 후 intra-period 드리프트) 기반 수익률
_daily_returns = rawdata_FG_TOTAL_RET_IDX_daily_drop_SP50.loc[target_weights_at_rebal_time.index[0]:]
_daily_weights = compute_daily_weights_from_rebal_targets(target_weights_at_rebal_time, _daily_returns)
pfl_return_series = (_daily_weights.shift(1) * _daily_returns).sum(axis=1).dropna()

transaction_cost_at_rebal_time = turnover_at_rebal_time * -transaction_cost_rate
indexer = pfl_return_series.index.get_indexer(transaction_cost_at_rebal_time.index, method='ffill')
transaction_cost_at_rebal_time.index = pfl_return_series.index[indexer]
pfl_return_series_after_cost = pfl_return_series + transaction_cost_at_rebal_time.reindex(pfl_return_series.index).fillna(0)

bm_weights_at_rebal_time=bm_weights.reindex(target_weights_at_rebal_time.index,method='ffill')
active_weight_monthly_end=target_weights_at_rebal_time-bm_weights_at_rebal_time

# --- 데이터 전처리 ---
# 1. 컬럼 이름 정리 (가독성 향상)
df = active_weight_monthly_end.copy()
df.columns = [name.replace('S&P 500 / ', '').replace(' -SEC', '').replace(' - SEC', '') for name in df.columns.get_level_values(0)]

fig, ax = plt.subplots(figsize=(16, 8))
# --- 한글 폰트 설정 (Windows/Mac에 맞게) ---
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

# 3. 누적 막대 그래프 그리기
# Overweight (양수) 부분
df.plot(kind='bar', stacked=True, ax=ax, width=0.8, 
                   colormap='tab20',legend=False) # 색상 맵 지정



# 4. 그래프 서식 설정
ax.set_title('월별 섹터별 액티브 비중(Active Weight) 추이', fontsize=18, pad=20)
ax.set_xlabel('날짜', fontsize=12)
ax.set_ylabel('액티브 비중', fontsize=12)

# y축 서식을 퍼센트로 변경
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# x축 레이블 형식 변경 (YYYY-MM) 및 간격 조정
tick_labels = [item.strftime('%Y-%m') for item in df.index]
ax.set_xticklabels(tick_labels, rotation=45, ha='right')
ax.xaxis.set_major_locator(plt.MaxNLocator(20)) # x축 틱 개수 조절

# 범례 설정
ax.legend(title='Sectors', bbox_to_anchor=(1.02, 1), loc='upper left')

# 수평선 추가 (0 기준선)
ax.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.show()


def process_tunable_parameter_combination(tunable_parameter, EP_spread, bm_weights, return_daily, transaction_cost_rate, start_date='2007-09-01', todays_date=pd.Timestamp.today()):
    """
    주어진 단일 튜너블 파라미터 (N, S, K, active_bet_multiplier, rebal_freq)에 대해
    - 시그널 → 최종 일별 비중 계산
    - 리밸런싱 시점의 타깃 비중 산출
    - intra-period return을 활용한 리밸런싱 직전 비중 계산
    - turnover 및 거래비용 반영한 일별 수익률 계산
    을 수행하고 결과를 반환한다.

    Parameters
    ----------
    tunable_parameter : tuple
        (N, S, K, active_bet_multiplier, rebal_freq)
    EP_spread : pd.DataFrame
        섹터별 EP 스프레드 (BM 제외)
    bm_weights : pd.DataFrame
        섹터별 BM 비중 (BM 제외)
    return_daily : pd.DataFrame
        섹터별 일별 수익률 (BM 제외)
    transaction_cost_rate : float
        거래비용률
    start_date : str
        백테스트 시작일(포맷: YYYY-MM-DD)
    todays_date : pd.Timestamp
        계산 상한 날짜

    Returns
    -------
    dict
        {
          'params': (N, S, K, active_bet_multiplier, rebal_freq),
          'returns_after_cost': pd.Series,
          'turnover_at_rebal_time': pd.Series,
          'target_weights_at_rebal_time': pd.DataFrame
        }
    """
    N, S, K, active_bet_multiplier, rebal_freq = tunable_parameter

    # 1) 밴드 및 시그널 계산
    middle_band = EP_spread.rolling(window=N).mean()
    std_dev = EP_spread.rolling(window=S).std()
    upper_band = middle_band + (K * std_dev)
    lower_band = middle_band - (K * std_dev)
    percent_b = (EP_spread - lower_band) / (upper_band - lower_band)
    investment_attractiveness = percent_b - 0.5
    weighted_signal = investment_attractiveness * np.sqrt(bm_weights)

    # 2) 액티브 비중 및 최종 비중
    mean_weighted_signal = weighted_signal.mean(axis=1)
    neutralized_signal = weighted_signal.sub(mean_weighted_signal, axis=0)
    active_weight = neutralized_signal * active_bet_multiplier
    final_weight = bm_weights + active_weight
    final_weight[final_weight < 0] = 0
    final_weight = final_weight.div(final_weight.sum(axis=1), axis=0)

    # 3) 분석 구간 및 리밸런싱 타깃 비중
    weight_df = final_weight.loc[start_date:].dropna()
    if weight_df.empty:
        return {
            'params': tunable_parameter,
            'returns_after_cost': pd.Series(dtype=float),
            'turnover_at_rebal_time': pd.Series(dtype=float),
            'target_weights_at_rebal_time': pd.DataFrame()
        }

    rebal_freq_str = f"{rebal_freq}W"
    target_weights_at_rebal_time = weight_df.resample(rebal_freq_str).last()[:todays_date]

    # 4) 가격 변동 반영한 리밸런싱 직전 비중 (루프 없이)
    ret_plus_one = (return_daily.loc[weight_df.index[0]:] + 1.0)
    period_return_prod = ret_plus_one.resample(rebal_freq_str).prod().reindex(target_weights_at_rebal_time.index)
    prev_target_weights = target_weights_at_rebal_time.shift(1)
    pre_rebal_unnorm = prev_target_weights * period_return_prod
    pre_rebal_weights = pre_rebal_unnorm.div(pre_rebal_unnorm.sum(axis=1), axis=0)
    turnover_at_rebal_time = (pre_rebal_weights - target_weights_at_rebal_time).abs().sum(axis=1).dropna()
    if not turnover_at_rebal_time.empty:
        turnover_at_rebal_time.iloc[0] = 1

    # 5) 일별 포트 수익률 및 거래비용 반영: intra-period 드리프트 기반 실제 비중 사용
    daily_returns = return_daily.loc[target_weights_at_rebal_time.index[0]:]
    daily_eod_weights = compute_daily_weights_from_rebal_targets(target_weights_at_rebal_time, daily_returns)
    pfl_return_series = (daily_eod_weights.shift(1) * daily_returns).sum(axis=1).dropna()

    transaction_cost_at_rebal_time = turnover_at_rebal_time * -transaction_cost_rate
    indexer = pfl_return_series.index.get_indexer(transaction_cost_at_rebal_time.index, method='ffill')
    transaction_cost_at_rebal_time.index = pfl_return_series.index[indexer]
    pfl_return_series_after_cost = pfl_return_series + transaction_cost_at_rebal_time.reindex(pfl_return_series.index).fillna(0)

    return {
        'params': tunable_parameter,
        'returns_after_cost': pfl_return_series_after_cost,
        'turnover_at_rebal_time': turnover_at_rebal_time,
        'target_weights_at_rebal_time': target_weights_at_rebal_time
    }


def compute_param_partial_risk_return_profiles(
    base_params,
    param_offset_ranges,
    EP_spread,
    bm_weights,
    return_daily,
    transaction_cost_rate,
    cash_return_daily_BenchmarkFrequency,
    BM,
    start_date='2007-09-01',
    todays_date=pd.Timestamp.today()
):
    """
    각 튜너블 파라미터(N, S, K, active_bet_multiplier, rebal_freq)에 대해
    시작점(base_params)을 기준으로 주어진 offset range를 더해가며
    partial(편미분) 관점의 risk_return_profile을 계산한다.

    Parameters
    ----------
    base_params : tuple
        (N, S, K, active_bet_multiplier, rebal_freq)
    param_offset_ranges : dict
        {
          'N': iterable of int offsets,
          'S': iterable of int offsets,
          'K': iterable of float offsets,
          'active_bet_multiplier': iterable of float offsets,
          'rebal_freq': iterable of int offsets,
        }
        각 값은 base_params에 더하는 offset으로 해석된다.
    EP_spread, bm_weights, return_daily : pd.DataFrame
        본 스크립트 상단에서 계산된 동일 포맷의 데이터프레임
    transaction_cost_rate : float
    cash_return_daily_BenchmarkFrequency : pd.Series
        현금(무위험) 일별 수익률
    BM : pd.Series
        벤치마크(S&P500) 일별 수익률
    start_date : str
    todays_date : pd.Timestamp

    Returns
    -------
    dict
        {
          'N': risk_return_profile_df_for_N_offsets,
          'S': risk_return_profile_df_for_S_offsets,
          'K': risk_return_profile_df_for_K_offsets,
          'active_bet_multiplier': risk_return_profile_df_for_multiplier_offsets,
          'rebal_freq': risk_return_profile_df_for_rebal_offsets,
        }
    """
    base_N, base_S, base_K, base_mult, base_rebal = base_params

    results_profiles_by_param = {}

    def build_and_score(series_map):
        if len(series_map) == 0:
            return pd.DataFrame()
        returns_df = pd.DataFrame(series_map)
        # 지표 계산 시 현금/벤치마크를 해당 인덱스에 정렬
        cash_series = cash_return_daily_BenchmarkFrequency.reindex(returns_df.index).dropna()
        bm_series = BM.reindex(returns_df.index).dropna()
        # 공통 구간으로 정렬
        aligned_returns, aligned_cash = returns_df.align(cash_series, join='inner', axis=0)
        aligned_returns, aligned_bm = aligned_returns.align(bm_series, join='inner', axis=0)
        profile = topquant_ksk.get_RiskReturnProfile(aligned_returns, aligned_cash, aligned_bm)
        return profile

    # 각 파라미터별로만 변화시킨 조합 생성 및 스코어링
    # 1) N
    if 'N' in param_offset_ranges and param_offset_ranges['N'] is not None:
        col_to_series = {}
        for offset in param_offset_ranges['N']:
            new_N = int(base_N + offset)
            if new_N <= 0:
                continue
            params = (new_N, base_S, base_K, base_mult, base_rebal)
            out = process_tunable_parameter_combination(params, EP_spread, bm_weights, return_daily, transaction_cost_rate, start_date, todays_date)
            ser = out['returns_after_cost']
            if ser.empty:
                continue
            col_to_series[(f'N', new_N)] = ser
        results_profiles_by_param['N'] = build_and_score(col_to_series)

    # 2) S
    if 'S' in param_offset_ranges and param_offset_ranges['S'] is not None:
        col_to_series = {}
        for offset in param_offset_ranges['S']:
            new_S = int(base_S + offset)
            if new_S <= 0:
                continue
            params = (base_N, new_S, base_K, base_mult, base_rebal)
            out = process_tunable_parameter_combination(params, EP_spread, bm_weights, return_daily, transaction_cost_rate, start_date, todays_date)
            ser = out['returns_after_cost']
            if ser.empty:
                continue
            col_to_series[(f'S', new_S)] = ser
        results_profiles_by_param['S'] = build_and_score(col_to_series)

    # 3) K
    if 'K' in param_offset_ranges and param_offset_ranges['K'] is not None:
        col_to_series = {}
        for offset in param_offset_ranges['K']:
            new_K = float(base_K + offset)
            if new_K <= 0:
                continue
            params = (base_N, base_S, new_K, base_mult, base_rebal)
            out = process_tunable_parameter_combination(params, EP_spread, bm_weights, return_daily, transaction_cost_rate, start_date, todays_date)
            ser = out['returns_after_cost']
            if ser.empty:
                continue
            col_to_series[(f'K', new_K)] = ser
        results_profiles_by_param['K'] = build_and_score(col_to_series)

    # 4) active_bet_multiplier
    if 'active_bet_multiplier' in param_offset_ranges and param_offset_ranges['active_bet_multiplier'] is not None:
        col_to_series = {}
        for offset in param_offset_ranges['active_bet_multiplier']:
            new_mult = float(base_mult + offset)
            if new_mult <= 0:
                continue
            params = (base_N, base_S, base_K, new_mult, base_rebal)
            out = process_tunable_parameter_combination(params, EP_spread, bm_weights, return_daily, transaction_cost_rate, start_date, todays_date)
            ser = out['returns_after_cost']
            if ser.empty:
                continue
            col_to_series[(f'active_bet_multiplier', new_mult)] = ser
        results_profiles_by_param['active_bet_multiplier'] = build_and_score(col_to_series)

    # 5) rebal_freq
    if 'rebal_freq' in param_offset_ranges and param_offset_ranges['rebal_freq'] is not None:
        col_to_series = {}
        for offset in param_offset_ranges['rebal_freq']:
            new_rebal = int(base_rebal + offset)
            if new_rebal <= 0:
                continue
            params = (base_N, base_S, base_K, base_mult, new_rebal)
            out = process_tunable_parameter_combination(params, EP_spread, bm_weights, return_daily, transaction_cost_rate, start_date, todays_date)
            ser = out['returns_after_cost']
            if ser.empty:
                continue
            col_to_series[(f'rebal_freq', new_rebal)] = ser
        results_profiles_by_param['rebal_freq'] = build_and_score(col_to_series)

    return results_profiles_by_param


profiles = compute_param_partial_risk_return_profiles(
    base_params=best_param_SR,
    param_offset_ranges={
        'N': list(np.arange(-240, 240+20, 20)),
        'S': list(np.arange(-240, 240*3+20, 20)),
        'K': list(np.arange(-3.0, 3.5, 0.5)),
        'active_bet_multiplier': [np.floor(i*10)/10 for i in np.arange(-1.0, 1.1, 0.1)],
        'rebal_freq': list(np.arange(-4, 9, 1)),
    },
    EP_spread=EP_spread,
    bm_weights=bm_weights,
    return_daily=rawdata_FG_TOTAL_RET_IDX_daily_drop_SP50,
    transaction_cost_rate=transaction_cost_rate,
    cash_return_daily_BenchmarkFrequency=cash_return_daily_BenchmarkFrequency,
    BM=BM,
    start_date='2007-09-01'
)

# 예: N에 대한 결과 확인
profiles['rebal_freq']