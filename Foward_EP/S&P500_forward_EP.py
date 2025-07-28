import pandas as pd
import numpy as np
import topquant_ksk 
from tqdm import tqdm


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

################################################################################################################    
transaction_cost_rate=0.0010
################################################################################################################    
## 2단계: 볼린저 밴드 지표 계산
# 파라미터 설정
N = 252  # 가격 이동평균 기간
S = 252  # Simga 이동평균 기간
K = 2   # 표준편차 승수
active_bet_multiplier = 0.5  # 트래킹 에러 조절 변수
rebal_freq_weekly_range=[i for i in range(1,13)]
rebal_freq=4

# 볼린저 밴드 계산
middle_band = EP_spread.rolling(window=N).mean()
std_dev = EP_spread.rolling(window=S).std()
upper_band = middle_band + (K * std_dev)
lower_band = middle_band - (K * std_dev)

## 3단계: 시그널 정량화 (%B 지표 활용)
# %B 계산
# 상단밴드와 하단밴드의 차이가 0이 되는 경우를 방지하기 위해 아주 작은 값(epsilon)을 더함
percent_b = (EP_spread - lower_band) / (upper_band - lower_band)

## 4단계: 액티브 시그널 생성 및 가중
# 투자 매력도 변환 (E/P 스프레드이므로 %B가 높을수록 매력적 -> %B - 0.5)
investment_attractiveness = percent_b - 0.5

# 시가총액 가중 시그널 계산 
weighted_signal = investment_attractiveness * np.sqrt(bm_weights)

## 5단계: 최종 투자 비중 계산
# 중립화
mean_weighted_signal = weighted_signal.mean(axis=1)

neutralized_signal = weighted_signal.sub(mean_weighted_signal, axis=0)

# 스케일링 (k값은 백테스팅을 통해 최적화 필요)
active_weight = neutralized_signal * active_bet_multiplier

# 최종 비중 산출 (BM 비중 + 액티브 비중)
final_weight = bm_weights + active_weight

# 제약조건 1: 비중 하한선 설정 (공매도 방지)
final_weight[final_weight < 0] = 0

# 제약조건 2: 재배분 (비중의 합을 100%로)
total_weight = final_weight.sum(axis=1)
final_weight = final_weight.div(total_weight, axis=0)

#rebalancing 시뮬레이션
start_date='2007-09-01'
weight_df=final_weight.loc[start_date:]
start_date=weight_df.index[0]
start_date_week_end=start_date.to_period('W').to_timestamp('W')
return_daily=rawdata_FG_TOTAL_RET_IDX_daily_ret.loc[start_date_week_end:].drop([('S&P 500', 'SP50')],axis=1)
return_daily_multiplier=return_daily+1

# 리밸런싱 주기별 목표 비중 추출
rebal_freq_str = str(rebal_freq) + 'W' # 대신 'M' 사용이 더 일반적
target_weights_at_rebal_time = weight_df.resample(rebal_freq_str).last()[:todays_date]

#turnover 계산
turnover_at_rebal_time=target_weights_at_rebal_time.diff().abs().sum(axis=1)[1:]

# 기간별 시작 비중 계산 = 금요일 Target 비중 -> 월요일 시작 비중으로 감
start_of_period_weights = target_weights_at_rebal_time.shift(1, freq='D').reindex(weight_df.index, method='ffill').dropna()

# # 기간별 누적 수익률 계산 = 월요일 시작 비중에 곱할 수익률 
intra_period_cum_returns = return_daily_multiplier.groupby(pd.Grouper(freq=rebal_freq_str)).cumprod()

# # 정규화되지 않은 가치 및 월요일 종가 기준 실제 비중 계산
unnormalized_values = (start_of_period_weights * intra_period_cum_returns).dropna(how='all')
total_daily_values = unnormalized_values.sum(axis=1)
actual_weights_final_df = unnormalized_values.divide(total_daily_values, axis=0).fillna(0)

pfl_return_series=(actual_weights_final_df.shift(1)*return_daily[actual_weights_final_df.index[0]:]).sum(axis=1)

#transaction cost 계산
pfl_return_series.iloc[0]=-transaction_cost_rate
transaction_cost_at_rebal_time=turnover_at_rebal_time*-transaction_cost_rate
transaction_cost_at_rebal_time.index.intersection(pfl_return_series.index)
indexer = pfl_return_series.index.get_indexer(transaction_cost_at_rebal_time.index, method='ffill')
matched_dates = pfl_return_series.index[indexer]
transaction_cost_at_rebal_time.index=matched_dates

#거래비용후 최종 수익률
pfl_return_series_after_transaction_cost=pfl_return_series-transaction_cost_at_rebal_time.reindex(pfl_return_series.index).fillna(0)
pfl_return_series_after_transaction_cost






# all_results[(window, n_etf, rebal_freq)] = pfl_return_series

# # 3. 저장된 모든 결과를 하나의 DataFrame으로 합치기
# #    딕셔너리의 키가 자동으로 MultiIndex 컬럼으로 변환됨
# results_df = pd.concat(all_results, axis=1)
# # MultiIndex 컬럼에 이름 부여
# results_df.columns.names = ['window', 'n_etf', 'rebal_freq_month']

# return_index_pfl=pd.concat([index_return_daily,results_df],axis=1).dropna()
# # SettingWithCopyWarning 방지를 위해 copy() 사용
# return_index_pfl = return_index_pfl.copy()
# return_index_pfl.iloc[0] = 0


# risk_return_profile=ksk_quant.get_RiskReturnPorfile(return_index_pfl,cash_return_daily_BenchmarkFrequency)
# Sharpe_grid=risk_return_profile['샤프비율'][3:]
# Sharpe_grid.index=pd.MultiIndex.from_tuples(Sharpe_grid.index,names=['window', 'n_etf', 'rebal_freq_month'])
# Sharpe_grid.idxmax()

# risk_return_profile.loc[['코스피 200','S&P500(H)','S&P500(UH)',(40,15,3)]].to_clipboard()





