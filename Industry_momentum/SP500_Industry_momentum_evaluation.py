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
BM_return_daily=fg_price.iloc[:,0].pct_change()

fg_market_value=topquant_ksk.load_FactSet_TimeSeriesData(
    filename='FG_MKT_VALUE.csv',
    column_spec=['Symbol Name','Symbol','Symbol Type','Asset Type','Item Name'],
    sheet_name=None  # CSV 파일이므로 sheet_name을 None으로 설정
) # 데이터 로드

# 지수, rf 데이터 로드
economic_data=topquant_ksk.load_FactSet_TimeSeriesData(
    filename='Eco.csv',
    column_spec=['Symbol Name','Symbol','Symbol Type','Asset Type','Item Name'],
    sheet_name=None  # CSV 파일이므로 sheet_name을 None으로 설정
) # 데이터

rawdata_rf=economic_data[('US Benchmark Bond - 1 Year', 'TRYUS1Y-FDS', 'Yield', 'Bond', 'FG_YIELD')]/100/365
rawdata_rf_pfl_value=(rawdata_rf+1).cumprod()                 #누적 수익률 계산
rawdata_rf_pfl_value=rawdata_rf_pfl_value.reindex(fg_price.index,method='ffill') #인덱스 맞추기 'ffill은 결측값을 앞의 값으로 채워줌'
cash_return_daily_BenchmarkFrequency= rawdata_rf_pfl_value.pct_change(fill_method=None).dropna() #일별 수익률 계산

result_dir = 'result'
pfl_return_df = pd.read_pickle('result/meta_pfl_return_df.pkl')
turnover_df=pd.read_pickle('result/meta_turnover_df.pkl')
meta_weight=pd.read_pickle('result/meta_weight_df.pkl')
meta_weight

split_point=int(len(pfl_return_df)*0.7)
pfl_return_df_train=pfl_return_df.iloc[:split_point]
pfl_return_df_test=pfl_return_df.iloc[split_point:]

risk_return_profile_train=topquant_ksk.get_RiskReturnProfile(rebalencing_ret=pfl_return_df_train,cash_return_daily_BenchmarkFrequency=cash_return_daily_BenchmarkFrequency,BM_ret=BM_return_daily)
max_sharpe_param=risk_return_profile_train[risk_return_profile_train['Sharpe_Ratio']==risk_return_profile_train['Sharpe_Ratio'].max()].index[0]
risk_return_profile_train.loc[[max_sharpe_param,'Benchmark']].T


risk_return_profile_test=topquant_ksk.get_RiskReturnProfile(rebalencing_ret=pfl_return_df_test,cash_return_daily_BenchmarkFrequency=cash_return_daily_BenchmarkFrequency,BM_ret=BM_return_daily)
risk_return_profile_test.loc[[max_sharpe_param,'Benchmark'],]

ret_daily=pd.concat([pfl_return_df[max_sharpe_param],BM_return_daily],axis=1,keys=['펀드','S&P500']).dropna()
(ret_daily+1).cumprod().to_clipboard()

meta_weight[max_sharpe_param].iloc[-1].round(3)*100

risk_return_profile_whole=topquant_ksk.get_RiskReturnProfile(rebalencing_ret=pfl_return_df,cash_return_daily_BenchmarkFrequency=cash_return_daily_BenchmarkFrequency,BM_ret=BM_return_daily)
risk_return_profile_whole.loc[[max_sharpe_param,'Benchmark']].T.to_clipboard()


risk_return_profile_whole.sort_values(by=['Sharpe_Ratio'],ascending=False).head()
risk_return_profile_train.sort_values(by=['Sharpe_Ratio'],ascending=False).head()
risk_return_profile_test.sort_values(by=['Sharpe_Ratio'],ascending=False).head()


max_sharpe_param_whole=risk_return_profile_whole[risk_return_profile_whole['Sharpe_Ratio']==risk_return_profile_whole['Sharpe_Ratio'].max()].index[0]


max_sharpe_param_whole

topquant_ksk.get_yearly_monthly_ER(pfl_return_df[max_sharpe_param_whole],BM_return_daily).heatmap()

# topquant_ksk.get_yearly_monthly_ER(pfl_return_df_train[max_sharpe_param],BM_return_daily)

(meta_weight[max_sharpe_param].iloc[-1].round(3)*100)