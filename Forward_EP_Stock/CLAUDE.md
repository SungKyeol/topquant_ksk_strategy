# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

S&P 500 개별 종목 대상 Forward 12M E/P(Earnings/Price) 전략 백테스팅 프로젝트.
Bollinger Band 기반 E/P Spread 시그널로 종목별 액티브 비중을 산출하고, 그리드 서치 최적화를 통해 최적 파라미터를 탐색한다.

### Main Strategy File
- `SP500_stock_forward12M_EP_strategy.py`: 전략 구현 단일 파일
  - DB에서 데이터 로드 (topquant_ksk.db)
  - Bollinger Band E/P Spread 시그널 생성
  - 그리드 서치 파라미터 최적화 (멀티프로세싱)
  - 백테스팅 및 성과 분석

## 참고 전략 코드 주의사항

`Foward_EP_Sector/SP500_forward_EP_sector_value_bet.py`를 참고하되 다음 사항에 주의:

1. **커스텀 백테스트 로직 사용 금지**: 참고코드의 `compute_daily_weights_from_rebal_targets()`, 턴오버 계산, 거래비용 적용 로직에 오류 가능성이 있음. 반드시 `topquant_ksk.compute_daily_weights_rets_from_rebal_targets()` 라이브러리 함수를 사용할 것
2. **시그널 가중 방식 변경**: 참고코드의 `weighted_signal = investment_attractiveness * np.sqrt(bm_weights)`를 `weighted_signal = investment_attractiveness * bm_weights`로 변경 (np.sqrt 제거)
3. **데이터 소스 변경**: 참고코드의 로컬 CSV/Excel 로드 대신 `topquant_ksk.db.download` 사용

## Custom Package: topquant_ksk

Located at `../../topquant-ksk/src/topquant_ksk/`. pip install -e로 설치하여 사용.

### 1. DB 데이터 로드 (topquant_ksk.db)

#### 터널 연결
```python
import topquant_ksk

# Cloudflare 터널 연결 (원격 DB 접속 시 필수)
tunnel = topquant_ksk.db.tunnel.manage_db_tunnel(
    hostname="db.alphawaves.vip",  # default
    local_port=15432               # default
)

# 작업 완료 후 터널 종료
topquant_ksk.db.tunnel.kill_tunnel(tunnel)
```

#### 시계열 데이터 로드
```python
df = topquant_ksk.db.download.fetch_timeseries_table(
    table_name="public.daily_price_data",   # 테이블명
    columns=['ticker'],                      # MultiIndex용 컬럼
    item_names=['close_pr', 'volume'],       # 가져올 항목 (None=전체)
    db_user="username",                      # PostgreSQL 사용자명
    db_password="password",                  # PostgreSQL 비밀번호
    local_host=False,                        # False: 터널(15432), True: localhost(5432)
    start_date='2024-01-01',                 # 시작일 (None=테이블 최소)
    end_date='2024-12-31',                   # 종료일 (None=테이블 최대)
    sedols="all",                            # 종목 필터 (list 또는 "all")
    limit=None                               # 행 수 제한 (None=전체)
)
# Returns: DatetimeIndex + MultiIndex(item_name, *columns, sedol, company_name, ticker)
```

#### 마스터 테이블 로드
```python
master = topquant_ksk.db.download.fetch_master_table(
    columns=['ticker', 'company_name', 'sedol'],
    db_user="username",
    db_password="password",
    local_host=False,
    table_name="public.master_table"  # default
)
```

#### DB 테이블 확인
```python
topquant_ksk.db.tools.check_existing_tables(
    db_user="username",
    db_password="password",
    local_host=False,
    detailed_column_date=False  # True: 컬럼별 min/max 날짜 표시
)
```

### 2. 백테스팅 (topquant_ksk.tools)

리밸런싱 타깃 비중 기반 일별 수익률, 비중, 턴오버를 한번에 계산:
```python
pfl_return_after_cost, daily_eod_weights, turnover_series = \
    topquant_ksk.compute_daily_weights_rets_from_rebal_targets(
        target_weights_at_rebal_time=weight_df,    # 리밸런싱 날짜별 타깃 비중 (DataFrame)
        price_return_daily=price_ret,               # 일별 가격 수익률 (DataFrame)
        total_return_daily=total_ret,               # 일별 총 수익률 (DataFrame)
        transaction_cost_rate=0.0010                # 거래비용률 (10bp)
    )
# Returns:
#   pfl_return_after_cost: pd.Series - 거래비용 차감 후 일별 포트폴리오 수익률
#   daily_eod_weights: pd.DataFrame - End-of-Day 일별 비중 (intra-period drift 반영)
#   turnover_series: pd.Series - 리밸런싱 시점별 턴오버
```

### 3. 성과 지표 (topquant_ksk.risk_return_metrics)

```python
profile = topquant_ksk.get_RiskReturnProfile(
    rebalencing_ret=daily_returns_df,                         # 일별 수익률 (DataFrame/Series)
    cash_return_daily_BenchmarkFrequency=cash_return_series,  # 일별 무위험 수익률
    BM_ret=benchmark_return_series                            # 벤치마크 일별 수익률 (optional)
)
# Returns DataFrame with:
# - CAGR(%), STD_annualized(%), Sharpe_Ratio, MDD(%), MDD시점, UnderWaterPeriod(년)
# - Weekly Hit Ratio(%), 1M/3M/6M/1Y/3Y Ret(%)
# - (BM 제공 시) BM_ret excess_return(%), tracking_error(%), Information_Ratio,
#   BM대비주간승률(%), BM대비최대손실(%), BM_ret Max Underwater(년)
```

#### 연도/월별 초과수익 히트맵
```python
yearly_monthly = topquant_ksk.get_yearly_monthly_ER(
    strategy_return=strategy_daily_ret,  # 전략 일별 수익률
    BM_return=bm_daily_ret               # 벤치마크 일별 수익률
)
yearly_monthly.heatmap(figsize=(18, 10), fontsize=9)
```

## Strategy Architecture

### 전략 흐름
1. **데이터 로드**: DB에서 개별 종목 가격, Forward EPS, 시가총액, 총 수익률 데이터 로드
2. **E/P Spread 계산**:
   - Forward E/P = Forward 12M EPS / Price
   - E/P Spread = 종목 E/P - S&P 500 E/P (벤치마크 대비 상대 밸류에이션)
3. **BM 비중 계산**: 시가총액 기반 비중 (S&P 500 제외한 개별 종목)
4. **Bollinger Band 시그널 생성**:
   - Middle Band = EP_spread.rolling(window=N).mean()
   - Std Dev = EP_spread.rolling(window=S).std()
   - Upper Band = Middle Band + K * Std Dev
   - Lower Band = Middle Band - K * Std Dev
   - %B = (EP_spread - Lower Band) / (Upper Band - Lower Band)
   - investment_attractiveness = %B - 0.5
5. **포트폴리오 비중 산출**:
   - weighted_signal = investment_attractiveness * bm_weights
   - neutralized_signal = weighted_signal - mean(weighted_signal)
   - active_weight = neutralized_signal * active_bet_multiplier
   - final_weight = bm_weights + active_weight (비음수, 합 1 정규화)
6. **리밸런싱 & 백테스팅**: `topquant_ksk.compute_daily_weights_rets_from_rebal_targets()` 사용
7. **성과 분석**: `topquant_ksk.get_RiskReturnProfile()` 으로 Train/Test 성과 비교

### Grid Search Parameters
| Parameter | Description | Range |
|-----------|-------------|-------|
| N | Moving average lookback (days) | 20-780 (20-day step) |
| S | Std dev lookback (days) | 20-780 (20-day step) |
| K | Bollinger Band width multiplier | [1, 1.5, 2, 2.5, 3] |
| active_bet_multiplier | Position sizing multiplier | [0.5] |
| rebal_freq | Rebalancing frequency (weeks) | [4] |

### Multiprocessing
```python
from joblib import Parallel, delayed
from itertools import product

ns_combinations = list(product(param_grid['N'], param_grid['S']))
n_cores = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

results = Parallel(n_jobs=n_cores)(
    delayed(process_ns_combination)(N, S, ...)
    for N, S in tqdm(ns_combinations)
)
```

### Train/Test Split
- 70/30 split: `split_point = int(len(all_returns_df) * 0.7)`
- 최적 파라미터: Information Ratio 또는 Sharpe Ratio 기준

## Commands

### Running the Strategy
```bash
cd C:\Users\SungKyeol\Desktop\github\topquant_ksk_strategy\Forward_EP_Stock
python SP500_stock_forward12M_EP_strategy.py
```

### Installing Dependencies
```bash
# topquant_ksk 패키지 설치
cd C:\Users\SungKyeol\Desktop\github\topquant-ksk
pip install -e .
```

Required packages:
- pandas >= 1.5.0
- numpy >= 1.20.0
- tqdm
- joblib (parallel processing)
- matplotlib (visualization)
- topquant_ksk (local package - includes psycopg2, sqlalchemy, polars for DB access)

## Important Notes

- 개별 종목(500+개) 레벨이므로 섹터(11개) 대비 연산량이 크게 증가함. 그리드 서치 범위 축소 또는 메모리 관리 필요
- DB 접속 시 db_user, db_password 필요 (코드에 하드코딩 금지, 환경변수 또는 input() 사용 권장)
- 원격 DB 접속 시 반드시 `manage_db_tunnel()` → 작업 → `kill_tunnel()` 순서로 터널 관리
- Transaction cost: 0.0010 (10bp per turnover)
- S&P 500 벤치마크는 투자 유니버스에서 제외하고 BM 수익률 비교용으로만 사용
- Forward EPS 데이터는 1일 shift 적용 (`shift(1)`) - look-ahead bias 방지
