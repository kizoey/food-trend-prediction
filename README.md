<h1> food-trend-prediction </h1>
<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/Tableau-E97627?style=flat-square&logo=Tableau&logoColor=white"/></a>
</p>
<b><i>Food trend prediction</i></b> shows the current trend and predicts future trend based on LSTM model. Term <b>'trend'</b> is defined and quantified as 3 aspects: long-term, short-term and persistence. <i>Long-term</i> condition is satisfied when the interest increased when compared to the interest level of the previous year. <i>Short-term</i> condition is met when the interest increased for the past 2 weeks. Lastly, <i>persistence</i> condition is satisfied when the upward trend continues within the target period (1 week). If all 3 conditions are met, we call the term is a <b>'trend'</b> term. <br>
<br>
To understand the current trend, we employed a powerful time-series technique <b>STL</b> (Seasonal and Trend decomposition using Loess) and <b>MA</b> (Moving Average). <br>
To predict the future trend, we used <b>LSTM</b> (Long-Short Term Memory).


<h2> major Contributions </h2>

- Define trend using **STL** and **Moving Average**
- Define trend intensity based on the difference between short-term MA and long-term MA
- Crawling trend keywords and related search words: https://datalab.naver.com/shoppingInsight/sCategory.naver
- Data and graph visualization using **Tableau** interactive tool
- Prediction of future trend using ARIMA and **LSTM**
- Provision of marketing suggestions to data analysts

<h2> Directory </h2>

### _algorithms_
- **collector**: 네이버 데이터랩에서 입력값 (조회기간/성별/연령대/1,2,3차 카테고리)에 따라 트렌드지수 데이터 크롤링
- **config**: 네이버 API 접근코드와 카테고리명 지정
- **daily_update**: 매일 크롤링한 트렌드지수 데이터로 LSTM 예측기 업데이트
- **initial**: 초기 트렌드지수 데이터로 LSTM 예측기 초기화/학습
- **lstm**: LSTM에 유의미한 변수조합으로 모델 학습
- **lstm_predictor**: 학습한 LSTM으로 미래 트렌드 예측
- **stl**: Moving Average로 계절성 제거한 현재 트렌드 파악
- **utils**: 데이터 전처리 단에서 필요한 함수 정의


### _data_
- DATALAB_트렌드지수_20대남성_20210815.csv: 21.08.15 기준 트렌드지수 크롤링 데이터 (20대남성)
- DATALAB_트렌드지수_20대여성_20210815.csv
- DATALAB_트렌드지수_30대남성_20210815.csv
- DATALAB_트렌드지수_30대여성_20210815.csv
- NAVER_카테고리목록_식품_210708: NAVER에서 21.07.08 기준 네이버 검색어트렌드 카테고리목록
