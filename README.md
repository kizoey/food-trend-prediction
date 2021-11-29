<h1> food-trend-prediction </h1>
<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/></a>&nbsp 
</p>
<b><i>Food trend prediction</i></b> shows the current trend and predicts future trend based on LSTM model. Term <b>'trend'</b> is defined and quantified as 3 aspects: long-term, short-term and persistence. <i>Long-term</i> condition is satisfied when the interest increased when compared to the interest level of the previous year. <i>Short-term</i> condition is met when the interest increased for the past 2 weeks. Lastly, <i>persistence</i> condition is satisfied when the upward trend continues within the target period (1 week). If all 3 conditions are met, we call the term is a <b>'trend'</b> term. <br>
<br>
To understand the current trend, we employed a powerful time-series technique <b>STL</b> (Seasonal and Trend decomposition using Loess) and <b>MA</b> (Moving Average). To predict the future trend, we used <b>LSTM</b> (Long-Short Term Memory).


<h2> major Contributions </h2>

- Define trend using **STL** and **Moving Average**
- Define trend intensity based on the difference between short-term MA and long-term MA
- Crawling trend keywords and related search words: https://datalab.naver.com/shoppingInsight/sCategory.naver
- Data and graph visualization using **Tableau** interactive tool
- Prediction of future trend using ARIMA and **LSTM**
- Provision of marketing suggestions to data analysts

<h2> Directory </h2>

### _algorithms_
- **collector**: KLUE(고려대학교 강의평가 사이트)에서 강의평, 학점/학습량/난이도/성취감 4개 지표 데이터 크롤링
- **config**: KLUE(고려대학교 강의평가 사이트)에서 강의평, 학점/학습량/난이도/성취감 4개 지표 데이터 크롤링
- **daily_update**: KLUE(고려대학교 강의평가 사이트)에서 강의평, 학점/학습량/난이도/성취감 4개 지표 데이터 크롤링
- **initial**: KLUE(고려대학교 강의평가 사이트)에서 강의평, 학점/학습량/난이도/성취감 4개 지표 데이터 크롤링
- **lstm**: KLUE(고려대학교 강의평가 사이트)에서 강의평, 학점/학습량/난이도/성취감 4개 지표 데이터 크롤링
- **lstm_predictor**: KLUE(고려대학교 강의평가 사이트)에서 강의평, 학점/학습량/난이도/성취감 4개 지표 데이터 크롤링
- **stl**: KLUE(고려대학교 강의평가 사이트)에서 강의평, 학점/학습량/난이도/성취감 4개 지표 데이터 크롤링
- **utils**: KLUE(고려대학교 강의평가 사이트)에서 강의평, 학점/학습량/난이도/성취감 4개 지표 데이터 크롤링


### _data_
- DATALAB_트렌드지수_20대남성_20210815.csv: 21.08.15 기준 트렌드지수 크롤링 데이터 (20대남성)
- DATALAB_트렌드지수_20대여성_20210815.csv
- DATALAB_트렌드지수_30대남성_20210815.csv
- DATALAB_트렌드지수_30대여성_20210815.csv
- NAVER_카테고리목록_식품_210708: NAVER에서 21.07/08에 제공하는 카테고리목록
