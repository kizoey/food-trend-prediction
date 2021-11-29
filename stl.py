import pandas as pd
from functools import reduce
import pandas as pd
from statsmodels.tsa.seasonal import STL
import numpy as np
from datetime import datetime


class MovingAverage():
    src_dir = "./data/source/"
    out_dir = "./data/output/"

    TERMS = (5, 10, 20, 60)
    def __init__(self, fname, target_dt, persona):
        # read source file
        self.search_idx = pd.read_csv(self.src_dir + fname, encoding='UTF8', header = 0, index_col=['period'] , parse_dates=['period'])

        # filter until target_dt
        self.search_idx = self.search_idx[self.search_idx.index <= target_dt]
        self.search_idx.loc[:, 'year'] = self.search_idx.index.year
        self.search_idx.loc[:, 'week'] = self.search_idx.index.isocalendar().week

        # target year, week
        self.target_year, self.target_week = self.search_idx.loc[target_dt, 'year'], self.search_idx.loc[target_dt, 'week']

        # category lists
        self.catgs = self.search_idx.drop(['year', 'week'], axis = 1).columns

        # processed data
        self.processed = self.search_idx.loc[:, ['year', 'week']].copy()

        # final output
        # self.trend = self.search_idx.loc[:, ['year', 'week']].copy()
        self.trend_chart = pd.DataFrame()

    def elimSeasonality(self, processed=None):
        if processed:
            self.processed = pd.read_csv(self.src_dir + processed, encoding='utf8', header = 0, index_col=['period'], parse_dates=['period'])
            print("Finished reading STL result")
        else:
            # Eliminate seasonality using STL
            for catg_nm in self.catgs:
                curr = self.search_idx.loc[:, catg_nm]

                # Handle NA values
                curr.replace(0, np.nan, inplace=True)
                curr.dropna(inplace=True)

                stl = STL(curr, period = 365)
                decomposed = stl.fit()

                _deseasonal = curr - decomposed.seasonal
                _deseasonal.name = catg_nm

                self.processed = pd.merge(self.processed, _deseasonal, "left", on=["period"])


            self.processed.to_csv(self.src_dir + f'DATALAB_트렌드지수_{persona}_{target_dt.replace("-","")}_deseasonal.csv')
            print("Finished saving STL result")


    def findTrend(self, shortterm, longterm):
        for catg_nm in self.catgs:
            curr = self.processed[['year', 'week', catg_nm]].copy()

            # moving average
            for term in self.TERMS:
                curr.loc[:, f"{catg_nm}_ma{str(term)}"] = curr[catg_nm].rolling(window=term).mean()
            curr.dropna(inplace=True)

            # golden cross
            curr.loc[:, f'{catg_nm}_golden'] = self.checkGolden(catg_nm, curr, shortterm, longterm)

            if self.isTrend(catg_nm, curr):
                # get trend score and add to chart
                diff, diff_med, diff_avg = self.getScore(catg_nm, curr, shortterm, longterm)
                self.trend_chart.loc[catg_nm, 'diff'] = diff
                self.trend_chart.loc[catg_nm, 'diff_med'] = diff_med
                self.trend_chart.loc[catg_nm, 'diff_avg'] = diff_avg

                # # get golden cross counts during targeted week
                # targeted = (curr["year"] == self.target_year) & (curr["week"] == self.target_week)
                # try:
                #     self.trend_chart.loc[catg_nm, 'accumulator(goledn_cnt)'] = curr.loc[targeted, f'{catg_nm}_golden']\
                #                                                                     .value_counts()[True]
                # except KeyError:
                #     self.trend_chart.loc[catg_nm, 'accumulator(goledn_cnt)'] = 0

                # add index log
                curr.drop(['year', 'week'], axis=1, inplace=True)
                try:
                    self.trend = pd.merge(self.trend, curr, "left", on="period")
                except AttributeError:
                    self.trend = curr

        # drop unneccessary period and add year, week
        self.trend.dropna(inplace=True)
        self.trend = pd.merge(self.search_idx.loc[:, ['year', 'week']], self.trend, "right", on="period")

        self.trend_chart.sort_values('diff', ascending=False, inplace=True)

    def checkGolden(self, catg_nm, catg_log, shortterm, longterm):
        shortterm, longterm = f"{catg_nm}_ma{str(shortterm)}", f"{catg_nm}_ma{str(longterm)}"
        return catg_log[shortterm] >= catg_log[longterm]

    def isTrend(self, catg_nm, catg_log):
        # get rid of stable trend = being trends for 1 month
        # before = (catg_log['year'] == self.target_year) \
        #          & (self.target_week - 4 <= catg_log['week']) \
        #          & (catg_log['week'] < self.target_week)
        # catg_log_bf = catg_log[before]
        # threshold = int(len(catg_log_bf) * 0.7)
        # try:
        #     if catg_log_bf['golden'].value_counts()[True] >= threshold: return False
        # except KeyError: pass

        # not stable 기준으로 이어나가기
        target = (catg_log['year'] == self.target_year) \
                 & (catg_log['week'] == self.target_week)
        catg_log = catg_log[target]

        # threshold = int(len(catg_log) * 0.8)
        threshold = 4
        try:
            golden_cnt = catg_log[f'{catg_nm}_golden'].value_counts()[True]
            if golden_cnt >= threshold:
                self.trend_chart.loc[catg_nm, "accumulator(golden_cnt)"] = golden_cnt
                return True
            return False
        except KeyError:
            return False


    def getScore(self, catg_nm, catg_log, shortterm, longterm):
        shortterm, longterm = f"{catg_nm}_ma{str(shortterm)}", f"{catg_nm}_ma{str(longterm)}"

        targeted = (catg_log['year'] == self.target_year) \
                   & (catg_log['week'] == self.target_week)

        # get difference bwt index of two terms
        catg_log = catg_log[targeted].copy()
        catg_log['diff'] = catg_log[shortterm] - catg_log[longterm]

        # change diff into relative value
        # base = catg_log[[shortterm]].iloc[0, :].values[0]
        diff = reduce(lambda acc, diff: acc + (diff if diff>0 else 0), catg_log['diff'], 0)

        # get median of recent 1 year
        med = catg_log[np.datetime64(target_dt) - np.timedelta64(365, 'D') <= catg_log.index].loc[:, catg_nm].median()
        diff_med = diff / med

        # get mean of recent 1 year
        avg = catg_log[np.datetime64(target_dt) - np.timedelta64(365, 'D') <= catg_log.index].loc[:, catg_nm].mean()
        diff_avg = diff / avg

        # return score/base * 100
        return diff, diff_med, diff_avg

    def save(self, df, fname):
        df.to_csv(self.out_dir + fname, encoding='cp949')

if __name__=='__main__':
    # target_dt = '2018-08-26' # 마라탕
    # target_dt = '2018-08-26' # 마라탕
    target_dt = '2020-08-13' # 크로플
    persona = "20대여성"

    # weaken seasonality
    seasondrop = MovingAverage(fname=f'DATALAB_트렌드지수_{persona}_20210815.csv', target_dt=target_dt, persona=persona)
    seasondrop.elimSeasonality()
    # seasondrop.elimSeasonality(processed = f'DATALAB_트렌드지수_{persona}_20200214_deseasonal.csv')
    seasondrop.findTrend(5, 20)
    seasondrop.save(seasondrop.trend, f"Trend Logs_{target_dt}_STL_{persona}.csv")
    seasondrop.save(seasondrop.trend_chart, f"Trend Chart_{target_dt}_STL_{persona}.csv")
    print("seasondrop saved")

    # short term trend
    # shorttrend = MovingAverage(fname='NAVER_카테고리별_트렌드지수_20210708.csv', target_dt=target_dt)
    # shorttrend.findTrend(5, 20)
    # shorttrend.save(shorttrend.trend, f"Trend Logs_{target_dt}_shorttrend.csv")
    # shorttrend.save(shorttrend.trend_chart, f"Trend Chart_{target_dt}_shorttrend.csv")
    # print("shortterm saved")

    # # long term trend
    # longtrend = MovingAverage(fname='NAVER_카테고리별_트렌드지수_20210708.csv', target_dt=target_dt)
    # longtrend.findTrend(20, 60)
    # longtrend.save(longtrend.trend, f"Trend Logs_{target_dt}_longtrend.csv")
    # longtrend.save(longtrend.trend_chart, f"Trend Chart_{target_dt}_longtrend.csv")
    # print("longterm saved")







