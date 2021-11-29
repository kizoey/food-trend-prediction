import pandas as pd
import numpy as np
from functools import reduce
from korean_lunar_calendar import KoreanLunarCalendar
import joblib
from sklearn.preprocessing import MinMaxScaler

from collector import collect_related, collect_search_index, collect_shopping_index


def process_features(df):
    # Transfer solar date to lunar date
    def add_lunar(date_str, df):
        # idx = date_str.index
        calendar = KoreanLunarCalendar()
        solar_y, solar_m, solar_d = map(int, date_str.split("-"))
        calendar.setSolarDate(solar_y, solar_m, solar_d)

        df.loc[date_str, "lunar_m"], df.loc[date_str, "lunar_d"] = calendar.lunarMonth, calendar.lunarDay

    def onehot(arr, num_classes):
        return np.eye(num_classes, k=-1)[arr.reshape(-1)]

    # Lunar date
    feat_df = pd.DataFrame(df.index).set_index(df.index)
    feat_df.applymap(lambda date_str: add_lunar(date_str, feat_df))
    feat_df = feat_df.astype({'lunar_m': 'int',
                              'lunar_d': 'int'})
    # Day of Week
    feat_df["dow"] = pd.to_datetime(df.index.to_series()).dt.weekday

    # One hot encoding
    m_arr = onehot(feat_df["lunar_m"].to_numpy(), 13)  # dummy dimension +  12 month
    d_arr = onehot(feat_df["lunar_d"].to_numpy(), 32)  # dummy dimension + 31 days
    dow_arr = onehot(feat_df["dow"].to_numpy(), 8)  # dummy dimension + 7 DoW

    feat_df = pd. \
        DataFrame(np.concatenate((m_arr, d_arr, dow_arr), axis=1)). \
        set_index(df.index)
    return feat_df


def select_features(feat_df, use_lunar, use_dow):
    LUNAR_COL = [_ for _ in range(0, 46)]
    DOW_COL = [_ for _ in range(46, 52)]

    # Select feature
    if use_lunar:
        if use_dow:
            return feat_df
        else:
            return feat_df.iloc[:, [LUNAR_COL]]
    else:
        if use_dow:
            return feat_df.iloc[:, [DOW_COL]]
        else:
            return None


def normalize(df, catg_nm, has_scaler):
    fname = f"./utils/scalers/{catg_nm}.save"
    if has_scaler:
        scaler = joblib.load(fname)
        scaled = scaler.transform(df)
    else:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
        joblib.dump(scaler, fname)

    df[catg_nm] = scaled
    return df



def prepare_curr_data(catg_logs=None, related_logs=None):
    if catg_logs is None:
        collect_shopping_index()
    else:
        catg_logs = pd.read_csv("./data/output/" + catg_logs)

    if related_logs is None:
        related_list = collect_related()
        related_logs = collect_search_index(related_list)
    else:
        related_logs = pd.read_csv("./data/output/" + related_logs)


def prepare_future_data(predicted_logs=None, model="LSTM"):
    if predicted_logs is None():
        predicted_logs = model()
    else:
        predicted_logs = pd.read_csv("./data/output/" + predicted_logs)


def preprocess_log(log_df):
    pass


class MovingAverage():
    TERMS = (5, 10, 20, 60)

    def __init__(self, fname, target_dt):
        # read source file
        data_dir = "./data/output/"
        self.search_idx = pd.read_csv(data_dir + fname, encoding='UTF8', header=0, index_col=['period'],
                                      parse_dates=['period'])

        # filter until target_dt
        self.search_idx = self.search_idx[self.search_idx.index <= target_dt]
        self.search_idx.loc[:, 'year'] = self.search_idx.index.year
        self.search_idx.loc[:, 'week'] = self.search_idx.index.isocalendar().week

        # target year, week
        self.target_year, self.target_week = self.search_idx.loc[target_dt, 'year'], self.search_idx.loc[
            target_dt, 'week']

        # category lists
        self.catgs = self.search_idx.drop(['year', 'week'], axis=1).columns

        # final output
        self.trend = self.search_idx.loc[:, ['year', 'week']].copy()
        self.trend_chart = pd.DataFrame()

    def elimSeasonality(self):
        for catg_nm in self.catgs:
            original = self.search_idx.loc[:, [catg_nm]]
            shifted = original.shift(365)
            self.search_idx[catg_nm] = original - shifted

        # drop first year
        self.trend = self.trend.iloc[366:, :]
        self.search_idx = self.search_idx.iloc[366:, :]

    def findTrend(self, shortterm, longterm):
        for catg_nm in self.catgs:
            curr = self.search_idx[['year', 'week', catg_nm]].copy()

            # moving average
            for term in self.TERMS:
                curr.loc[:, 'ma' + str(term)] = curr[catg_nm].rolling(window=term).mean()
            curr.dropna(inplace=True)

            # golden cross
            curr.loc[:, 'golden'] = self.checkGolden(curr, shortterm, longterm)

            if self.isTrend(curr):
                # get trend score and add to chart
                score = self.getScore(curr, shortterm, longterm)
                self.trend_chart.loc[catg_nm, 'score'] = score

                # add index log
                curr.drop(['year', 'week'], axis=1, inplace=True)
                self.trend = pd.concat([self.trend, curr], axis=1)

        self.trend.dropna(inplace=True)  # drop unneccessary year, week
        self.trend_chart.sort_values('score', ascending=False, inplace=True)

    def checkGolden(self, catg_log, shortterm, longterm):
        shortterm, longterm = 'ma' + str(shortterm), 'ma' + str(longterm)
        return catg_log[shortterm] >= catg_log[longterm]

    def isTrend(self, catg_log):
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

        threshold = int(len(catg_log) * 0.8)
        try:
            passed = catg_log['golden'].value_counts()[True] >= threshold
        except KeyError:
            passed = False

        return passed

    def getScore(self, catg_log, shortterm, longterm):
        shortterm, longterm = 'ma' + str(shortterm), 'ma' + str(longterm)

        targeted = (catg_log['year'] == self.target_year) \
                   & (catg_log['week'] == self.target_week)

        # get difference bwt index of two terms
        catg_log = catg_log[targeted].copy()
        catg_log['diff'] = catg_log[shortterm] - catg_log[longterm]

        # change diff into relative value
        base = catg_log[[shortterm]].iloc[0, :].values[0]
        score = reduce(lambda acc, diff: acc + (diff if diff > 0 else 0), catg_log['diff'], 0)

        return score / base * 100

    def save(self, df, fname):
        out_dir = "./data/output/"
        df.to_csv(out_dir + fname, encoding='cp949')


def trend_detector(fname='.csv'):
    target_df = pd.read_csv("./data/output/" + fname)

    non_seasonal = remove_seasonal(target_df)
    trend = detect_trend(non_seasonal)
    save_charts(trend)
    return trend
