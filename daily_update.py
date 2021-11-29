from lstm_predictor import *
from utils import *
from os import listdir

LOOKBACK_WINDOW = 30
FORWARD_WINDOW = 14
USE_LUNAR=True
USE_DOW=True


if __name__ == "__main__":
    dir = "./data/click index_future/"
    for fname in listdir(dir):
        persona = fname.split("_")[2]
        # src_df_new : new source of data, src_df: original data
        src_df_new = pd.read_csv(dir + fname,
                                index_col=["period"],
                                encoding='utf8')
        catgs = src_df_new.columns  # cateogries of crawled data

        # Preprocess features
        feat_df = process_features(src_df_new)
        # Select features (lunar date, dayofweek)
        feat_df = select_features(feat_df, USE_LUNAR, USE_DOW)

        for catg_nm_n in catgs:
            # if scalar data, normalize with MinMaxScaler()
            normalized = normalize(src_df_new[[catg_nm]], catg_nm=catg_nm, has_scaler=False)
            catg_df = pd.concat((feat_df, normalized), axis=1)

            lstm = LSTM_predictor(persona, catg_nm, catg_df, LOOKBACK_WINDOW, FORWARD_WINDOW)
            lstm.daily_update(lstm_units=32)