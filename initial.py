#-*-coding:utf-8-*-
from lstm_predictor import *
from utils import *
from os import listdir

LOOKBACK_WINDOW = 30
FORWARD_WINDOW = 14
USE_LUNAR=True
USE_DOW=True


if __name__ == "__main__":
    dir = "./data/click index_current/"
    for fname in listdir(dir):
        persona = fname.split("_")[2]
        src_df = pd.read_csv(dir + fname,
                             index_col=["period"],
                             encoding='utf8')
        catgs = src_df.columns

        # Preprocess features
        feat_df = process_features(src_df)
        feat_df = select_features(feat_df, USE_LUNAR, USE_DOW)

        for catg_nm in catgs:
            normalized = normalize(src_df[[catg_nm]], catg_nm=catg_nm, has_scaler=False)
            catg_df = pd.concat((feat_df, normalized), axis=1)

            lstm = LSTM_predictor(persona, catg_nm, catg_df, LOOKBACK_WINDOW, FORWARD_WINDOW)
            lstm.train(val_size=0.1, lstm_units=32)