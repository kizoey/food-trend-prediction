import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil
from sklearn.externals import joblib
import torch
import torch.nn as nn
import torchvision.datasets as dsets

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from korean_lunar_calendar import KoreanLunarCalendar

from keras.layers import Embedding, LSTM
from keras import backend as K

dir_path = '/content'
dir_name = 'scaler'
os.mkdir(dir_path + '/' + dir_name + '/')

dir_path = '/content'
dir_name = 'image'
os.mkdir(dir_path + '/' + dir_name + '/')

df = pd.read_csv("/content/drive/MyDrive/푸드코트/NAVER_카테고리별_트렌드지수_20210708.csv")
df_date = df.copy()

# datetime 형식으로 변환
df_date['date'] = pd.to_datetime(df_date['period'])

df_date['year'] = df_date['date'].dt.year  # 년
df_date['month'] = df_date['date'].dt.month  # 월
df_date['day'] = df_date['date'].dt.day  # 일
df_date['dayname'] = df_date['date'].dt.weekday  # 요일

calendar = KoreanLunarCalendar()

lunar = []
for i in range(len(df_date)):
    calendar.setSolarDate(df_date['year'][i], df_date['month'][i], df_date['day'][i])
    lun = calendar.LunarIsoFormat()
    lunar.append(lun)
print(type(lun))
lunar = pd.DataFrame(lunar)
lunar.columns = ["lunar"]

lunar['lyear'] = lunar['lunar'].str[0:4]  # 년
lunar['lmonth'] = lunar['lunar'].str[5:7]  # 월
lunar['lday'] = lunar['lunar'].str[8:10]  # 일

df_date['year'] = df_date['date'].dt.year.astype(str)
df_date['month'] = df_date['date'].dt.month.astype(str)
df_date['day'] = df_ddate['date'].dt.day.astype(str)
df_date['dayname'] = df_date['date'].dt.weekday.astype(str)

new_df = df_date[['period', 'year', 'month', 'day', 'dayname']]
new_df = pd.concat([new_df, lunar], axis=1)
data = pd.merge(new_df, df.copy())


# 원핫인코딩으로 변환
data_drop = data[['year', 'month', 'day', 'dayname', 'lyear', 'lmonth', 'lday']]

data_drop_year = pd.get_dummies(data_drop['year'])
data_drop_month = pd.get_dummies(data_drop['month'])
data_drop_day = pd.get_dummies(data_drop['day'])
data_drop_dayname = pd.get_dummies(data_drop['dayname'])
data_drop_lyear = pd.get_dummies(data_drop['lyear'])
data_drop_lmonth = pd.get_dummies(data_drop['lmonth'])
data_drop_lday = pd.get_dummies(data_drop['lday'])

list_a = [None, data_drop_year, data_drop_month, data_drop_day, data_drop_dayname, data_drop_lyear, data_drop_lmonth, data_drop_lday]

shutil.rmtree('/content/scaler/')

len(range(9, len(data.columns)))


# 객체를 pickled binary file 형태로 저장한다.
for i in range(9, len(data.columns)):
    data_std = pd.DataFrame(data.iloc[:, i])
    std = MinMaxScaler()
    data.iloc[:, i] = std.fit_transform(data_std)
    file_name = os.path.join('/content/scaler/', 'scaler_' + str(i) + '.pkl')
    joblib.dump(std, file_name)

'''
from sklearn.externals import joblib 
# pickled binary file 형태로 저장된 객체를 로딩한다 
file_name = 'object_01.pkl' 
obj = joblib.load(file_name) 
'''


data.to("cuda")

train_data = data.copy()[:1100]
test_data = data.copy()[1100:-30]
show_data = data.copy()[-30:]


class multi_LSTM:
    import pandas as pd
    import numpy as np
    import os
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from keras import backend as K

    def __init__(self, df, target, colnum, feature, feature_one_hot, train_prop, history, future):
        """
      df: 데이터프레임, 정규화 전 상태여야 함
      target: 테스트 데이터프레임
      colnum: LSTM을 구하기 원하는 열 번호
      feature: 사용할 feature 열 번호 // ex. [1,2,4,6], []
      feature_one_hot:
      train_prop: 훈련데이터 비율
      history: future를 구할 기반 데이터 개수(우리는 21)
      future: LSTM으로 예상할 데이터 개수(우리는 7)
        """
        self.df = df
        self.target = target
        self.colnum = colnum
        self.feature = feature
        self.history = history
        self.future = future

        # 훈련 + 검증 나누기
        self.TRAIN_SPLIT = int(len(df) * train_prop)

        # 데이터
        train_dataset = df.iloc[:, colnum]
        test_dataset = target.iloc[:, colnum]

        # feature + 데이터
        if len(feature) != 0:
            train_feature = pd.DataFrame()
            for i in feature:
                train_feature = pd.concat([train_feature, feature_one_hot[i]], axis=1)
            target_feature = pd.DataFrame()
            for i in feature:
                target_feature = pd.concat([target_feature, feature_one_hot[i]], axis=1)
            target_feature.reset_index(drop=True, inplace=True)

        # x:(history)일치 데이터 y:다음날 데이터 형태로 만들기
        multivariate_past_history = history
        multivariate_future_target = 0

        if len(feature) != 0:
            x_train_multi, y_train_multi = self.multivariate_data(
                pd.concat([train_feature[:1100], train_dataset], axis=1),
                train_dataset,
                0, self.TRAIN_SPLIT,
                multivariate_past_history,
                multivariate_future_target)

            x_val_multi, y_val_multi = self.multivariate_data(pd.concat([train_feature[:1100], train_dataset], axis=1),
                                                              train_dataset,
                                                              self.TRAIN_SPLIT, None,
                                                              multivariate_past_history,
                                                              multivariate_future_target)

            x_test_multi, y_test_multi = self.multivariate_data(
                pd.concat([target_feature[1100:-30], test_dataset], axis=1, ignore_index=True),
                test_dataset,
                0, len(test_dataset) - history,
                multivariate_past_history,
                multivariate_future_target)

        else:
            x_train_multi, y_train_multi = self.multivariate_data(train_dataset,
                                                                  train_dataset,
                                                                  0, self.TRAIN_SPLIT,
                                                                  multivariate_past_history,
                                                                  multivariate_future_target)
            x_val_multi, y_val_multi = self.multivariate_data(train_dataset,
                                                              train_dataset,
                                                              self.TRAIN_SPLIT, None,
                                                              multivariate_past_history,
                                                              multivariate_future_target)
            x_test_multi, y_test_multi = self.multivariate_data(test_dataset,
                                                                test_dataset,
                                                                0, len(test_dataset) - history,
                                                                multivariate_past_history,
                                                                multivariate_future_target)

        # 배치 사이즈, 버퍼 사이즈 조정
        if len(x_val_multi) > 256:
            BATCH_SIZE = 128
        elif len(x_val_multi) <= 256 and len(x_val_multi) > 128:
            BATCH_SIZE = 64
        elif len(x_val_multi) <= 128 and len(x_val_multi) > 64:
            BATCH_SIZE = 32
        elif len(x_val_multi) <= 64 and len(x_val_multi) > 32:
            BATCH_SIZE = 16
        else:
            BATCH_SIZE = 8
        BUFFER_SIZE = x_train_multi.shape[0]

        train_multivariate = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
        train_multivariate = train_multivariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_multivariate = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
        val_multivariate = val_multivariate.batch(BATCH_SIZE).repeat()

        # 모델
        model = Sequential()

        embedding_length = 5

        # model.add(Embedding( , ,))
        model.add(LSTM(32, input_shape=x_train_multi.shape[-2:]))  # (21일치, 465개, ~)일 때 input = 465
        model.add(Dense(1))

        model.compile(optimizer='adam', loss=my_loss)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        model_path = 'model'
        filename = os.path.join(model_path, 'tmp_checkpoint' + str(colnum) + '.h5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True,
                                                        mode='auto')

        h = model.fit(train_multivariate, epochs=100,
                      steps_per_epoch=200,
                      validation_data=val_multivariate, validation_steps=10,
                      callbacks=[early_stop, checkpoint])

        # 향후 (future)일 데이터를 예측한다.
        j = 0

        nFuture = future
        if len(y_test_multi) > 50:
            self.lastData = np.copy(y_test_multi[j:j + history])  # 원 데이터의 앞 21개만 그려본다
        else:
            self.lastData = np.copy(y_test_multi)

        dx = x_test_multi[j + history].copy()

        if len(feature) != 0:
            self.estimate = [x_test_multi[j + history][-1][train_feature.shape[1]]]
        else:
            self.estimate = [x_test_multi[j + history][-1][len(feature)]]

        self.realData = [y_test_multi[j + history - 1]]

        for i in range(nFuture):
            # (history)일 만큼 입력데이로 다음 값을 예측한다
            if len(feature) != 0:
                px = dx[-history:].reshape(1, history, train_feature.shape[1] + 1)
            else:
                px = dx[-history:].reshape(1, history, len(feature) + 1)
            # 다음 값을 예측한다.
            yHat = model.predict(px)[0][0]
            # 예측값을 저장해 둔다
            self.estimate.append(yHat)
            # 이전 예측값을 포함하여 또 다음 값을 예측하기위해 예측한 값을 저장해 둔다
            dx = np.insert(dx, len(dx), x_test_multi[j + history + i + 1].copy()[-1], axis=0)
            if len(feature) != 0:
                dx[-1][train_feature.shape[1] - 1] = yHat
            else:
                dx[-1] = yHat
            # plot하기 위해 y값도 같은 폼으로 고친다
            self.realData.append(y_test_multi[j + history + i])

        # 결과물
        self.real = self.realData[1:]
        self.prediction = self.estimate[1:]
        self.deviation = []
        for day in range(len(self.real)):
            dev = self.real[day] - self.prediction[day]
            self.deviation.append(dev)

    ########## history, future 함수 ###########
    def multivariate_data(self, dataset, target, start_index, end_index, history_size, target_size):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i)
            if len(self.feature) != 0:
                data.append(dataset.iloc[indices])
            else:
                data.append(np.reshape([dataset.iloc[indices]], (history_size, 1)))
            labels.append(target.iloc[i + target_size])
        return np.array(data, dtype=np.float), np.array(labels, dtype=np.float)

    def loss_figure(self):
        plt.figure(figsize=(8, 4))
        plt.plot(h.history['loss'], color='red')
        plt.title("Loss History for " + df.columns[colnum])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

    def show_plot(self):
        ax1 = self.target['period'][self.history:self.history * 2].values
        ax2 = self.target['period'][self.history * 2 - 1:self.history * 2 + self.future].values

        plt.figure(figsize=(16, 4))
        plt.plot(ax1, self.lastData, 'b-o', color='blue', markersize=3, label='Time series', linewidth=1)
        plt.plot(ax2, self.realData, 'b-o', color='green', markersize=3, label='Origin Data')
        plt.plot(ax2, self.estimate, 'b-o', color='red', markersize=3, label='Estimate')
        plt.axvline(x=ax1[-1], linestyle='dashed', linewidth=1)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    def save_plot(self):
        ax1 = self.target['period'][self.history:self.history * 2].values
        ax2 = self.target['period'][self.history * 2 - 1:self.history * 2 + self.future].values

        plt.figure(figsize=(16, 4))
        plt.plot(ax1, self.lastData, 'b-o', color='blue', markersize=3, label='Time series', linewidth=1)
        plt.plot(ax2, self.realData, 'b-o', color='green', markersize=3, label='Origin Data')
        plt.plot(ax2, self.estimate, 'b-o', color='red', markersize=3, label='Estimate')
        plt.axvline(x=ax1[-1], linestyle='dashed', linewidth=1)
        plt.xticks(rotation=45)
        plt.legend()

        image_path = 'image'
        image_name = os.path.join(image_path, 'graph_' + self.df.columns[self.colnum] + '.png')
        print(image_name)
        plt.savefig(image_name)

    def output(self):
        return self.real, self.prediction, self.deviation

    def all_table(self):
        a = pd.concat([pd.DataFrame(self.real), pd.DataFrame(self.prediction), pd.DataFrame(self.deviation)], axis=1)
        a.columns = ['real', 'prediction', 'deviation']
        return a

    def dev_table(self):
        b = pd.DataFrame(self.deviation)
        b.columns = [self.df.columns[self.colnum]]
        b.index.name = "deviation"
        return b


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# 손실함수 정의
def my_loss(y_true, y_pred):
    # 내가 정의한 손실 함수
    loss = K.sqrt(K.mean(K.square(y_true - y_pred))) / K.mean(y_true)
    return loss


test0001 = multi_LSTM(train_data, test_data, 10, [], list_a, 0.7, 60, 30)

# 60일로 30일 추측
test0001.show_plot()

data_drop

test0001 = multi_LSTM(train_data, test_data, 10, [1], list_a, 0.7, 60, 30)
test0001.show_plot()

test0002 = multi_LSTM(train_data, test_data, 10, [2], list_a, 0.7, 60, 30)
test0002.show_plot()

test0003 = multi_LSTM(train_data, test_data, 10, [3], list_a, 0.7, 60, 30)
test0003.show_plot()

test0004 = multi_LSTM(train_data, test_data, 10, [4], list_a, 0.7, 60, 30)
test0004.show_plot()

test0005 = multi_LSTM(train_data, test_data, 10, [5], list_a, 0.7, 60, 30)
test0005.show_plot()

test0006 = multi_LSTM(train_data, test_data, 10, [6], list_a, 0.7, 60, 30)
test0006.show_plot()

test0007 = multi_LSTM(train_data, test_data, 10, [7], list_a, 0.7, 60, 30)
test0007.show_plot()

test0008 = multi_LSTM(train_data, test_data, 10, [1, 2], list_a, 0.7, 60, 30)
test0008.show_plot()

test0009 = multi_LSTM(train_data, test_data, 10, [1, 3], list_a, 0.7, 60, 30)
test0009.show_plot()

test0010 = multi_LSTM(train_data, test_data, 10, [1, 4], list_a, 0.7, 60, 30)
test0010.show_plot()

test0011 = multi_LSTM(train_data, test_data, 10, [2, 3], list_a, 0.7, 60, 30)
test0011.show_plot()

test0012 = multi_LSTM(train_data, test_data, 10, [2, 4], list_a, 0.7, 60, 30)
test0012.show_plot()

test0013 = multi_LSTM(train_data, test_data, 10, [3, 4], list_a, 0.7, 60, 30)
test0013.show_plot()

test0014 = multi_LSTM(train_data, test_data, 10, [5, 6], list_a, 0.7, 60, 30)
test0014.show_plot()

test0015 = multi_LSTM(train_data, test_data, 10, [5, 7], list_a, 0.7, 60, 30)
test0015.show_plot()

test0016 = multi_LSTM(train_data, test_data, 10, [6, 7], list_a, 0.7, 60, 30)
test0016.show_plot()

test0017 = multi_LSTM(train_data, test_data, 11, [2, 4], list_a, 0.7, 60, 30)
test0017.show_plot()

test0017 = multi_LSTM(train_data, test_data, 11, [7], list_a, 0.7, 60, 30)
test0017.show_plot()

test0017 = multi_LSTM(train_data, test_data, 11, [4], list_a, 0.7, 60, 30)
test0017.show_plot()

test0017 = multi_LSTM(train_data, test_data, 9, [4], list_a, 0.7, 60, 30)
test0017.show_plot()

test0017 = multi_LSTM(train_data, test_data, 9, [3], list_a, 0.7, 60, 30)
test0017.show_plot()

test0017 = multi_LSTM(train_data, test_data, 9, [3, 4], list_a, 0.7, 60, 30)
test0017.show_plot()
