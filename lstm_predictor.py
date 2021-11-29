import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow import keras
from keras.models import load_model
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import RootMeanSquaredError
from keras import backend as K

# lunar_col = month + days
# weekday_col = 7 days
LUNAR_COL = [_ for _ in range(0, 46)]
DOW_COL = [_ for _ in range(46, 52)]
CATG_COL = 53

class LSTM_predictor:
    def __init__(self, persona, catg_nm, catg_df, lookback_window, forward_window):
        self.persona = persona
        self.catg_nm = catg_nm
        self.catg_df = catg_df
        self.lookback_window = lookback_window
        self.forward_window = forward_window

    def create_dataset(self, df):
        """
        Split time series data into a set of samples consists of (lookback data, data to predict)
        """

        lookback_window = self.lookback_window
        until = len(df) - lookback_window - 1
        for i in range(until):
            sample_X = np.array(df.iloc[i:i + lookback_window + 1, :])
            sample_y = np.array([df.iloc[i + lookback_window + 1, :][self.catg_nm]])

            sample_X = np.expand_dims(sample_X, axis=0)
            sample_y = np.expand_dims(sample_y, axis=0)

            try:
                data_X = np.concatenate((data_X, sample_X), axis=0)
                data_y = np.concatenate((data_y, sample_y), axis=0)
            except NameError:
                data_X = sample_X
                data_y = sample_y

        return data_X, data_y

    def relative_rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_true - y_pred))) / K.mean(y_true)

    def train(self, val_size = 0.1, lstm_units=32):
        # Split train and test data
        boundary = -(self.lookback_window + self.forward_window)
        train_df = self.catg_df.iloc[:boundary, :]
        test_df = self.catg_df.iloc[boundary:, :]

        train_X, train_y = self.create_dataset(train_df)
        test_X, test_y = self.create_dataset(test_df)

        # Split train and validation data
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y,
                                                          train_size=(1-val_size), test_size=val_size,
                                                          random_state=321)

        # Define and Compile LSTM model
        model = keras.Sequential()
        model.add(LSTM(lstm_units, input_shape = train_X.shape[-2:]))  # ({lookback}, feature dim)
        model.add(Dense(1))
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=[RootMeanSquaredError(), self.relative_rmse])

        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = ModelCheckpoint(f"./utils/models/{self.catg_nm}_{self.persona}_{lstm_units}unit.h5",
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto')
        # Fitting
        history = model.fit(train_X, train_y,
                            batch_size = 16,
                            epochs=100,
                            validation_data= (val_X, val_y),
                            callbacks=[early_stop, checkpoint],
                            verbose=1)

        # Draw and save train_loss, val_loss graph
        self.plot_loss(history, ["loss", "val_root_mean_squared_error", "val_relative_rmse"], ['red', 'yellow', 'green'])
        print(f"Finished training and saving loss graph")

        # Test model
        predicted = model.predict(test_X)
        rmse = np.sqrt(mean_squared_error(test_y, predicted))
        r2 = r2_score(test_y, predicted)
        print(f"RMSE: {rmse}\n"
              f"R2 score: {r2}")

        # Final report chart
        plt.plot(self.full_df[self.catg_nm], 'b-o', color='blue', markersize=3, label='original data', linewidth=1)
        plt.plot(test_df.index, predicted, 'b-o', color='red', markersize=3, label='predicted data')
        plt.axvline(x=self.full_df.index[boundary], linestyle='dashed', linewidth=1)
        plt.xticks(rotation=45)
        plt.legend()
        plt.savefig(f"./utils/charts/{self.catg_nm}_model report.png")

    def daily_update(self, lstm_units=32):
        """
        Given a model, we use train on batch method to update the model with new data.
        train_on_batch function accepts a single batch of data, performs backpropagation, and then updates the model parameters
        """
        for epoch in range(100):
            num_epochs = 100
            print("Epoch {0}/{1}".format(epoch+1, num_epochs))
            mean_tr_accuracy = []
            mean_tr_loss = []
            
            for i in range(len(self.catg_df)):
                if i % 100 == 0:
                    print("Done with {0}/{1}".format(i, len(self.catg_df)))
                for j in range(len(self.catg_df[i])):
                    train_accuracy, train_loss = model.train_on_batch(np.expand_dims(self.catg_df[i][j], axis=0))
                    mean_tr_accuracy.append(train_accuracy)
                    mean_tr_loss.append(train_loss)
                model.reset_states()
            
            mean_accuracy = np.mean(mean_tr_accuracy)
            mean_loss = np.mean(mean_tr_loss)
            print("Mean Accuracy", mean_tr_accuracy)
            print("Mean Loss", mean_tr_loss)
            filepath = f"./utils/charts/{0}_model report(update)".format(self.catg_nm)
            model.save_weights(filepath)
        

    def plot_loss(self, history, metrics, colors):
        plt.figure(figsize=(8, 4))
        for metric, color in zip(metrics, colors):
            plt.plot(history.history[metric], color=color)

        plt.title(f"Loss history for {self.catg_nm}_{self.persona}")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()

        plt.savefig(f"./utils/charts/{self.catg_nm}_{self.persona}_loss graph.png")

    def infer(self, model_nm, infer_df):
        infer_df = self.full_df.iloc[-self.config["lookback_window"]:, :]
        infer_X = self.select_features(infer_df)
        infer_X = self.normalize(infer_X, has_scaler=True)

        model = load_model(f"./utils/models/{model_nm}")
        infer_y = model.predict(infer_X)

        # Infer chart
        plt.plot(self.full_df[self.catg_nm], 'b-o', color='blue', markersize=3, label='original data', linewidth=1)
        plt.plot(infer_df.index, infer_y, 'b-o', color='red', markersize=3, label='inferred data')
        plt.xticks(rotation=45)
        plt.legend()
        plt.savefig(f"./utils/charts/{self.catg_nm}_inferred.png")

