import os
import sys
import pdb
import time
import datetime
import numpy as np
import pandas as pd
import random
import multiprocessing as mp
from loguru import logger
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from model_lstmwiththoughts import LSTMWithThoughts

class RNN():
    def __init__(self, params):
        self.model_path = params['model_path']
        self.prediction_path = params['prediction_path']
        self.model_params = params['model_params']
        
        self.timesteps = params['timesteps']
        self.num_features = params['num_features']
        self.valid_length = params['valid_length']
        self.label = params['label']

        self.lr = self.model_params['lr']
        self.batch_size = self.model_params['batch_size']
        self.epochs = self.model_params['epochs']
        self.units = self.model_params['units']
        self.thought_dim = self.model_params['thought_dim']
        self.num_thoughts = self.model_params['num_thoughts']
        self.dropout_rate = self.model_params['dropout_rate']

        self.random_seed = 666
        # self.patience = 5

        self.stocks = None

    def load_data(self):
        self.X_data = pd.read_csv('上证股票技术因子数据200101-241217_cleaned.csv')
        self.y_data = pd.read_csv('label_monthly_return.csv')[['Stkcd', 'Trdmnt', 'Mretwd', 'Mretwd_scaled']]
        self.y_data['Trdmnt'] = pd.to_datetime(self.y_data['Trdmnt'], format='%y-%b').dt.strftime('%Y%m')
        self.stocks = self.y_data['Stkcd'].unique().tolist()

    def load_single_sample(self, date):
        X_data = self.X_data[self.X_data['trade_date'] < int(date)].sort_values(['ts_code', 'trade_date'], ascending=[True, False])
        X_data = X_data.groupby('ts_code').head(self.timesteps).sort_values(['ts_code', 'trade_date'], ascending=[True, True])
        y_data = self.y_data[self.y_data['Trdmnt'] == date[:6]][self.label]
        return X_data.iloc[:, -self.num_features:], y_data

    def model_load_sample(self, sample_dates):
        tasks = []
        X_data, y_data = [], []

        pool = mp.Pool(10)
        for date in sample_dates:
            tasks.append(pool.apply_async(self.load_single_sample, args=(date,)))
        pool.close()
        pool.join()
        
        for task in tasks:
            X, y = task.get()
            X_data.append(X)
            y_data.append(y)

        X_train, y_train, X_valid, y_valid = X_data[:-self.valid_length], y_data[:-self.valid_length], X_data[-self.valid_length:], y_data[-self.valid_length:]
        X_train, y_train, X_valid, y_valid = pd.concat(X_train).values, pd.concat(y_train).values, pd.concat(X_valid).values, pd.concat(y_valid).values
        X_train = X_train.reshape(-1, self.timesteps, self.num_features)
        X_valid = X_valid.reshape(-1, self.timesteps, self.num_features)
        return X_train, y_train, X_valid, y_valid
    
    def correlation_coefficient_loss(self, y_true, y_pred):
        x = y_true
        y = y_pred
        mx = K.mean(x)
        my = K.mean(y)
        xm, ym = x - mx, y - my
        r_num = K.sum(tf.multiply(xm, ym))
        r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
        r = - r_num / r_den
        return r
    
    def get_model(self, sample_dates):
        # fix random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        # load sample data
        X_train, y_train, X_valid, y_valid = self.model_load_sample(sample_dates)
        ## shuffle
        train_shuffle_index, valid_shuffle_index = np.random.permutation(len(X_train)), np.random.permutation(len(X_valid))
        X_train, y_train = X_train[train_shuffle_index], y_train[train_shuffle_index]
        X_valid, y_valid = X_valid[valid_shuffle_index], y_valid[valid_shuffle_index]
        logger.info('Make sample success.')

        # define callbacks
        log_path = os.path.join(self.model_path, 'logs')
        os.makedirs(log_path, exist_ok=True)

        checkpoint = ModelCheckpoint(filepath=os.path.join(log_path, 'model_{epoch:02d}.keras'), monitor='val_loss', save_best_only=False, mode='auto')
        csvlogger = CSVLogger(filename=os.path.join(log_path, 'train_log.csv'), append=True)
        callbacks = [checkpoint, csvlogger]

        # compile model
        loss_fn = self.correlation_coefficient_loss
        model = LSTMWithThoughts(hidden_dim=self.units, thought_dim=self.thought_dim, num_thoughts=self.num_thoughts, dropout_rate=self.dropout_rate)  
        model.compile(optimizer=Adam(float(self.lr)), loss=loss_fn)
        model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks)
        model.save_weights(os.path.join(self.model_path, 'model.weights.h5'))

    def label_predict(self, predict_date):
        # load predict data
        X_pred, y_true = self.load_single_sample(predict_date)
        X_pred = X_pred.values.reshape(-1, self.timesteps, self.num_features)

        # load model
        loss_fn = self.correlation_coefficient_loss
        pred_model = LSTMWithThoughts(hidden_dim=self.units, thought_dim=self.thought_dim, num_thoughts=self.num_thoughts, dropout_rate=self.dropout_rate) 
        pred_model.compile(optimizer=Adam(float(self.lr)), loss=loss_fn)
        dummy_x = tf.ones((1, self.timesteps, self.num_features))
        _ = pred_model(dummy_x)

        # predict outputs
        for epoch in range(1, self.epochs + 1):
            cur_ep = str(epoch).zfill(2)
            pred_model.load_weights(os.path.join(self.model_path, f'logs/model_{cur_ep}.keras'))
            predictions = pred_model.predict(X_pred, batch_size=len(X_pred))
            predictions = pd.DataFrame(predictions)
            predictions.insert(0, 'Stock', self.stocks)
            predictions.columns = ['Stock', 'Prediction']
            save_path = self.prediction_path + '/' + cur_ep
            os.makedirs(save_path, exist_ok=True)
            predictions.to_csv(os.path.join(save_path, predict_date + '.csv'), index=False)


def paralle_run_model(params, retrain_dates, all_retrain_dates):
    params['model_path'] = os.path.join(params['output_path'], 'Models')
    params['prediction_path'] = os.path.join(params['output_path'], 'Predictions')
    # execstr = model + '(params)'
    cur_model = RNN(params)

    logger.info('Load data started')
    cur_model.load_data()
    logger.info('Load data ended')

    for retrain_date in retrain_dates:
        index = all_retrain_dates.index(retrain_date)
        train_dates = all_retrain_dates[index - params['train_length'] : index]
        
        cur_model.model_path = os.path.join(params['output_path'], 'Models', retrain_date)

        logger.info(f'{retrain_date} Training started')
        os.makedirs(cur_model.model_path, exist_ok=True)
        cur_model.get_model(train_dates)
        logger.info(f'{retrain_date} Training ended')

        logger.info(f'{retrain_date} Predict started')
        os.makedirs(cur_model.prediction_path, exist_ok=True)
        cur_model.label_predict(retrain_date)
        logger.info(f'{retrain_date} Predict ended')


if __name__ == '__main__':
    # set parameters
    train_years = ['2023', '2024']
    params = {'output_path': '/Users/sunshine/Desktop/杨光专用/研究生/研一上/投资学/小组项目/Output_v2', 'timesteps': 22, 'num_features': 101, 'train_length': 34, 'valid_length': 6, 'label': 'Mretwd', 'model_params': {'lr': 1e-4, 'batch_size': 256, 'epochs': 10, 'units': 64, 'thought_dim': 32, 'num_thoughts': 5, 'dropout_rate': 0.2}}

    trade_dates = pd.read_csv('trade_dates.csv')['trade_date'].tolist()
    trade_dates = [str(date) for date in trade_dates]
    all_retrain_dates = pd.read_csv('retrain_dates.csv')['retrain_date'].tolist()
    all_retrain_dates = [str(date) for date in all_retrain_dates]
    retrain_dates = [date for date in all_retrain_dates if date[:4] in train_years and date[4:6] in ['01', '04', '07', '10']]

    paralle_run_model(params, retrain_dates, all_retrain_dates)