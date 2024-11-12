from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor, BayesianRidge, LinearRegression, ARDRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import librosa
#from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_percentage_error
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from python_speech_features import mfcc, logfbank, fbank, ssc, delta
import numpy as np

from utils.process_excel import convert_excel2dataframe
from sklearn.metrics import mean_squared_error
#from utils.data_helper import rmse
import time

class FinalModel(BaseEstimator, ClassifierMixin):
    def __init__(self, bits_size=65536, use_mel=True, nfilt=26,
                 samplerate = 16000, winlen=0.025, winstep=0.01, nfft=1024, preemph=0.97,
                 preprocessing_func=2):

        self.bits_size = bits_size
        self.use_mel = use_mel
        self.winlen = winlen
        self.winstep = winstep
        self.nfft = nfft
        self.preemph = preemph
        self.samplerate = samplerate
        self.nfilt = nfilt
        self.preprocessing_func = preprocessing_func
        self.x = 1
    def transform(self, m_signal):
        m1_fft = np.asarray(m_signal)
        m_signal = self.bits_size*(m1_fft-m1_fft.min())/(m1_fft.max() - m1_fft.min())
        
#         mfcc_signal = mfcc(m_signal.astype(int), nfilt=30, samplerate=16000, lowfreq=10, numcep=15)
        
        if self.preprocessing_func == 2:
            # mfcc 與 nearest neighbour filter 
            mfcc_signal = librosa.decompose.nn_filter(mfcc(m_signal.astype(int),
                                    samplerate=self.samplerate,
                                    winlen=self.winlen,
                                    winstep=self.winstep,
                                    nfilt=self.nfilt,
                                    nfft=self.nfft,
                                    preemph=self.preemph))
            print('mfcc_signal.shape:',mfcc_signal.shape)
            print(self.x)
            self.x = self.x + 1
            
            
        
        else:
            # log fbank 前處理，類似 mfcc 但少了DCT步驟，出來的是頻率窗口，不是時間
            mfcc_signal = np.log(fbank(m_signal.astype(int),
                                    
                                    samplerate=self.samplerate,
                                    winlen=self.winlen,
                                    winstep=self.winstep,
                                    nfilt=self.nfilt,
                                    nfft=self.nfft,
                                    preemph=self.preemph)[0])
#         mfcc_signal = ssc(m_signal.astype(int))

        return mfcc_signal

    def fit(self, X,y):
        X = np.array(X)
        
        x_train = self.transform_data(X) # 前處理
        
        print('x_train.shape:',x_train.shape)   
        pca = PCA(0.99, whiten=True) # 利用PCA 還原回input 有0.99 準確率
        pca_train = pca.fit_transform(x_train)
        
        print('fit.shape:',pca_train.shape)
        #adaboost = AdaBoostRegressor(base_estimator=LinearRegression())
        adaboost = LinearRegression() # better
        adaboost.fit(pca_train, y)

        bagging = BaggingRegressor(base_estimator=BayesianRidge(
             lambda_1=1e-06,
             lambda_2=1e-07,
             alpha_1=1e-07,
             alpha_2=1e-05,
        ))
        #bagging = SVR(C=50)
        bagging.fit(pca_train, y)

        self.trained_pca = pca
        self.trained_adaboost = adaboost
        self.trained_bagging = bagging

        # save model
        # joblib.dump(pca, 'pca.pkl') 
        # joblib.dump(adaboost, 'adaboost.pkl')
        # joblib.dump(bagging, 'bagging.pkl')

    def predict(self, X):
        # load model
        # self.trained_pca = joblib.load('pca.pkl')
        # self.trained_adaboost = joblib.load('adaboost.pkl')
        # self.trained_bagging = joblib.load('bagging.pkl')
        X = np.array(X)

        x_input = self.trained_pca.transform(self.transform_data(X))
        y_adaboost = self.trained_adaboost.predict(x_input)
        y_bagging = self.trained_bagging.predict(x_input)

        #return (y_adaboost + y_bagging)/2.0
        return (y_adaboost + y_bagging)/2.0


    def get_params(self, deep=True):
        return {'nfilt': self.nfilt, 'samplerate': self.samplerate, 'preemph': self.preemph,
               'nfft': self.nfft, 'winstep': self.winstep, 'winlen': self.winlen,
               'use_mel': self.use_mel, 'bits_size': self.bits_size}

    def transform_data(self, signal):
        
        print(signal.shape)
        return np.asarray([ np.asarray([ self.transform(e) for e in row ]).flatten() for row in signal ] )

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def generate_prediction(data_path=r'C:\KTF\台科\大數據比賽\2018初賽訓練數據\初賽訓練數據\806初賽訓練數據', predict_path=r'C:\KTF\台科\大數據比賽\2018比賽測驗集'):
    from utils.process_excel import convert_excel2dataframe
    df = convert_excel2dataframe(data_path)
    X = np.asarray([ np.asarray(df.m1.values), np.asarray(df.m2.values), np.asarray(df.m3.values), np.asarray(df.m4.values) ])
    y = np.asarray(df.quality.values)
    X = X.T
    final = FinalModel()
    final.fit(X, y)

    df = convert_excel2dataframe(predict_path, is_training_data=False)
    X = np.asarray([ np.asarray(df.m1.values), np.asarray(df.m2.values), np.asarray(df.m3.values), np.asarray(df.m4.values) ])
    X = X.T

    y_results = final.predict(X)
    print(y_results)

if __name__ == "__main__":
    from utils.process_excel import convert_excel2dataframe
    from utils.data_helper import rmse
    from sklearn.model_selection import StratifiedKFold, KFold
    data_path = r'C:\KTF\台科\大數據比賽\2018初賽訓練數據\初賽訓練數據\806初賽訓練數據'
    df = convert_excel2dataframe(data_path)

    X = np.asarray([ np.asarray(df.m1.values), np.asarray(df.m2.values), np.asarray(df.m3.values), np.asarray(df.m4.values) ])
    y = np.asarray(df.quality.values)
    X = X.T
    for n_split in [4,5,8]: # [4,5,8]
        total = []
        for i in range(10):
            kfold = KFold(n_splits=n_split, shuffle=True)
            for k, (train, test) in enumerate(kfold.split(X, y)):
                final = FinalModel()
                final.fit(X[train], y[train])
                print('------')
                y_pred = final.predict(X[test])
                rmse_result = mean_squared_error(y_pred, y[test],squared=False)
                print('Testing quality')
                print(y[test])
                print('Predicted quality')
                print(y_pred)
                print('RMSE: {:.4f}\n'.format(rmse_result))
                total.append(rmse_result)
        total = np.array(total)
        print('{} fold, RMSE Avg: {:.4f}, Var: {:.4f}'.format(n_split, total.mean(), total.var()))
