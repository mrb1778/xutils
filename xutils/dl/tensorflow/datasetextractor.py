# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import io

from google.colab import files
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


# DatasetExtractor v1.1
class DatasetExtractor:

    def __init__(self):
        self.upload = self.Upload()
        self.Upload()
        self.partition = self.Partition()
        self.Partition()
        self.extract = self.Extract()
        self.Extract()
        self.statistics = self.Statistics()
        self.Statistics()

    def cross_sectional_split(self, test_size, random_state=42):  # Per cross-sectional data
        X = pd.DataFrame()
        X = pd.concat([X_numerical, X_categorical], axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def time_series_split(self, date_col, train_lambda, test_lambda, only_X=True):  # Per time-series
        # train_lambda contiene tutte le date prima di una data
        # train_lambda contiene tutte le date dopo di una data
        global X
        global y
        K_slim = X
        if y.empty == False:
            y_col = y.columns
            X = pd.concat([X, y], axis=1)
        K = X  # backup
        X_train = self.partition.select_rows(date_col, train_lambda)
        # ora che abbiamo selezionato le colonne, non esistono colonne di altro tipo, bisogna resettare X
        X_train = X
        X = K
        self.partition.select_rows(date_col, test_lambda)
        X_test = X
        if only_X == True:  # divide in X_train e X_test
            return X_train, X_test
        if only_X == False:  # divide in X_train, y_train, X_test, y_test
            y_train = pd.DataFrame()
            for col in y_col:
                c = X_train.pop(col)
                # c.rename(columns={0:col}, inplace=True)
                y_train = pd.concat([y_train, c])

            y_test = pd.DataFrame()
            for col in y_col:
                c = X_test.pop(col)
                # c.rename(columns={0:col}, inplace=True)
                y_test = pd.concat([y_test, c])
            X = K_slim
            return X_train, X_test, y_train, y_test

    def build_VocabularyList(self, numerical, categorical):
        feature_columns = []
        if categorical.size != 0:
            categorical_columns = categorical.columns
            for col in categorical_columns:
                unique1 = categorical[col].unique()
                feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(col, unique1))

        if numerical.size != 0:
            numerical_columns = numerical.columns
            for col in numerical_columns:
                feature_columns.append(tf.feature_column.numeric_column(col, dtype=tf.float32))

        return feature_columns

    class Upload:
        def __init__(self):
            pass

        def offline_csv(self):
            
            uploaded = files.upload()
            print(uploaded)
            
            global X
            global y
            y = pd.DataFrame()
            X = pd.read_csv(
                io.BytesIO(uploaded[next(iter(uploaded))]),
                # parse_dates=['Date'],
                date_parser=True,
                # index_col="date"
            )
            global col_selected
            global col_total
            col_total = list(X.columns)

        def online_csv(self, csv, header='infer'):
            global X
            global y
            X = pd.read_csv(csv, header=header)
            global col_selected
            global col_total
            col_total = list(X.columns)

        def make_backup(self, *argv):
            # non fa che una copia come backup
            if len(argv) == 0:
                global X
                return X.copy()  # ADD, senza copy non si copia il database, ma come se si esportasse

        def retrieve_backup(self, X_1):
            global X
            global col_total
            X = X_1.copy()
            col_total = list(X_1.columns)

        def remove(self, columns, df=pd.DataFrame()):
            global X
            if df.empty == True:  # non passiamo alcun df
                df = X
            for col in columns:
                df.pop(col)

        def add_single_predict(self, predict_tuple):
            global X
            X.loc[-1] = predict_tuple  # adding a row
            X.index = X.index + 1  # shifting index
            X = X.sort_index()  # sorting by index

        def pop_single_predict(self, position):
            global X
            global y
            # dobbiamo toglierlo sia da X, sia da y

            pred = X.iloc[[position]]
            X.drop(X.index[:position + 1], inplace=True)
            y.drop(y.index[:position + 1], inplace=True)
            # y.iloc[[position]]
            return pred

    class Extract:

        def __init__(self):
            pass

        def partition_safe(self, partition):  # del? ancora si deve vedere se ci serve per categorical_label_encoder
            if str(type(partition)) == "<class 'pandas.core.frame.DataFrame'>":
                # column esiste
                pass
            elif str(type(partition)) == "<class 'pandas.core.series.Series'>":
                # passiamo una sola colonna come Series: column non esiste
                # Quindi trasformiamo la Series in un DataFrame
                empty = pd.DataFrame()
                partition = pd.concat([empty, partition], axis=1)
            return partition

        def categorical_label_encoder(self, col):  # se viene passato un dataframe a + colonne ok, altrimenti err
            # che sia una Series o un Datafram non ci sono problemi, la corregge
            global X_categorical

            # prima si opera su X, poi si staccano y, categorical, numerical
            le = preprocessing.LabelEncoder()
            df = pd.DataFrame()
            le.fit(list(X_categorical[col].unique()))
            partition = X_categorical.pop(col)
            col_encoded = le.transform(partition)
            col_encoded = pd.DataFrame(col_encoded)
            # rinominiamo la heading
            col_encoded.rename(columns={0: col}, inplace=True)
            X_categorical = pd.concat([X_categorical, col_encoded], axis=1)

        def labels(self, y_1):  # None nel caso abbiamo solo labels e categorical / labels e numerical
            global y
            global X
            y = pd.DataFrame()
            # y_1 non è che una lista ['sex', 'generation]
            for k in y_1:
                y = pd.concat([y, X.pop(k)], axis=1)

        def categorical(self, X_categorical_1,
                        label_encoder=False):  # mentre per numerical è default, meglio mettere opzione automatica per categorical
            global X_categorical
            global X
            X_categorical = pd.DataFrame()
            # y_1 non è che una lista ['sex', 'generation]
            if X_categorical_1 == 'all':
                if len(X.columns) != 0:
                    for k in X.columns:
                        p = X.pop(k)
                        X_categorical = pd.concat([X_categorical, p], axis=1)
                        if label_encoder == True:  # trasformiamo tutti in label da qui
                            self.categorical_label_encoder(k)
                    return None

                else:
                    print('Dataset is empty')
                    return None
            for k in X_categorical_1:
                p = X.pop(k)
                X_categorical = pd.concat([X_categorical, p], axis=1)
                if label_encoder == True:
                    self.categorical_label_encoder(k)  # trasformiamo tutti in label da qui

        def numerical(self, X_numerical_1):
            global X_numerical
            global X
            X_numerical = pd.DataFrame()
            # y_1 non è che una lista ['sex', 'generation]
            if X_numerical_1 == 'all':
                if len(X.columns) != 0:
                    for k in X.columns:
                        X_numerical = pd.concat([X_numerical, X.pop(k)], axis=1)
                        X_numerical = X_numerical.astype('float32')
                    return None
                else:
                    print('Dataset is empty')
                    return None
            for k in X_numerical_1:
                X_numerical = pd.concat([X_numerical, X.pop(k)], axis=1)
                X_numerical = X_numerical.astype('float32')

    class Partition:

        def __init__(self):
            pass

        def label_encoder(self, partitions, dict1=None,
                          to_float=False):  # se viene passato un dataframe a + colonne ok, altrimenti err
            # che sia una Series o un Datafram non ci sono problemi, la corregge
            global X
            # prima si opera su X, poi si staccano y, categorical, numerical
            # partition = self.partition_safe(partition) #DEL? Non ne abbiamo bisogno, passiamo sempre una colonna di X
            if dict1 == None:
                for col in partitions:  # we x what we want manually
                    le = preprocessing.LabelEncoder()
                    df = pd.DataFrame()
                    le.fit(list(X[col].unique()))
                    partition = X.pop(col)
                    col_encoded = le.transform(partition)
                    col_encoded = pd.DataFrame(col_encoded)
                    # rinominiamo la heading
                    col_encoded.rename(columns={0: col}, inplace=True)
                    # lo riattacchaimo
                    if to_float == True:
                        col_encoded = col_encoded.astype('float32')
                    X = pd.concat([X, col_encoded], axis=1)

            else:
                # automatic labeling
                dict_counter = 0
                for col in partitions:
                    # stacchiamo una colonna
                    r = X.pop(col)
                    r = r.map(dict1[dict_counter])
                    X = pd.concat([X, r], axis=1)
                    dict_counter += 1

        # we can edit continuous data, adding, multiplying... no row is removed in the process
        def edit_continuous(self, partitions, formula_lambda):
            global X
            for col in partitions:
                r = X.pop(col)
                empty = pd.DataFrame()
                r = pd.concat([empty, r], axis=1)
                for index, row in r.iterrows():
                    row[col] = formula_lambda(row[col])
                X = pd.concat([X, r], axis=1)

        def iterate_list(self, partitions, algo):
            if partitions == 'all':
                for col in df.columns:
                    algo(col)
            else:
                for col in partitions:
                    algo(col)

        # ATTENZIONE, there is a big difference between scaling a partitiono and the entire df
        def scale(self, partitions, scaler='MinMaxScaler', df=pd.DataFrame(), to_float=False, return_df=False):
            # partitions = 'all', le fa una ad una e le riattacca
            # partitions = 'all_df', le fa tutte insieme e trasforma il df in un numpy.array
            global X

            if scaler == 'RobustScaler':
                f_transformer = RobustScaler()
            elif scaler == 'MinMaxScaler':
                f_transformer = MinMaxScaler(feature_range=(0, 1))
            elif scaler == 'StandardScaler':
                f_transformer = StandardScaler()

            if partitions == 'all_df':
                if to_float == True:
                    df = df.astype('float32')
                if df.empty == True:
                    X = df.copy()

                # tutto df deve essere con float32
                df_col = df.columns
                df = f_transformer.fit_transform(df.values)  # ne esce un inspiegabile numpy array
                df = pd.DataFrame(df)
                df.columns = df_col
                if return_df == True:
                    return f_transformer, df
                else:
                    X = df.copy()
                return f_transformer

            def algo(col):
                global X
                f_columns = [] * len(partitions)
                f_columns.append(col)
                f_transformer = f_transformer.fit(X[f_columns].to_numpy())
                X.loc[:, f_columns] = f_transformer.transform(X[f_columns].to_numpy())
                # X.loc[:, f_columns] = f_transformer.transform(X[f_columns].to_numpy())

            if scaler == 'StandardScaler':
                X.loc[:, f_columns] = scaler.inverse_transform(X.loc[:, f_columns])
                # X.loc[:, f_columns] = f_transformer.transform(X[f_columns].to_numpy())
            self.iterate_list(partitions, algo)
            return f_transformer

        def one_hot(self, partitions):

            def algo(col):
                global X
                k = X.pop(col)
                k = pd.get_dummies(k, prefix=col)
                X = pd.concat([X, k], axis=1)

            self.iterate_list(partitions, algo)

        # rows that are not in the threshold are removed
        def select_rows(self, partitions, formula_lambda):
            global X
            for col in partitions:
                X = X[formula_lambda(X[col])]

    class Statistics:

        def cross_validation(self, clf, X, y, cv, return_scores=False):
            
            scores = cross_val_score(clf, X, y, cv=cv)
            acc = ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            if return_scores == True:
                return scores
            else:
                return acc, scores

    def tuning(self, clf, param_grid, X_train, y_train, tuner='GridSearch'):
        # ex.
        # param_grid = {
        # 'n_estimators': [1, 2, 3], 
        # 'criterion': ['gini', 'entropy'],
        # 'min_samples_split': [2, 3, 4, 5],
        # 'bootstrap': [False, True]
        # }

        
        
        # Instantiate the grid search model
        if tuner == 'GridSearch':
            grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, verbose=2, cv=None)
        # Fit the grid search to the data
        clf = grid_search.fit(X_train, y_train)
        return (clf.best_params_)

    def cross_validation(self, clf, X, y, cv, return_scores=False):
        
        scores = cross_val_score(clf, X, y, cv=cv)
        acc = ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        if return_scores == True:
            return scores
        else:
            return acc, scores


v = DatasetExtractor()
# v.upload.online_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
# v.upload.online_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv', header=None)
# v.upload.offline_csv()
# K = v.upload.export_X() #backup
# possiamo permetterci di runnare questo modulo una volta sola, tanto abbiamo i backup
# X.head()

# Ci guida alla costruzione di LSTM e preprocessing






class LSTM_creator:

    def __init__(self):
        self.upload = self.Upload()
        self.preprocessing = self.Preprocessing()
        self.model = self.Model()
        self.Upload()
        self.Preprocessing()
        self.Model()

    class Upload:

        def __init__(self):
            pass

        def offline_csv_parsed(self, parse_dates):  # ['year', 'month', 'day', 'hour']
            
            
            
            
            global X
            global y

            def parse(x):
                return datetime.strptime(x, '%Y %m %d %H')

            y = pd.DataFrame()
            uploaded = files.upload()
            X = pd.read_csv(io.BytesIO(uploaded[next(iter(uploaded))]),
                            # header=0, 
                            index_col=0,
                            parse_dates=[parse_dates],
                            date_parser=parse,
                            )

        # importiamo un csv in modo normalissimo
        def offline_csv(self):  # ['year', 'month', 'day', 'hour']
            
            
            
            
            global X
            global y

            y = pd.DataFrame()
            uploaded = files.upload()
            X = pd.read_csv(io.BytesIO(uploaded[next(iter(uploaded))]),
                            # header=0,
                            )

        def make_backup(self, *argv):
            # non fa che una copia come backup
            if len(argv) == 0:
                global X
                return X.copy()  # ADD, senza copy non si copia il database, ma come se si esportasse

        def retrieve_backup(self, X_1):
            global X
            global col_total
            X = X_1.copy()
            col_total = list(X_1.columns)

    class Preprocessing:
        def chunk_extractor():
            pass

        # date-time parsing function for loading the dataset
        def parser(self, x):
            return datetime.strptime('190' + x, '%Y-%m')

        def transform_to_stationary(self):
            # transform data to be stationary
            global X
            X = X.values  # al di fuori delle funzioni voglio operare solo su un DataFrame
            X = self.difference(X, 1)  # X ritorna ad essere un df

        # create a differenced series
        def difference(self, dataset, interval=1):
            diff = list()
            for i in range(interval, len(dataset)):
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
            return DataFrame(diff)

        # ATTENZIONE: prima riga si elimina
        # convert series to supervised learning
        def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True, drop_col=False, y_var=1):
            n_features = int(len(data.columns))
            n_vars = 1 if type(data) is list else data.shape[1]
            df = DataFrame(data)
            cols, names = list(), list()
            # x sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
            # put it all together
            agg = concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
                data = agg.copy()

            if drop_col == True:
                tot = n_features * n_in + n_features  # 24+8 = 32

                y_name = list(data.columns)[n_features * n_in - 1 + y_var]
                y = data[y_name]
                for i in range(n_features * n_in, tot):
                    data.drop(data.columns[[tot - n_features]], axis=1, inplace=True)
                data = pd.concat([data, y], axis=1)

            return data

        # split data into train and test-sets
        def split(self, test_size, df=pd.DataFrame()):
            # 0.2
            if df.empty == True:
                global X
                df = X.values
            else:
                df = df.values
            len_df = df.shape[0]
            test_size = int(len_df * test_size)
            train, test = df[0:-test_size], df[-test_size:]
            return train, test

        # scale train and test data to [-1, 1]
        def scale(self):  # DEL???
            # fit scaler
            global X
            X = X.values
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler = scaler.fit(X)
            # transform train
            X = X.reshape(X.shape[0], X.shape[1])
            X = scaler.transform(X)
            X = DataFrame(data=X)
            # transform test
            return scaler

    class Model:
        def __init__(self):
            pass

        # make a one-step forecast
        def forecast_lstm(self, model, batch_size, X):
            X = X.reshape(1, 1, len(X))
            yhat = model.predict(X, batch_size=batch_size)
            return yhat[0, 0]

        # inverse scaling for a forecasted value
        def invert_scale(self, scaler, X, value):
            new_row = [x for x in X] + [value]
            array = np.array(new_row)
            array = array.reshape(1, len(array))
            inverted = scaler.inverse_transform(array)
            return inverted[0, -1]

        # invert differenced value
        def inverse_difference(self, history, yhat, interval=1):
            return yhat + history[-interval]


l = LSTM_creator()


class grapher:

    def __init__(self):
        pass

    def graph_rows(self, dataset, columns='all'):
        # load dataset
        values = dataset.values
        # specify columns to plot
        groups = [0] * len(dataset.columns)
        col_counter = 0
        i = 1
        if columns == 'all':
            # si stampano tutte le colonne
            for col in range(0, len(dataset.columns)):
                groups[col] = col_counter
                col_counter += 1
        else:
            groups = []
            max_col = 0
            for col in columns:
                for c in range(0, len(dataset.columns)):
                    if col == list(dataset.columns)[c]:
                        groups.append(c)
                        max_col += 1
                        col_counter += 1
        # plot each column
        pyplot.figure()
        for group in groups:
            pyplot.subplot(len(groups), 1, i)
            pyplot.plot(values[:, group])
            pyplot.title(dataset.columns[group], y=0.5, loc='right')
            i += 1
        pyplot.show()


g = grapher()
