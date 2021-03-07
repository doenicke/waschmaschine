"""Waschmaschine

Predict the resuming time
"""
import pandas as pd
import config
# import datetime
import logging
import os
import pickle
from sqlalchemy import create_engine
import datetime
import fire
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

logging.basicConfig()
logger = logging.getLogger(__name__)


class Waschmaschine:
    def __init__(self):
        self.db_connection = None
        self.cache_filename = config.db_cache_file
        self.df_raw = None
        self.model = None
        self.model_filename = config.model_file

    def establish_db_connection(self):
        if not self.db_connection:
            print("Establishing DB connection...")
            # db_connection_str = 'mysql+pymysql://mysql_user:mysql_password@mysql_host/mysql_db'
            self.db_connection = create_engine(config.db_connection_str)

    def read_from_sql(self, date_from=None, date_til=None):
        """Read raw Watt data from MySQL database and write it into pickle cache file."""
        query = "SELECT timestamp, CAST(value AS DECIMAL(8,1)) AS value FROM history " \
                "WHERE device = 'Gosund_Waschmaschine' AND value > 0 "
        if date_from:
            query += f"AND timestamp >= '{str(date_from)}' "
        if date_til:
            query += f"AND timestamp <= '{str(date_til)}' "
        query += "ORDER BY timestamp ASC"

        print("Reading data from MySQL database...")
        self.establish_db_connection()
        self.df_raw = pd.read_sql(query, con=self.db_connection)
        return self.df_raw

    def load_cache_file(self):
        print(f"Reading data from cache file {self.cache_filename}...")
        self.df_raw = pd.read_pickle(self.cache_filename)
        return self.df_raw

    def write_cache_file(self):
        print("Writing cache file...", end='')
        self.df_raw.to_pickle(self.cache_filename)
        print(" {} ({:.1f} kB)".format(self.cache_filename, os.path.getsize(self.cache_filename) / 1024))

    def split_into_sessions(self, duration_min=0, verbose=False):
        """
        DataFrame mit allen Waschvorgängen auftrennen und unrelevante Messwerte verwerfen:
        """
        if duration_min == 0:
            duration_min = config.duration_min
        df_sessions = []
        watt = []
        ende = False
        for index, row in self.df_raw.iterrows():
            value = row['value']
            if value > 1:
                watt.append(value)
                ende = False
            elif value <= 1 and not ende:

                if len(watt) > duration_min:  # Nur Vorgänge berücksichtigen, die länger als x Minuten dauern
                    watt.append(value)
                    watt = [0] * config.n_features + watt
                    rest = list(range(len(watt) - 1, -1, -1))
                    betrieb = [0] * config.n_features + list(range(0, len(watt) - config.n_features))
                    if verbose:
                        print('Watt:', watt, len(watt))
                        print('Betrieb:', betrieb, len(betrieb))
                        print('Rest:', rest, len(rest))
                        print('')

                    session_dict = {'watt': watt, 'betrieb': betrieb, 'rest': rest}
                    df_sessions.append(pd.DataFrame(session_dict))

                watt = []
                ende = True

        return df_sessions

    def get_last_session(self):
        df_sessions = self.split_into_sessions(duration_min=2, verbose=True)
        return df_sessions[-1]

    @staticmethod
    def transform_session(df_src, n_features):
        """
        Jeder Waschvorgang wird in n Abschnitte unterteilt.
        Jeder Abschnitt ist dann n Minuten lang.
        Hinzu kommt noch das Feature der Betriebszeit.

        :param df_src:
        :param n_features: Anzahl der Features mit Messwerten
        :return:
        """
        data_list = []
        for row in range(len(df_src) - n_features + 1):
            df_tmp = df_src.iloc[row:row + n_features]
            # print(df_tmp)
            row = [list(df_tmp['betrieb'])[-1]] + list(df_tmp['watt']) + [list(df_tmp['rest'])[-1]]
            # print(row)
            data_list.append(row)
        return data_list

    def create_model(self, test_size=0.15):
        # For prediction we create a wide table (Dataframe) having following features:
        # past minutes and n watt (e.g. 60 watt values per 60 minutes)
        # Label is the resuming time:
        cols_features = ['betrieb'] + list(range(config.n_features))
        cols_label = ['rest']
        df = pd.DataFrame(columns=cols_features + cols_label)

        # Create list of washing sessions (list of Dataframes):
        df_sessions = self.split_into_sessions()

        # Each washing session gets this layout:
        for df_session in df_sessions:
            data = self.transform_session(df_session, config.n_features)
            df_tmp = pd.DataFrame(data, columns=cols_features + cols_label)
            df = df.append(df_tmp, ignore_index=True)

        print(df)
        print('Count washing sessions:', len(df_sessions))

        df_features, df_label = df[cols_features], df[cols_label]
        x_train, x_test, y_train, y_test = train_test_split(
            df_features, df_label, test_size=test_size, random_state=42)

        print("Training the model...")
        regr = ExtraTreesRegressor(random_state=42, n_estimators=15, max_depth=10)
        regr.fit(x_train, y_train.values.ravel())
        y_pred = regr.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        print('Score: ', regr.score(x_test, y_test))
        print('MSE:   ', mse)
        self.model = regr
        return self.model

    def dump_model(self):
        print("Dumping model into file...", end='')
        pickle.dump(self.model, open(self.model_filename, 'wb'))
        print(" {} ({:.1f} MB)".format(self.model_filename, os.path.getsize(self.model_filename) / 1024 / 1024))

    def load_model(self):
        """Load trained model from pickle file and predict resuming time."""
        regr = pickle.load(open(self.model_filename, 'rb'))
        self.model = regr
        return self.model

    @staticmethod
    def get_test_data():
        df = wm.load_cache_file()
        df = df[(df['timestamp'] >= '2021-02-27 00:00:00')
                & (df['timestamp'] <= '2021-02-27 11:30:00')]
        return df


class MainApp(object):
    def __init__(self):
        self._wm = Waschmaschine()

    def train(self, src='db'):
        if src.lower() == 'db':
            self._wm.read_from_sql()
        else:
            self._wm.load_cache_file()

        print(self._wm.df_raw)
        self._wm.create_model()
        self._wm.dump_model()

    def predict(self, test=True):
        if not test:
            self._wm.read_from_sql(date_from=datetime.date.today())
        else:
            df = self._wm.load_cache_file()
            self._wm.df_raw = df[(df['timestamp'] >= '2021-02-27 00:00:00')
                                 & (df['timestamp'] <= '2021-02-27 11:30:00')]

        print("Last record:")
        print(self._wm.df_raw.tail(1))

        # Session is only running if last value is greater than 1:
        if self._wm.df_raw.tail(1)[['value']].values[0] <= 1:
            return


if __name__ == '__main__':
    # date_start = datetime.date(2021, 1, 8)
    # date_end = datetime.date(2021, 1, 9)

    # fire.Fire(MainApp)

    wm = Waschmaschine()
    wm_runtime = Waschmaschine()
    wm_runtime.df_raw = wm.get_test_data()
    print(wm_runtime.df_raw)
    df = wm_runtime.get_last_session()
    data = wm_runtime.transform_session(df, config.n_features)
    data_x = data # [data[0], data[1]]
    print(data_x)

    cols_features = ['betrieb'] + list(range(config.n_features))
    df_x = pd.DataFrame(data_x, columns=cols_features)
    print(df_x)
