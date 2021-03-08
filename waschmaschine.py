"""Waschmaschine

Predict the resuming time
"""
import pandas as pd
import config
from datetime import datetime, timedelta
import logging
import os
import pickle
from sqlalchemy import create_engine
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

logging.basicConfig()
logger = logging.getLogger(__name__)


class Waschmaschine:
    def __init__(self):
        self.db_connection = None
        self.cache_filename = config.db_cache_file
        self.df_orig = None
        self.sessions = []  # list of Dataframes
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
        print(query)
        self.establish_db_connection()
        self.df_orig = pd.read_sql(query, con=self.db_connection)
        return self.df_orig

    def load_cache_file(self):
        print(f"Reading data from cache file {self.cache_filename}...")
        self.df_orig = pd.read_pickle(self.cache_filename)
        return self.df_orig

    def write_cache_file(self):
        print("Writing cache file...", end='')
        self.df_orig.to_pickle(self.cache_filename)
        print(" {} ({:.1f} kB)".format(self.cache_filename, os.path.getsize(self.cache_filename) / 1024))

    def split_into_sessions(self):
        sessions = []
        session = []

        for index, row in self.df_orig.iterrows():
            if row['value'] > config.standby_watt:
                session.append(row)
            elif len(session) > 0:
                sessions.append(pd.DataFrame(session))
                session = []

        if len(session) > 0:
            sessions.append(pd.DataFrame(session))
        self.sessions = sessions
        return sessions

    def drop_short_sessions(self):
        sessions_new = []
        for session in self.sessions:
            duration = len(session)
            if duration >= config.duration_min:
                sessions_new.append(session)
        self.sessions = sessions_new
        return sessions_new

    def mean_duration(self):
        m = 0
        for session in self.sessions:
            m += len(session)
        return m / len(self.sessions)

    def list_sessions(self):
        for idx, session in enumerate(self.sessions):
            print(idx, session.iloc[0, 0], len(session))

    def get_running_session(self, src='sql'):
        if src == 'sql':
            self.establish_db_connection()
            from_timestamp = datetime.now() - timedelta(hours=3)
            # from_str = from_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            self.read_from_sql(date_from=from_timestamp)
        else:
            df = self.load_cache_file()
            self.df_orig = df[(df['timestamp'] >= '2021-02-27 00:00:00')
                              & (df['timestamp'] <= '2021-02-27 11:30:00')]

        self.split_into_sessions()
        if len(self.sessions) > 0:
            return self.sessions[-1]

    @staticmethod
    def transform_session(df_src, n_features=0):
        """
        Jeder Waschvorgang wird in n Abschnitte unterteilt.
        Jeder Abschnitt ist dann n Minuten lang.
        Hinzu kommt noch das Feature der Betriebszeit.

        :param df_src:
        :param n_features: Anzahl der Features mit Messwerten
        :return:
        """
        if n_features == 0:
            n_features = config.n_features

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


def list_all():
    wm = Waschmaschine()
    wm.load_cache_file()
    wm.split_into_sessions()
    wm.drop_short_sessions()
    wm.list_sessions()


if __name__ == '__main__':
    # list_all()

    run = Waschmaschine()
    df = run.get_running_session(src='test')
    print(df)
    data = run.transform_session(df)
    print(data)
    # run.split_into_sessions()
    # for idx, session in enumerate(run.sessions):
    #     print(idx, session.iloc[0, 0], len(session))
