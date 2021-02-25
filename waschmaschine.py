"""Waschmaschine

Predict the resuming time
"""
import pickle
import pandas as pd
import config
# import datetime
import fire
import logging
import os
from sqlalchemy import create_engine
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from transform import transform_data

logging.basicConfig()
logger = logging.getLogger(__name__)


class Waschmaschine:
    def read_sql(self):
        """Read raw Watt data from MySQL database and write it into pickle cache file."""
        print("Reading data from MySQL database...")
        # db_connection_str = 'mysql+pymysql://mysql_user:mysql_password@mysql_host/mysql_db'
        db_connection = create_engine(config.db_connection_str)

        # date_start = datetime.date(2021, 1, 8)
        # date_end = datetime.date(2021, 1, 9)
        query = "SELECT timestamp, CAST(value AS DECIMAL(8,1)) AS value FROM history " \
                "WHERE device = 'Gosund_Waschmaschine' AND value > 0 " \
                "ORDER BY timestamp ASC"
        #    "AND timestamp < '2025-01-10' " \
        #    "AND timestamp BETWEEN '{}' AND '{}' " \
        # .format(date_start, date_end)
        df = pd.read_sql(query, con=db_connection)

        print("Writing pickle file as local cache...", end='')
        df.to_pickle(config.sql_cache_file)
        print(" {} ({:.1f} kB)".format(config.sql_cache_file, os.path.getsize(config.sql_cache_file) / 1024))

    def train(self):
        """Read data from cache file, train the model and store it at pickle file."""
        print("Reading data from local cache...", config.sql_cache_file)
        df = pd.read_pickle(config.sql_cache_file)
        print(df)

        df_features, df_label = transform_data(df)
        x_train, x_test, y_train, y_test = train_test_split(
            df_features, df_label, test_size=0.2, random_state=42)

        regr = ExtraTreesRegressor(random_state=42, n_estimators=22, max_depth=9)
        regr.fit(x_train, y_train.values.ravel())
        y_pred = regr.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        print('Score: ', regr.score(x_test, y_test))
        print('MSE:   ', mse)

        print("Dumping model into file...", end='')
        pickle.dump(regr, open(config.model_file, 'wb'))
        print(" {} ({:.1f} MB)".format(config.model_file, os.path.getsize(config.model_file) / 1024 / 1024))

    def predict(self):
        """Load trained model from pickle file and predict resuming time."""
        regr = pickle.load(open(config.model_file, 'rb'))


if __name__ == '__main__':
    Waschmaschine.__doc__ = __doc__
    fire.Fire(Waschmaschine)
