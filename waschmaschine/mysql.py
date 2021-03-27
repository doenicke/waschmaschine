import pandas as pd
import config
import logging
import os
from sqlalchemy import create_engine

logging.basicConfig()
logger = logging.getLogger(__name__)


def establish_db_connection():
    print("Establishing DB connection...")
    # db_connection_str = 'mysql+pymysql://mysql_user:mysql_password@mysql_host/mysql_db'
    return create_engine(config.db_connection_str)


def read_from_sql(date_from=None, date_til=None):
    """Read raw Watt data from MySQL database and write it into pickle cache file.

    Returns:
        DataFrame: timestamp, value (float)
    """
    query = "SELECT timestamp, CAST(value AS DECIMAL(8,1)) AS value FROM history " \
            "WHERE device = 'Gosund_Waschmaschine' AND value > 0 "
    if date_from:
        query += f"AND timestamp >= '{str(date_from)}' "
    if date_til:
        query += f"AND timestamp <= '{str(date_til)}' "
    query += "ORDER BY timestamp ASC"

    print("Reading data from MySQL database...")
    print(query)
    db_connection = establish_db_connection()
    df = pd.read_sql(query, con=db_connection)
    return df


def load_cache_file(pkl_file):
    print(f"Reading data from cache file {pkl_file}...")
    df = pd.read_pickle(pkl_file)
    return df


def write_cache_file(df, pkl_file):
    print("Writing cache file...", end='')
    df.to_pickle(pkl_file)
    print(" {} ({:.1f} kB)".format(pkl_file, os.path.getsize(pkl_file) / 1024))


if __name__ == '__main__':
    df = read_from_sql()
    print(df)
    write_cache_file(df, config.db_cache_file)

    df1 = load_cache_file(config.db_cache_file)
    print(df1)
