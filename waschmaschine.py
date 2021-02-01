import pandas as pd
import config
# import datetime
# import argparse
# import textwrap
import logging
from sqlalchemy import create_engine
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

logging.basicConfig()
logger = logging.getLogger(__name__)


def read_sql(db_connection_str):
    # db_connection_str = 'mysql+pymysql://mysql_user:mysql_password@mysql_host/mysql_db'
    db_connection = create_engine(db_connection_str)

    # date_start = datetime.date(2021, 1, 8)
    # date_end = datetime.date(2021, 1, 9)

    query = "SELECT timestamp, CAST(value AS DECIMAL(8,1)) AS value FROM history " \
            "WHERE device = 'Gosund_Waschmaschine' AND value > 0 " \
            "AND timestamp < '2025-01-10' " \
            "ORDER BY timestamp ASC"
    # AND timestamp BETWEEN '{}' AND '{}' " \
    # .format(date_start, date_end)

    return pd.read_sql(query, con=db_connection)


def split_into_train_df(df_src, n_features):
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
        # print(df)
        row = [list(df_tmp['betrieb'])[-1]] + list(df_tmp['watt']) + [list(df_tmp['rest'])[-1]]
        # print(row)
        data_list.append(row)
    return data_list


def split_into_session_df(df_src):
    """
    DataFrame mit allen Waschvorgängen auftrennen und unrelevante Messwerte verwerfen:
    """
    df_sessions = []
    watt = []
    ende = False
    for index, row in df_src.iterrows():
        value = row['value']
        if value > 1:
            watt.append(value)
            ende = False
        elif value <= 1 and not ende:

            if len(watt) > config.duration_min:  # Nur Vorgänge berücksichtigen, die länger als x Minuten dauern
                watt.append(value)
                watt = [0]*config.n_features + watt
                rest = list(range(len(watt) - 1, -1, -1))
                betrieb = [0]*(config.n_features) + list(range(0, len(watt)-config.n_features))
                print('Watt:', watt, len(watt))
                print('Betrieb:', betrieb, len(betrieb))
                print('Rest:', rest, len(rest))
                print('')

                session_dict = {'watt': watt, 'betrieb': betrieb, 'rest': rest}
                df_sessions.append(pd.DataFrame(session_dict))

            watt = []
            ende = True

    return df_sessions


def model_test(df_x, df_y):
    X_train, X_test, y_train, y_test = train_test_split(
        df_x, df_y, test_size=0.2, random_state=42)

    print("ExtraTreesRegressor")
    model = ExtraTreesRegressor(random_state=42)
    model.fit(X_train, y_train.values.ravel())
    # model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print('Score: ', model.score(X_test, y_test))
    print('MSE:   ', mse)

# def commandline_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("session_file", help="Load session file")
#     # ap.add_argument("-p", "--predict", default=None, help="Predict")
#     ap.formatter_class = argparse.RawDescriptionHelpFormatter
#     ap.description = textwrap.dedent(__doc__)
#     return vars(ap.parse_args())


if __name__ == '__main__':
    # args = commandline_args()
    if config.use_sql_cache:
        print("Lese Daten aus SQL-Cache:", config.sql_cache_file)
        df_sql = pd.read_pickle(config.sql_cache_file)
    else:
        print("Lese Daten aus MySQL-Tabelle.")
        df_sql = read_sql(config.db_connection_str)
        df_sql.to_pickle('df_sql.pkl')

    print(df_sql)
    df_sessions = split_into_session_df(df_sql)

    cols_features = ['betrieb'] + list(range(config.n_features))
    cols_label = ['rest']
    df = pd.DataFrame(columns=cols_features+cols_label)

    for df_session in df_sessions:
        data = split_into_train_df(df_session, config.n_features)
        df_tmp = pd.DataFrame(data, columns=cols_features+cols_label)
        df = df.append(df_tmp, ignore_index=True)
    print(df)
    print('Anz. Waschvorgänge:', len(df_sessions))

    df_features = df[cols_features]
    df_label = df[cols_label]
    model_test(df_features, df_label)

    model = ExtraTreesRegressor(random_state=42)
    model.fit(df_features, df_features)



