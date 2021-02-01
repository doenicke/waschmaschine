import pandas as pd
import config
# import datetime
# import argparse
# import textwrap
import logging
from sqlalchemy import create_engine
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

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

            if len(watt) > 20:  # Nur Vorgänge berücksichtigen, die länger als 20 Minuten dauern
                watt.append(value)
                watt = [0] + watt
                rest = list(range(len(watt) - 1, -1, -1))
                betrieb = list(range(0, len(watt)))
                print('Watt:', watt, len(watt))
                print('Betrieb:', betrieb, len(betrieb))
                print('Rest:', rest, len(rest))
                print('')

                session_dict = {'watt': watt, 'betrieb': betrieb, 'rest': rest}
                df_sessions.append(pd.DataFrame(session_dict))

            watt = []
            ende = True

    return df_sessions


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
    df_label = df[cols_label].values

    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_label, test_size=0.2, random_state=42)

    # https://www.kaggle.com/junkal/selecting-the-best-regression-model

    from sklearn.feature_selection import RFE
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    pipelines = []
    pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])))
    pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])))
    pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])))
    pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])))
    pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])))
    pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))

    results = []
    names = []
    for name, model in pipelines:
        # kfold = KFold(n_splits=10, random_state=21)
        # kfold = KFold(n_splits=10, random_state=None)
        kfold = KFold(n_splits=10, random_state=21, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # KNN
