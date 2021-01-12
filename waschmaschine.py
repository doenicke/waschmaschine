import pandas as pd
import config
# import datetime

from sqlalchemy import create_engine


def read_sql(db_connection_str):
    # db_connection_str = 'mysql+pymysql://mysql_user:mysql_password@mysql_host/mysql_db'
    db_connection = create_engine(db_connection_str)

    # date_start = datetime.date(2021, 1, 8)
    # date_end = datetime.date(2021, 1, 9)

    query = "SELECT timestamp, CAST(value AS DECIMAL(8,1)) AS value FROM history " \
            "WHERE device = 'Gosund_Waschmaschine' AND value > 0 " \
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
    df_waschvorgaenge = []
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

                waschvorgang_dict = {'watt': watt, 'betrieb': betrieb, 'rest': rest}
                df_waschvorgaenge.append(pd.DataFrame(waschvorgang_dict))

            watt = []
            ende = True

    return df_waschvorgaenge


if __name__ == '__main__':
    if config.use_sql_cache:
        print("Lese Daten aus SQL-Cache:", config.sql_cache_file)
        df_sql = pd.read_pickle(config.sql_cache_file)
    else:
        print("Lese Daten aus MySQL-Tabelle.")
        df_sql = read_sql(config.db_connection_str)
        df_sql.to_pickle('df_sql.pkl')

    df_waschvorgaenge = split_into_session_df(df_sql)

    cols_features = ['betrieb'] + list(range(config.n_features))
    cols_label = ['rest']
    df_train = pd.DataFrame(columns=cols_features+cols_label)

    for df_waschvorgang in df_waschvorgaenge:
        data = split_into_train_df(df_waschvorgang, config.n_features)
        df_tmp = pd.DataFrame(data, columns=cols_features+cols_label)
        df_train = df_train.append(df_tmp, ignore_index=True)
    print(df_train)
    print('Anz. Waschvorgänge:', len(df_waschvorgaenge))
